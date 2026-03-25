"""PII detection -- scan data for personally identifiable information.

ML models can memorize and reproduce PII from training data. GDPR fines
reach 4% of global revenue. This module ports battle-tested regex patterns
from ShrimPK's pii.rs to detect emails, phones, SSNs, credit cards,
API keys, and passwords in DataFrame text columns.

Sprint 18: Israel PII patterns added -- Teudat Zehut (national ID with
Luhn checksum), Israel phone numbers, and IBAN with MOD-97 validation.

Sprint 23: Tier 3 international PII patterns added -- UK NHS Number (MOD-11),
UK National Insurance Number (NINO), Germany Steuer-ID (structural),
India Aadhaar (Verhoeff checksum), and India PAN (format).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

from mltk.core.assertion import assert_true, timed_assertion
from mltk.core.result import Severity, TestResult

# ---------------------------------------------------------------------------
# Checksum validators
# ---------------------------------------------------------------------------


def _validate_tz(digits: str) -> bool:
    """Validate an Israeli Teudat Zehut (national ID) via its Luhn variant.

    The Israeli ID checksum algorithm:
      1. Multiply alternating digits by 1 and 2 (starting at index 0 with 1).
      2. If a product is >= 10, sum its two digits (e.g., 14 -> 1+4 = 5).
      3. Sum all processed values; result must be divisible by 10.

    Args:
        digits: Exactly 9-digit string (no dashes).

    Returns:
        True if the checksum is valid, False otherwise.
    """
    if len(digits) != 9 or not digits.isdigit():
        return False
    total = 0
    for i, ch in enumerate(digits):
        val = int(ch) * (1 if i % 2 == 0 else 2)
        if val >= 10:
            val = val // 10 + val % 10
        total += val
    return total % 10 == 0


def _validate_tz_match(text: str) -> bool:
    """Validate a TZ match string (strips optional dashes before checking)."""
    digits = text.replace("-", "")
    return _validate_tz(digits)


def _validate_iban(text: str) -> bool:
    """Validate an IBAN string using the MOD-97 algorithm (ISO 13616).

    Algorithm:
      1. Remove spaces; ensure uppercase.
      2. Move the first 4 characters to the end.
      3. Replace each letter with its numeric value (A=10 .. Z=35).
      4. Compute the resulting integer mod 97; result must equal 1.

    Args:
        text: Raw IBAN string (may contain spaces).

    Returns:
        True if MOD-97 checksum passes, False otherwise.
    """
    iban = text.replace(" ", "").upper()
    if len(iban) < 4:
        return False
    rearranged = iban[4:] + iban[:4]
    numeric_str = "".join(
        str(ord(ch) - ord("A") + 10) if ch.isalpha() else ch
        for ch in rearranged
    )
    try:
        return int(numeric_str) % 97 == 1
    except ValueError:
        return False


def _validate_nhs(text: str) -> bool:
    """Validate a UK NHS Number using the MOD-11 algorithm.

    The NHS Number is a 10-digit identifier issued to every patient registered
    with the National Health Service in England, Wales, and the Isle of Man.

    Algorithm (NHS Digital specification):
      1. Strip spaces to get exactly 10 digits.
      2. Multiply the first 9 digits by weights 10, 9, 8, 7, 6, 5, 4, 3, 2.
      3. Sum the products.
      4. Compute remainder = sum mod 11.
      5. Check digit = 11 - remainder.
      6. If check digit == 11, check digit is treated as 0.
      7. If check digit == 10, the number is invalid (no valid NHS number exists).
      8. Check digit must equal the 10th digit.

    Args:
        text: Raw NHS Number string (may contain spaces).

    Returns:
        True if the MOD-11 checksum passes, False otherwise.
    """
    digits = text.replace(" ", "")
    if len(digits) != 10 or not digits.isdigit():
        return False
    weights = [10, 9, 8, 7, 6, 5, 4, 3, 2]
    total = sum(int(d) * w for d, w in zip(digits[:9], weights, strict=False))
    remainder = total % 11
    check = 11 - remainder
    if check == 11:
        check = 0
    if check == 10:
        return False
    return check == int(digits[9])


def _validate_steuerid(text: str) -> bool:
    """Validate a German Steuer-Identifikationsnummer (tax ID) structurally.

    The German tax ID (Steuer-ID / IdNr) has 11 digits. The structural rules
    apply to positions 1-10 (the first 10 digits); position 11 is a check digit
    computed separately (ISO 7064 MOD-11,10) and is not constrained structurally.

    Rules applied here (Bundeszentralamt fuer Steuern specification):
      1. First digit must not be 0.
      2. Among positions 1-10, exactly one digit value (0-9) appears more than once
         (i.e., exactly one digit is duplicated in those first 10 positions).
      3. Among positions 1-10, exactly one digit value (0-9) does not appear at all.
      (Rules 2+3 together: 9 distinct values across 10 positions, one repeated once,
       one absent -- 2 + 8*1 + 0 = 10 positions with 9 distinct values.)

    Note: Full Elster check digit verification (ISO 7064 MOD-11,10) requires the
    actual check digit formula; these structural rules are the publicly documented
    format constraints sufficient for PII detection (high specificity, low FPR).

    Args:
        text: Raw Steuer-ID string (11 digits, no spaces or dashes).

    Returns:
        True if the structural constraints pass, False otherwise.
    """
    s = text.strip()
    if len(s) != 11 or not s.isdigit():
        return False
    # Rule 1: first digit must not be 0
    if s[0] == "0":
        return False
    # Rules 2 & 3: applied to the first 10 digits (positions 1-10)
    # Position 11 (s[10]) is the check digit -- not structurally constrained here.
    body = s[:10]
    counts = [0] * 10
    for ch in body:
        counts[int(ch)] += 1
    # Rule 2: exactly one digit appears more than once in positions 1-10
    duplicated = sum(1 for c in counts if c > 1)
    if duplicated != 1:
        return False
    # Rule 3: exactly one digit does not appear at all in positions 1-10
    absent = sum(1 for c in counts if c == 0)
    if absent != 1:
        return False
    return True


# Verhoeff algorithm tables (standard, per Verhoeff 1969).
#
# D is the Cayley table of the dihedral group D5 (order 10).
# P is the permutation table derived from the cyclic permutation
#   sigma = (0 1 5 8 9 4 3 7 6 2), applied 0..7 times to the identity.
#   P[i][j] = sigma^i(j).  Computed analytically from sigma.
_VERHOEFF_D = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
    [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
    [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
    [4, 0, 1, 2, 3, 9, 5, 6, 7, 8],
    [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
    [6, 5, 9, 8, 7, 1, 0, 4, 3, 2],
    [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
    [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
    [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
]

_VERHOEFF_P = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],  # sigma^0 = identity
    [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],  # sigma^1
    [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],  # sigma^2
    [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],  # sigma^3
    [9, 4, 5, 3, 1, 2, 6, 8, 7, 0],  # sigma^4
    [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],  # sigma^5
    [2, 7, 9, 3, 8, 0, 6, 4, 1, 5],  # sigma^6
    [7, 0, 4, 6, 9, 1, 3, 2, 5, 8],  # sigma^7
]


def _validate_aadhaar(text: str) -> bool:
    """Validate an India Aadhaar number using the Verhoeff checksum algorithm.

    Aadhaar is a 12-digit Unique Identification number issued by the Unique
    Identification Authority of India (UIDAI). The last digit is a Verhoeff
    check digit computed over the first 11 digits.

    Algorithm:
      1. Strip spaces to obtain 12 digits.
      2. Process digits right-to-left with position index i starting at 0.
      3. c = 0; for each digit: c = D[c][P[i % 8][digit]]; i++.
      4. Result c must equal 0 for a valid number.

    The first digit of an Aadhaar number is never 0 or 1 (UIDAI policy).

    Args:
        text: Raw Aadhaar string (may contain spaces).

    Returns:
        True if the Verhoeff checksum passes and format is valid, False otherwise.
    """
    digits = text.replace(" ", "")
    if len(digits) != 12 or not digits.isdigit():
        return False
    # UIDAI: first digit must be 2-9
    if digits[0] in ("0", "1"):
        return False
    c = 0
    for i, ch in enumerate(reversed(digits)):
        c = _VERHOEFF_D[c][_VERHOEFF_P[i % 8][int(ch)]]
    return c == 0


@dataclass
class PiiMatch:
    """A single PII detection result."""

    type: str
    start: int
    end: int
    matched_text: str


# Patterns ported from ShrimPK pii.rs (battle-tested, production-proven)
_PII_PATTERNS: dict[str, re.Pattern[str]] = {
    "api_key_openai_project": re.compile(r"sk-proj-[a-zA-Z0-9_-]{20,}"),
    "api_key_anthropic": re.compile(r"sk-ant-[a-zA-Z0-9_-]{20,}"),
    "api_key_openai": re.compile(r"sk-[a-zA-Z0-9_-]{20,}"),
    "api_key_stripe": re.compile(r"pk_[a-zA-Z0-9_-]{20,}"),
    "api_key_aws": re.compile(r"AKIA[A-Z0-9]{16}"),
    "api_key_groq": re.compile(r"gsk_[a-zA-Z0-9]{20,}"),
    "api_key_xai": re.compile(r"xai-[a-zA-Z0-9]{20,}"),
    "api_key_github": re.compile(r"ghp_[a-zA-Z0-9]{36}"),
    "api_key_gitlab": re.compile(r"glpat-[a-zA-Z0-9_-]{20,}"),
    "credit_card": re.compile(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "email": re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),
    "phone": re.compile(r"\b\d{3}[\s.\-]?\d{3}[\s.\-]?\d{4}\b"),
    "password": re.compile(r"(?i)(?:password|pwd|passwd|pass)\s*[:=]\s*\S+"),
    # Sprint 13: Tier 1 expansion (10 new patterns)
    "ipv4": re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"),
    "jwt": re.compile(r"eyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"),
    "pem_private_key": re.compile(r"-----BEGIN (?:RSA |DSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "db_connection_string": re.compile(
        r"(?:postgres|mysql|mongodb|redis|amqp)://[^:]+:[^@]+@[^\s]+"
    ),
    "api_key_stripe_secret": re.compile(r"(?:sk_live|sk_test)_[0-9a-zA-Z]{24,}"),
    "bearer_token": re.compile(r"Bearer\s+[a-zA-Z0-9\-._~+/]{20,}=*"),
    "api_key_google": re.compile(r"AIza[0-9A-Za-z_\-]{35}"),
    # IBAN -- regex catches the shape; MOD-97 validation is applied in scan_pii
    "iban": re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{4,30}[A-Z0-9]\b"),
    "url_auth_token": re.compile(
        r"https?://[^\s]*[?&](?:token|key|api_key|access_token|auth)=[^\s&]+"
    ),
    "slack_webhook": re.compile(
        r"https://hooks\.slack\.com/services/T[a-zA-Z0-9_]{8,10}/B[a-zA-Z0-9_]{8,10}/[a-zA-Z0-9_]{24}"
    ),
    # Sprint 18: Israel-specific PII patterns
    # Israeli national ID (Teudat Zehut) -- 9 digits, optional dashes.
    # Checksum validation (_validate_tz_match) is applied in scan_pii to
    # eliminate false positives (plain 9-digit numbers that fail Luhn).
    "israel_tz": re.compile(r"\b\d{3}-?\d{3}-?\d{3}\b"),
    # Israeli mobile numbers: 05X-XXX-XXXX (10 digits).
    # Israeli landline numbers: 0[2-9]-XXX-XXXX / 0[2-9]X-XXX-XXXX (9-10 digits).
    # Combined into one pattern; the mobile prefix (05) is the most specific.
    "israel_phone": re.compile(
        r"\b0(?:5\d-?\d{3}-?\d{4}|[2-9]\d?-?\d{3}-?\d{4})\b"
    ),
    # Sprint 23: Tier 3 international PII patterns
    # UK NHS Number -- 10 digits, optional spaces after 3rd and 6th digit.
    # MOD-11 checksum validation (_validate_nhs) applied in scan_pii.
    "uk_nhs": re.compile(r"\b\d{3}\s?\d{3}\s?\d{4}\b"),
    # UK National Insurance Number (NINO) -- 2 letters + 6 digits + 1 letter.
    # First two letters exclude certain characters per HMRC rules.
    # No checksum; format validation is sufficient.
    "uk_nino": re.compile(
        r"\b[A-CEGHJ-PR-TW-Z]{2}\s?\d{2}\s?\d{2}\s?\d{2}\s?[A-D]\b"
    ),
    # Germany Steuer-Identifikationsnummer (tax ID) -- 11 digits.
    # Structural validator (_validate_steuerid) applied in scan_pii.
    "de_steuerid": re.compile(r"\b\d{11}\b"),
    # India Aadhaar -- 12 digits, optional spaces after 4th and 8th digit.
    # Verhoeff checksum validation (_validate_aadhaar) applied in scan_pii.
    "in_aadhaar": re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b"),
    # India Permanent Account Number (PAN) -- 5 letters + 4 digits + 1 letter.
    # No checksum; format is sufficiently specific to avoid false positives.
    "in_pan": re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"),
}

# Checksum validators keyed by pattern name.
# scan_pii applies these after a regex match to filter out false positives.
# Only matches that pass the validator are included in results.
_VALIDATORS: dict[str, object] = {
    "israel_tz": _validate_tz_match,
    "iban": _validate_iban,
    "uk_nhs": _validate_nhs,
    "de_steuerid": _validate_steuerid,
    "in_aadhaar": _validate_aadhaar,
}

# Group API key patterns under a single category for filtering
_PATTERN_CATEGORIES: dict[str, list[str]] = {
    "api_key": [k for k in _PII_PATTERNS if k.startswith("api_key")],
    "credit_card": ["credit_card"],
    "ssn": ["ssn"],
    "email": ["email"],
    "phone": ["phone"],
    "password": ["password"],
    "ipv4": ["ipv4"],
    "jwt": ["jwt"],
    "pem_private_key": ["pem_private_key"],
    "db_connection_string": ["db_connection_string"],
    "bearer_token": ["bearer_token"],
    "iban": ["iban"],
    "url_auth_token": ["url_auth_token"],
    "slack_webhook": ["slack_webhook"],
    "israel_tz": ["israel_tz"],
    "israel_phone": ["israel_phone"],
    # Sprint 23 Tier 3
    "uk_nhs": ["uk_nhs"],
    "uk_nino": ["uk_nino"],
    "de_steuerid": ["de_steuerid"],
    "in_aadhaar": ["in_aadhaar"],
    "in_pan": ["in_pan"],
}


def scan_pii(
    text: str,
    patterns: list[str] | None = None,
) -> list[PiiMatch]:
    """Scan text for PII patterns.

    Args:
        text: Text to scan.
        patterns: Pattern categories to check (e.g., ["email", "phone"]).
            None = all patterns.

    Returns:
        List of PiiMatch objects with type, position, and matched text.

    Example:
        >>> matches = scan_pii("Contact john@example.com or 555-123-4567")
        >>> [m.type for m in matches]
        ['email', 'phone']
    """
    matches: list[PiiMatch] = []

    # Determine which patterns to run
    if patterns is not None:
        active_names: set[str] = set()
        for p in patterns:
            if p in _PATTERN_CATEGORIES:
                active_names.update(_PATTERN_CATEGORIES[p])
            elif p in _PII_PATTERNS:
                active_names.add(p)
    else:
        active_names = set(_PII_PATTERNS.keys())

    for name in sorted(active_names):
        pattern = _PII_PATTERNS.get(name)
        if pattern is None:
            continue
        validator = _VALIDATORS.get(name)
        for match in pattern.finditer(text):
            matched = match.group()
            # Apply checksum validator when present; skip invalid matches to
            # reduce false positives (e.g., random 9-digit numbers, malformed IBANs).
            if validator is not None and not validator(matched):  # type: ignore[operator]
                continue

            # Determine the user-facing type (collapse api_key subtypes)
            display_type = name
            for category, members in _PATTERN_CATEGORIES.items():
                if name in members and category != name:
                    display_type = category
                    break

            matches.append(
                PiiMatch(
                    type=display_type,
                    start=match.start(),
                    end=match.end(),
                    matched_text=matched,
                )
            )

    return matches


@timed_assertion
def assert_no_pii(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    patterns: list[str] | None = None,
) -> TestResult:
    """Assert no PII detected in DataFrame text columns.

    Args:
        df: DataFrame to scan.
        columns: Columns to scan. None = all object/string columns.
        patterns: Pattern categories to check. None = all.

    Returns:
        TestResult with match details per column and type.

    Example:
        >>> assert_no_pii(df, columns=["feedback_text", "notes"])
    """
    if columns is None:
        columns = [
            col
            for col in df.columns
            if pd.api.types.is_string_dtype(df[col])
            or pd.api.types.is_object_dtype(df[col])
        ]

    total_matches = 0
    matches_by_column: dict[str, int] = {}
    matches_by_type: dict[str, int] = {}

    for col in columns:
        col_matches = 0
        for value in df[col].dropna().astype(str):
            found = scan_pii(str(value), patterns=patterns)
            col_matches += len(found)
            for m in found:
                matches_by_type[m.type] = matches_by_type.get(m.type, 0) + 1

        if col_matches > 0:
            matches_by_column[col] = col_matches
        total_matches += col_matches

    passed = total_matches == 0
    message = (
        f"No PII detected in {len(columns)} column(s)"
        if passed
        else f"{total_matches} PII match(es) in columns: {list(matches_by_column.keys())}"
    )

    return assert_true(
        passed,
        name="data.pii",
        message=message,
        severity=Severity.CRITICAL,
        total_matches=total_matches,
        matches_by_column=matches_by_column,
        matches_by_type=matches_by_type,
        columns_scanned=columns,
    )
