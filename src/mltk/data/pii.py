"""PII detection -- scan data for personally identifiable information.

ML models can memorize and reproduce PII from training data. GDPR fines
reach 4% of global revenue. This module ports battle-tested regex patterns
from ShrimPK's pii.rs to detect emails, phones, SSNs, credit cards,
API keys, and passwords in DataFrame text columns.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

from mltk.core.assertion import assert_true, timed_assertion
from mltk.core.result import Severity, TestResult


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
    "iban": re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{4}\d{7}[A-Z0-9]{0,16}\b"),
    "url_auth_token": re.compile(
        r"https?://[^\s]*[?&](?:token|key|api_key|access_token|auth)=[^\s&]+"
    ),
    "slack_webhook": re.compile(
        r"https://hooks\.slack\.com/services/T[a-zA-Z0-9_]{8,10}/B[a-zA-Z0-9_]{8,10}/[a-zA-Z0-9_]{24}"
    ),
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
        for match in pattern.finditer(text):
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
                    matched_text=match.group(),
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
