"""NER-based PII detection -- context-aware entity recognition for ML data.

WHY NER-BASED PII DETECTION MATTERS:

Regex patterns are excellent at catching structured PII -- credit card numbers,
SSNs, API keys, IBANs -- because these follow rigid, predictable formats.
But regex fundamentally cannot detect *contextual* PII:

  - Person names: "John Smith submitted the report" -- no regex can reliably
    distinguish a person's name from any other capitalized words.
  - Organizations: "Acme Corp processed the payment" -- company names are
    free-form and infinite.
  - Locations: "123 Main Street, Springfield" -- street addresses are too
    varied for regex to cover without massive false positive rates.
  - Medical record numbers: "MRN: A12345678" -- domain-specific identifiers
    vary by hospital system with no universal format.

Named Entity Recognition (NER) solves this by using language models trained
on millions of annotated documents to recognize entities by *context*, not
just by pattern. When the model sees "Dr. Sarah Chen treated the patient",
it understands "Sarah Chen" is a PERSON because of the surrounding context
("Dr.", "treated"), not because the name matches a regex.

This module provides three NER backends:

1. **Presidio** (Microsoft) -- production-grade PII detection built on spaCy
   NER models. Supports 17+ entity types out of the box. Best for general-
   purpose PII scanning with high precision.

2. **GLiNER** -- zero-shot NER that detects entities you *define at runtime*
   without fine-tuning. Specify ["medical record number", "patient id",
   "legal case number"] and it finds them. Best for domain-specific PII
   that Presidio's fixed entity list doesn't cover.

3. **Hybrid** -- runs BOTH regex (from pii.py) and Presidio NER, then
   merges results with intelligent deduplication. Regex handles structured
   patterns (API keys, credit cards); NER handles contextual entities
   (names, organizations, locations). Best overall coverage.

Sprint 73: Initial NER backend implementation.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from typing import Any

from mltk.data.pii import PiiMatch

__all__ = [
    "scan_pii_ner",
    "scan_pii_gliner",
    "scan_pii_hybrid",
    "DEFAULT_PRESIDIO_ENTITIES",
    "DEFAULT_GLINER_ENTITIES",
]

# -----------------------------------------------------------------------
# Entity type mapping: Presidio entity names -> mltk PiiMatch.type values
# -----------------------------------------------------------------------

_PRESIDIO_TO_MLTK: dict[str, str] = {
    "PERSON": "person_name",
    "LOCATION": "location",
    "PHONE_NUMBER": "phone",
    "EMAIL_ADDRESS": "email",
    "CREDIT_CARD": "credit_card",
    "CRYPTO": "crypto_address",
    "DATE_TIME": "date_time",
    "IBAN_CODE": "iban",
    "IP_ADDRESS": "ipv4",
    "MEDICAL_LICENSE": "medical_license",
    "NRP": "nationality",
    "US_BANK_NUMBER": "bank_account",
    "US_DRIVER_LICENSE": "driver_license",
    "US_ITIN": "tax_id",
    "US_PASSPORT": "passport",
    "US_SSN": "ssn",
    "ORGANIZATION": "organization",
}

# GLiNER uses free-form labels -- map common ones to mltk types.
_GLINER_TO_MLTK: dict[str, str] = {
    "person name": "person_name",
    "person": "person_name",
    "organization": "organization",
    "location": "location",
    "medical record number": "medical_record",
    "patient id": "patient_id",
    "date of birth": "date_of_birth",
    "social security number": "ssn",
    "phone number": "phone",
    "email": "email",
    "credit card": "credit_card",
    "address": "address",
}

# Maximum text length for NER scanning. Transformer models have O(n^2)
# attention complexity; texts longer than this are rejected with a clear
# error to prevent OOM on the server. 100K chars covers virtually all
# real-world PII scanning use cases (a 50-page document is ~25K chars).
_MAX_NER_TEXT_LENGTH: int = 100_000

# Pinned model revisions for supply-chain safety. GLiNER models are
# loaded from HuggingFace Hub -- pinning by commit SHA ensures a
# compromised model push doesn't affect users. Update these hashes
# when intentionally upgrading to a newer model version.
# SEC: These MUST be pinned to commit SHAs before production use.
# Run: `git ls-remote https://huggingface.co/urchade/gliner_medium-v2.1`
# to get the current HEAD SHA, then replace "main" below.
_GLINER_REVISIONS: dict[str, str] = {
    "urchade/gliner_medium-v2.1": "main",  # SECURITY: pin before release
    "urchade/gliner_large-v2.1": "main",   # SECURITY: pin before release
}

# Default entity types for each backend.
DEFAULT_PRESIDIO_ENTITIES: list[str] = [
    "PERSON",
    "ORGANIZATION",
    "LOCATION",
    "PHONE_NUMBER",
    "EMAIL_ADDRESS",
    "CREDIT_CARD",
    "CRYPTO",
    "DATE_TIME",
    "IBAN_CODE",
    "IP_ADDRESS",
    "MEDICAL_LICENSE",
    "NRP",
    "US_BANK_NUMBER",
    "US_DRIVER_LICENSE",
    "US_ITIN",
    "US_PASSPORT",
    "US_SSN",
]

DEFAULT_GLINER_ENTITIES: list[str] = [
    "person name",
    "organization",
    "location",
    "medical record number",
    "patient id",
    "date of birth",
    "social security number",
]


# -----------------------------------------------------------------------
# Private helpers -- lazy-load expensive NLP models
# -----------------------------------------------------------------------


@lru_cache(maxsize=1)
def _get_presidio_analyzer() -> Any:
    """Create and cache the Presidio AnalyzerEngine with spaCy backend.

    Presidio's AnalyzerEngine is expensive to initialize (~2-5 seconds on
    first call) because it loads the spaCy NLP model (en_core_web_lg) and
    registers all built-in recognizers. We cache it so subsequent calls
    within the same process are instant.

    Returns:
        A configured ``presidio_analyzer.AnalyzerEngine`` instance.

    Raises:
        ImportError: If presidio-analyzer or spaCy model is not installed.
            The error message includes the exact pip command to fix it.

    Example:
        >>> analyzer = _get_presidio_analyzer()
        >>> results = analyzer.analyze(text="John Smith", language="en",
        ...                            entities=["PERSON"])
    """
    try:
        from presidio_analyzer import AnalyzerEngine
    except ImportError as exc:
        raise ImportError(
            "Presidio is required for NER-based PII detection. "
            "Install it with: pip install mltk[ner]  "
            "(or: pip install presidio-analyzer spacy && "
            "python -m spacy download en_core_web_lg)"
        ) from exc

    return AnalyzerEngine()


@lru_cache(maxsize=4)
def _get_gliner_model(model_name: str) -> Any:
    """Load and cache a GLiNER zero-shot NER model.

    GLiNER models are ~500MB and take 5-15 seconds to load from HuggingFace
    Hub on first use. We cache by model_name so switching between models
    (e.g., medium vs. large) keeps both in memory. maxsize=4 allows up to
    4 different models cached simultaneously.

    Args:
        model_name: HuggingFace model identifier, e.g.,
            ``"urchade/gliner_medium-v2.1"``.

    Returns:
        A loaded ``gliner.GLiNER`` model instance.

    Raises:
        ImportError: If the gliner package is not installed.

    Example:
        >>> model = _get_gliner_model("urchade/gliner_medium-v2.1")
        >>> entities = model.predict_entities("John Smith", ["person name"])
    """
    try:
        from gliner import GLiNER
    except ImportError as exc:
        raise ImportError(
            "GLiNER is required for zero-shot NER PII detection. "
            "Install it with: pip install mltk[gliner]  "
            "(or: pip install gliner)"
        ) from exc

    revision = _GLINER_REVISIONS.get(model_name)
    if revision and revision != "main":
        return GLiNER.from_pretrained(
            model_name, revision=revision
        )
    if revision == "main":
        import warnings

        warnings.warn(
            f"GLiNER model '{model_name}' is not pinned to a "
            f"specific revision (using 'main'). Pin to a commit "
            f"SHA in _GLINER_REVISIONS before production use to "
            f"prevent supply-chain attacks.",
            UserWarning,
            stacklevel=2,
        )
    return GLiNER.from_pretrained(model_name)


# -----------------------------------------------------------------------
# Public API -- NER scanning functions
# -----------------------------------------------------------------------


def scan_pii_ner(
    text: str,
    entity_types: list[str] | None = None,
    score_threshold: float = 0.5,
    language: str = "en",
) -> list[PiiMatch]:
    """Scan text for PII using Microsoft Presidio (spaCy NER engine).

    Presidio combines rule-based recognizers with spaCy's NER model to
    detect entities that regex alone cannot find -- person names,
    organizations, locations, and other contextual PII. Each detected
    entity includes a confidence score (0.0-1.0) so you can tune the
    sensitivity vs. false-positive trade-off.

    Args:
        text: The text string to scan for PII entities.
        entity_types: Presidio entity type names to detect (e.g.,
            ``["PERSON", "LOCATION"]``). ``None`` uses the full default
            list of 16 entity types.
        score_threshold: Minimum confidence score (0.0-1.0) to include
            a detection. Presidio recommends 0.5 as the minimum; raise
            to 0.7+ for fewer false positives. Default: 0.5.
        language: ISO 639-1 language code for the spaCy model. Default:
            ``"en"`` (English). Must match an installed spaCy model.

    Returns:
        List of ``PiiMatch`` objects, one per detected entity. The
        ``PiiMatch.type`` field uses mltk's normalized type names
        (e.g., ``"person_name"`` not ``"PERSON"``).

    Example:
        >>> from mltk.data.pii_ner import scan_pii_ner
        >>> matches = scan_pii_ner("Dr. Sarah Chen lives in Boston")
        >>> [m.type for m in matches]
        ['person_name', 'location']
    """
    if not text or not text.strip():
        return []
    if len(text) > _MAX_NER_TEXT_LENGTH:
        raise ValueError(
            f"Text length {len(text):,} exceeds maximum "
            f"{_MAX_NER_TEXT_LENGTH:,} chars for NER scanning. "
            f"Split into smaller chunks first."
        )

    analyzer = _get_presidio_analyzer()
    entities = entity_types or DEFAULT_PRESIDIO_ENTITIES

    results = analyzer.analyze(
        text=text,
        entities=entities,
        language=language,
        score_threshold=score_threshold,
    )

    matches: list[PiiMatch] = []
    for r in results:
        mltk_type = _PRESIDIO_TO_MLTK.get(
            r.entity_type, r.entity_type.lower()
        )
        matches.append(
            PiiMatch(
                type=mltk_type,
                start=r.start,
                end=r.end,
                matched_text=text[r.start : r.end],
            )
        )

    return matches


def scan_pii_gliner(
    text: str,
    entity_types: list[str] | None = None,
    score_threshold: float = 0.7,
    model_name: str = "urchade/gliner_medium-v2.1",
) -> list[PiiMatch]:
    """Scan text for PII using GLiNER zero-shot NER.

    GLiNER is a generalist model for Named Entity Recognition that works
    in zero-shot mode -- you pass the entity labels you want to detect
    at inference time, and it finds them without any fine-tuning. This
    makes it ideal for domain-specific PII that Presidio's fixed entity
    list doesn't cover:

      - Healthcare: "medical record number", "patient id", "diagnosis"
      - Legal: "case number", "court name", "attorney name"
      - Finance: "account number", "routing number", "portfolio id"

    The trade-off: GLiNER is slower than Presidio (~100ms vs ~10ms per
    text) and has lower precision on standard entities. Use Presidio for
    general PII; use GLiNER for domain-specific extensions.

    Args:
        text: The text string to scan for PII entities.
        entity_types: Free-form entity labels to detect (e.g.,
            ``["medical record number", "patient id"]``). These are
            natural language descriptions, not fixed codes. ``None``
            uses the default list of 7 common PII types.
        score_threshold: Minimum confidence score (0.0-1.0). GLiNER
            scores tend lower than Presidio, so the default is 0.7
            (stricter) to reduce false positives. Default: 0.7.
        model_name: HuggingFace model identifier. Default:
            ``"urchade/gliner_medium-v2.1"`` (good balance of speed
            and accuracy). Alternative: ``"urchade/gliner_large-v2.1"``
            for higher accuracy at ~2x latency.

    Returns:
        List of ``PiiMatch`` objects. The ``PiiMatch.type`` field is
        mapped from GLiNER's label to mltk's normalized type names
        where possible (e.g., ``"person name"`` -> ``"person_name"``);
        unknown labels are lowercased with spaces replaced by
        underscores.

    Example:
        >>> from mltk.data.pii_ner import scan_pii_gliner
        >>> matches = scan_pii_gliner(
        ...     "Patient MRN: A12345678",
        ...     entity_types=["medical record number"],
        ... )
        >>> matches[0].type
        'medical_record'
    """
    if not text or not text.strip():
        return []
    if len(text) > _MAX_NER_TEXT_LENGTH:
        raise ValueError(
            f"Text length {len(text):,} exceeds maximum "
            f"{_MAX_NER_TEXT_LENGTH:,} chars for NER scanning. "
            f"Split into smaller chunks first."
        )

    model = _get_gliner_model(model_name)
    labels = entity_types or DEFAULT_GLINER_ENTITIES

    entities = model.predict_entities(
        text, labels, threshold=score_threshold
    )

    matches: list[PiiMatch] = []
    for ent in entities:
        label = ent.get("label", ent.get("type", "unknown"))
        label_lower = label.lower()
        mltk_type = _GLINER_TO_MLTK.get(
            label_lower, label_lower.replace(" ", "_")
        )
        start = ent.get("start", 0)
        end = ent.get("end", 0)
        matched_text = ent.get("text", text[start:end])
        matches.append(
            PiiMatch(
                type=mltk_type,
                start=start,
                end=end,
                matched_text=matched_text,
            )
        )

    return matches


def scan_pii_hybrid(
    text: str,
    patterns: list[str] | None = None,
    entity_types: list[str] | None = None,
    score_threshold: float = 0.5,
    allowlist: list[str] | None = None,
) -> list[PiiMatch]:
    """Scan text with BOTH regex and NER, then merge results.

    The hybrid approach gives the best of both worlds:

      - **Regex** (from ``pii.scan_pii``) excels at structured patterns:
        API keys (``sk-proj-...``), credit cards (``4111-1111-...``),
        SSNs (``123-45-6789``), JWTs, IBANs, and other format-defined
        PII. These have near-zero false positive rates because the
        patterns are highly specific.

      - **NER** (from ``scan_pii_ner``) excels at contextual entities:
        person names, organizations, locations, and other free-form PII
        that regex fundamentally cannot detect.

    When both methods detect the same span of text (e.g., both regex
    and NER find an email address), the deduplication logic keeps the
    match with higher priority:
      - NER wins for names, organizations, locations (regex can't
        reliably detect these).
      - Regex wins for structured patterns like API keys and credit
        cards (higher precision for fixed formats).

    Overlap is determined by character span intersection: if two matches
    share any characters, they are considered duplicates.

    Args:
        text: The text string to scan.
        patterns: Regex pattern categories to check (passed to
            ``scan_pii``). ``None`` = all regex patterns.
        entity_types: Presidio entity types to detect. ``None`` = all
            default NER entity types.
        score_threshold: Minimum NER confidence score. Default: 0.5.
        allowlist: Exact strings to suppress (passed to ``scan_pii``
            for regex matches).

    Returns:
        Unified list of ``PiiMatch`` objects from both regex and NER,
        deduplicated by span overlap.

    Example:
        >>> from mltk.data.pii_ner import scan_pii_hybrid
        >>> text = "John Smith's key: sk-proj-abc123def456ghi789jkl"
        >>> matches = scan_pii_hybrid(text)
        >>> sorted(m.type for m in matches)
        ['api_key', 'person_name']
    """
    from mltk.data.pii import scan_pii

    regex_matches = scan_pii(
        text, patterns=patterns, allowlist=allowlist
    )
    ner_matches = scan_pii_ner(
        text,
        entity_types=entity_types,
        score_threshold=score_threshold,
    )

    return _merge_matches(regex_matches, ner_matches)


# -----------------------------------------------------------------------
# NER-preferred entity types for deduplication
# -----------------------------------------------------------------------

# These entity types are better detected by NER than regex. When both
# methods detect an overlapping span, NER's result is preferred for
# these types because regex tends to produce false positives on
# contextual entities.
_NER_PREFERRED_TYPES: frozenset[str] = frozenset({
    "person_name",
    "organization",
    "location",
    "nationality",
    "date_time",
    "medical_license",
})


def _spans_overlap(a: PiiMatch, b: PiiMatch) -> bool:
    """Check whether two PiiMatch spans share any characters.

    Two spans [a.start, a.end) and [b.start, b.end) overlap if and
    only if one starts before the other ends.

    Args:
        a: First PII match.
        b: Second PII match.

    Returns:
        True if the spans share at least one character position.
    """
    return a.start < b.end and b.start < a.end


def _merge_matches(
    regex_matches: Sequence[PiiMatch],
    ner_matches: Sequence[PiiMatch],
) -> list[PiiMatch]:
    """Merge regex and NER matches, deduplicating overlapping spans.

    Deduplication strategy:
      1. Start with all NER matches.
      2. For each regex match, check if it overlaps with any NER match.
      3. If overlap exists AND the NER match type is in the NER-preferred
         set, keep the NER match and discard the regex match.
      4. If overlap exists but the NER type is NOT NER-preferred (e.g.,
         both found an email), keep the regex match (higher precision
         for structured patterns).
      5. If no overlap, add the regex match.

    Args:
        regex_matches: Matches from regex-based scanning.
        ner_matches: Matches from NER-based scanning.

    Returns:
        Deduplicated list of PiiMatch objects.
    """
    # Index-based tracking: use list indices (not id()) to avoid
    # equality/identity mismatch when PiiMatch objects have identical
    # field values. consumed_indices tracks NER matches already
    # replaced by regex so subsequent regex matches skip them.
    ner_list = list(ner_matches)
    merged: list[PiiMatch] = list(ner_list)
    consumed_indices: set[int] = set()

    for rm in regex_matches:
        overlapping = [
            (i, nm) for i, nm in enumerate(ner_list)
            if _spans_overlap(rm, nm)
            and i not in consumed_indices
        ]

        if not overlapping:
            merged.append(rm)
        else:
            ner_wins = any(
                nm.type in _NER_PREFERRED_TYPES
                for _, nm in overlapping
            )
            if not ner_wins:
                for idx, nm in overlapping:
                    try:
                        merged.remove(nm)
                    except ValueError:
                        pass
                    consumed_indices.add(idx)
                merged.append(rm)

    return merged
