"""ISO/IEC 42001:2023 AI Management System — Annex A control mappings.

Pure data + helper functions — no I/O, no external dependencies.
Maps mltk assertion prefixes to ISO 42001 Annex A controls for
compliance gap analysis and coverage reporting.
"""

from __future__ import annotations

from mltk.core.assertion import assert_true, timed_assertion
from mltk.core.result import Severity, TestResult

# ---------------------------------------------------------------------------
# Annex A control definitions
# ---------------------------------------------------------------------------

ANNEX_A_CONTROLS: dict[str, dict] = {
    "A.2": {
        "title": "AI Policies",
        "description": (
            "Organization-level AI governance policies covering data handling, "
            "privacy, and acceptable use of AI systems (ISO 42001 A.2)."
        ),
        "assertions": ["data.pii", "data.schema"],
    },
    "A.4": {
        "title": "AI Risk Assessment",
        "description": (
            "Identification, analysis, and evaluation of risks arising from AI "
            "systems, including bias, adversarial threats, and calibration "
            "uncertainty (ISO 42001 A.4)."
        ),
        "assertions": ["model.bias", "model.adversarial", "model.calibration"],
    },
    "A.5": {
        "title": "Data Quality",
        "description": (
            "Data used for AI system development and operation must meet quality "
            "requirements including schema conformance, completeness, type "
            "correctness, distribution stability, timeliness, and privacy "
            "(ISO 42001 A.5)."
        ),
        "assertions": [
            "data.schema",
            "data.no_nulls",
            "data.dtypes",
            "data.drift",
            "data.freshness",
            "data.no_pii",
        ],
    },
    "A.6": {
        "title": "System Performance",
        "description": (
            "Ongoing monitoring and evaluation of AI system performance, "
            "including accuracy metrics, regression detection, subgroup analysis, "
            "latency, throughput, and SLA compliance (ISO 42001 A.6)."
        ),
        "assertions": [
            "model.metric",
            "model.regression",
            "model.slice",
            "inference.latency",
            "inference.throughput",
            "monitor.degradation",
            "monitor.sla",
        ],
    },
    "A.7": {
        "title": "Third Party",
        "description": (
            "Management of supply-chain and third-party AI components, ensuring "
            "integrity verification and reproducibility of external artifacts "
            "(ISO 42001 A.7)."
        ),
        "assertions": ["pipeline.checksum", "pipeline.reproducible"],
    },
    "A.8": {
        "title": "Documentation",
        "description": (
            "Maintenance of records and documentation for AI system lifecycle "
            "activities, decisions, and changes (ISO 42001 A.8)."
        ),
        "assertions": [],
    },
    "A.9": {
        "title": "Incident Response",
        "description": (
            "AI incident management processes for detecting, reporting, and "
            "responding to performance degradation and service-level breaches "
            "(ISO 42001 A.9)."
        ),
        "assertions": ["monitor.degradation", "monitor.sla"],
    },
    "A.10": {
        "title": "Bias and Fairness",
        "description": (
            "Non-discrimination and fairness controls ensuring AI systems do not "
            "produce biased outcomes across protected subgroups (ISO 42001 A.10)."
        ),
        "assertions": ["model.bias", "model.slice"],
    },
}

# Ordered list of control IDs for deterministic iteration and report rendering.
ANNEX_A_IDS: list[str] = list(ANNEX_A_CONTROLS.keys())


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _assertion_to_clause_ids(assertion_name: str) -> list[str]:
    """Return all Annex A clause IDs whose assertion prefixes match *assertion_name*.

    A match occurs when *assertion_name* starts with any prefix listed in
    ``ANNEX_A_CONTROLS[clause_id]["assertions"]``.

    Args:
        assertion_name: Fully-qualified assertion name, e.g. ``"model.bias.demographic_parity"``.

    Returns:
        List of matching clause IDs (may be empty; may contain more than one).
    """
    matched: list[str] = []
    for clause_id, meta in ANNEX_A_CONTROLS.items():
        for prefix in meta["assertions"]:
            if assertion_name.startswith(prefix):
                matched.append(clause_id)
                break  # one match per clause is enough
    return matched


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def map_results_to_clauses(results: list[dict]) -> dict[str, list]:
    """Group test results by ISO 42001 Annex A clause.

    Each result dict is expected to have at least:
    - ``name`` (str): assertion name, e.g. ``"model.bias.demographic_parity"``
    - ``passed`` (bool): whether the assertion passed
    - ``message`` (str, optional): human-readable message

    Results whose names do not match any known assertion prefix are placed in
    the special ``"uncategorised"`` bucket.

    Args:
        results: List of result dicts (from JSON or TestResult serialisation).

    Returns:
        Dict mapping clause strings (e.g. ``"A.5"``) to lists of result dicts.
        Each result dict is enriched with a ``"clause"`` key.

    Example:
        >>> r = [{"name": "model.bias.x", "passed": True, "message": "ok"}]
        >>> grouped = map_results_to_clauses(r)
        >>> "A.4" in grouped
        True
    """
    grouped: dict[str, list] = {}
    for r in results:
        name = str(r.get("name", ""))
        clause_ids = _assertion_to_clause_ids(name)
        if not clause_ids:
            enriched = dict(r)
            enriched["clause"] = "uncategorised"
            grouped.setdefault("uncategorised", []).append(enriched)
        else:
            for clause_id in clause_ids:
                enriched = dict(r)
                enriched["clause"] = clause_id
                grouped.setdefault(clause_id, []).append(enriched)
    return grouped


def find_gaps(results: list[dict]) -> list[str]:
    """Find ISO 42001 Annex A clauses not covered by any test results.

    A clause is considered *covered* if at least one result exists whose
    assertion name matches any of the clause's assertion prefixes.
    Clauses with empty assertion lists (e.g. A.8 Documentation) are always
    reported as gaps since they cannot be covered by automated tests.

    Args:
        results: List of result dicts (same format as :func:`map_results_to_clauses`).

    Returns:
        Sorted list of clause IDs that have no matching test results.

    Example:
        >>> gaps = find_gaps([])
        >>> "A.8" in gaps
        True
    """
    covered: set[str] = set()
    for r in results:
        name = str(r.get("name", ""))
        for clause_id in _assertion_to_clause_ids(name):
            covered.add(clause_id)
    all_clauses = set(ANNEX_A_IDS)
    return sorted(all_clauses - covered)


@timed_assertion
def assert_iso_42001_coverage(
    results: list[dict],
    min_coverage: float = 0.8,
) -> TestResult:
    """Assert minimum ISO 42001 Annex A coverage.

    Coverage is defined as:
    ``clauses_with_at_least_one_test / total_clauses``

    At the default threshold of 0.8, at least 6 of the 8 Annex A clauses
    must have corresponding test results.

    Note: A.8 (Documentation) has no mapped assertion prefixes and can only
    be covered through manual/process controls. It counts against coverage.

    Args:
        results: List of result dicts (same format as :func:`map_results_to_clauses`).
        min_coverage: Minimum fraction of Annex A clauses that must be
            covered (0.0-1.0, default 0.8).

    Returns:
        :class:`~mltk.core.result.TestResult` indicating pass/fail with
        ``covered_count``, ``total``, ``coverage``, and ``min_coverage``
        in ``details``.

    Raises:
        :class:`~mltk.core.assertion.MltkAssertionError`: If coverage is
            below *min_coverage*.

    Example:
        >>> r = [{"name": "model.bias.x", "passed": True, "message": "ok"}]
        >>> result = assert_iso_42001_coverage(r, min_coverage=0.1)
        >>> result.passed
        True
    """
    total = len(ANNEX_A_IDS)
    gaps = find_gaps(results)
    covered_count = total - len(gaps)
    coverage = covered_count / total if total > 0 else 0.0
    passed = coverage >= min_coverage

    message = (
        f"ISO 42001 Annex A coverage {coverage:.0%} "
        f"({covered_count}/{total} clauses) "
        + ("meets" if passed else "below")
        + f" minimum {min_coverage:.0%}"
    )

    return assert_true(
        passed,
        name="compliance.iso_42001.coverage",
        message=message,
        severity=Severity.CRITICAL,
        covered_count=covered_count,
        total=total,
        coverage=round(coverage, 4),
        min_coverage=min_coverage,
        gaps=gaps,
    )
