"""Federal Reserve SR 11-7 model risk management mapping.

Pure data + helper functions -- no I/O, no external dependencies.

Maps mltk assertion name prefixes to the three SR 11-7 pillars:
Model Development, Model Validation, Ongoing Monitoring & Governance.

Reference: Federal Reserve SR 11-7, Guidance on Model Risk Management
(April 2011).
"""

from __future__ import annotations

from mltk.core.assertion import assert_true, timed_assertion
from mltk.core.result import Severity, TestResult

# -------------------------------------------------------------------
# SR 11-7 section definitions
# -------------------------------------------------------------------
# Keys are section identifiers.
# "assertions" lists assertion-name prefixes (matched via startswith,
# same convention as nist_ai_rmf and eu_ai_act).

SR_11_7_SECTIONS: dict[str, dict] = {
    "development": {
        "title": "Model Development",
        "description": (
            "Sound model development including theory, "
            "methodology, and testing."
        ),
        "assertions": [
            "model.metric",
            "model.regression",
            "model.calibration",
            "model.adversarial",
            "model.bias",
            "model.overfitting",
            "data.schema",
            "data.drift",
            "data.no_nulls",
            "training.no_target_leakage",
            "training.no_train_test_overlap",
        ],
    },
    "validation": {
        "title": "Model Validation",
        "description": (
            "Independent review of model performance "
            "and limitations."
        ),
        "assertions": [
            "model.metric",
            "model.slice",
            "model.calibration",
            "model.counterfactual",
            "model.causal",
            "model.attribution",
            "model.conformal",
            "data.synthetic",
        ],
    },
    "governance": {
        "title": "Ongoing Monitoring & Governance",
        "description": (
            "Ongoing monitoring, outcomes analysis, "
            "and model inventory."
        ),
        "assertions": [
            "monitor.degradation",
            "monitor.sla",
            "monitor.streaming_drift",
            "monitor.concept_drift",
            "data.drift",
            "data.freshness",
            "inference.latency",
            "inference.throughput",
        ],
    },
}

# Ordered list of section IDs for deterministic iteration.
SR_11_7_SECTION_IDS: list[str] = list(SR_11_7_SECTIONS.keys())

# Canonical section metadata for rendering.
SECTION_META: list[dict] = [
    {
        "section": sec_id,
        "title": meta["title"],
        "description": meta["description"],
    }
    for sec_id, meta in SR_11_7_SECTIONS.items()
]

# -------------------------------------------------------------------
# Compliance-level classification
# -------------------------------------------------------------------

COMPLIANCE_LEVELS = [
    "non_compliant",
    "minimal",
    "partial",
    "compliant",
]

COMPLIANCE_CLASSIFICATION: dict[str, dict] = {
    "non_compliant": {
        "label": "Non-Compliant",
        "description": (
            "No SR 11-7 pillars are covered. Model risk "
            "management practices are absent."
        ),
        "color": "#f85149",
        "badge_class": "fail",
    },
    "minimal": {
        "label": "Minimal",
        "description": (
            "At least one SR 11-7 pillar is covered. "
            "Significant gaps remain in model risk "
            "management."
        ),
        "color": "#d29922",
        "badge_class": "warn",
    },
    "partial": {
        "label": "Partial",
        "description": (
            "Two of three SR 11-7 pillars are covered. "
            "Some model risk management practices are "
            "in place but not comprehensive."
        ),
        "color": "#58a6ff",
        "badge_class": "info",
    },
    "compliant": {
        "label": "Compliant",
        "description": (
            "All three SR 11-7 pillars are covered. "
            "Model risk management practices address "
            "development, validation, and governance."
        ),
        "color": "#3fb950",
        "badge_class": "pass",
    },
}


# -------------------------------------------------------------------
# Internal helpers
# -------------------------------------------------------------------


def _assertion_to_section_ids(
    assertion_name: str,
) -> list[str]:
    """Return section IDs whose prefixes match *assertion_name*.

    A match occurs when *assertion_name* starts with any prefix
    listed in ``SR_11_7_SECTIONS[sec_id]["assertions"]``.

    Args:
        assertion_name: Fully-qualified assertion name,
            e.g. ``"model.bias.demographic_parity"``.

    Returns:
        List of matching section IDs (may be empty).
    """
    matched: list[str] = []
    for sec_id, meta in SR_11_7_SECTIONS.items():
        for prefix in meta["assertions"]:
            if assertion_name.startswith(prefix):
                matched.append(sec_id)
                break  # one match per section is enough
    return matched


# -------------------------------------------------------------------
# Public API
# -------------------------------------------------------------------


def classify_compliance(coverage: float) -> str:
    """Classify SR 11-7 compliance level based on coverage.

    The level is determined by the fraction of sections covered:

    - ``< 0.34`` -- Non-Compliant
    - ``< 0.67`` -- Minimal
    - ``< 1.0``  -- Partial
    - ``>= 1.0`` -- Compliant

    Args:
        coverage: Coverage ratio between 0.0 and 1.0.

    Returns:
        Compliance level key string.

    Example:
        >>> classify_compliance(1.0)
        'compliant'
        >>> classify_compliance(0.5)
        'minimal'
    """
    if coverage < 0.34:
        return "non_compliant"
    if coverage < 0.67:
        return "minimal"
    if coverage < 1.0:
        return "partial"
    return "compliant"


def map_results_to_sections(
    results: list[dict],
) -> dict[str, list]:
    """Group test results by SR 11-7 section.

    Each result dict is expected to have at least:
    - ``name`` (str): assertion name
    - ``passed`` (bool): whether the assertion passed
    - ``message`` (str, optional): human-readable message

    Results whose names do not match any known section prefix
    are placed in the ``"uncategorised"`` bucket.

    Args:
        results: List of result dicts.

    Returns:
        Dict mapping section IDs to lists of result dicts.
        Each result dict is enriched with a ``"section"`` key.

    Example:
        >>> r = [{"name": "model.bias.x", "passed": True}]
        >>> grouped = map_results_to_sections(r)
        >>> "development" in grouped
        True
    """
    grouped: dict[str, list] = {}
    for r in results:
        name = str(r.get("name", ""))
        sec_ids = _assertion_to_section_ids(name)
        if not sec_ids:
            enriched = dict(r)
            enriched["section"] = "uncategorised"
            grouped.setdefault(
                "uncategorised", []
            ).append(enriched)
        else:
            for sec_id in sec_ids:
                enriched = dict(r)
                enriched["section"] = sec_id
                grouped.setdefault(sec_id, []).append(
                    enriched
                )
    return grouped


def find_gaps(results: list[dict]) -> list[str]:
    """Find SR 11-7 sections not covered by any results.

    A section is considered *covered* if at least one result
    exists whose assertion name matches a prefix in that
    section's assertion list.

    Args:
        results: List of result dicts.

    Returns:
        Sorted list of section IDs that have no matching
        test results.

    Example:
        >>> gaps = find_gaps([])
        >>> "development" in gaps
        True
    """
    covered: set[str] = set()
    for r in results:
        name = str(r.get("name", ""))
        for sec_id in _assertion_to_section_ids(name):
            covered.add(sec_id)
    all_sections = set(SR_11_7_SECTION_IDS)
    return sorted(all_sections - covered)


@timed_assertion
def assert_sr_11_7_coverage(
    results: list[dict],
    min_coverage: float = 0.8,
) -> TestResult:
    """Assert minimum SR 11-7 section coverage.

    Coverage is defined as:
    ``sections_with_at_least_one_test / total_sections``

    At the default threshold of 0.8, at least 3 of the 3
    SR 11-7 sections (Development, Validation, Governance)
    must have corresponding test results.

    Args:
        results: List of result dicts.
        min_coverage: Minimum fraction of sections that must
            be covered (0.0--1.0, default 0.8).

    Returns:
        :class:`~mltk.core.result.TestResult` indicating
        pass/fail with ``covered_count``, ``total``,
        ``coverage``, and ``min_coverage`` in ``details``.

    Raises:
        :class:`~mltk.core.assertion.MltkAssertionError`:
            If coverage is below *min_coverage*.

    Example:
        >>> r = [{"name": "model.bias.x", "passed": True}]
        >>> result = assert_sr_11_7_coverage(r, 0.1)
        >>> result.passed
        True
    """
    total = len(SR_11_7_SECTION_IDS)
    covered: set[str] = set()
    for r in results:
        name = str(r.get("name", ""))
        for sec_id in _assertion_to_section_ids(name):
            covered.add(sec_id)
    covered_count = len(covered)
    coverage = (
        covered_count / total if total > 0 else 0.0
    )
    passed = coverage >= min_coverage

    level = classify_compliance(coverage)
    level_label = COMPLIANCE_CLASSIFICATION[level][
        "label"
    ]

    message = (
        f"SR 11-7 coverage {coverage:.0%} "
        f"({covered_count}/{total} sections) "
        + ("meets" if passed else "below")
        + f" minimum {min_coverage:.0%}"
        + f" [{level_label}]"
    )

    return assert_true(
        passed,
        name="compliance.sr_11_7.coverage",
        message=message,
        severity=Severity.CRITICAL,
        covered_count=covered_count,
        total=total,
        coverage=round(coverage, 4),
        min_coverage=min_coverage,
        compliance_level=level,
        compliance_label=level_label,
    )
