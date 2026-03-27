"""NIST AI RMF mapping -- map mltk assertions to NIST AI Risk Management Framework.

Pure data + helper functions -- no I/O, no external dependencies.

Maps assertion name prefixes to the four NIST AI RMF functions:
GOVERN (GV), MAP (MP), MEASURE (MS), MANAGE (MN).

Reference: NIST AI 100-1, AI Risk Management Framework (January 2023).
"""

from __future__ import annotations

from mltk.core.assertion import assert_true, timed_assertion
from mltk.core.result import Severity, TestResult

# ---------------------------------------------------------------------------
# NIST AI RMF tier definitions
# ---------------------------------------------------------------------------

TIERS = ["partial", "risk_informed", "repeatable", "adaptive"]

TIER_CLASSIFICATION: dict[str, dict] = {
    "partial": {
        "label": "Partial (Tier 1)",
        "description": (
            "Risk management is ad hoc and reactive. Organisational awareness "
            "of AI risks is limited; processes are not formalised."
        ),
        "color": "#f85149",
        "badge_class": "fail",
    },
    "risk_informed": {
        "label": "Risk-Informed (Tier 2)",
        "description": (
            "Risk management practices are approved by management but may not "
            "be established organisation-wide. Risk awareness exists but "
            "processes are inconsistently applied."
        ),
        "color": "#d29922",
        "badge_class": "warn",
    },
    "repeatable": {
        "label": "Repeatable (Tier 3)",
        "description": (
            "Risk management practices are formally approved, expressed as "
            "policy, and regularly updated. Organisation-wide approach to "
            "managing AI risks."
        ),
        "color": "#58a6ff",
        "badge_class": "info",
    },
    "adaptive": {
        "label": "Adaptive (Tier 4)",
        "description": (
            "Risk management practices adapt based on lessons learned and "
            "predictive indicators. Continuous improvement is embedded across "
            "the organisation."
        ),
        "color": "#3fb950",
        "badge_class": "pass",
    },
}

# ---------------------------------------------------------------------------
# NIST AI RMF function and subcategory definitions
# ---------------------------------------------------------------------------
# Keys are function codes (GV, MP, MS, MN).
# "assertions" lists assertion-name prefixes (matched via startswith,
# same convention as eu_ai_act.ARTICLE_MAPPING and owasp_llm).

NIST_RMF_FUNCTIONS: dict[str, dict] = {
    "GV": {
        "title": "GOVERN",
        "description": (
            "Cultivate and implement a culture of risk management. "
            "Establish policies, define roles and responsibilities, and set "
            "risk tolerance thresholds for AI systems."
        ),
        "subcategories": {
            "GV-1": "Policies and procedures",
            "GV-2": "Accountability structures",
        },
        "assertions": [
            "data.pii",
            "data.schema",
            "data.synthetic.dcr_safe",
            "data.synthetic.novelty",
            "model.bias",
        ],
    },
    "MP": {
        "title": "MAP",
        "description": (
            "Categorise and contextualise AI systems. Identify stakeholders, "
            "intended use, and potential risks across the AI lifecycle."
        ),
        "subcategories": {
            "MP-1": "System context and capabilities",
            "MP-2": "Stakeholder identification",
        },
        "assertions": ["model.metric", "model.slice", "data.drift"],
    },
    "MS": {
        "title": "MEASURE",
        "description": (
            "Analyse, assess, and track AI risks using quantitative and "
            "qualitative methods. Employ appropriate metrics and benchmarks "
            "to evaluate model performance, robustness, and fairness."
        ),
        "subcategories": {
            "MS-1": "Appropriate metrics identified",
            "MS-2": "Performance evaluated",
            "MS-3": "Risks tracked over time",
        },
        "assertions": [
            "model.metric",
            "model.regression",
            "model.calibration",
            "model.adversarial",
            "model.interval_coverage",
            "model.prediction_set_size",
            "inference.latency",
            "inference.throughput",
            "training.gradient",
            "training.weight_divergence",
        ],
    },
    "MN": {
        "title": "MANAGE",
        "description": (
            "Prioritise and act on identified risks. Allocate resources, "
            "implement treatment plans, and continuously monitor deployed AI "
            "systems to maintain acceptable risk levels."
        ),
        "subcategories": {
            "MN-1": "Treatment plans",
            "MN-2": "Monitoring deployed systems",
        },
        "assertions": [
            "monitor.degradation",
            "monitor.sla",
            "monitor.streaming_drift",
            "monitor.concept_drift",
            "data.drift",
        ],
    },
}

# Ordered list of function codes for deterministic iteration.
NIST_RMF_FUNCTION_IDS: list[str] = list(NIST_RMF_FUNCTIONS.keys())

# Canonical function metadata -- used to render per-function sections in order.
FUNCTION_META: list[dict] = [
    {
        "function": func_id,
        "title": meta["title"],
        "description": meta["description"],
        "subcategories": meta["subcategories"],
    }
    for func_id, meta in NIST_RMF_FUNCTIONS.items()
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _assertion_to_function_ids(assertion_name: str) -> list[str]:
    """Return all RMF function IDs whose assertion prefixes match *assertion_name*.

    A match occurs when *assertion_name* starts with any prefix listed in
    ``NIST_RMF_FUNCTIONS[func_id]["assertions"]``.

    Args:
        assertion_name: Fully-qualified assertion name, e.g. ``"model.bias.demographic_parity"``.

    Returns:
        List of matching function IDs (may be empty; may contain more than one).
    """
    matched: list[str] = []
    for func_id, meta in NIST_RMF_FUNCTIONS.items():
        for prefix in meta["assertions"]:
            if assertion_name.startswith(prefix):
                matched.append(func_id)
                break  # one match per function is enough
    return matched


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def classify_tier(coverage: float) -> str:
    """Classify NIST AI RMF tier based on coverage ratio.

    The tier is determined by the fraction of RMF functions covered:

    - ``< 0.25`` -- Partial (Tier 1)
    - ``< 0.50`` -- Risk-Informed (Tier 2)
    - ``< 0.75`` -- Repeatable (Tier 3)
    - ``>= 0.75`` -- Adaptive (Tier 4)

    Args:
        coverage: Coverage ratio between 0.0 and 1.0.

    Returns:
        Tier key string (one of ``TIERS``).

    Example:
        >>> classify_tier(0.9)
        'adaptive'
        >>> classify_tier(0.3)
        'risk_informed'
    """
    if coverage < 0.25:
        return "partial"
    if coverage < 0.50:
        return "risk_informed"
    if coverage < 0.75:
        return "repeatable"
    return "adaptive"


def map_results_to_measures(results: list[dict]) -> dict[str, list]:
    """Group test results by NIST AI RMF function.

    Each result dict is expected to have at least:
    - ``name`` (str): assertion name, e.g. ``"model.bias.demographic_parity"``
    - ``passed`` (bool): whether the assertion passed
    - ``message`` (str, optional): human-readable message

    Results whose names do not match any known function prefix are placed in
    the special ``"uncategorised"`` bucket.

    Args:
        results: List of result dicts (from JSON or TestResult serialisation).

    Returns:
        Dict mapping function codes (e.g. ``"GV"``, ``"MS"``) to lists of
        result dicts. Each result dict is enriched with a ``"function"`` key.

    Example:
        >>> r = [{"name": "model.bias.x", "passed": True, "message": "ok"}]
        >>> grouped = map_results_to_measures(r)
        >>> "GV" in grouped
        True
    """
    grouped: dict[str, list] = {}
    for r in results:
        name = str(r.get("name", ""))
        func_ids = _assertion_to_function_ids(name)
        if not func_ids:
            enriched = dict(r)
            enriched["function"] = "uncategorised"
            grouped.setdefault("uncategorised", []).append(enriched)
        else:
            for func_id in func_ids:
                enriched = dict(r)
                enriched["function"] = func_id
                grouped.setdefault(func_id, []).append(enriched)
    return grouped


def find_gaps(results: list[dict]) -> list[str]:
    """Find NIST AI RMF functions not covered by any test results.

    A function is considered *covered* if at least one result exists whose
    assertion name matches a prefix in that function's assertion list.

    Args:
        results: List of result dicts (same format as :func:`map_results_to_measures`).

    Returns:
        Sorted list of function codes (e.g. ``["MN", "MS"]``) that have no
        matching test results.

    Example:
        >>> gaps = find_gaps([])
        >>> "GV" in gaps
        True
    """
    covered: set[str] = set()
    for r in results:
        name = str(r.get("name", ""))
        for func_id in _assertion_to_function_ids(name):
            covered.add(func_id)
    all_funcs = set(NIST_RMF_FUNCTION_IDS)
    return sorted(all_funcs - covered)


@timed_assertion
def assert_nist_rmf_coverage(
    results: list[dict],
    min_coverage: float = 0.8,
) -> TestResult:
    """Assert minimum NIST AI RMF function coverage.

    Coverage is defined as:
    ``functions_with_at_least_one_test / total_functions``

    At the default threshold of 0.8, at least 4 of the 4 RMF functions
    (GOVERN, MAP, MEASURE, MANAGE) must have corresponding test results.

    Args:
        results: List of result dicts (same format as :func:`map_results_to_measures`).
        min_coverage: Minimum fraction of RMF functions that must be
            covered (0.0--1.0, default 0.8).

    Returns:
        :class:`~mltk.core.result.TestResult` indicating pass/fail with
        ``covered_count``, ``total``, ``coverage``, and ``min_coverage``
        in ``details``.

    Raises:
        :class:`~mltk.core.assertion.MltkAssertionError`: If coverage is
            below *min_coverage*.

    Example:
        >>> r = [{"name": "model.bias.x", "passed": True, "message": "ok"}]
        >>> result = assert_nist_rmf_coverage(r, min_coverage=0.1)
        >>> result.passed
        True
    """
    total = len(NIST_RMF_FUNCTION_IDS)
    covered: set[str] = set()
    for r in results:
        name = str(r.get("name", ""))
        for func_id in _assertion_to_function_ids(name):
            covered.add(func_id)
    covered_count = len(covered)
    coverage = covered_count / total if total > 0 else 0.0
    passed = coverage >= min_coverage

    tier = classify_tier(coverage)
    tier_label = TIER_CLASSIFICATION[tier]["label"]

    message = (
        f"NIST AI RMF coverage {coverage:.0%} "
        f"({covered_count}/{total} functions) "
        + ("meets" if passed else "below")
        + f" minimum {min_coverage:.0%}"
        + f" [{tier_label}]"
    )

    return assert_true(
        passed,
        name="compliance.nist_ai_rmf.coverage",
        message=message,
        severity=Severity.CRITICAL,
        covered_count=covered_count,
        total=total,
        coverage=round(coverage, 4),
        min_coverage=min_coverage,
        tier=tier,
        tier_label=tier_label,
    )
