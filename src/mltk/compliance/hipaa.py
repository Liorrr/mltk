"""HIPAA compliance mapping -- map mltk assertions to HIPAA rules.

Pure data + helper functions -- no I/O, no external dependencies.

The Health Insurance Portability and Accountability Act (HIPAA) governs
how Protected Health Information (PHI) is used, stored, and transmitted
by covered entities and business associates.  When machine-learning
models are trained on, or make predictions about, health data, HIPAA
compliance requires demonstrable controls around privacy, security,
and breach notification.

This module maps mltk assertion-name prefixes to the four major HIPAA
rule categories so that organisations can:

1. **Map** existing test results to the HIPAA rules they satisfy.
2. **Find gaps** where no automated test covers a rule.
3. **Assert** minimum coverage as a pass/fail gate in CI pipelines.

Reference: 45 CFR Parts 160, 162, and 164 (HIPAA Administrative
Simplification Regulations).

Why ML teams care about HIPAA
-----------------------------
If your model touches patient records, diagnostic images, claims data,
or any individually identifiable health information, you are likely a
covered entity or business associate under HIPAA.  Even if the data is
de-identified, you need to *prove* it was de-identified correctly --
and mltk's ``data.pii`` / ``data.synthetic.*`` assertions help you do
exactly that.

Mapping strategy
----------------
Each HIPAA rule lists the mltk assertion *prefixes* that provide
evidence of compliance.  A test result whose ``name`` starts with any
listed prefix is considered relevant to that rule.  One assertion can
satisfy multiple rules (e.g. ``data.pii`` is relevant to both the
Privacy Rule and the Breach Notification Rule).
"""

from __future__ import annotations

from mltk.core.assertion import assert_true, timed_assertion
from mltk.core.result import Severity, TestResult

# ---------------------------------------------------------------------------
# HIPAA rule definitions
# ---------------------------------------------------------------------------
# Keys are short snake_case identifiers.
# "assertions" lists assertion-name prefixes (matched via startswith,
# same convention as eu_ai_act.ARTICLE_MAPPING and nist_ai_rmf).

HIPAA_RULES: dict[str, dict] = {
    "privacy_rule": {
        "title": "Privacy Rule (45 CFR 164.500-534)",
        "description": (
            "Protects individually identifiable health information (PHI). "
            "Establishes national standards for when PHI may be used or "
            "disclosed, and gives patients rights over their health data. "
            "ML systems must demonstrate that training data either contains "
            "no PHI, or that PHI has been properly de-identified."
        ),
        "assertions": [
            "data.pii",
            "data.no_pii",
            "data.synthetic.dcr_safe",
            "data.synthetic.novelty",
        ],
    },
    "security_rule_admin": {
        "title": "Security Rule - Administrative (45 CFR 164.308)",
        "description": (
            "Requires administrative safeguards including risk analysis, "
            "workforce training, and contingency plans. For ML systems this "
            "translates to bias audits (risk analysis), calibration checks "
            "(workforce trust in model outputs), and leakage detection "
            "(preventing training data from memorising PHI)."
        ),
        "assertions": [
            "model.bias",
            "model.calibration",
            "training.no_target_leakage",
        ],
    },
    "security_rule_technical": {
        "title": "Security Rule - Technical (45 CFR 164.312)",
        "description": (
            "Requires technical safeguards: access controls, audit controls, "
            "integrity mechanisms, and transmission security. For ML systems "
            "this maps to SLA monitoring (audit trail of model behaviour), "
            "degradation detection (integrity of predictions), and latency "
            "monitoring (ensuring the system remains operational)."
        ),
        "assertions": [
            "monitor.sla",
            "monitor.degradation",
            "inference.latency",
        ],
    },
    "breach_notification": {
        "title": "Breach Notification Rule (45 CFR 164.400-414)",
        "description": (
            "Requires notification when unsecured PHI is breached. ML "
            "systems must detect when data privacy controls fail (PII "
            "leakage) and when model behaviour degrades in ways that could "
            "expose sensitive information (degradation monitoring)."
        ),
        "assertions": [
            "data.pii",
            "monitor.degradation",
        ],
    },
}

# Ordered list of rule IDs for deterministic iteration and report rendering.
HIPAA_RULE_IDS: list[str] = list(HIPAA_RULES.keys())


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _assertion_to_rule_ids(assertion_name: str) -> list[str]:
    """Return all HIPAA rule IDs whose assertion prefixes match *assertion_name*.

    A match occurs when *assertion_name* starts with any prefix listed in
    ``HIPAA_RULES[rule_id]["assertions"]``.

    Args:
        assertion_name: Fully-qualified assertion name,
            e.g. ``"data.pii.email_scan"``.

    Returns:
        List of matching rule IDs (may be empty; may contain more than one).
    """
    matched: list[str] = []
    for rule_id, meta in HIPAA_RULES.items():
        for prefix in meta["assertions"]:
            if assertion_name.startswith(prefix):
                matched.append(rule_id)
                break  # one match per rule is enough
    return matched


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def map_results_to_rules(results: list[dict]) -> dict[str, list]:
    """Group test results by HIPAA rule.

    Each result dict is expected to have at least:

    - ``name`` (str): assertion name, e.g. ``"data.pii.email_scan"``
    - ``passed`` (bool): whether the assertion passed
    - ``message`` (str, optional): human-readable message

    Results whose names do not match any known HIPAA rule prefix are
    placed in the special ``"uncategorised"`` bucket.

    Why this matters
    ~~~~~~~~~~~~~~~~
    When preparing a HIPAA compliance report, auditors want to see which
    tests satisfy which rules.  This function creates that grouping so
    that downstream report generators can render per-rule sections.

    Args:
        results: List of result dicts (from JSON or TestResult serialisation).

    Returns:
        Dict mapping rule IDs (e.g. ``"privacy_rule"``) to lists of
        result dicts.  Each result dict is enriched with a ``"rule"`` key.

    Example:
        >>> r = [{"name": "data.pii.scan", "passed": True, "message": "ok"}]
        >>> grouped = map_results_to_rules(r)
        >>> "privacy_rule" in grouped
        True
    """
    grouped: dict[str, list] = {}
    for r in results:
        name = str(r.get("name", ""))
        rule_ids = _assertion_to_rule_ids(name)
        if not rule_ids:
            enriched = dict(r)
            enriched["rule"] = "uncategorised"
            grouped.setdefault("uncategorised", []).append(enriched)
        else:
            for rule_id in rule_ids:
                enriched = dict(r)
                enriched["rule"] = rule_id
                grouped.setdefault(rule_id, []).append(enriched)
    return grouped


def find_gaps(results: list[dict]) -> list[str]:
    """Find HIPAA rules not covered by any test results.

    A rule is considered *covered* if at least one result exists whose
    assertion name matches any of the rule's assertion prefixes.

    Why gap analysis matters
    ~~~~~~~~~~~~~~~~~~~~~~~~
    HIPAA violations carry penalties of $100 to $50,000 per violation,
    with a maximum of $1.5 million per year per violation category.
    Identifying untested rules *before* an audit lets teams prioritise
    where to add test coverage.

    Args:
        results: List of result dicts (same format as
            :func:`map_results_to_rules`).

    Returns:
        Sorted list of rule IDs (e.g. ``["breach_notification",
        "security_rule_technical"]``) that have no matching test results.

    Example:
        >>> gaps = find_gaps([])
        >>> "privacy_rule" in gaps
        True
    """
    covered: set[str] = set()
    for r in results:
        name = str(r.get("name", ""))
        for rule_id in _assertion_to_rule_ids(name):
            covered.add(rule_id)
    all_rules = set(HIPAA_RULE_IDS)
    return sorted(all_rules - covered)


@timed_assertion
def assert_hipaa_coverage(
    results: list[dict],
    min_coverage: float = 0.8,
) -> TestResult:
    """Assert minimum HIPAA rule coverage.

    Coverage is defined as::

        rules_with_at_least_one_test / total_rules

    At the default threshold of 0.8, at least 3 of the 4 HIPAA rules
    must have corresponding test results (3/4 = 75% < 80%, so actually
    all 4 must be covered at the default threshold -- this is intentional
    since HIPAA is a strict regulatory framework).

    Why this belongs in CI
    ~~~~~~~~~~~~~~~~~~~~~~
    Running ``assert_hipaa_coverage`` as a CI gate ensures that no model
    ships to production without a minimum level of HIPAA-relevant test
    coverage.  If a team removes a PII scan or degrades monitoring, the
    pipeline fails before the model reaches patients.

    Args:
        results: List of result dicts (same format as
            :func:`map_results_to_rules`).
        min_coverage: Minimum fraction of HIPAA rules that must be
            covered (0.0--1.0, default 0.8).

    Returns:
        :class:`~mltk.core.result.TestResult` indicating pass/fail with
        ``covered_count``, ``total``, ``coverage``, ``min_coverage``,
        and ``gaps`` in ``details``.

    Raises:
        :class:`~mltk.core.assertion.MltkAssertionError`: If coverage is
            below *min_coverage*.

    Example:
        >>> r = [{"name": "data.pii.scan", "passed": True, "message": "ok"}]
        >>> result = assert_hipaa_coverage(r, min_coverage=0.1)
        >>> result.passed
        True
    """
    total = len(HIPAA_RULE_IDS)
    gaps = find_gaps(results)
    covered_count = total - len(gaps)
    coverage = covered_count / total if total > 0 else 0.0
    passed = coverage >= min_coverage

    message = (
        f"HIPAA rule coverage {coverage:.0%} "
        f"({covered_count}/{total} rules) "
        + ("meets" if passed else "below")
        + f" minimum {min_coverage:.0%}"
    )

    return assert_true(
        passed,
        name="compliance.hipaa.coverage",
        message=message,
        severity=Severity.CRITICAL,
        covered_count=covered_count,
        total=total,
        coverage=round(coverage, 4),
        min_coverage=min_coverage,
        gaps=gaps,
    )
