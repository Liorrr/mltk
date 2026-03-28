"""Tests for HIPAA compliance mapping module (mltk.compliance.hipaa).

Covers: result-to-rule mapping, gap detection (full/partial/no coverage),
coverage assertion pass/fail, Privacy Rule assertion mapping, empty results,
cross-rule mapping, uncategorised results, and timing decorator.

HIPAA (Health Insurance Portability and Accountability Act) is a US federal
law that protects patient health information.  These tests verify that mltk
correctly maps ML test assertions to the four major HIPAA rule categories:

1. Privacy Rule -- PHI de-identification checks
2. Security Rule (Administrative) -- bias, calibration, leakage
3. Security Rule (Technical) -- SLA, degradation, latency monitoring
4. Breach Notification Rule -- PII detection + degradation alerts
"""

from __future__ import annotations

import pytest

from mltk.compliance.hipaa import (
    HIPAA_RULE_IDS,
    HIPAA_RULES,
    assert_hipaa_coverage,
    find_gaps,
    map_results_to_rules,
)
from mltk.core.assertion import MltkAssertionError

# ---------------------------------------------------------------------------
# Sample result fixtures
# ---------------------------------------------------------------------------

# Covers all 4 HIPAA rules:
#   privacy_rule: data.pii, data.synthetic.dcr_safe
#   security_rule_admin: model.bias, model.calibration
#   security_rule_technical: monitor.sla, monitor.degradation, inference.latency
#   breach_notification: data.pii (shared), monitor.degradation (shared)
FULL_COVERAGE_RESULTS: list[dict] = [
    # Privacy Rule
    {"name": "data.pii.email_scan",          "passed": True,  "message": "no emails found"},
    {"name": "data.no_pii.ssn_check",        "passed": True,  "message": "no SSNs found"},
    {"name": "data.synthetic.dcr_safe.v1",   "passed": True,  "message": "DCR=0.02"},
    {"name": "data.synthetic.novelty.score",  "passed": True,  "message": "novelty=0.95"},
    # Security Rule - Administrative
    {"name": "model.bias.demographic_parity", "passed": False, "message": "disparity=0.15"},
    {"name": "model.calibration.ece",         "passed": True,  "message": "ECE=0.03"},
    {"name": "training.no_target_leakage.v1", "passed": True,  "message": "no leakage"},
    # Security Rule - Technical
    {"name": "monitor.sla.uptime",            "passed": True,  "message": "99.95%"},
    {"name": "monitor.degradation.weekly",    "passed": True,  "message": "stable"},
    {"name": "inference.latency.p99",         "passed": True,  "message": "p99=85ms"},
]

# Covers only privacy_rule (data.pii) and security_rule_technical (monitor.sla)
# -- 2 of 4 rules
PARTIAL_RESULTS: list[dict] = [
    {"name": "data.pii.scan",       "passed": True, "message": "clean"},
    {"name": "monitor.sla.uptime",  "passed": True, "message": "99.9%"},
]

# Covers nothing -- unrecognised assertion prefixes
NO_COVERAGE_RESULTS: list[dict] = [
    {"name": "custom.internal.check", "passed": True, "message": "ok"},
]


# ---------------------------------------------------------------------------
# Test 1 -- map_results_to_rules groups results by HIPAA rule
# ---------------------------------------------------------------------------


def test_map_results_to_rules_grouping():
    """Verify that each assertion lands in the correct HIPAA rule bucket.

    WHY: The mapper is the core grouping function that drives per-rule
    report sections.  Every assertion prefix in HIPAA_RULES must route
    to its declared rule when matched via startswith.
    """
    grouped = map_results_to_rules(FULL_COVERAGE_RESULTS)

    # Privacy Rule
    assert "privacy_rule" in grouped
    pr_names = [r["name"] for r in grouped["privacy_rule"]]
    assert "data.pii.email_scan" in pr_names
    assert "data.no_pii.ssn_check" in pr_names
    assert "data.synthetic.dcr_safe.v1" in pr_names
    assert "data.synthetic.novelty.score" in pr_names

    # Security Rule - Administrative
    assert "security_rule_admin" in grouped
    admin_names = [r["name"] for r in grouped["security_rule_admin"]]
    assert "model.bias.demographic_parity" in admin_names
    assert "model.calibration.ece" in admin_names
    assert "training.no_target_leakage.v1" in admin_names

    # Security Rule - Technical
    assert "security_rule_technical" in grouped
    tech_names = [r["name"] for r in grouped["security_rule_technical"]]
    assert "monitor.sla.uptime" in tech_names
    assert "monitor.degradation.weekly" in tech_names
    assert "inference.latency.p99" in tech_names

    # Breach Notification -- shares prefixes with privacy + technical
    assert "breach_notification" in grouped
    bn_names = [r["name"] for r in grouped["breach_notification"]]
    assert "data.pii.email_scan" in bn_names
    assert "monitor.degradation.weekly" in bn_names


# ---------------------------------------------------------------------------
# Test 2 -- find_gaps returns all rules when no results
# ---------------------------------------------------------------------------


def test_find_gaps_empty_results():
    """Empty results must report every HIPAA rule as a gap.

    WHY: When no tests have been run, every rule is uncovered.  This is
    the worst-case scenario and the gap list must be complete so that
    teams know exactly what needs to be added.
    """
    gaps = find_gaps([])
    assert gaps == sorted(HIPAA_RULE_IDS)
    assert len(gaps) == 4


# ---------------------------------------------------------------------------
# Test 3 -- find_gaps returns empty list when fully covered
# ---------------------------------------------------------------------------


def test_find_gaps_full_coverage():
    """Full test coverage must produce zero gaps.

    WHY: A fully covered system is HIPAA-ready (from a test-coverage
    perspective).  Reporting false gaps would cause unnecessary remediation.
    """
    gaps = find_gaps(FULL_COVERAGE_RESULTS)
    assert gaps == [], f"Expected no gaps but got: {gaps}"


# ---------------------------------------------------------------------------
# Test 4 -- find_gaps returns uncovered rules for partial coverage
# ---------------------------------------------------------------------------


def test_find_gaps_partial_coverage():
    """Partial coverage must correctly identify the missing rules.

    PARTIAL_RESULTS covers:
    - privacy_rule (data.pii) -- also satisfies breach_notification
    - security_rule_technical (monitor.sla)
    - breach_notification (data.pii shared prefix)

    Missing: security_rule_admin (no model.bias/calibration/leakage)
    """
    gaps = find_gaps(PARTIAL_RESULTS)

    assert "privacy_rule" not in gaps, "privacy_rule is covered by data.pii"
    assert "security_rule_technical" not in gaps, "tech rule covered by monitor.sla"
    assert "breach_notification" not in gaps, "breach covered by data.pii"
    assert "security_rule_admin" in gaps, "admin rule has no matching results"


# ---------------------------------------------------------------------------
# Test 5 -- assert_hipaa_coverage passes when all rules are covered
# ---------------------------------------------------------------------------


def test_coverage_assertion_pass():
    """Full coverage must pass the assertion at the default 80% threshold.

    WHY: This is the happy path -- all 4/4 rules covered (100% >= 80%).
    The returned TestResult must contain accurate details for reporting.
    """
    result = assert_hipaa_coverage(FULL_COVERAGE_RESULTS, min_coverage=0.8)

    assert result.passed is True
    assert result.details["covered_count"] == 4
    assert result.details["total"] == 4
    assert result.details["coverage"] == 1.0
    assert result.details["min_coverage"] == 0.8
    assert result.details["gaps"] == []
    assert "meets" in result.message
    assert result.name == "compliance.hipaa.coverage"


# ---------------------------------------------------------------------------
# Test 6 -- assert_hipaa_coverage fails when below threshold
# ---------------------------------------------------------------------------


def test_coverage_assertion_fail():
    """Partial coverage below threshold must raise MltkAssertionError.

    WHY: The assertion is a CI gate.  When HIPAA coverage is insufficient,
    the pipeline must fail with a clear error showing what is missing.
    """
    with pytest.raises(MltkAssertionError) as exc_info:
        assert_hipaa_coverage(PARTIAL_RESULTS, min_coverage=0.8)

    result = exc_info.value.result
    assert result.passed is False
    assert result.details["coverage"] < 0.8
    assert "below" in result.message
    assert len(result.details["gaps"]) > 0


# ---------------------------------------------------------------------------
# Test 7 -- Privacy Rule assertions are mapped correctly
# ---------------------------------------------------------------------------


def test_privacy_rule_assertions():
    """Verify that all declared Privacy Rule prefixes produce matches.

    WHY: The Privacy Rule is the most critical HIPAA component for ML
    systems handling health data.  Every declared assertion prefix must
    actually route results to this rule -- a mapping error here could
    cause a false "all clear" on a PHI audit.
    """
    privacy_prefixes = HIPAA_RULES["privacy_rule"]["assertions"]
    assert len(privacy_prefixes) == 4, (
        "Privacy Rule should have exactly 4 assertion prefixes"
    )

    # Generate a result for each prefix and verify mapping
    results = [
        {"name": f"{prefix}.test", "passed": True, "message": "ok"}
        for prefix in privacy_prefixes
    ]
    grouped = map_results_to_rules(results)

    assert "privacy_rule" in grouped
    mapped_names = [r["name"] for r in grouped["privacy_rule"]]
    for prefix in privacy_prefixes:
        expected_name = f"{prefix}.test"
        assert expected_name in mapped_names, (
            f"Privacy Rule prefix {prefix!r} did not map correctly"
        )


# ---------------------------------------------------------------------------
# Test 8 -- Unrecognised assertions go to "uncategorised" bucket
# ---------------------------------------------------------------------------


def test_uncategorised_results():
    """Assertions with unrecognised prefixes must not be silently dropped.

    WHY: When teams add custom assertions that do not match any HIPAA
    rule, those results must still appear in the output (in the
    "uncategorised" bucket) so they are visible in reports and not lost.
    """
    grouped = map_results_to_rules(NO_COVERAGE_RESULTS)

    assert "uncategorised" in grouped
    names = [r["name"] for r in grouped["uncategorised"]]
    assert "custom.internal.check" in names

    # Verify the enriched "rule" key is set
    for item in grouped["uncategorised"]:
        assert item["rule"] == "uncategorised"


# -------------------------------------------------------------------
# Parametrized & edge-case tests (hardening)
# -------------------------------------------------------------------


@pytest.mark.parametrize(
    "prefix,rule_id",
    [
        ("data.pii", "privacy_rule"),
        ("model.bias", "security_rule_admin"),
        ("monitor.sla", "security_rule_technical"),
        ("monitor.degradation", "breach_notification"),
    ],
)
def test_prefix_maps_to_correct_rule(
    prefix: str, rule_id: str
):
    """Each assertion prefix routes to its HIPAA rule."""
    results = [
        {
            "name": f"{prefix}.hardening_test",
            "passed": True,
            "message": "ok",
        }
    ]
    grouped = map_results_to_rules(results)
    assert rule_id in grouped
    names = [r["name"] for r in grouped[rule_id]]
    assert f"{prefix}.hardening_test" in names


def test_full_coverage_passes_at_100():
    """100% coverage (4/4 rules) always passes."""
    r = assert_hipaa_coverage(
        FULL_COVERAGE_RESULTS, min_coverage=1.0
    )
    assert r.passed is True
    assert r.details["coverage"] == 1.0
    assert r.details["gaps"] == []


def test_single_result_maps_to_multiple_rules():
    """data.pii maps to both privacy_rule and breach."""
    results = [
        {
            "name": "data.pii.multi_rule",
            "passed": True,
            "message": "ok",
        }
    ]
    grouped = map_results_to_rules(results)
    assert "privacy_rule" in grouped
    assert "breach_notification" in grouped


def test_custom_assertion_names_uncategorised():
    """Assertions with no matching prefix go uncategorised."""
    results = [
        {
            "name": "perf.latency.custom",
            "passed": True,
            "message": "ok",
        },
        {
            "name": "xyz.unknown.check",
            "passed": True,
            "message": "ok",
        },
    ]
    grouped = map_results_to_rules(results)
    assert "uncategorised" in grouped
    assert len(grouped["uncategorised"]) == 2
    gaps = find_gaps(results)
    assert len(gaps) == 4  # none of the 4 rules covered


def test_coverage_at_exact_threshold_boundary():
    """3/4 = 0.75 coverage at min_coverage=0.75 passes."""
    r = assert_hipaa_coverage(
        PARTIAL_RESULTS, min_coverage=0.75
    )
    assert r.passed is True
    assert r.details["covered_count"] == 3
