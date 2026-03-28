"""Tests for SR 11-7 mapping module (mltk.compliance.sr_11_7).

Covers: result mapping, gap detection (full/partial/no coverage),
coverage assertion pass/fail, compliance classification, edge cases
(empty results, uncategorised), cross-section mapping, and timing.
"""

from __future__ import annotations

import pytest

from mltk.compliance.sr_11_7 import (
    COMPLIANCE_CLASSIFICATION,
    COMPLIANCE_LEVELS,
    SR_11_7_SECTION_IDS,
    SR_11_7_SECTIONS,
    SECTION_META,
    assert_sr_11_7_coverage,
    classify_compliance,
    find_gaps,
    map_results_to_sections,
)
from mltk.core.assertion import MltkAssertionError

# ---------------------------------------------------------------
# Sample result fixtures
# ---------------------------------------------------------------

# Covers all 3 sections:
#   development (model.metric, model.regression, model.bias,
#                data.schema, data.drift,
#                training.no_target_leakage)
#   validation  (model.metric, model.slice, model.calibration,
#                model.counterfactual, data.synthetic)
#   governance  (monitor.degradation, monitor.sla, data.drift,
#                inference.latency)
FULL_COVERAGE_RESULTS: list[dict] = [
    # development
    {
        "name": "model.metric.accuracy",
        "passed": True,
        "message": "accuracy=0.94",
    },
    {
        "name": "model.regression.v2_vs_v1",
        "passed": True,
        "message": "no regression",
    },
    {
        "name": "model.bias.demographic_parity",
        "passed": False,
        "message": "disparity=0.18",
    },
    {
        "name": "data.schema.columns",
        "passed": True,
        "message": "schema ok",
    },
    {
        "name": "data.drift.psi",
        "passed": True,
        "message": "PSI=0.04",
    },
    {
        "name": "training.no_target_leakage.check",
        "passed": True,
        "message": "no leakage",
    },
    # validation
    {
        "name": "model.slice.gender",
        "passed": True,
        "message": "slice ok",
    },
    {
        "name": "model.calibration.ece",
        "passed": True,
        "message": "ECE=0.04",
    },
    {
        "name": "model.counterfactual.flip",
        "passed": True,
        "message": "stable",
    },
    {
        "name": "data.synthetic.dcr_safe",
        "passed": True,
        "message": "DCR ok",
    },
    # governance
    {
        "name": "monitor.degradation.week",
        "passed": True,
        "message": "no degradation",
    },
    {
        "name": "monitor.sla.uptime",
        "passed": True,
        "message": "99.9%",
    },
    {
        "name": "inference.latency.p99",
        "passed": True,
        "message": "p99=120ms",
    },
]

# Only covers development (model.bias) -- 1 of 3 sections
PARTIAL_RESULTS: list[dict] = [
    {
        "name": "model.bias.demographic_parity",
        "passed": True,
        "message": "fair",
    },
]

# Covers nothing -- unrecognised assertion prefix
NO_COVERAGE_RESULTS: list[dict] = [
    {
        "name": "custom.unknown.check",
        "passed": True,
        "message": "ok",
    },
]

# Covers exactly 2 of 3 sections: development + governance
TWO_SECTION_RESULTS: list[dict] = [
    {
        "name": "model.bias.dp",
        "passed": True,
        "message": "ok",
    },
    {
        "name": "monitor.degradation.week",
        "passed": True,
        "message": "ok",
    },
]


# ---------------------------------------------------------------
# Test 1 -- map_results_to_sections groups by SR 11-7 section
# ---------------------------------------------------------------


def test_map_results_to_sections_grouping():
    # SCENARIO: full results list covering all 3 sections
    # WHY: mapper must place each assertion in correct bucket
    # EXPECTED: each section has its expected assertions

    grouped = map_results_to_sections(FULL_COVERAGE_RESULTS)

    # development
    assert "development" in grouped
    dev_names = [r["name"] for r in grouped["development"]]
    assert "model.regression.v2_vs_v1" in dev_names
    assert "model.bias.demographic_parity" in dev_names
    assert "data.schema.columns" in dev_names
    assert "training.no_target_leakage.check" in dev_names

    # validation
    assert "validation" in grouped
    val_names = [r["name"] for r in grouped["validation"]]
    assert "model.slice.gender" in val_names
    assert "model.counterfactual.flip" in val_names
    assert "data.synthetic.dcr_safe" in val_names

    # governance
    assert "governance" in grouped
    gov_names = [r["name"] for r in grouped["governance"]]
    assert "monitor.degradation.week" in gov_names
    assert "monitor.sla.uptime" in gov_names
    assert "inference.latency.p99" in gov_names


# ---------------------------------------------------------------
# Test 2 -- map enriches results with "section" key
# ---------------------------------------------------------------


def test_map_results_enriches_section_key():
    # SCENARIO: mapped results checked for injected "section" key
    # WHY: downstream consumers need section key for rendering
    # EXPECTED: every result in a named bucket has matching key

    grouped = map_results_to_sections(FULL_COVERAGE_RESULTS)
    for sec_id, items in grouped.items():
        if sec_id == "uncategorised":
            continue
        for item in items:
            assert item.get("section") == sec_id, (
                f"Result {item['name']!r} in {sec_id!r} "
                f"has section={item.get('section')!r}"
            )


# ---------------------------------------------------------------
# Test 3 -- map with empty results
# ---------------------------------------------------------------


def test_map_results_empty():
    # SCENARIO: empty results list
    # WHY: mapper must return empty dict without raising
    # EXPECTED: empty dict

    grouped = map_results_to_sections([])
    assert grouped == {}


# ---------------------------------------------------------------
# Test 4 -- find_gaps returns all sections when no results
# ---------------------------------------------------------------


def test_find_gaps_no_results():
    # SCENARIO: no test results at all
    # WHY: all 3 sections should appear as gaps
    # EXPECTED: sorted list of all 3 section IDs

    gaps = find_gaps([])
    assert gaps == sorted(SR_11_7_SECTION_IDS)
    assert len(gaps) == 3


# ---------------------------------------------------------------
# Test 5 -- find_gaps returns empty list when fully covered
# ---------------------------------------------------------------


def test_find_gaps_full_coverage():
    # SCENARIO: results cover all 3 sections
    # WHY: no gaps should be reported
    # EXPECTED: empty list

    gaps = find_gaps(FULL_COVERAGE_RESULTS)
    assert gaps == [], f"Expected no gaps but got: {gaps}"


# ---------------------------------------------------------------
# Test 6 -- find_gaps returns uncovered sections
# ---------------------------------------------------------------


def test_find_gaps_partial_coverage():
    # SCENARIO: PARTIAL_RESULTS covers only development
    # WHY: gaps must reflect which sections lack coverage
    # EXPECTED: validation and governance in gaps

    gaps = find_gaps(PARTIAL_RESULTS)

    assert "development" not in gaps
    assert "validation" in gaps
    assert "governance" in gaps


# ---------------------------------------------------------------
# Test 7 -- assert_sr_11_7_coverage passes at full coverage
# ---------------------------------------------------------------


def test_coverage_assertion_pass():
    # SCENARIO: FULL_COVERAGE_RESULTS covers all 3 sections
    # WHY: at 80% threshold, 3/3 (100%) must pass
    # EXPECTED: passed is True; covered_count == 3

    result = assert_sr_11_7_coverage(
        FULL_COVERAGE_RESULTS, min_coverage=0.8
    )

    assert result.passed is True
    assert result.details["covered_count"] == 3
    assert result.details["total"] == 3
    assert result.details["coverage"] == 1.0
    assert result.details["min_coverage"] == 0.8
    assert "meets" in result.message
    assert result.name == "compliance.sr_11_7.coverage"


# ---------------------------------------------------------------
# Test 8 -- assert_sr_11_7_coverage fails below threshold
# ---------------------------------------------------------------


def test_coverage_assertion_fail():
    # SCENARIO: PARTIAL_RESULTS covers 1/3 (33%), below 80%
    # WHY: assertion must raise on low coverage
    # EXPECTED: MltkAssertionError; passed is False

    with pytest.raises(MltkAssertionError) as exc_info:
        assert_sr_11_7_coverage(
            PARTIAL_RESULTS, min_coverage=0.8
        )

    result = exc_info.value.result
    assert result.passed is False
    assert result.details["covered_count"] == 1
    assert result.details["coverage"] == round(1 / 3, 4)
    assert "below" in result.message


# ---------------------------------------------------------------
# Test 9 -- assert_sr_11_7_coverage with empty results
# ---------------------------------------------------------------


def test_coverage_assertion_empty_results():
    # SCENARIO: no results at all, 80% threshold
    # WHY: empty input should fail gracefully
    # EXPECTED: MltkAssertionError; coverage is 0.0

    with pytest.raises(MltkAssertionError) as exc_info:
        assert_sr_11_7_coverage([], min_coverage=0.8)

    result = exc_info.value.result
    assert result.passed is False
    assert result.details["covered_count"] == 0
    assert result.details["coverage"] == 0.0


# ---------------------------------------------------------------
# Test 10 -- classify_compliance returns correct levels
# ---------------------------------------------------------------


def test_classify_compliance():
    # SCENARIO: various coverage values classified
    # WHY: level drives badge and label in reports
    # EXPECTED: correct level for each bracket

    assert classify_compliance(0.0) == "non_compliant"
    assert classify_compliance(0.1) == "non_compliant"
    assert classify_compliance(0.33) == "non_compliant"
    assert classify_compliance(0.34) == "minimal"
    assert classify_compliance(0.5) == "minimal"
    assert classify_compliance(0.66) == "minimal"
    assert classify_compliance(0.67) == "partial"
    assert classify_compliance(0.8) == "partial"
    assert classify_compliance(0.99) == "partial"
    assert classify_compliance(1.0) == "compliant"


# ---------------------------------------------------------------
# Test 11 -- compliance classification data is complete
# ---------------------------------------------------------------


def test_compliance_classification_data():
    # SCENARIO: all level keys have complete metadata
    # WHY: report rendering depends on label, description, etc.
    # EXPECTED: every level has all required keys

    required_keys = {
        "label", "description", "color", "badge_class",
    }
    for level in COMPLIANCE_LEVELS:
        assert level in COMPLIANCE_CLASSIFICATION, (
            f"Level {level!r} missing from classification"
        )
        info = COMPLIANCE_CLASSIFICATION[level]
        assert required_keys.issubset(info.keys()), (
            f"Missing keys for {level!r}: "
            f"{required_keys - set(info.keys())}"
        )


# ---------------------------------------------------------------
# Test 12 -- coverage assertion includes compliance level
# ---------------------------------------------------------------


def test_coverage_assertion_includes_level():
    # SCENARIO: assert returns compliance level in details
    # WHY: downstream consumers need level for badges
    # EXPECTED: details contain level and label

    result = assert_sr_11_7_coverage(
        FULL_COVERAGE_RESULTS, min_coverage=0.5
    )

    assert result.details["compliance_level"] == "compliant"
    assert "Compliant" in result.details["compliance_label"]
    assert "Compliant" in result.message


# ---------------------------------------------------------------
# Test 13 -- timed_assertion populates duration_ms
# ---------------------------------------------------------------


def test_coverage_assertion_timing():
    # SCENARIO: assert is decorated with @timed_assertion
    # WHY: all mltk assertions must populate duration_ms
    # EXPECTED: duration_ms is a non-negative float

    result = assert_sr_11_7_coverage(
        FULL_COVERAGE_RESULTS, min_coverage=0.1
    )
    assert isinstance(result.duration_ms, float)
    assert result.duration_ms >= 0.0


# ---------------------------------------------------------------
# Test 14 -- uncategorised results go to correct bucket
# ---------------------------------------------------------------


def test_uncategorised_results():
    # SCENARIO: results with unrecognised prefixes
    # WHY: unknown prefixes must not be silently dropped
    # EXPECTED: they appear in "uncategorised" bucket

    grouped = map_results_to_sections(NO_COVERAGE_RESULTS)

    assert "uncategorised" in grouped
    names = [
        r["name"] for r in grouped["uncategorised"]
    ]
    assert "custom.unknown.check" in names


# ---------------------------------------------------------------
# Test 15 -- cross-section mapping
# ---------------------------------------------------------------


def test_cross_section_mapping():
    # SCENARIO: model.metric maps to development + validation;
    #           data.drift maps to development + governance
    # WHY: prefix table allows multi-section mapping
    # EXPECTED: assertions appear in both buckets

    results = [
        {
            "name": "model.metric.accuracy",
            "passed": True,
            "message": "ok",
        },
        {
            "name": "data.drift.psi",
            "passed": True,
            "message": "ok",
        },
    ]
    grouped = map_results_to_sections(results)

    # model.metric -> development and validation
    assert "development" in grouped
    assert "validation" in grouped
    dev_names = [
        r["name"] for r in grouped["development"]
    ]
    val_names = [
        r["name"] for r in grouped["validation"]
    ]
    assert "model.metric.accuracy" in dev_names
    assert "model.metric.accuracy" in val_names

    # data.drift -> development and governance
    assert "governance" in grouped
    dev_drift = [
        r["name"] for r in grouped["development"]
    ]
    gov_names = [
        r["name"] for r in grouped["governance"]
    ]
    assert "data.drift.psi" in dev_drift
    assert "data.drift.psi" in gov_names


# ---------------------------------------------------------------
# Test 16 -- section metadata is consistent
# ---------------------------------------------------------------


def test_section_meta_consistency():
    # SCENARIO: SECTION_META matches SR_11_7_SECTIONS
    # WHY: metadata list is used for deterministic rendering
    # EXPECTED: same count and IDs as SR_11_7_SECTIONS

    assert len(SECTION_META) == len(SR_11_7_SECTION_IDS)
    meta_ids = [m["section"] for m in SECTION_META]
    assert meta_ids == SR_11_7_SECTION_IDS


# ---------------------------------------------------------------
# Test 17 -- two-section partial coverage assertion
# ---------------------------------------------------------------


def test_two_section_coverage():
    # SCENARIO: TWO_SECTION_RESULTS covers 2/3 (67%)
    # WHY: validates boundary between minimal and partial
    # EXPECTED: passes at 0.6 threshold, fails at 0.8

    result = assert_sr_11_7_coverage(
        TWO_SECTION_RESULTS, min_coverage=0.6
    )
    assert result.passed is True
    assert result.details["covered_count"] == 2
    assert result.details["coverage"] == round(2 / 3, 4)

    with pytest.raises(MltkAssertionError):
        assert_sr_11_7_coverage(
            TWO_SECTION_RESULTS, min_coverage=0.8
        )


# ---------------------------------------------------------------
# Test 18 -- low min_coverage always passes with any result
# ---------------------------------------------------------------


def test_coverage_low_threshold():
    # SCENARIO: min_coverage=0.0 with partial results
    # WHY: zero threshold must always pass (even empty)
    # EXPECTED: passed is True regardless of coverage

    result = assert_sr_11_7_coverage(
        PARTIAL_RESULTS, min_coverage=0.0
    )
    assert result.passed is True

    result_empty = assert_sr_11_7_coverage(
        [], min_coverage=0.0
    )
    assert result_empty.passed is True
