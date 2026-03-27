"""Tests for NIST AI RMF mapping module (mltk.compliance.nist_ai_rmf).

Covers: result mapping, gap detection (full/partial/no coverage),
coverage assertion pass/fail, tier classification, edge cases (empty results),
and timing decorator.
"""

from __future__ import annotations

import pytest

from mltk.compliance.nist_ai_rmf import (
    NIST_RMF_FUNCTION_IDS,
    TIER_CLASSIFICATION,
    TIERS,
    assert_nist_rmf_coverage,
    classify_tier,
    find_gaps,
    map_results_to_measures,
)
from mltk.core.assertion import MltkAssertionError

# ---------------------------------------------------------------------------
# Sample result fixtures
# ---------------------------------------------------------------------------

# Covers all 4 functions: GV (data.pii, model.bias), MP (model.metric, data.drift),
# MS (model.metric, model.regression, inference.latency), MN (monitor.degradation, data.drift)
FULL_COVERAGE_RESULTS: list[dict] = [
    # GV -- GOVERN
    {"name": "data.pii.scan",                   "passed": True,  "message": "no PII"},
    {"name": "data.schema.columns",             "passed": True,  "message": "schema ok"},
    {"name": "model.bias.demographic_parity",   "passed": False, "message": "disparity=0.18"},
    # MP -- MAP
    {"name": "model.metric.accuracy",           "passed": True,  "message": "accuracy=0.94"},
    {"name": "model.slice.gender",              "passed": True,  "message": "slice ok"},
    {"name": "data.drift.psi",                  "passed": True,  "message": "PSI=0.04"},
    # MS -- MEASURE
    {"name": "model.regression.v2_vs_v1",       "passed": True,  "message": "no regression"},
    {"name": "model.calibration.ece",           "passed": True,  "message": "ECE=0.04"},
    {"name": "model.adversarial.fgsm",          "passed": True,  "message": "robust"},
    {"name": "inference.latency.p99",           "passed": True,  "message": "p99=120ms"},
    {"name": "inference.throughput.rps",         "passed": True,  "message": "rps=320"},
    # MN -- MANAGE
    {"name": "monitor.degradation.week",        "passed": True,  "message": "no degradation"},
    {"name": "monitor.sla.uptime",              "passed": True,  "message": "99.9%"},
]

# Only covers GV (data.pii) and MS (inference.latency) -- 2 of 4 functions
PARTIAL_RESULTS: list[dict] = [
    {"name": "data.pii.scan",           "passed": True, "message": "no PII"},
    {"name": "inference.latency.p99",   "passed": True, "message": "p99=120ms"},
]

# Covers nothing -- unrecognised assertion prefixes
NO_COVERAGE_RESULTS: list[dict] = [
    {"name": "custom.unknown.check",    "passed": True, "message": "ok"},
]

# Covers exactly 2 of 4 functions = 50%
HALF_COVERAGE_RESULTS: list[dict] = [
    {"name": "data.pii.scan",          "passed": True, "message": "no PII"},         # GV
    {"name": "model.metric.accuracy",  "passed": True, "message": "accuracy=0.94"},  # MP + MS
]


# ---------------------------------------------------------------------------
# Test 1 -- map_results_to_measures groups results by RMF function
# ---------------------------------------------------------------------------


def test_map_results_to_measures_grouping():
    # SCENARIO: a rich results list covering all 4 RMF functions is mapped
    # WHY:      the mapper is the core grouping function; every assertion must
    #           land in the correct function bucket(s) according to prefix table
    # EXPECTED: GV bucket has data.pii/schema/bias; MS has metric/regression/etc.

    grouped = map_results_to_measures(FULL_COVERAGE_RESULTS)

    # GV -- GOVERN
    assert "GV" in grouped
    gv_names = [r["name"] for r in grouped["GV"]]
    assert "data.pii.scan" in gv_names
    assert "data.schema.columns" in gv_names
    assert "model.bias.demographic_parity" in gv_names

    # MP -- MAP
    assert "MP" in grouped
    mp_names = [r["name"] for r in grouped["MP"]]
    assert "model.metric.accuracy" in mp_names
    assert "model.slice.gender" in mp_names
    assert "data.drift.psi" in mp_names

    # MS -- MEASURE
    assert "MS" in grouped
    ms_names = [r["name"] for r in grouped["MS"]]
    assert "model.regression.v2_vs_v1" in ms_names
    assert "model.calibration.ece" in ms_names
    assert "model.adversarial.fgsm" in ms_names
    assert "inference.latency.p99" in ms_names
    assert "inference.throughput.rps" in ms_names
    # model.metric maps to both MP and MS
    assert "model.metric.accuracy" in ms_names

    # MN -- MANAGE
    assert "MN" in grouped
    mn_names = [r["name"] for r in grouped["MN"]]
    assert "monitor.degradation.week" in mn_names
    assert "monitor.sla.uptime" in mn_names
    # data.drift maps to both MP and MN
    assert "data.drift.psi" in mn_names


# ---------------------------------------------------------------------------
# Test 2 -- map_results_to_measures enriches results with "function" key
# ---------------------------------------------------------------------------


def test_map_results_enriches_function_key():
    # SCENARIO: mapped results are checked for the injected "function" key
    # WHY:      downstream consumers (e.g. report generators) need the function
    #           key on each result dict to render per-function sections
    # EXPECTED: every result dict in a named bucket has a "function" key matching

    grouped = map_results_to_measures(FULL_COVERAGE_RESULTS)
    for func_id, items in grouped.items():
        if func_id == "uncategorised":
            continue
        for item in items:
            assert item.get("function") == func_id, (
                f"Result {item['name']!r} in bucket {func_id!r} "
                f"has function={item.get('function')!r}"
            )


# ---------------------------------------------------------------------------
# Test 3 -- map_results_to_measures with empty results
# ---------------------------------------------------------------------------


def test_map_results_empty():
    # SCENARIO: an empty results list is passed
    # WHY:      the mapper must return an empty dict without raising
    # EXPECTED: empty dict, no keys

    grouped = map_results_to_measures([])
    assert grouped == {}


# ---------------------------------------------------------------------------
# Test 4 -- find_gaps returns all functions when no results
# ---------------------------------------------------------------------------


def test_find_gaps_no_results():
    # SCENARIO: no test results at all
    # WHY:      all 4 RMF functions should be reported as gaps
    # EXPECTED: sorted list of all 4 function codes

    gaps = find_gaps([])
    assert gaps == sorted(NIST_RMF_FUNCTION_IDS)
    assert len(gaps) == 4


# ---------------------------------------------------------------------------
# Test 5 -- find_gaps returns empty list when fully covered
# ---------------------------------------------------------------------------


def test_find_gaps_full_coverage():
    # SCENARIO: results cover all 4 RMF functions
    # WHY:      no gaps should be reported when every function is covered
    # EXPECTED: empty list

    gaps = find_gaps(FULL_COVERAGE_RESULTS)
    assert gaps == [], f"Expected no gaps but got: {gaps}"


# ---------------------------------------------------------------------------
# Test 6 -- find_gaps returns uncovered functions for partial coverage
# ---------------------------------------------------------------------------


def test_find_gaps_partial_coverage():
    # SCENARIO: PARTIAL_RESULTS covers GV (data.pii) and MS (inference.latency)
    #           but NOT MP or MN
    # WHY:      gaps must accurately reflect which functions lack test coverage
    # EXPECTED: MP and MN appear in gaps; GV and MS do not

    gaps = find_gaps(PARTIAL_RESULTS)

    assert "GV" not in gaps, "GV is covered by data.pii"
    assert "MS" not in gaps, "MS is covered by inference.latency"
    assert "MP" in gaps, "MP has no matching results"
    assert "MN" in gaps, "MN has no matching results"


# ---------------------------------------------------------------------------
# Test 7 -- assert_nist_rmf_coverage passes when all functions are covered
# ---------------------------------------------------------------------------


def test_coverage_assertion_pass():
    # SCENARIO: FULL_COVERAGE_RESULTS covers all 4 RMF functions
    # WHY:      at the default 80% threshold, 4/4 (100%) must pass
    # EXPECTED: TestResult.passed is True; covered_count == 4; no exception

    result = assert_nist_rmf_coverage(FULL_COVERAGE_RESULTS, min_coverage=0.8)

    assert result.passed is True
    assert result.details["covered_count"] == 4
    assert result.details["total"] == 4
    assert result.details["coverage"] == 1.0
    assert result.details["min_coverage"] == 0.8
    assert "meets" in result.message
    assert result.name == "compliance.nist_ai_rmf.coverage"


# ---------------------------------------------------------------------------
# Test 8 -- assert_nist_rmf_coverage fails when below threshold
# ---------------------------------------------------------------------------


def test_coverage_assertion_fail():
    # SCENARIO: PARTIAL_RESULTS covers only 2/4 functions (50%),
    #           below the 80% minimum threshold
    # WHY:      the assertion must raise MltkAssertionError on low coverage
    # EXPECTED: MltkAssertionError raised; result.passed is False;
    #           message contains "below"

    with pytest.raises(MltkAssertionError) as exc_info:
        assert_nist_rmf_coverage(PARTIAL_RESULTS, min_coverage=0.8)

    result = exc_info.value.result
    assert result.passed is False
    assert result.details["covered_count"] == 2
    assert result.details["coverage"] == 0.5
    assert "below" in result.message


# ---------------------------------------------------------------------------
# Test 9 -- assert_nist_rmf_coverage with empty results
# ---------------------------------------------------------------------------


def test_coverage_assertion_empty_results():
    # SCENARIO: no results at all, 80% threshold
    # WHY:      empty input should fail gracefully, not crash
    # EXPECTED: MltkAssertionError raised; coverage is 0.0

    with pytest.raises(MltkAssertionError) as exc_info:
        assert_nist_rmf_coverage([], min_coverage=0.8)

    result = exc_info.value.result
    assert result.passed is False
    assert result.details["covered_count"] == 0
    assert result.details["coverage"] == 0.0


# ---------------------------------------------------------------------------
# Test 10 -- classify_tier returns correct tiers
# ---------------------------------------------------------------------------


def test_classify_tier():
    # SCENARIO: various coverage values are classified into NIST tiers
    # WHY:      tier classification drives the badge and label in reports
    # EXPECTED: correct tier for each coverage bracket

    assert classify_tier(0.0) == "partial"
    assert classify_tier(0.1) == "partial"
    assert classify_tier(0.24) == "partial"
    assert classify_tier(0.25) == "risk_informed"
    assert classify_tier(0.4) == "risk_informed"
    assert classify_tier(0.49) == "risk_informed"
    assert classify_tier(0.5) == "repeatable"
    assert classify_tier(0.6) == "repeatable"
    assert classify_tier(0.74) == "repeatable"
    assert classify_tier(0.75) == "adaptive"
    assert classify_tier(0.9) == "adaptive"
    assert classify_tier(1.0) == "adaptive"


# ---------------------------------------------------------------------------
# Test 11 -- tier classification data is complete
# ---------------------------------------------------------------------------


def test_tier_classification_data():
    # SCENARIO: all tier keys have complete metadata
    # WHY:      report rendering depends on label, description, color, badge_class
    # EXPECTED: every tier in TIERS has all required keys in TIER_CLASSIFICATION

    required_keys = {"label", "description", "color", "badge_class"}
    for tier in TIERS:
        assert tier in TIER_CLASSIFICATION, f"Tier {tier!r} missing from classification"
        info = TIER_CLASSIFICATION[tier]
        assert required_keys.issubset(info.keys()), (
            f"Missing keys for tier {tier!r}: {required_keys - set(info.keys())}"
        )


# ---------------------------------------------------------------------------
# Test 12 -- coverage assertion includes tier in details and message
# ---------------------------------------------------------------------------


def test_coverage_assertion_includes_tier():
    # SCENARIO: assert_nist_rmf_coverage returns tier info in details
    # WHY:      downstream consumers need the tier for badge rendering
    # EXPECTED: details contain tier and tier_label; message mentions the label

    result = assert_nist_rmf_coverage(FULL_COVERAGE_RESULTS, min_coverage=0.5)

    assert result.details["tier"] == "adaptive"
    assert "Adaptive" in result.details["tier_label"]
    assert "Adaptive" in result.message


# ---------------------------------------------------------------------------
# Test 13 -- timed_assertion populates duration_ms
# ---------------------------------------------------------------------------


def test_coverage_assertion_timing():
    # SCENARIO: assert_nist_rmf_coverage is decorated with @timed_assertion
    # WHY:      all mltk assertions must populate duration_ms for perf tracking
    # EXPECTED: result.duration_ms is a non-negative float

    result = assert_nist_rmf_coverage(FULL_COVERAGE_RESULTS, min_coverage=0.1)
    assert isinstance(result.duration_ms, float)
    assert result.duration_ms >= 0.0


# ---------------------------------------------------------------------------
# Test 14 -- uncategorised results go to "uncategorised" bucket
# ---------------------------------------------------------------------------


def test_uncategorised_results():
    # SCENARIO: results with unrecognised assertion prefixes are mapped
    # WHY:      unknown prefixes must not be silently dropped
    # EXPECTED: they appear in the "uncategorised" bucket

    grouped = map_results_to_measures(NO_COVERAGE_RESULTS)

    assert "uncategorised" in grouped
    names = [r["name"] for r in grouped["uncategorised"]]
    assert "custom.unknown.check" in names


# ---------------------------------------------------------------------------
# Test 15 -- cross-function mapping (assertions map to multiple functions)
# ---------------------------------------------------------------------------


def test_cross_function_mapping():
    # SCENARIO: model.metric maps to both MP and MS; data.drift maps to MP and MN
    # WHY:      the prefix table allows one assertion to satisfy multiple functions
    # EXPECTED: model.metric appears in both MP and MS buckets

    results = [
        {"name": "model.metric.accuracy", "passed": True, "message": "ok"},
        {"name": "data.drift.psi",        "passed": True, "message": "ok"},
    ]
    grouped = map_results_to_measures(results)

    # model.metric -> MP and MS
    assert "MP" in grouped
    assert "MS" in grouped
    mp_names = [r["name"] for r in grouped["MP"]]
    ms_names = [r["name"] for r in grouped["MS"]]
    assert "model.metric.accuracy" in mp_names
    assert "model.metric.accuracy" in ms_names

    # data.drift -> MP and MN
    assert "MN" in grouped
    mp_names_drift = [r["name"] for r in grouped["MP"]]
    mn_names = [r["name"] for r in grouped["MN"]]
    assert "data.drift.psi" in mp_names_drift
    assert "data.drift.psi" in mn_names
