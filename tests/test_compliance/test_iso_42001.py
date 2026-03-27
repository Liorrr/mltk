"""Tests for ISO 42001 compliance module (mltk.compliance.iso_42001).

Covers: clause mapping, gap detection, coverage assertion pass/fail,
empty-input edge cases, partial-coverage scenarios, and timing.
"""

from __future__ import annotations

import pytest

from mltk.compliance.iso_42001 import (
    ANNEX_A_CONTROLS,
    ANNEX_A_IDS,
    assert_iso_42001_coverage,
    find_gaps,
    map_results_to_clauses,
)
from mltk.core.assertion import MltkAssertionError

# ---------------------------------------------------------------------------
# Sample result fixtures
# ---------------------------------------------------------------------------

# Covers: A.2 (data.pii, data.schema), A.4 (model.bias, model.adversarial, model.calibration),
#         A.5 (data.schema, data.no_nulls, data.dtypes, data.drift, data.freshness, data.no_pii),
#         A.6 (model.metric, model.regression, model.slice, inference.latency, inference.throughput,
#              monitor.degradation, monitor.sla),
#         A.7 (pipeline.checksum, pipeline.reproducible),
#         A.9 (monitor.degradation, monitor.sla),
#         A.10 (model.bias, model.slice)
# NOT covered: A.8 (Documentation — no assertion prefixes exist)
FULL_SAMPLE_RESULTS: list[dict] = [
    # A.2 — AI Policies
    {"name": "data.pii.scan",                    "passed": True,  "message": "no PII detected"},
    {"name": "data.schema.columns",              "passed": True,  "message": "schema ok"},
    # A.4 — AI Risk Assessment
    {"name": "model.bias.demographic_parity",    "passed": False, "message": "disparity=0.18"},
    {"name": "model.adversarial.fgsm",           "passed": True,  "message": "robust"},
    {"name": "model.calibration.ece",            "passed": True,  "message": "ECE=0.04"},
    # A.5 — Data Quality (some overlap with A.2)
    {"name": "data.no_nulls.train",              "passed": True,  "message": "0 nulls"},
    {"name": "data.dtypes.numeric",              "passed": True,  "message": "types ok"},
    {"name": "data.drift.psi",                   "passed": True,  "message": "PSI=0.03"},
    {"name": "data.freshness.check",             "passed": True,  "message": "fresh"},
    {"name": "data.no_pii.emails",               "passed": True,  "message": "no PII"},
    # A.6 — System Performance
    {"name": "model.metric.accuracy",            "passed": True,  "message": "accuracy=0.94"},
    {"name": "model.regression.baseline",        "passed": True,  "message": "no regression"},
    {"name": "model.slice.gender",               "passed": True,  "message": "balanced"},
    {"name": "inference.latency.p99",            "passed": True,  "message": "p99=120ms"},
    {"name": "inference.throughput.rps",          "passed": True,  "message": "rps=320"},
    {"name": "monitor.degradation.week",         "passed": True,  "message": "stable"},
    {"name": "monitor.sla.uptime",               "passed": True,  "message": "99.9%"},
    # A.7 — Third Party
    {"name": "pipeline.checksum.model",          "passed": True,  "message": "checksum ok"},
    {"name": "pipeline.reproducible.train",      "passed": True,  "message": "reproducible"},
]

# Only covers A.6 and A.10 partially — well below 80% threshold
SPARSE_RESULTS: list[dict] = [
    {"name": "model.metric.accuracy",  "passed": True,  "message": "accuracy=0.91"},
    {"name": "model.slice.age",        "passed": True,  "message": "balanced"},
]

# Covers exactly 4 of 8 clauses (50%): A.2, A.4, A.5, A.10
HALF_COVERAGE_RESULTS: list[dict] = [
    {"name": "data.pii.scan", "passed": True, "message": ""},  # A.2 + A.5
    {"name": "data.schema.x",                  "passed": True, "message": ""},  # A.2 + A.5
    {"name": "model.bias.x",                   "passed": True, "message": ""},  # A.4 + A.10
    {"name": "data.no_nulls.x",                "passed": True, "message": ""},  # A.5
]


# ---------------------------------------------------------------------------
# Test 1 — map_results_to_clauses groups by clause correctly
# ---------------------------------------------------------------------------


def test_map_results_to_clauses():
    # SCENARIO: a rich results list covering many assertion prefixes is grouped
    # WHY:      the clause mapper is the core grouping function; each assertion
    #           must land in every clause that claims it via prefix matching
    # EXPECTED: A.4 contains model.bias; A.6 contains model.metric;
    #           A.10 contains model.bias AND model.slice

    grouped = map_results_to_clauses(FULL_SAMPLE_RESULTS)

    # A.4 — AI Risk Assessment (model.bias, model.adversarial, model.calibration)
    assert "A.4" in grouped
    a4_names = [r["name"] for r in grouped["A.4"]]
    assert "model.bias.demographic_parity" in a4_names
    assert "model.adversarial.fgsm"        in a4_names
    assert "model.calibration.ece"         in a4_names

    # A.6 — System Performance (model.metric, inference.latency, etc.)
    assert "A.6" in grouped
    a6_names = [r["name"] for r in grouped["A.6"]]
    assert "model.metric.accuracy"     in a6_names
    assert "inference.latency.p99"     in a6_names
    assert "monitor.degradation.week"  in a6_names

    # A.10 — Bias and Fairness (model.bias, model.slice)
    assert "A.10" in grouped
    a10_names = [r["name"] for r in grouped["A.10"]]
    assert "model.bias.demographic_parity" in a10_names
    assert "model.slice.gender"            in a10_names

    # A.7 — Third Party (pipeline.checksum, pipeline.reproducible)
    assert "A.7" in grouped
    a7_names = [r["name"] for r in grouped["A.7"]]
    assert "pipeline.checksum.model"      in a7_names
    assert "pipeline.reproducible.train"  in a7_names


# ---------------------------------------------------------------------------
# Test 2 — map_results_to_clauses enriches clause key
# ---------------------------------------------------------------------------


def test_map_results_to_clauses_enriches_clause_key():
    # SCENARIO: mapped results are checked for the injected "clause" key
    # WHY:      downstream consumers need the clause key on each result dict
    # EXPECTED: every result dict in a named bucket has a "clause" key matching its bucket

    grouped = map_results_to_clauses(FULL_SAMPLE_RESULTS)
    for clause, items in grouped.items():
        if clause == "uncategorised":
            continue
        for item in items:
            assert item.get("clause") == clause, (
                f"Result {item['name']!r} in bucket {clause!r} "
                f"has clause={item.get('clause')!r}"
            )


# ---------------------------------------------------------------------------
# Test 3 — map_results_to_clauses handles unrecognised assertions
# ---------------------------------------------------------------------------


def test_map_results_to_clauses_uncategorised():
    # SCENARIO: results list includes an assertion that matches no clause prefix
    # WHY:      unknown assertions should land in "uncategorised" rather than
    #           being silently dropped or raising an error
    # EXPECTED: "uncategorised" bucket contains the unknown assertion

    results = [
        {"name": "custom.check.something", "passed": True, "message": "ok"},
    ]
    grouped = map_results_to_clauses(results)

    assert "uncategorised" in grouped
    names = [r["name"] for r in grouped["uncategorised"]]
    assert "custom.check.something" in names


# ---------------------------------------------------------------------------
# Test 4 — find_gaps with empty results returns all clauses
# ---------------------------------------------------------------------------


def test_find_gaps_empty_results():
    # SCENARIO: no test results are provided at all
    # WHY:      every clause should be reported as a gap when nothing is tested
    # EXPECTED: all 8 Annex A clause IDs appear in the gaps list

    gaps = find_gaps([])

    assert len(gaps) == len(ANNEX_A_IDS), (
        f"Expected {len(ANNEX_A_IDS)} gaps for empty results, got {len(gaps)}"
    )
    for clause_id in ANNEX_A_IDS:
        assert clause_id in gaps, f"Expected {clause_id} in gaps"


# ---------------------------------------------------------------------------
# Test 5 — find_gaps always includes A.8 (Documentation)
# ---------------------------------------------------------------------------


def test_find_gaps_always_includes_documentation():
    # SCENARIO: full sample results cover every clause except A.8
    # WHY:      A.8 (Documentation) has no assertion prefixes and cannot be
    #           covered by automated tests; it must always appear as a gap
    # EXPECTED: "A.8" is in gaps even with comprehensive test coverage

    gaps = find_gaps(FULL_SAMPLE_RESULTS)
    assert "A.8" in gaps, (
        "A.8 (Documentation) should always be a gap since it has no "
        "mapped assertion prefixes"
    )


# ---------------------------------------------------------------------------
# Test 6 — find_gaps returns only uncovered clauses
# ---------------------------------------------------------------------------


def test_find_gaps_partial_coverage():
    # SCENARIO: results cover A.4 and A.10 via model.bias, but nothing else
    # WHY:      only the clauses with matching assertions should be excluded
    #           from the gaps list; all others should be present
    # EXPECTED: A.4 and A.10 are NOT in gaps; A.2, A.5, A.6, A.7, A.8, A.9 are

    results = [
        {"name": "model.bias.x", "passed": True, "message": ""},
    ]
    gaps = find_gaps(results)

    # model.bias maps to A.4 and A.10 — these should NOT be gaps
    assert "A.4"  not in gaps, "A.4 is covered by model.bias"
    assert "A.10" not in gaps, "A.10 is covered by model.bias"

    # Everything else should be a gap
    assert "A.2" in gaps
    assert "A.5" in gaps
    assert "A.6" in gaps
    assert "A.7" in gaps
    assert "A.8" in gaps
    assert "A.9" in gaps


# ---------------------------------------------------------------------------
# Test 7 — assert_iso_42001_coverage passes when enough clauses are covered
# ---------------------------------------------------------------------------


def test_coverage_assertion_pass():
    # SCENARIO: FULL_SAMPLE_RESULTS covers 7 of 8 clauses (only A.8 is missing)
    # WHY:      with 87.5% coverage, the default 80% threshold should be met
    # EXPECTED: TestResult.passed is True; covered_count == 7; message says "meets"

    result = assert_iso_42001_coverage(FULL_SAMPLE_RESULTS, min_coverage=0.8)

    assert result.passed is True
    assert result.details["covered_count"] == 7
    assert result.details["total"] == 8
    assert result.details["coverage"] >= 0.8
    assert result.details["min_coverage"] == 0.8
    assert "meets" in result.message
    assert result.name == "compliance.iso_42001.coverage"


# ---------------------------------------------------------------------------
# Test 8 — assert_iso_42001_coverage fails when too few clauses are covered
# ---------------------------------------------------------------------------


def test_coverage_assertion_fail():
    # SCENARIO: SPARSE_RESULTS covers only A.6 and A.10 (2 of 8 = 25%),
    #           well below the 80% minimum threshold
    # WHY:      the assertion must raise MltkAssertionError on low coverage
    # EXPECTED: MltkAssertionError raised; result.passed is False;
    #           message contains "below"

    with pytest.raises(MltkAssertionError) as exc_info:
        assert_iso_42001_coverage(SPARSE_RESULTS, min_coverage=0.8)

    result = exc_info.value.result
    assert result.passed is False
    assert result.details["covered_count"] < 7
    assert "below" in result.message


# ---------------------------------------------------------------------------
# Test 9 — assert_iso_42001_coverage stores timing in duration_ms
# ---------------------------------------------------------------------------


def test_coverage_assertion_timing():
    # SCENARIO: assert_iso_42001_coverage is decorated with @timed_assertion
    # WHY:      all mltk assertions must populate duration_ms for perf tracking
    # EXPECTED: result.duration_ms is a non-negative float

    result = assert_iso_42001_coverage(FULL_SAMPLE_RESULTS, min_coverage=0.1)
    assert isinstance(result.duration_ms, float)
    assert result.duration_ms >= 0.0


# ---------------------------------------------------------------------------
# Test 10 — coverage boundary: passes at exact threshold, fails just above
# ---------------------------------------------------------------------------


def test_coverage_boundary():
    # SCENARIO: HALF_COVERAGE_RESULTS covers exactly 4 of 8 clauses (50%)
    # WHY:      boundary behaviour — coverage must pass at exactly 0.5
    #           but fail at 0.6
    # EXPECTED: passes at min_coverage=0.5; raises at min_coverage=0.6

    result_pass = assert_iso_42001_coverage(
        HALF_COVERAGE_RESULTS, min_coverage=0.5,
    )
    assert result_pass.passed is True
    assert result_pass.details["coverage"] >= 0.5

    with pytest.raises(MltkAssertionError):
        assert_iso_42001_coverage(HALF_COVERAGE_RESULTS, min_coverage=0.6)


# ---------------------------------------------------------------------------
# Test 11 — ANNEX_A_CONTROLS data integrity
# ---------------------------------------------------------------------------


def test_annex_a_controls_structure():
    # SCENARIO: verify that every entry in ANNEX_A_CONTROLS has the
    #           required keys and correct types
    # WHY:      downstream code depends on title, description, assertions
    #           being present and well-typed
    # EXPECTED: all 8 controls have title (str), description (str),
    #           assertions (list of str)

    assert len(ANNEX_A_CONTROLS) == 8, (
        f"Expected 8 Annex A controls, got {len(ANNEX_A_CONTROLS)}"
    )

    for clause_id, meta in ANNEX_A_CONTROLS.items():
        assert "title" in meta, f"{clause_id} missing 'title'"
        assert "description" in meta, f"{clause_id} missing 'description'"
        assert "assertions" in meta, f"{clause_id} missing 'assertions'"
        assert isinstance(meta["title"], str), f"{clause_id} title not str"
        assert isinstance(meta["description"], str), f"{clause_id} description not str"
        assert isinstance(meta["assertions"], list), f"{clause_id} assertions not list"
        for prefix in meta["assertions"]:
            assert isinstance(prefix, str), (
                f"{clause_id} has non-string assertion prefix: {prefix!r}"
            )
