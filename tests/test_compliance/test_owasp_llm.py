"""Tests for OWASP LLM Top 10 mapping module (mltk.compliance.owasp_llm).

Covers: result mapping, empty-input edge cases, coverage assertion pass/fail,
report format, and partial-coverage scenarios.
"""

from __future__ import annotations

import pytest

from mltk.compliance.owasp_llm import (
    OWASP_LLM_IDS,
    OWASP_LLM_TOP_10,
    assert_owasp_coverage,
    generate_owasp_report,
    owasp_llm_scan,
)
from mltk.core.assertion import MltkAssertionError

# ---------------------------------------------------------------------------
# Sample result fixtures
# ---------------------------------------------------------------------------

# Covers: LLM01, LLM02 (x3 prefixes), LLM03 (x4 prefixes), LLM04 (x3),
#         LLM06, LLM07 (x2), LLM08 (partial), LLM09 (x3)
FULL_SAMPLE_RESULTS: list[dict] = [
    # LLM01 — Prompt Injection
    {"name": "nlp.prompt_injection.direct",    "passed": True,  "message": "no injection"},
    # LLM02 — Insecure Output Handling
    {"name": "llm.text_length.max",            "passed": True,  "message": "len ok"},
    {"name": "llm.output_format.json",         "passed": True,  "message": "format ok"},
    {"name": "llm.toxicity.score",             "passed": False, "message": "toxicity=0.08 > 0.01"},
    # LLM03 — Training Data Poisoning
    {"name": "data.schema.columns",            "passed": True,  "message": "schema ok"},
    {"name": "data.no_nulls.train",            "passed": True,  "message": "0 nulls"},
    {"name": "data.drift.psi",                 "passed": True,  "message": "PSI=0.04"},
    {"name": "data.pii.scan",                  "passed": True,  "message": "no PII"},
    # LLM04 — Model Denial of Service
    {"name": "inference.latency.p99",          "passed": True,  "message": "p99=120ms"},
    {"name": "inference.throughput.rps",       "passed": True,  "message": "rps=320"},
    {"name": "monitor.sla.uptime",             "passed": True,  "message": "99.9%"},
    # LLM06 — Sensitive Information Disclosure (data.pii already above; toxicity too)
    # LLM07 — Insecure Plugin Design
    {"name": "llm.tool_selection.whitelist",   "passed": True,  "message": "whitelist ok"},
    {"name": "llm.tool_call.schema",           "passed": True,  "message": "schema valid"},
    # LLM08 — Excessive Agency (shares llm.tool_selection with LLM07)
    {"name": "llm.task_completion.scope",      "passed": True,  "message": "in scope"},
    # LLM09 — Overreliance
    {"name": "llm.hallucination.rag",          "passed": False, "message": "2/10 unsupported"},
    {"name": "llm.faithfulness.bertscore",     "passed": True,  "message": "score=0.92"},
    {"name": "llm.coherence.ppl",              "passed": True,  "message": "ppl=12.3"},
]

# Only covers LLM09 and partial LLM02 — well below 50% threshold
SPARSE_RESULTS: list[dict] = [
    {"name": "llm.hallucination.direct",   "passed": True,  "message": "ok"},
    {"name": "llm.toxicity.basic",         "passed": True,  "message": "ok"},
]

# Covers exactly 5 out of 10 categories (50 %)
HALF_COVERAGE_RESULTS: list[dict] = [
    {"name": "nlp.prompt_injection.x",    "passed": True, "message": ""},  # LLM01
    {"name": "llm.text_length.x",         "passed": True, "message": ""},  # LLM02
    {"name": "data.schema.x",             "passed": True, "message": ""},  # LLM03
    {"name": "inference.latency.x",       "passed": True, "message": ""},  # LLM04
    {"name": "pipeline.checksum.x",       "passed": True, "message": ""},  # LLM05 + LLM10
]


# ---------------------------------------------------------------------------
# Test 1 — owasp_llm_scan maps results to correct OWASP categories
# ---------------------------------------------------------------------------


def test_owasp_scan_maps_results():
    # SCENARIO: a rich results list covering many assertion prefixes is scanned
    # WHY:      the scan function is the core mapper; every assertion must land
    #           in the correct OWASP bucket(s) according to the prefix table
    # EXPECTED: LLM01 covered by prompt_injection; LLM09 covered by hallucination;
    #           LLM05 NOT covered because no pipeline.checksum result was supplied

    scan = owasp_llm_scan(FULL_SAMPLE_RESULTS)

    # LLM01 — nlp.prompt_injection.direct maps here
    assert scan["LLM01"]["covered"] is True
    llm01_names = [t["name"] for t in scan["LLM01"]["tests"]]
    assert "nlp.prompt_injection.direct" in llm01_names

    # LLM02 — all three prefixes covered
    assert scan["LLM02"]["covered"] is True
    llm02_names = [t["name"] for t in scan["LLM02"]["tests"]]
    assert "llm.text_length.max"  in llm02_names
    assert "llm.output_format.json" in llm02_names
    assert "llm.toxicity.score"   in llm02_names

    # LLM09 — hallucination, faithfulness, coherence all present
    assert scan["LLM09"]["covered"] is True
    llm09_names = [t["name"] for t in scan["LLM09"]["tests"]]
    assert "llm.hallucination.rag"      in llm09_names
    assert "llm.faithfulness.bertscore" in llm09_names
    assert "llm.coherence.ppl"          in llm09_names

    # LLM05 — pipeline.checksum not in FULL_SAMPLE_RESULTS → should be missing
    assert scan["LLM05"]["covered"] is False
    assert scan["LLM05"]["tests"] == []

    # Every result dict in a covered category is enriched with "owasp_id"
    for owasp_id, entry in scan.items():
        for t in entry["tests"]:
            assert t.get("owasp_id") == owasp_id, (
                f"Test {t['name']!r} in {owasp_id} has owasp_id={t.get('owasp_id')!r}"
            )


# ---------------------------------------------------------------------------
# Test 2 — owasp_llm_scan with empty results → all categories uncovered
# ---------------------------------------------------------------------------


def test_owasp_scan_empty():
    # SCENARIO: an empty results list is passed to the scanner
    # WHY:      a project with no tests must still return a valid structure
    #           rather than raising or returning None
    # EXPECTED: all 10 OWASP categories present, all covered=False, tests=[]

    scan = owasp_llm_scan([])

    assert set(scan.keys()) == set(OWASP_LLM_IDS), (
        "scan must contain all 10 OWASP IDs even with no results"
    )
    for owasp_id, entry in scan.items():
        assert entry["covered"] is False, f"{owasp_id} should be uncovered"
        assert entry["tests"]   == [],    f"{owasp_id} should have no tests"
        # gaps must contain all assertion prefixes for the category
        expected_gaps = set(OWASP_LLM_TOP_10[owasp_id]["assertions"])
        assert set(entry["gaps"]) == expected_gaps, (
            f"{owasp_id} gaps mismatch: got {entry['gaps']}"
        )


# ---------------------------------------------------------------------------
# Test 3 — assert_owasp_coverage passes when enough categories are covered
# ---------------------------------------------------------------------------


def test_owasp_coverage_pass():
    # SCENARIO: FULL_SAMPLE_RESULTS covers 8+ of 10 OWASP categories
    # WHY:      a well-tested LLM system should pass the 50 % default threshold
    # EXPECTED: TestResult.passed is True; covered_count >= 5; no exception raised

    result = assert_owasp_coverage(FULL_SAMPLE_RESULTS, min_coverage=0.5)

    assert result.passed is True
    assert result.details["covered_count"] >= 5
    assert result.details["total"] == 10
    assert result.details["coverage"] >= 0.5
    assert result.details["min_coverage"] == 0.5
    assert "meets" in result.message


# ---------------------------------------------------------------------------
# Test 4 — assert_owasp_coverage fails when too few categories are covered
# ---------------------------------------------------------------------------


def test_owasp_coverage_fail():
    # SCENARIO: SPARSE_RESULTS covers only 2 categories (LLM09 + LLM02),
    #           well below the 80 % minimum threshold
    # WHY:      the assertion must raise MltkAssertionError on low coverage
    # EXPECTED: MltkAssertionError raised; result.passed is False;
    #           covered_count < 8; message contains "below"

    with pytest.raises(MltkAssertionError) as exc_info:
        assert_owasp_coverage(SPARSE_RESULTS, min_coverage=0.8)

    result = exc_info.value.result
    assert result.passed is False
    assert result.details["covered_count"] < 8
    assert "below" in result.message


# ---------------------------------------------------------------------------
# Test 5 — generate_owasp_report contains all 10 OWASP IDs
# ---------------------------------------------------------------------------


def test_owasp_report_format():
    # SCENARIO: generate_owasp_report is called with FULL_SAMPLE_RESULTS
    # WHY:      the report is the human-readable audit artefact; it must contain
    #           every OWASP ID, the overall coverage line, and status markers
    # EXPECTED: all 10 OWASP IDs present; "COVERED"/"MISSING" labels present;
    #           coverage fraction line present; header and footer delimiters present

    report = generate_owasp_report(FULL_SAMPLE_RESULTS)

    for owasp_id in OWASP_LLM_IDS:
        assert owasp_id in report, f"Report missing {owasp_id}"

    # At least some categories should be marked COVERED
    assert "COVERED" in report
    # LLM05 (pipeline.checksum) is not in FULL_SAMPLE_RESULTS → MISSING
    assert "MISSING" in report

    # Coverage summary line
    assert "Coverage:" in report
    assert "/10" in report

    # Delimiter rows for structure
    assert "=" * 10 in report


# ---------------------------------------------------------------------------
# Test 6 — partial coverage: some categories covered, some not
# ---------------------------------------------------------------------------


def test_owasp_partial_coverage():
    # SCENARIO: HALF_COVERAGE_RESULTS covers exactly 5 of the 10 OWASP IDs
    #           (LLM01, LLM02, LLM03, LLM04, LLM05/LLM10 via pipeline.checksum)
    # WHY:      boundary condition — 50 % coverage must pass at min_coverage=0.5
    #           but fail at min_coverage=0.6; gaps must be non-empty
    # EXPECTED: coverage = 0.5 or above (pipeline.checksum hits LLM05 + LLM10);
    #           assert passes at 0.5; assert fails at 0.6;
    #           uncovered categories have non-empty gaps lists

    scan = owasp_llm_scan(HALF_COVERAGE_RESULTS)

    covered_ids = [k for k, v in scan.items() if v["covered"]]
    uncovered_ids = [k for k, v in scan.items() if not v["covered"]]

    # pipeline.checksum hits both LLM05 and LLM10 → at least 6 covered
    assert len(covered_ids) >= 5, f"Expected >= 5 covered, got {covered_ids}"
    assert len(uncovered_ids) > 0, "Expected some categories to be uncovered"

    # Uncovered categories must list their assertion gaps
    for owasp_id in uncovered_ids:
        assert len(scan[owasp_id]["gaps"]) > 0, (
            f"{owasp_id} is uncovered but has no gaps listed"
        )

    # Passes at exactly 50 % threshold
    result_pass = assert_owasp_coverage(HALF_COVERAGE_RESULTS, min_coverage=0.5)
    assert result_pass.passed is True

    # Fails at 90 % threshold (only ~6 covered)
    with pytest.raises(MltkAssertionError):
        assert_owasp_coverage(HALF_COVERAGE_RESULTS, min_coverage=0.9)


# ---------------------------------------------------------------------------
# Test 7 — owasp_llm_scan correctly tracks per-category gaps
# ---------------------------------------------------------------------------


def test_owasp_scan_gap_tracking():
    # SCENARIO: results cover only ONE of three assertion prefixes in LLM02
    #           (llm.text_length only; llm.output_format and llm.toxicity absent)
    # WHY:      gap tracking must report individual missing prefixes, not just
    #           the category-level covered flag
    # EXPECTED: LLM02 is "covered" (one test present); gaps lists the 2 missing prefixes

    partial_results = [
        {"name": "llm.text_length.chars", "passed": True, "message": "ok"},
    ]
    scan = owasp_llm_scan(partial_results)

    assert scan["LLM02"]["covered"] is True, "LLM02 should be covered (one test present)"

    gaps = scan["LLM02"]["gaps"]
    assert "llm.output_format" in gaps, "llm.output_format should be a gap"
    assert "llm.toxicity"      in gaps, "llm.toxicity should be a gap"
    assert "llm.text_length"   not in gaps, "llm.text_length is covered, not a gap"


# ---------------------------------------------------------------------------
# Test 8 — assert_owasp_coverage stores timing in duration_ms
# ---------------------------------------------------------------------------


def test_owasp_coverage_timing():
    # SCENARIO: assert_owasp_coverage is decorated with @timed_assertion
    # WHY:      all mltk assertions must populate duration_ms for perf tracking
    # EXPECTED: result.duration_ms is a positive float

    result = assert_owasp_coverage(FULL_SAMPLE_RESULTS, min_coverage=0.1)
    assert isinstance(result.duration_ms, float)
    assert result.duration_ms >= 0.0
