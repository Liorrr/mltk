"""Tests for mltk.report.summarizer -- test history trend analysis.

Each test follows the pattern:
  # SCENARIO: <what situation is being tested>
  # WHY: <why this matters / what could go wrong>
  # EXPECTED: <the concrete assertion>
"""

from __future__ import annotations

from mltk.report.summarizer import summarize_test_history

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_run(
    score: float,
    passed: int,
    failed: int,
    timestamp: str = "2025-01-01T00:00:00",
    results: list[dict] | None = None,  # type: ignore[type-arg]
) -> dict:  # type: ignore[type-arg]
    """Build a minimal run dict accepted by summarize_test_history."""
    return {
        "score": score,
        "passed": passed,
        "failed": failed,
        "total": passed + failed,
        "timestamp": timestamp,
        "results": results or [],
    }


def _make_result(name: str, passed: bool = True) -> dict:  # type: ignore[type-arg]
    """Build a minimal per-test result dict."""
    return {"name": name, "passed": passed}


# ---------------------------------------------------------------------------
# Test 1 -- improving trend
# ---------------------------------------------------------------------------

def test_improving_trend() -> None:
    # SCENARIO: Scores rise steadily over 6 runs
    # WHY: The summarizer must detect upward momentum so users see positive
    #       feedback when their test suite is getting healthier
    # EXPECTED: trend == "improving"

    runs = [
        _make_run(score=50.0, passed=5, failed=5, timestamp="2025-01-01T00:00:00"),
        _make_run(score=55.0, passed=6, failed=4, timestamp="2025-01-02T00:00:00"),
        _make_run(score=60.0, passed=6, failed=4, timestamp="2025-01-03T00:00:00"),
        _make_run(score=70.0, passed=7, failed=3, timestamp="2025-01-04T00:00:00"),
        _make_run(score=80.0, passed=8, failed=2, timestamp="2025-01-05T00:00:00"),
        _make_run(score=90.0, passed=9, failed=1, timestamp="2025-01-06T00:00:00"),
    ]

    summary = summarize_test_history(runs)

    assert summary["trend"] == "improving"
    assert summary["avg_score"] > 0


# ---------------------------------------------------------------------------
# Test 2 -- degrading trend
# ---------------------------------------------------------------------------

def test_degrading_trend() -> None:
    # SCENARIO: Scores drop steadily over 6 runs
    # WHY: Regression detection is the main value prop of summarization;
    #       missing a downward trend could let quality erode silently
    # EXPECTED: trend == "degrading"

    runs = [
        _make_run(score=90.0, passed=9, failed=1, timestamp="2025-01-01T00:00:00"),
        _make_run(score=85.0, passed=9, failed=1, timestamp="2025-01-02T00:00:00"),
        _make_run(score=80.0, passed=8, failed=2, timestamp="2025-01-03T00:00:00"),
        _make_run(score=70.0, passed=7, failed=3, timestamp="2025-01-04T00:00:00"),
        _make_run(score=60.0, passed=6, failed=4, timestamp="2025-01-05T00:00:00"),
        _make_run(score=50.0, passed=5, failed=5, timestamp="2025-01-06T00:00:00"),
    ]

    summary = summarize_test_history(runs)

    assert summary["trend"] == "degrading"
    assert "downward" in summary["recommendations"][0].lower() or \
           "regression" in summary["recommendations"][0].lower()


# ---------------------------------------------------------------------------
# Test 3 -- stable trend
# ---------------------------------------------------------------------------

def test_stable_trend() -> None:
    # SCENARIO: Scores hover around the same value
    # WHY: The summarizer should not cry wolf when nothing is changing;
    #       stable should mean no urgent action needed
    # EXPECTED: trend == "stable"

    runs = [
        _make_run(score=75.0, passed=8, failed=2, timestamp="2025-01-01T00:00:00"),
        _make_run(score=76.0, passed=8, failed=2, timestamp="2025-01-02T00:00:00"),
        _make_run(score=74.0, passed=8, failed=2, timestamp="2025-01-03T00:00:00"),
        _make_run(score=75.5, passed=8, failed=2, timestamp="2025-01-04T00:00:00"),
        _make_run(score=75.0, passed=8, failed=2, timestamp="2025-01-05T00:00:00"),
    ]

    summary = summarize_test_history(runs)

    assert summary["trend"] == "stable"
    assert 74.0 <= summary["avg_score"] <= 76.0


# ---------------------------------------------------------------------------
# Test 4 -- most common failures
# ---------------------------------------------------------------------------

def test_common_failures() -> None:
    # SCENARIO: Several runs share recurring test failures
    # WHY: Identifying the most-failed test lets teams prioritize fixes;
    #       if counting is wrong, teams may chase the wrong failures
    # EXPECTED: The test that fails most often is ranked first

    runs = [
        _make_run(
            score=70.0, passed=2, failed=1, timestamp="2025-01-01T00:00:00",
            results=[
                _make_result("data.schema.types", passed=True),
                _make_result("model.metric.accuracy", passed=False),
                _make_result("data.drift.psi", passed=True),
            ],
        ),
        _make_run(
            score=65.0, passed=1, failed=2, timestamp="2025-01-02T00:00:00",
            results=[
                _make_result("data.schema.types", passed=False),
                _make_result("model.metric.accuracy", passed=False),
                _make_result("data.drift.psi", passed=True),
            ],
        ),
        _make_run(
            score=60.0, passed=1, failed=2, timestamp="2025-01-03T00:00:00",
            results=[
                _make_result("data.schema.types", passed=True),
                _make_result("model.metric.accuracy", passed=False),
                _make_result("data.drift.psi", passed=False),
            ],
        ),
    ]

    summary = summarize_test_history(runs)

    # model.metric.accuracy failed in all 3 runs -- should be #1
    failures = summary["most_common_failures"]
    assert len(failures) >= 1
    top_name, top_count = failures[0]
    assert top_name == "model.metric.accuracy"
    assert top_count == 3


# ---------------------------------------------------------------------------
# Test 5 -- flaky tests
# ---------------------------------------------------------------------------

def test_flaky_tests() -> None:
    # SCENARIO: A test passes in some runs and fails in others
    # WHY: Flaky tests erode trust in the test suite; the summarizer must
    #       surface them so they can be investigated
    # EXPECTED: The intermittent test name appears in flaky_tests

    runs = [
        _make_run(
            score=80.0, passed=2, failed=0, timestamp="2025-01-01T00:00:00",
            results=[
                _make_result("data.schema.types", passed=True),
                _make_result("model.metric.accuracy", passed=True),
            ],
        ),
        _make_run(
            score=70.0, passed=1, failed=1, timestamp="2025-01-02T00:00:00",
            results=[
                _make_result("data.schema.types", passed=True),
                _make_result("model.metric.accuracy", passed=False),
            ],
        ),
        _make_run(
            score=80.0, passed=2, failed=0, timestamp="2025-01-03T00:00:00",
            results=[
                _make_result("data.schema.types", passed=True),
                _make_result("model.metric.accuracy", passed=True),
            ],
        ),
    ]

    summary = summarize_test_history(runs)

    assert "model.metric.accuracy" in summary["flaky_tests"]
    # data.schema.types always passes -- not flaky
    assert "data.schema.types" not in summary["flaky_tests"]
    # Recommendation should mention flaky
    assert any("flaky" in r.lower() for r in summary["recommendations"])


# ---------------------------------------------------------------------------
# Test 6 -- empty runs list
# ---------------------------------------------------------------------------

def test_empty_runs() -> None:
    # SCENARIO: No runs submitted yet
    # WHY: A fresh project with zero history should not crash; it should
    #       return a safe default summary
    # EXPECTED: Stable trend, zero score, no failures, no flaky tests

    summary = summarize_test_history([])

    assert summary["trend"] == "stable"
    assert summary["avg_score"] == 0.0
    assert summary["most_common_failures"] == []
    assert summary["flaky_tests"] == []
    assert len(summary["recommendations"]) >= 1


# ---------------------------------------------------------------------------
# Test 7 -- single run (edge case)
# ---------------------------------------------------------------------------

def test_single_run() -> None:
    # SCENARIO: Only one run exists
    # WHY: Trend calculation divides by window size; a single run must
    #       not cause division errors or misleading trend signals
    # EXPECTED: trend == "stable" (not enough data to determine direction)

    runs = [
        _make_run(
            score=85.0, passed=9, failed=1, timestamp="2025-01-01T00:00:00",
            results=[_make_result("model.metric.accuracy", passed=False)],
        ),
    ]

    summary = summarize_test_history(runs)

    assert summary["trend"] == "stable"
    assert summary["avg_score"] == 85.0
    assert len(summary["most_common_failures"]) == 1
