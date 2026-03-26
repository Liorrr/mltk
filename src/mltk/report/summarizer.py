"""Test history summarization -- trends, common failures, recommendations."""

from __future__ import annotations

from collections import Counter
from typing import Any


def summarize_test_history(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Analyze multiple test runs and produce a summary.

    Args:
        runs: List of run dicts, each with ``score``, ``passed``, ``failed``,
              ``total``, ``timestamp``, and optionally ``results`` (list of
              per-test result dicts with at least ``name`` and ``passed``).

    Returns:
        Dict with:

        - **trend**: ``"improving"`` | ``"degrading"`` | ``"stable"``
        - **avg_score**: mean score across all runs
        - **most_common_failures**: list of ``(test_name, fail_count)`` tuples,
          sorted by fail count descending
        - **flaky_tests**: list of test names that passed in some runs and
          failed in others
        - **recommendations**: list of human-readable action strings
    """
    if not runs:
        return {
            "trend": "stable",
            "avg_score": 0.0,
            "most_common_failures": [],
            "flaky_tests": [],
            "recommendations": ["No test runs available -- submit results to begin tracking."],
        }

    # ------------------------------------------------------------------
    # Trend: compare average of first 3 vs last 3 scores (chronological)
    # ------------------------------------------------------------------
    # Runs may arrive in any order; sort by timestamp ascending so
    # index 0 is the earliest.
    sorted_runs = sorted(runs, key=lambda r: r.get("timestamp", ""))

    scores = [r.get("score", 0.0) or 0.0 for r in sorted_runs]
    n = len(scores)

    window = min(3, n)
    early_avg = sum(scores[:window]) / window
    late_avg = sum(scores[-window:]) / window

    # Use a small epsilon to avoid calling noise "improving"/"degrading".
    eps = 2.0  # percent-points
    if late_avg - early_avg > eps:
        trend = "improving"
    elif early_avg - late_avg > eps:
        trend = "degrading"
    else:
        trend = "stable"

    avg_score = sum(scores) / n

    # ------------------------------------------------------------------
    # Per-test failure counts and flaky detection
    # ------------------------------------------------------------------
    fail_counter: Counter[str] = Counter()
    pass_set: set[str] = set()
    fail_set: set[str] = set()

    for run in sorted_runs:
        for result in run.get("results", []):
            name = result.get("name", "<unknown>")
            if result.get("passed"):
                pass_set.add(name)
            else:
                fail_set.add(name)
                fail_counter[name] += 1

    most_common_failures = fail_counter.most_common()
    flaky_tests = sorted(pass_set & fail_set)

    # ------------------------------------------------------------------
    # Recommendations
    # ------------------------------------------------------------------
    recommendations: list[str] = []

    if trend == "degrading":
        recommendations.append(
            "Scores are trending downward -- investigate recent changes "
            "for regressions."
        )

    if flaky_tests:
        recommendations.append(
            f"Found {len(flaky_tests)} flaky test(s): {', '.join(flaky_tests[:5])}. "
            "Stabilize these before trusting pass/fail signals."
        )

    if most_common_failures:
        top_name, top_count = most_common_failures[0]
        recommendations.append(
            f"Most frequent failure: '{top_name}' failed {top_count} time(s). "
            "Prioritize fixing this test or the code it covers."
        )

    if not recommendations:
        recommendations.append("All clear -- test suite looks healthy.")

    return {
        "trend": trend,
        "avg_score": round(avg_score, 2),
        "most_common_failures": most_common_failures,
        "flaky_tests": flaky_tests,
        "recommendations": recommendations,
    }
