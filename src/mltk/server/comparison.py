"""Run comparison — diff test results between two runs."""
from __future__ import annotations


def compare_runs(run_a: dict, run_b: dict) -> dict:  # type: ignore[type-arg]
    """Compare two test runs and return a structured diff.

    Looks at the ``results`` list inside each run dict (as returned by
    ``Storage.get_run``).  Each result must have a ``name`` key and a
    ``passed`` boolean.

    Returns:
        dict with keys:
        - new_failures: tests that passed in A but failed in B
        - fixed: tests that failed in A but passed in B
        - still_failing: tests that failed in both
        - still_passing: tests that passed in both
        - new_tests: tests only in B (not in A)
        - removed_tests: tests only in A (not in B)
        - score_change: B.score - A.score  (0.0 if scores absent)
    """
    results_a: dict[str, bool] = {
        r["name"]: bool(r.get("passed", False))
        for r in run_a.get("results", [])
    }
    results_b: dict[str, bool] = {
        r["name"]: bool(r.get("passed", False))
        for r in run_b.get("results", [])
    }

    names_a = set(results_a)
    names_b = set(results_b)
    common = names_a & names_b

    new_failures: list[str] = []
    fixed: list[str] = []
    still_failing: list[str] = []
    still_passing: list[str] = []

    for name in sorted(common):
        passed_a = results_a[name]
        passed_b = results_b[name]
        if passed_a and not passed_b:
            new_failures.append(name)
        elif not passed_a and passed_b:
            fixed.append(name)
        elif not passed_a and not passed_b:
            still_failing.append(name)
        else:
            still_passing.append(name)

    score_a = float(run_a.get("score", 0.0))
    score_b = float(run_b.get("score", 0.0))

    return {
        "new_failures": new_failures,
        "fixed": fixed,
        "still_failing": still_failing,
        "still_passing": still_passing,
        "new_tests": sorted(names_b - names_a),
        "removed_tests": sorted(names_a - names_b),
        "score_change": round(score_b - score_a, 4),
    }
