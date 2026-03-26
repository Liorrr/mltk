"""Tests for mltk.report.visual_diff -- HTML visual diff between two test runs.

Each test follows the pattern:
  # SCENARIO: <what situation is being tested>
  # WHY: <why this matters / what could go wrong>
  # EXPECTED: <the concrete assertion>
"""

from __future__ import annotations

from pathlib import Path

from mltk.report.visual_diff import generate_diff_report

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_run(
    results: list[tuple[str, bool]],
    score: float = 0.0,
) -> dict:
    """Build a minimal run dict from (name, passed) tuples."""
    return {
        "results": [{"name": n, "passed": p} for n, p in results],
        "score": score,
    }


# ---------------------------------------------------------------------------
# Test 1 — file creation
# ---------------------------------------------------------------------------

def test_creates_html_file(tmp_path: Path) -> None:
    # SCENARIO: generate_diff_report is called with two valid runs
    # WHY: The function must produce an HTML file; if it doesn't, CI
    #       dashboards and reviewers have nothing to inspect
    # EXPECTED: The output path exists, has .html suffix, and content is non-empty

    run_a = _make_run([("t1", True), ("t2", False)], score=50.0)
    run_b = _make_run([("t1", True), ("t2", True)], score=100.0)

    out = tmp_path / "diff.html"
    returned = generate_diff_report(run_a, run_b, output_path=out)

    assert returned == out
    assert out.exists()
    assert out.suffix == ".html"
    content = out.read_text(encoding="utf-8")
    assert len(content) > 100
    assert "<!DOCTYPE html>" in content


# ---------------------------------------------------------------------------
# Test 2 — regressions shown in red
# ---------------------------------------------------------------------------

def test_shows_regressions_in_red(tmp_path: Path) -> None:
    # SCENARIO: A test that passed in Run A now fails in Run B
    # WHY: Regressions are the highest-priority signal; if they aren't
    #       visually highlighted, users miss critical breakage
    # EXPECTED: The HTML contains the "regressed" CSS class for the row

    run_a = _make_run([("model.accuracy", True)], score=95.0)
    run_b = _make_run([("model.accuracy", False)], score=80.0)

    out = tmp_path / "regression.html"
    generate_diff_report(run_a, run_b, output_path=out)

    content = out.read_text(encoding="utf-8")

    # The regressed test row should carry the regressed CSS class
    assert 'class="regressed"' in content
    # The test name should appear
    assert "model.accuracy" in content
    # Score change should be negative
    assert "score-down" in content


# ---------------------------------------------------------------------------
# Test 3 — fixes shown in green
# ---------------------------------------------------------------------------

def test_shows_fixes_in_green(tmp_path: Path) -> None:
    # SCENARIO: A test that failed in Run A now passes in Run B
    # WHY: Fixes are positive signals that should be visually distinct;
    #       without green highlighting, they blend in with unchanged tests
    # EXPECTED: The HTML contains the "fixed" CSS class for the row

    run_a = _make_run([("data.schema", False), ("data.drift", True)], score=50.0)
    run_b = _make_run([("data.schema", True), ("data.drift", True)], score=100.0)

    out = tmp_path / "fixes.html"
    generate_diff_report(run_a, run_b, output_path=out)

    content = out.read_text(encoding="utf-8")

    # The fixed test row should carry the fixed CSS class
    assert 'class="fixed"' in content
    assert "data.schema" in content
    # Score went up
    assert "score-up" in content


# ---------------------------------------------------------------------------
# Test 4 — empty runs do not crash
# ---------------------------------------------------------------------------

def test_empty_runs(tmp_path: Path) -> None:
    # SCENARIO: Both runs have zero test results
    # WHY: Edge case — users may compare placeholder runs before tests exist;
    #       a crash would block the pipeline
    # EXPECTED: File is generated without errors; basic HTML structure present

    run_a: dict = {"results": [], "score": 0.0}
    run_b: dict = {"results": [], "score": 0.0}

    out = tmp_path / "empty.html"
    returned = generate_diff_report(run_a, run_b, output_path=out)

    assert returned.exists()
    content = out.read_text(encoding="utf-8")
    assert "<!DOCTYPE html>" in content
    # Score change should be zero
    assert "score-same" in content
    # Summary cards should show 0
    assert "0" in content


# ---------------------------------------------------------------------------
# Test 5 — new tests highlighted in yellow
# ---------------------------------------------------------------------------

def test_new_tests_highlighted(tmp_path: Path) -> None:
    # SCENARIO: Run B has a test that did not exist in Run A
    # WHY: New tests need distinct styling so reviewers can distinguish
    #       added coverage from regressions or fixes
    # EXPECTED: The row carries the "new-test" CSS class

    run_a = _make_run([("t1", True)], score=100.0)
    run_b = _make_run([("t1", True), ("t2_brand_new", True)], score=100.0)

    out = tmp_path / "new-test.html"
    generate_diff_report(run_a, run_b, output_path=out)

    content = out.read_text(encoding="utf-8")
    assert 'class="new-test"' in content
    assert "t2_brand_new" in content
