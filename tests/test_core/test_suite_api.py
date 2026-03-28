"""Tests for mltk.core.suite -- composable MltkSuite API.

MltkSuite wraps mltk assertion functions so they never raise, enabling
programmatic usage in notebooks, scripts, CI, and monitoring jobs.
These tests verify the core contract:

1. Assertions are deferred until run() is called
2. Failures are captured (never raised) as TestResult objects
3. Method chaining works for fluent API usage
4. Export formats (JSON, JUnit XML, HTML) produce valid output
5. SuiteResult provides correct aggregation and properties
6. Edge cases: empty suite, re-running, unexpected exceptions
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from mltk.core.assertion import assert_true
from mltk.core.result import Severity, TestResult
from mltk.core.suite import MltkSuite, SuiteResult

# ---------------------------------------------------------------------------
# Helpers -- deterministic assertion functions for testing
# ---------------------------------------------------------------------------

def _always_pass() -> TestResult:
    """Assertion that always passes."""
    return assert_true(
        True,
        name="test.always_pass",
        message="Passed as expected",
    )


def _always_fail() -> TestResult:
    """Assertion that always fails (raises MltkAssertionError)."""
    return assert_true(
        False,
        name="test.always_fail",
        message="Failed as expected",
    )


def _warning_fail() -> TestResult:
    """Assertion that fails with WARNING severity (does not raise)."""
    return assert_true(
        False,
        name="test.warning_fail",
        message="Warning failure",
        severity=Severity.WARNING,
    )


def _pass_with_details(value: float, threshold: float) -> TestResult:
    """Assertion that passes and carries detail kwargs."""
    return assert_true(
        value >= threshold,
        name="test.details",
        message=f"value {value} >= {threshold}",
        value=value,
        threshold=threshold,
    )


def _exploding_assertion() -> TestResult:
    """Assertion that raises an unexpected (non-mltk) exception."""
    raise ValueError("something broke unexpectedly")


# ---------------------------------------------------------------------------
# 1. Basic add + run
# ---------------------------------------------------------------------------

class TestAddAndRun:
    """Core add/run lifecycle."""

    def test_single_passing_assertion(self) -> None:
        """PASS: Single passing assertion produces SuiteResult
        with 1 total, 1 passed.

        WHY: The simplest use case -- one assertion that passes.
        If this fails, the entire suite API is broken.
        """
        suite = MltkSuite("single")
        suite.add(_always_pass)
        result = suite.run()

        assert result.total == 1
        assert result.passed_count == 1
        assert result.failed_count == 0
        assert result.passed is True

    def test_multiple_passing_assertions(self) -> None:
        """PASS: Multiple passing assertions produce correct counts.

        WHY: Suites typically contain many assertions.  Aggregation
        must count each one exactly once.
        """
        suite = MltkSuite("multi")
        suite.add(_always_pass)
        suite.add(_always_pass)
        suite.add(_always_pass)
        result = suite.run()

        assert result.total == 3
        assert result.passed_count == 3
        assert result.failed_count == 0

    def test_mixed_pass_and_fail(self) -> None:
        """MIXED: Suite with both passing and failing assertions
        reports correct split.

        WHY: Real suites have mixed results.  The suite must not
        short-circuit on the first failure -- every assertion must
        execute and be counted.
        """
        suite = MltkSuite("mixed")
        suite.add(_always_pass)
        suite.add(_always_fail)
        suite.add(_always_pass)
        result = suite.run()

        assert result.total == 3
        assert result.passed_count == 2
        assert result.failed_count == 1
        assert result.passed is False


# ---------------------------------------------------------------------------
# 2. Failure capture (never raises)
# ---------------------------------------------------------------------------

class TestFailureCapture:
    """Verify that MltkAssertionError is caught, not propagated."""

    def test_critical_failure_captured_not_raised(self) -> None:
        """FAIL-CAPTURE: A CRITICAL failure is captured in results,
        not raised to the caller.

        WHY: The entire point of MltkSuite is that failures become
        data, not exceptions.  If run() raises, notebooks crash.
        """
        suite = MltkSuite("capture")
        suite.add(_always_fail)
        result = suite.run()  # must NOT raise

        assert result.total == 1
        assert result.failed_count == 1
        failed = result.results[0]
        assert failed.passed is False
        assert failed.name == "test.always_fail"
        assert failed.severity == Severity.CRITICAL

    def test_warning_failure_captured(self) -> None:
        """WARN: WARNING-severity failure appears in results with
        passed=False.

        WHY: Warnings don't raise in assert_true either, but MltkSuite
        must still record them as failed results for accurate counts.
        """
        suite = MltkSuite("warn")
        suite.add(_warning_fail)
        result = suite.run()

        assert result.total == 1
        assert result.failed_count == 1
        failed = result.results[0]
        assert failed.passed is False
        assert failed.severity == Severity.WARNING

    def test_unexpected_exception_captured(self) -> None:
        """ERROR: Non-mltk exceptions produce a CRITICAL failure result.

        WHY: Assertion functions may have bugs (e.g., KeyError, TypeError).
        The suite must not crash -- it wraps the error in a synthetic
        TestResult so the rest of the suite still runs.
        """
        suite = MltkSuite("error")
        suite.add(_exploding_assertion)
        result = suite.run()  # must NOT raise

        assert result.total == 1
        assert result.failed_count == 1
        failed = result.results[0]
        assert failed.passed is False
        assert "Unexpected error" in failed.message
        assert failed.details["exception_type"] == "ValueError"


# ---------------------------------------------------------------------------
# 3. Method chaining
# ---------------------------------------------------------------------------

class TestMethodChaining:
    """Verify fluent API: suite.add(...).add(...).run()."""

    def test_add_returns_self(self) -> None:
        """CHAIN: add() returns the suite instance for chaining.

        WHY: Fluent APIs are idiomatic in Python testing libraries
        (e.g., Evidently TestSuite).  Returning self enables one-liner
        suite definitions in notebooks.
        """
        suite = MltkSuite("chain")
        returned = suite.add(_always_pass)
        assert returned is suite

    def test_chained_add_and_run(self) -> None:
        """CHAIN: Full chain suite.add().add().run() works end-to-end.

        WHY: Users will write this pattern in notebooks:
        ``result = MltkSuite("x").add(f).add(g).run()``
        """
        result = (
            MltkSuite("chained")
            .add(_always_pass)
            .add(_always_fail)
            .run()
        )
        assert isinstance(result, SuiteResult)
        assert result.total == 2
        assert result.passed_count == 1
        assert result.failed_count == 1


# ---------------------------------------------------------------------------
# 4. Export: JSON
# ---------------------------------------------------------------------------

class TestJsonExport:
    """Verify to_json() produces valid, complete JSON."""

    def test_to_json_writes_valid_file(self, tmp_path: Path) -> None:
        """JSON: Output file contains valid JSON with all results.

        WHY: JSON export feeds CI dashboards and downstream tools.
        Invalid JSON or missing fields breaks integrations silently.
        """
        out = tmp_path / "results.json"
        suite = MltkSuite("json-test")
        suite.add(_always_pass)
        suite.add(_always_fail)
        suite.run()

        written = suite.to_json(str(out))
        assert Path(written).exists()

        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["suite"] == "json-test"
        assert data["total"] == 2
        assert data["passed"] == 1
        assert data["failed"] == 1
        assert len(data["results"]) == 2

        # Verify each result has required fields
        for r in data["results"]:
            assert "name" in r
            assert "passed" in r
            assert "severity" in r
            assert "message" in r
            assert "duration_ms" in r

    def test_to_json_before_run_raises(self, tmp_path: Path) -> None:
        """ERROR: Calling to_json() before run() raises RuntimeError.

        WHY: Exporting empty results is a user mistake.  The error
        message should guide them to call run() first.
        """
        suite = MltkSuite("no-run")
        with pytest.raises(RuntimeError, match="run"):
            suite.to_json(str(tmp_path / "nope.json"))


# ---------------------------------------------------------------------------
# 5. Export: JUnit XML
# ---------------------------------------------------------------------------

class TestJunitExport:
    """Verify to_junit() produces valid JUnit XML."""

    def test_to_junit_writes_valid_xml(self, tmp_path: Path) -> None:
        """JUNIT: Output file is well-formed XML with correct counts.

        WHY: CI systems (Jenkins, GitLab) parse JUnit XML natively.
        Malformed XML or wrong counts breaks pipeline dashboards.
        """
        out = tmp_path / "results.xml"
        suite = MltkSuite("junit-test")
        suite.add(_always_pass)
        suite.add(_always_fail)
        suite.run()

        written = suite.to_junit(str(out))
        assert Path(written).exists()

        tree = ET.parse(out)
        root = tree.getroot()
        assert root.tag == "testsuites"

        ts = root.find("testsuite")
        assert ts is not None
        assert ts.get("name") == "junit-test"
        assert ts.get("tests") == "2"
        assert ts.get("failures") == "1"

        testcases = ts.findall("testcase")
        assert len(testcases) == 2

    def test_to_junit_before_run_raises(self, tmp_path: Path) -> None:
        """ERROR: Calling to_junit() before run() raises RuntimeError."""
        suite = MltkSuite("no-run")
        with pytest.raises(RuntimeError, match="run"):
            suite.to_junit(str(tmp_path / "nope.xml"))


# ---------------------------------------------------------------------------
# 6. Export: HTML
# ---------------------------------------------------------------------------

class TestHtmlExport:
    """Verify to_html() produces an HTML file (requires jinja2)."""

    def test_to_html_writes_file(self, tmp_path: Path) -> None:
        """HTML: Output file exists and contains HTML content.

        WHY: HTML reports are the primary visual output for notebooks
        and stakeholder review.  If jinja2 is not installed, the test
        is skipped rather than failing.
        """
        pytest.importorskip("jinja2")

        out = tmp_path / "report.html"
        suite = MltkSuite("html-test")
        suite.add(_always_pass)
        suite.add(_always_fail)
        suite.run()

        written = suite.to_html(str(out))
        assert Path(written).exists()
        content = Path(written).read_text(encoding="utf-8")
        assert "<html" in content.lower() or "<!doctype" in content.lower()


# ---------------------------------------------------------------------------
# 7. Summary
# ---------------------------------------------------------------------------

class TestSummary:
    """Verify summary() returns a human-readable string."""

    def test_summary_contains_counts(self) -> None:
        """SUMMARY: Output includes pass count, total, and percentage.

        WHY: summary() is for quick terminal/notebook output.  Users
        need to see at a glance how many tests passed.
        """
        suite = MltkSuite("summary-test")
        suite.add(_always_pass)
        suite.add(_always_fail)
        suite.add(_always_pass)
        suite.run()

        text = suite.summary()
        assert "2/3" in text
        assert "summary-test" in text
        assert "66.7%" in text

    def test_summary_before_run(self) -> None:
        """SUMMARY: Before run(), summary says 'not yet run'.

        WHY: Calling summary() before run() should not crash --
        it should indicate that no results exist yet.
        """
        suite = MltkSuite("pending")
        assert "not yet run" in suite.summary()


# ---------------------------------------------------------------------------
# 8. Empty suite
# ---------------------------------------------------------------------------

class TestEmptySuite:
    """Edge case: suite with zero assertions."""

    def test_empty_suite_run(self) -> None:
        """EMPTY: run() on empty suite returns SuiteResult with 0 total.

        WHY: An empty suite is valid (e.g., conditional adds that all
        got skipped).  It must not crash or produce NaN values.
        """
        suite = MltkSuite("empty")
        result = suite.run()

        assert result.total == 0
        assert result.passed_count == 0
        assert result.failed_count == 0
        assert result.passed is True  # vacuously true
        assert result.pass_rate == 0.0
        assert result.duration_ms >= 0.0


# ---------------------------------------------------------------------------
# 9. SuiteResult properties
# ---------------------------------------------------------------------------

class TestSuiteResultProperties:
    """Verify SuiteResult computed properties."""

    def test_pass_rate_all_pass(self) -> None:
        """RATE: 3/3 passed gives pass_rate 1.0.

        WHY: pass_rate is used for threshold checks in CI scripts
        (e.g., fail if pass_rate < 0.95).  Must be exact.
        """
        sr = SuiteResult(
            name="t", total=3, passed_count=3, failed_count=0
        )
        assert sr.pass_rate == 1.0

    def test_pass_rate_mixed(self) -> None:
        """RATE: 2/4 passed gives pass_rate 0.5."""
        sr = SuiteResult(
            name="t", total=4, passed_count=2, failed_count=2
        )
        assert sr.pass_rate == 0.5

    def test_pass_rate_empty(self) -> None:
        """RATE: Empty suite gives pass_rate 0.0, not ZeroDivisionError.

        WHY: Division by zero must be guarded.  0.0 signals "nothing
        was tested" which is distinct from "everything failed" (also 0.0
        but with total > 0).
        """
        sr = SuiteResult(name="t", total=0, passed_count=0, failed_count=0)
        assert sr.pass_rate == 0.0

    def test_duration_ms_is_positive(self) -> None:
        """TIMING: run() records a non-negative duration_ms.

        WHY: duration_ms feeds into reports and performance tracking.
        A negative value indicates a timing bug.
        """
        suite = MltkSuite("timed")
        suite.add(_always_pass)
        result = suite.run()
        assert result.duration_ms >= 0.0


# ---------------------------------------------------------------------------
# 10. Suite reuse (run twice)
# ---------------------------------------------------------------------------

class TestSuiteReuse:
    """Running a suite multiple times produces fresh results."""

    def test_run_twice_produces_fresh_results(self) -> None:
        """REUSE: Second run() replaces results from the first run.

        WHY: Monitoring jobs may re-run the same suite on new data.
        Each run must produce independent results without accumulating
        stale data from previous runs.
        """
        suite = MltkSuite("reuse")
        suite.add(_always_pass)

        r1 = suite.run()
        r2 = suite.run()

        assert r1.total == 1
        assert r2.total == 1
        # Results are independent objects
        assert r1 is not r2
        assert r1.results is not r2.results


# ---------------------------------------------------------------------------
# 11. Assertion with arguments
# ---------------------------------------------------------------------------

class TestAssertionArguments:
    """Verify that args and kwargs are forwarded correctly."""

    def test_args_forwarded(self) -> None:
        """ARGS: Positional and keyword args reach the assertion fn.

        WHY: suite.add(assert_metric, y_true, y_pred, metric="f1")
        must forward y_true and y_pred as positional args and metric
        as a keyword arg.  If forwarding is broken, assertions get
        wrong inputs and produce wrong results.
        """
        suite = MltkSuite("args")
        suite.add(_pass_with_details, 0.95, 0.9)
        result = suite.run()

        assert result.total == 1
        assert result.passed_count == 1
        r = result.results[0]
        assert r.details["value"] == 0.95
        assert r.details["threshold"] == 0.9

    def test_kwargs_forwarded(self) -> None:
        """KWARGS: Keyword arguments reach the assertion function."""
        suite = MltkSuite("kwargs")
        suite.add(
            _pass_with_details, value=0.88, threshold=0.80
        )
        result = suite.run()

        assert result.total == 1
        assert result.passed_count == 1
        r = result.results[0]
        assert r.details["value"] == 0.88
        assert r.details["threshold"] == 0.80


# ---------------------------------------------------------------------------
# 12. Results property
# ---------------------------------------------------------------------------

class TestResultsProperty:
    """Verify the .results and .passed properties."""

    def test_results_returns_list(self) -> None:
        """RESULTS: .results returns a list of TestResult objects.

        WHY: Programmatic access to individual results is the primary
        API for non-pytest consumers.
        """
        suite = MltkSuite("prop")
        suite.add(_always_pass)
        suite.add(_always_fail)
        suite.run()

        results = suite.results
        assert isinstance(results, list)
        assert len(results) == 2
        assert all(isinstance(r, TestResult) for r in results)

    def test_passed_property_true(self) -> None:
        """PASSED: .passed is True when all assertions pass."""
        suite = MltkSuite("all-pass")
        suite.add(_always_pass)
        suite.run()
        assert suite.passed is True

    def test_passed_property_false(self) -> None:
        """PASSED: .passed is False when any assertion fails."""
        suite = MltkSuite("has-fail")
        suite.add(_always_pass)
        suite.add(_always_fail)
        suite.run()
        assert suite.passed is False

    def test_passed_before_run_raises(self) -> None:
        """ERROR: .passed before run() raises RuntimeError."""
        suite = MltkSuite("no-run")
        with pytest.raises(RuntimeError, match="run"):
            _ = suite.passed

    def test_results_before_run_empty(self) -> None:
        """EDGE: .results before run() returns empty list (no error)."""
        suite = MltkSuite("no-run")
        assert suite.results == []


# ---------------------------------------------------------------------------
# 13. Repr and len
# ---------------------------------------------------------------------------

class TestReprAndLen:
    """Verify __repr__ and __len__ for debugging."""

    def test_repr_before_run(self) -> None:
        """REPR: Shows pending count before run."""
        suite = MltkSuite("debug")
        suite.add(_always_pass)
        r = repr(suite)
        assert "debug" in r
        assert "pending=1" in r

    def test_repr_after_run(self) -> None:
        """REPR: Shows results count after run."""
        suite = MltkSuite("debug")
        suite.add(_always_pass)
        suite.run()
        r = repr(suite)
        assert "results=1" in r

    def test_len_before_run(self) -> None:
        """LEN: len() returns pending count before run."""
        suite = MltkSuite("len")
        suite.add(_always_pass)
        suite.add(_always_pass)
        assert len(suite) == 2

    def test_len_after_run(self) -> None:
        """LEN: len() returns results count after run."""
        suite = MltkSuite("len")
        suite.add(_always_pass)
        suite.run()
        assert len(suite) == 1
