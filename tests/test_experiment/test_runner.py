"""Tests for mltk.experiment.runner -- the ExperimentRunner.

The ExperimentRunner takes ScanFindings, applies fix hypotheses,
re-runs assertions, and ranks results.  These tests verify:

1.  Single passing hypothesis selects the fix.
2.  Single failing hypothesis yields no selected fix.
3.  Multiple hypotheses pick the best one.
4.  apply_fns dict auto-builds hypotheses.
5.  No hypotheses and no apply_fns produces an empty result.
6.  Exception inside a hypothesis is captured safely.
7.  run_batch() across multiple findings.
8.  run_batch() with empty list returns empty.
9.  Baseline is captured from the original finding assertion.
10. Timeout on a slow hypothesis is captured as failure.
"""

from __future__ import annotations

import time

from mltk.core.assertion import MltkAssertionError
from mltk.core.result import Severity, TestResult
from mltk.experiment.hypothesis import Hypothesis
from mltk.experiment.runner import ExperimentRunner
from mltk.scan.finding import FixSuggestion, ScanFinding

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fix(
    title: str = "fix-a",
    confidence: str = "high",
) -> FixSuggestion:
    """Create a minimal FixSuggestion for testing."""
    return FixSuggestion(
        category="code",
        title=title,
        description=f"Apply {title}",
        confidence=confidence,
    )


def _make_failing_finding(
    scanner_name: str = "test_scanner",
    fixes: list[FixSuggestion] | None = None,
) -> ScanFinding:
    """Create a ScanFinding whose assertion always fails.

    The assertion_fn raises MltkAssertionError so the baseline
    is captured as a failed TestResult.
    """
    def _failing_assertion() -> TestResult:
        result = TestResult(
            name="baseline_check",
            passed=False,
            severity=Severity.CRITICAL,
            message="Original failure",
        )
        raise MltkAssertionError(result)

    return ScanFinding(
        result=TestResult(
            name="baseline_check",
            passed=False,
            severity=Severity.CRITICAL,
            message="Original failure",
        ),
        assertion_fn=_failing_assertion,
        scanner_name=scanner_name,
        suggested_fixes=fixes or [],
    )


def _make_passing_apply_fn() -> callable:
    """Apply function that returns a passing TestResult."""
    def apply_fn() -> TestResult:
        return TestResult(
            name="fixed",
            passed=True,
            severity=Severity.INFO,
            message="Fixed",
        )
    return apply_fn


def _make_failing_apply_fn() -> callable:
    """Apply function that returns a failing TestResult."""
    def apply_fn() -> TestResult:
        return TestResult(
            name="still_failing",
            passed=False,
            severity=Severity.WARNING,
            message="Still broken",
        )
    return apply_fn


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_single_passing_hypothesis_selects_fix() -> None:
    """PASS: One hypothesis that fixes the issue is selected.

    WHY: The most basic success case -- a single fix hypothesis that
    turns the failing assertion into a pass should be automatically
    selected as the chosen fix.
    Expected: selected_fix is the hypothesis's FixSuggestion.
    """
    fix = _make_fix("resample-data")
    finding = _make_failing_finding(fixes=[fix])

    hyp = Hypothesis(
        fix=fix,
        apply_fn=_make_passing_apply_fn(),
        description="Resample the training data",
    )
    runner = ExperimentRunner(strategy="passed")
    result = runner.run(finding, hypotheses=[hyp])

    assert result.selected_fix is fix
    assert result.any_fix_works is True
    assert result.hypotheses_tested == 1
    assert result.hypothesis_results[0].is_winning is True


def test_single_failing_hypothesis_no_fix() -> None:
    """FAIL: One hypothesis that does not fix the issue yields None.

    WHY: If the only available fix still fails the assertion, no fix
    should be selected -- the runner should honestly report that nothing
    worked rather than recommending an ineffective change.
    Expected: selected_fix is None.
    """
    fix = _make_fix("bad-fix")
    finding = _make_failing_finding(fixes=[fix])

    hyp = Hypothesis(
        fix=fix,
        apply_fn=_make_failing_apply_fn(),
        description="A fix that does not work",
    )
    runner = ExperimentRunner(strategy="passed")
    result = runner.run(finding, hypotheses=[hyp])

    assert result.selected_fix is None
    assert result.any_fix_works is False
    assert result.hypotheses_tested == 1


def test_multiple_hypotheses_best_selected() -> None:
    """RANK: With multiple hypotheses the best one is selected.

    WHY: When multiple fixes are available, the ranking strategy must
    pick the one that actually resolved the finding.  The passing
    hypothesis should be rank 1 and selected.
    Expected: selected_fix is the passing hypothesis's fix.
    """
    fix_good = _make_fix("good-fix", confidence="high")
    fix_bad = _make_fix("bad-fix", confidence="medium")
    finding = _make_failing_finding(fixes=[fix_good, fix_bad])

    hyp_good = Hypothesis(
        fix=fix_good,
        apply_fn=_make_passing_apply_fn(),
    )
    hyp_bad = Hypothesis(
        fix=fix_bad,
        apply_fn=_make_failing_apply_fn(),
    )

    runner = ExperimentRunner(strategy="passed")
    result = runner.run(finding, hypotheses=[hyp_bad, hyp_good])

    assert result.selected_fix is fix_good
    assert result.hypothesis_results[0].hypothesis.fix is fix_good
    assert result.hypothesis_results[0].rank == 1


def test_apply_fns_dict_builds_hypotheses() -> None:
    """AUTO: apply_fns dict is converted to Hypothesis objects.

    WHY: Users should be able to pass a simple dict of
    FixSuggestion -> callable without manually constructing
    Hypothesis objects.  The runner should auto-build them.
    Expected: result has one hypothesis result, fix is selected.
    """
    fix = _make_fix("auto-fix")
    finding = _make_failing_finding(fixes=[fix])

    runner = ExperimentRunner(strategy="passed")
    result = runner.run(
        finding,
        apply_fns=[(fix, _make_passing_apply_fn())],
    )

    assert result.selected_fix is fix
    assert result.hypotheses_tested == 1


def test_no_hypotheses_no_apply_fns_empty_result() -> None:
    """EMPTY: No hypotheses and no apply_fns gives empty result.

    WHY: A finding with no fix suggestions should produce a valid
    ExperimentResult with an empty hypothesis list and no selected
    fix, rather than raising an error.
    Expected: hypothesis_results is empty, selected_fix is None.
    """
    finding = _make_failing_finding()

    runner = ExperimentRunner(strategy="passed")
    result = runner.run(finding)

    assert result.selected_fix is None
    assert result.hypotheses_tested == 0
    assert result.hypothesis_results == []


def test_hypothesis_exception_captured() -> None:
    """ERROR: Exception inside apply_fn is captured as failed result.

    WHY: A buggy fix function must never crash the runner.  The
    exception should be caught and recorded as a CRITICAL failure
    so the experiment can continue with remaining hypotheses.
    Expected: TestResult with passed=False, severity=CRITICAL.
    """
    fix = _make_fix("crashy-fix")
    finding = _make_failing_finding(fixes=[fix])

    def _crashing_fn() -> TestResult:
        raise RuntimeError("oops")

    hyp = Hypothesis(fix=fix, apply_fn=_crashing_fn)
    runner = ExperimentRunner(strategy="passed")
    result = runner.run(finding, hypotheses=[hyp])

    assert result.selected_fix is None
    assert result.hypotheses_tested == 1
    hr = result.hypothesis_results[0]
    assert hr.fixed_result.passed is False
    assert hr.fixed_result.severity == Severity.CRITICAL
    assert "RuntimeError" in hr.fixed_result.message


def test_run_batch_multiple_findings() -> None:
    """BATCH: run_batch processes multiple findings.

    WHY: Real scans produce dozens of findings.  run_batch must
    iterate over all of them, applying fixes where available and
    returning one ExperimentResult per finding.
    Expected: len(results) == 2, each has its own baseline.
    """
    fix_a = _make_fix("fix-a")
    fix_b = _make_fix("fix-b")
    finding_a = _make_failing_finding(
        scanner_name="scanner_a", fixes=[fix_a],
    )
    finding_b = _make_failing_finding(
        scanner_name="scanner_b", fixes=[fix_b],
    )

    runner = ExperimentRunner(strategy="passed")
    apply_map = {
        "scanner_a": {"fix-a": _make_passing_apply_fn()},
        "scanner_b": {"fix-b": _make_failing_apply_fn()},
    }
    results = runner.run_batch(
        [finding_a, finding_b], apply_fns_map=apply_map,
    )

    assert len(results) == 2
    assert results[0].selected_fix is fix_a
    assert results[1].selected_fix is None


def test_run_batch_empty_findings() -> None:
    """EMPTY: run_batch with empty findings list returns empty.

    WHY: Edge case -- calling run_batch with no findings should
    simply return an empty list without errors.
    Expected: results == [].
    """
    runner = ExperimentRunner(strategy="passed")
    results = runner.run_batch([])

    assert results == []


def test_baseline_captured_from_finding() -> None:
    """BASELINE: The original assertion result is captured.

    WHY: The baseline is the control measurement -- it must faithfully
    re-run the original assertion so hypothesis results can be compared
    against it.  Both passing and failing baselines should work.
    Expected: baseline_result matches the assertion's output.
    """
    def _passing_assertion() -> TestResult:
        return TestResult(
            name="already_passing",
            passed=True,
            severity=Severity.INFO,
            message="Already fine",
        )

    finding = ScanFinding(
        result=TestResult(
            name="already_passing",
            passed=True,
            severity=Severity.INFO,
            message="Already fine",
        ),
        assertion_fn=_passing_assertion,
        scanner_name="test",
    )

    runner = ExperimentRunner(strategy="passed")
    result = runner.run(finding)

    assert result.baseline_result is not None
    assert result.baseline_result.passed is True
    assert result.baseline_result.name == "already_passing"


def test_timeout_captured_as_failure() -> None:
    """TIMEOUT: A slow hypothesis is captured as a failed TestResult.

    WHY: Production fix functions may hang on I/O or infinite loops.
    The runner must enforce a per-hypothesis timeout and report the
    timeout as a structured failure, not an uncaught exception.
    Expected: TestResult with passed=False, 'timed out' in message.
    """
    fix = _make_fix("slow-fix")
    finding = _make_failing_finding(fixes=[fix])

    def _slow_fn() -> TestResult:
        time.sleep(10)  # way longer than timeout
        return TestResult(
            name="never_reached",
            passed=True,
            severity=Severity.INFO,
            message="Should not get here",
        )

    hyp = Hypothesis(fix=fix, apply_fn=_slow_fn)
    runner = ExperimentRunner(strategy="passed", timeout=0.2)
    result = runner.run(finding, hypotheses=[hyp])

    assert result.selected_fix is None
    assert result.hypotheses_tested == 1
    hr = result.hypothesis_results[0]
    assert hr.fixed_result.passed is False
    assert "timed out" in hr.fixed_result.message.lower()
