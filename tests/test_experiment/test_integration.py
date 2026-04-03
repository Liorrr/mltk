"""Integration tests for the Experiment Runner full pipeline.

Tests the end-to-end flow: ScanFinding with fix suggestions goes
through ExperimentRunner.run() / run_batch(), producing ranked
ExperimentResults with selected fixes (or None when nothing works).

Uses real mltk dataclasses and assertion helpers rather than mocks
to verify the runner integrates correctly with the rest of the
experiment module.
"""

from __future__ import annotations

import pytest

from mltk.core.result import Severity, TestResult
from mltk.scan.finding import FixSuggestion, ScanFinding

try:
    from mltk.experiment import (
        ExperimentResult,
        ExperimentRunner,
        Hypothesis,
        HypothesisResult,
        rank_hypotheses,
    )

    # Detect the stub: if ExperimentRunner has no ``run`` method the
    # full implementation has not landed yet.
    _HAS_RUNNER = callable(getattr(ExperimentRunner, "run", None))
except ImportError:
    _HAS_RUNNER = False

pytestmark = pytest.mark.skipif(
    not _HAS_RUNNER,
    reason="ExperimentRunner.run() not yet implemented",
)


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------


def _make_result(passed: bool = False, **overrides) -> TestResult:
    """Build a TestResult with sensible defaults."""
    defaults = {
        "name": "test",
        "passed": passed,
        "severity": Severity.WARNING,
        "message": "msg",
    }
    defaults.update(overrides)
    return TestResult(**defaults)


def _make_fix(**overrides) -> FixSuggestion:
    """Build a FixSuggestion with sensible defaults."""
    defaults = {
        "category": "code",
        "title": "Fix it",
        "description": "Apply fix.",
        "confidence": "high",
    }
    defaults.update(overrides)
    return FixSuggestion(**defaults)


def _make_finding(fixes=None, **overrides) -> ScanFinding:
    """Build a ScanFinding with sensible defaults."""
    defaults = {
        "result": _make_result(),
        "assertion_fn": lambda: _make_result(),
        "suggested_test": "def test_x(): pass",
        "scanner_name": "test",
        "suggested_fixes": fixes or [],
    }
    defaults.update(overrides)
    return ScanFinding(**defaults)


# ---------------------------------------------------------------
# 1. Full pipeline: failing assertion -> fix -> passes
# ---------------------------------------------------------------


class TestFullPipelineFixPasses:
    """Run a finding through the runner where the fix resolves the issue."""

    def test_selected_fix_is_set_when_fix_passes(self) -> None:
        # SCENARIO: Baseline fails (low accuracy), apply_fn returns passing
        #   result (high accuracy).
        # WHY: The core value of the runner is selecting a winning fix.
        # EXPECTED: selected_fix is set and any_fix_works is True.
        baseline = _make_result(
            passed=False,
            details={"actual": 0.4},
        )
        fix = _make_fix(title="Retrain model")
        finding = _make_finding(
            fixes=[fix],
            result=baseline,
            assertion_fn=lambda: baseline,
        )

        fixed_result = _make_result(
            passed=True,
            details={"actual": 0.95},
        )

        runner = ExperimentRunner()
        hypotheses = [
            Hypothesis(
                fix=fix,
                apply_fn=lambda: fixed_result,
                description="Retrain with more data",
            ),
        ]
        result = runner.run(finding, hypotheses=hypotheses)

        assert isinstance(result, ExperimentResult)
        assert result.any_fix_works is True
        assert result.selected_fix is fix


# ---------------------------------------------------------------
# 2. Full pipeline: no fix works
# ---------------------------------------------------------------


class TestFullPipelineNoFixWorks:
    """All hypotheses fail -- no fix resolves the issue."""

    def test_selected_fix_is_none_when_all_fail(self) -> None:
        # SCENARIO: Two fixes, both return failing TestResults.
        # WHY: Runner must handle the case where nothing helps.
        # EXPECTED: selected_fix is None, any_fix_works is False.
        baseline = _make_result(passed=False)
        fix_a = _make_fix(title="Fix A")
        fix_b = _make_fix(title="Fix B")
        finding = _make_finding(
            fixes=[fix_a, fix_b],
            result=baseline,
            assertion_fn=lambda: baseline,
        )

        still_broken = _make_result(passed=False)

        runner = ExperimentRunner()
        hypotheses = [
            Hypothesis(fix=fix_a, apply_fn=lambda: still_broken),
            Hypothesis(fix=fix_b, apply_fn=lambda: still_broken),
        ]
        result = runner.run(finding, hypotheses=hypotheses)

        assert result.selected_fix is None
        assert result.any_fix_works is False


# ---------------------------------------------------------------
# 3. Multiple hypotheses ranked correctly
# ---------------------------------------------------------------


class TestMultipleHypothesesRankedCorrectly:
    """Three hypotheses with different outcomes are ranked properly."""

    def test_high_confidence_pass_ranks_first(self) -> None:
        # SCENARIO: One high-confidence pass, one low-confidence pass,
        #   one failure.
        # WHY: Default "passed" strategy should rank passes by
        #   confidence.
        # EXPECTED: high-conf pass > low-conf pass > fail.
        baseline = _make_result(
            passed=False,
            details={"actual": 0.3},
        )
        fix_hi = _make_fix(title="High conf", confidence="high")
        fix_lo = _make_fix(title="Low conf", confidence="low")
        fix_fail = _make_fix(title="Nope", confidence="medium")
        finding = _make_finding(
            fixes=[fix_hi, fix_lo, fix_fail],
            result=baseline,
            assertion_fn=lambda: baseline,
        )

        pass_result = _make_result(passed=True, details={"actual": 0.9})
        fail_result = _make_result(passed=False, details={"actual": 0.35})

        runner = ExperimentRunner()
        hypotheses = [
            Hypothesis(fix=fix_fail, apply_fn=lambda: fail_result),
            Hypothesis(fix=fix_lo, apply_fn=lambda: pass_result),
            Hypothesis(fix=fix_hi, apply_fn=lambda: pass_result),
        ]
        result = runner.run(finding, hypotheses=hypotheses)

        ranked = result.hypothesis_results
        assert len(ranked) == 3
        # Best result should be the high-confidence passing fix
        assert result.best_result is not None
        assert result.best_result.hypothesis.fix is fix_hi
        assert result.best_result.rank == 1
        # The failure should be last
        last = ranked[-1]
        assert last.hypothesis.fix is fix_fail


# ---------------------------------------------------------------
# 4. Runner with apply_fns dict
# ---------------------------------------------------------------


class TestRunnerWithApplyFnsDict:
    """Provide apply_fns via run_batch's string-keyed apply_fns_map."""

    def test_auto_builds_hypotheses_from_apply_fns_map(self) -> None:
        # SCENARIO: Use run_batch's apply_fns_map (scanner_name ->
        #   fix_title -> callable) so the runner auto-builds hypotheses
        #   for each finding's suggested_fixes.
        # WHY: Convenience API -- users provide a flat mapping of
        #   fix titles to callables and the runner resolves them.
        # EXPECTED: Runner auto-wraps into hypotheses and returns a
        #   valid ExperimentResult with the fix applied.
        baseline = _make_result(passed=False, details={"actual": 0.5})
        fix = _make_fix(title="Scale features")
        finding = _make_finding(
            fixes=[fix],
            result=baseline,
            assertion_fn=lambda: baseline,
            scanner_name="perf",
        )
        fixed = _make_result(passed=True, details={"actual": 0.85})

        runner = ExperimentRunner()
        results = runner.run_batch(
            [finding],
            apply_fns_map={
                "perf": {"Scale features": lambda: fixed},
            },
        )

        assert len(results) == 1
        result = results[0]
        assert isinstance(result, ExperimentResult)
        assert result.hypotheses_tested == 1
        assert result.any_fix_works is True


# ---------------------------------------------------------------
# 5. Runner handles exception in apply_fn
# ---------------------------------------------------------------


class TestRunnerHandlesApplyFnException:
    """apply_fn that raises should not crash the runner."""

    def test_exception_captured_as_failed_hypothesis(self) -> None:
        # SCENARIO: apply_fn raises RuntimeError instead of
        #   returning a TestResult.
        # WHY: Production apply_fns may fail -- runner must be
        #   resilient.
        # EXPECTED: Result captures it as a failed hypothesis;
        #   no unhandled exception propagates.
        baseline = _make_result(passed=False)
        fix = _make_fix(title="Risky fix")
        finding = _make_finding(
            fixes=[fix],
            result=baseline,
            assertion_fn=lambda: baseline,
        )

        def exploding_fn() -> TestResult:
            raise RuntimeError("apply_fn blew up")

        runner = ExperimentRunner()
        hypotheses = [Hypothesis(fix=fix, apply_fn=exploding_fn)]
        result = runner.run(finding, hypotheses=hypotheses)

        assert isinstance(result, ExperimentResult)
        assert result.hypotheses_tested == 1
        assert result.any_fix_works is False
        # The failed hypothesis result should not be marked as passing
        hr = result.hypothesis_results[0]
        assert hr.fixed_result.passed is False


# ---------------------------------------------------------------
# 6. run_batch with multiple findings
# ---------------------------------------------------------------


class TestRunBatchMultipleFindings:
    """run_batch processes multiple findings in one call."""

    def test_returns_one_result_per_finding(self) -> None:
        # SCENARIO: Two findings from different scanners, each with
        #   one fix. run_batch receives the findings list plus an
        #   apply_fns_map keyed by scanner_name -> fix_title -> fn.
        # WHY: Batch API should process all findings and return
        #   matching number of results.
        # EXPECTED: List of 2 ExperimentResults.
        baseline_a = _make_result(passed=False, name="finding_a")
        baseline_b = _make_result(passed=False, name="finding_b")
        fix_a = _make_fix(title="Fix A")
        fix_b = _make_fix(title="Fix B")
        finding_a = _make_finding(
            fixes=[fix_a],
            result=baseline_a,
            assertion_fn=lambda: baseline_a,
            scanner_name="scanner_a",
        )
        finding_b = _make_finding(
            fixes=[fix_b],
            result=baseline_b,
            assertion_fn=lambda: baseline_b,
            scanner_name="scanner_b",
        )
        fixed = _make_result(passed=True)

        runner = ExperimentRunner()
        results = runner.run_batch(
            [finding_a, finding_b],
            apply_fns_map={
                "scanner_a": {"Fix A": lambda: fixed},
                "scanner_b": {"Fix B": lambda: fixed},
            },
        )

        assert len(results) == 2
        assert all(isinstance(r, ExperimentResult) for r in results)


# ---------------------------------------------------------------
# 7. ExperimentResult properties
# ---------------------------------------------------------------


class TestExperimentResultProperties:
    """Verify aggregation properties on a real runner result."""

    def test_hypotheses_tested_winning_count_duration(self) -> None:
        # SCENARIO: Run 3 hypotheses -- 2 pass, 1 fails.
        # WHY: Verify hypotheses_tested, winning_count, best_result,
        #   and duration_ms all reflect accurate data from a real run.
        # EXPECTED: hypotheses_tested == 3, winning_count == 2,
        #   best_result is not None, duration_ms > 0.
        baseline = _make_result(passed=False, details={"actual": 0.3})
        fix_1 = _make_fix(title="Fix 1", confidence="high")
        fix_2 = _make_fix(title="Fix 2", confidence="medium")
        fix_3 = _make_fix(title="Fix 3", confidence="low")
        finding = _make_finding(
            fixes=[fix_1, fix_2, fix_3],
            result=baseline,
            assertion_fn=lambda: baseline,
        )

        pass_result = _make_result(passed=True, details={"actual": 0.9})
        fail_result = _make_result(passed=False, details={"actual": 0.35})

        runner = ExperimentRunner()
        hypotheses = [
            Hypothesis(fix=fix_1, apply_fn=lambda: pass_result),
            Hypothesis(fix=fix_2, apply_fn=lambda: pass_result),
            Hypothesis(fix=fix_3, apply_fn=lambda: fail_result),
        ]
        result = runner.run(finding, hypotheses=hypotheses)

        assert result.hypotheses_tested == 3
        assert result.winning_count == 2
        assert result.best_result is not None
        assert result.duration_ms > 0


# ---------------------------------------------------------------
# 8. Ranking strategy changes winner
# ---------------------------------------------------------------


class TestRankingStrategyChangesWinner:
    """Different strategies can produce different rankings."""

    def test_passed_vs_delta_produce_different_rank_one(self) -> None:
        # SCENARIO: Hypothesis A passes with low delta (0.1),
        #   Hypothesis B fails but has high delta (0.5).
        # WHY: "passed" strategy favours passing; "delta" strategy
        #   favours metric improvement regardless of pass/fail.
        # EXPECTED: "passed" -> A is rank 1; "delta" -> B is rank 1.
        baseline = _make_result(
            passed=False,
            details={"actual": 0.3},
        )
        fix_a = _make_fix(title="Low delta pass", confidence="high")
        fix_b = _make_fix(title="High delta fail", confidence="high")

        result_a = _make_result(passed=True, details={"actual": 0.4})
        result_b = _make_result(passed=False, details={"actual": 0.8})

        hr_a = HypothesisResult(
            hypothesis=Hypothesis(fix=fix_a, apply_fn=lambda: result_a),
            baseline_result=baseline,
            fixed_result=result_a,
        )
        hr_b = HypothesisResult(
            hypothesis=Hypothesis(fix=fix_b, apply_fn=lambda: result_b),
            baseline_result=baseline,
            fixed_result=result_b,
        )

        # "passed" strategy: A (passes) should be rank 1
        passed_ranked = rank_hypotheses(
            [hr_b, hr_a], strategy="passed"
        )
        assert passed_ranked[0] is hr_a
        assert hr_a.rank == 1

        # Reset ranks before re-ranking
        hr_a.rank = 0
        hr_b.rank = 0

        # "delta" strategy: B (delta=0.5) should be rank 1
        delta_ranked = rank_hypotheses(
            [hr_a, hr_b], strategy="delta"
        )
        assert delta_ranked[0] is hr_b
        assert hr_b.rank == 1


# ---------------------------------------------------------------
# 9. Empty hypotheses list
# ---------------------------------------------------------------


class TestEmptyHypothesesList:
    """run() with no hypotheses produces an empty result."""

    def test_empty_hypotheses_returns_empty_result(self) -> None:
        # SCENARIO: Call run() with an empty hypothesis list.
        # WHY: Edge case -- no fixes to test should not crash.
        # EXPECTED: ExperimentResult with empty results, no
        #   selected_fix.
        baseline = _make_result(passed=False)
        finding = _make_finding(
            result=baseline,
            assertion_fn=lambda: baseline,
        )

        runner = ExperimentRunner()
        result = runner.run(finding, hypotheses=[])

        assert isinstance(result, ExperimentResult)
        assert result.hypotheses_tested == 0
        assert result.hypothesis_results == []
        assert result.selected_fix is None
        assert result.any_fix_works is False


# ---------------------------------------------------------------
# 10. HypothesisResult.metric_delta
# ---------------------------------------------------------------


class TestHypothesisResultMetricDelta:
    """metric_delta computes fixed.actual - baseline.actual."""

    def test_metric_delta_equals_expected_value(self) -> None:
        # SCENARIO: Baseline details={"actual": 0.5}, fixed
        #   details={"actual": 0.8}.
        # WHY: metric_delta should be the arithmetic difference
        #   of the "actual" values between fixed and baseline.
        # EXPECTED: metric_delta == 0.3.
        baseline = _make_result(
            passed=False,
            details={"actual": 0.5},
        )
        fixed = _make_result(
            passed=True,
            details={"actual": 0.8},
        )
        fix = _make_fix()
        hr = HypothesisResult(
            hypothesis=Hypothesis(fix=fix, apply_fn=lambda: fixed),
            baseline_result=baseline,
            fixed_result=fixed,
        )

        assert hr.metric_delta == pytest.approx(0.3)
