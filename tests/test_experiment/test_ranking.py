"""Tests for mltk.experiment.ranking."""

from __future__ import annotations

import pytest

from mltk.core.result import Severity, TestResult
from mltk.experiment.hypothesis import Hypothesis, HypothesisResult
from mltk.experiment.ranking import rank_hypotheses
from mltk.scan.finding import FixSuggestion

# -- helper ----------------------------------------------------------------


def _make_hr(
    passed: bool = True,
    confidence: str = "high",
    delta: float = 0.1,
    duration_ms: float = 10.0,
) -> HypothesisResult:
    fix = FixSuggestion(
        category="code", title="fix", description="d", confidence=confidence
    )
    hyp = Hypothesis(fix=fix, apply_fn=lambda: None)
    baseline = TestResult(
        name="t",
        passed=False,
        severity=Severity.WARNING,
        message="baseline",
        details={"actual": 0.5},
    )
    fixed_val = 0.5 + delta
    fixed = TestResult(
        name="t",
        passed=passed,
        severity=Severity.INFO,
        message="fixed",
        details={"actual": fixed_val},
        duration_ms=duration_ms,
    )
    return HypothesisResult(
        hypothesis=hyp,
        baseline_result=baseline,
        fixed_result=fixed,
        improvement=delta,
    )


# -- "passed" strategy -----------------------------------------------------


class TestPassedStrategy:
    def test_passed_ranks_above_failed(self) -> None:
        hr_pass = _make_hr(passed=True)
        hr_fail = _make_hr(passed=False)
        ranked = rank_hypotheses([hr_fail, hr_pass], strategy="passed")
        assert ranked[0] is hr_pass
        assert ranked[1] is hr_fail
        assert hr_pass.rank == 1
        assert hr_fail.rank == 2

    def test_higher_confidence_wins_among_passed(self) -> None:
        hr_high = _make_hr(passed=True, confidence="high", delta=0.1)
        hr_low = _make_hr(passed=True, confidence="low", delta=0.1)
        ranked = rank_hypotheses([hr_low, hr_high], strategy="passed")
        assert ranked[0] is hr_high
        assert ranked[1] is hr_low

    def test_higher_delta_wins_at_same_confidence(self) -> None:
        hr_big = _make_hr(passed=True, confidence="medium", delta=0.3)
        hr_small = _make_hr(passed=True, confidence="medium", delta=0.1)
        ranked = rank_hypotheses([hr_small, hr_big], strategy="passed")
        assert ranked[0] is hr_big
        assert ranked[1] is hr_small


# -- "delta" strategy ------------------------------------------------------


class TestDeltaStrategy:
    def test_highest_delta_ranks_first(self) -> None:
        hr_big = _make_hr(delta=0.5)
        hr_small = _make_hr(delta=0.1)
        ranked = rank_hypotheses([hr_small, hr_big], strategy="delta")
        assert ranked[0] is hr_big
        assert hr_big.rank == 1

    def test_passed_wins_at_same_delta(self) -> None:
        hr_pass = _make_hr(passed=True, delta=0.2)
        hr_fail = _make_hr(passed=False, delta=0.2)
        ranked = rank_hypotheses([hr_fail, hr_pass], strategy="delta")
        assert ranked[0] is hr_pass


# -- "composite" strategy --------------------------------------------------


class TestCompositeStrategy:
    def test_basic_ranking(self) -> None:
        hr_good = _make_hr(passed=True, confidence="high", delta=0.3, duration_ms=10.0)
        hr_bad = _make_hr(passed=False, confidence="low", delta=0.05, duration_ms=900.0)
        ranked = rank_hypotheses([hr_bad, hr_good], strategy="composite")
        assert ranked[0] is hr_good
        assert hr_good.rank == 1

    def test_fast_beats_slow_same_pass_confidence(self) -> None:
        hr_fast = _make_hr(passed=True, confidence="medium", delta=0.1, duration_ms=50.0)
        hr_slow = _make_hr(passed=True, confidence="medium", delta=0.1, duration_ms=900.0)
        ranked = rank_hypotheses([hr_slow, hr_fast], strategy="composite")
        assert ranked[0] is hr_fast


# -- edge cases ------------------------------------------------------------


class TestEdgeCases:
    def test_empty_list(self) -> None:
        result = rank_hypotheses([], strategy="passed")
        assert result == []

    def test_single_result_gets_rank_one(self) -> None:
        hr = _make_hr()
        ranked = rank_hypotheses([hr], strategy="passed")
        assert len(ranked) == 1
        assert ranked[0].rank == 1

    def test_invalid_strategy_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown ranking strategy"):
            rank_hypotheses([_make_hr()], strategy="invalid")
