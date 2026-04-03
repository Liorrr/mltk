"""Tests for mltk.experiment.hypothesis and mltk.experiment.result.

Covers the Hypothesis, HypothesisResult, and ExperimentResult
dataclasses -- field defaults, is_winning logic, metric_delta
extraction, and aggregation properties.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from mltk.core.result import Severity, TestResult
from mltk.scan.finding import FixSuggestion, ScanFinding

try:
    from mltk.experiment.hypothesis import Hypothesis, HypothesisResult
    from mltk.experiment.result import ExperimentResult
except ImportError:
    Hypothesis = None  # type: ignore[assignment,misc]
    HypothesisResult = None  # type: ignore[assignment,misc]
    ExperimentResult = None  # type: ignore[assignment,misc]

pytestmark = pytest.mark.skipif(
    Hypothesis is None,
    reason="mltk.experiment not yet implemented",
)


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------


def _make_fix(**overrides) -> FixSuggestion:
    """Build a FixSuggestion with sensible defaults."""
    defaults = {
        "category": "code",
        "title": "Increase training epochs",
        "description": "Train longer to improve accuracy.",
        "confidence": "high",
    }
    defaults.update(overrides)
    return FixSuggestion(**defaults)


def _make_test_result(
    *,
    passed: bool = False,
    details: dict | None = None,
) -> TestResult:
    """Build a TestResult with sensible defaults."""
    return TestResult(
        name="assert_metric",
        passed=passed,
        severity=Severity.CRITICAL,
        message="accuracy below threshold" if not passed else "ok",
        details=details or {},
    )


def _make_hypothesis(**overrides) -> Hypothesis:
    """Build a Hypothesis with sensible defaults."""
    defaults = {
        "fix": _make_fix(),
        "apply_fn": lambda m, X, y: (m, X, y),
    }
    defaults.update(overrides)
    return Hypothesis(**defaults)


def _make_hypothesis_result(
    *,
    baseline_passed: bool = False,
    fixed_passed: bool = True,
    baseline_details: dict | None = None,
    fixed_details: dict | None = None,
    rank: int = 0,
) -> HypothesisResult:
    """Build a HypothesisResult with configurable pass/fail."""
    return HypothesisResult(
        hypothesis=_make_hypothesis(),
        baseline_result=_make_test_result(
            passed=baseline_passed,
            details=baseline_details or {},
        ),
        fixed_result=_make_test_result(
            passed=fixed_passed,
            details=fixed_details or {},
        ),
        rank=rank,
    )


def _make_finding() -> ScanFinding:
    """Build a ScanFinding with sensible defaults."""
    return ScanFinding(
        result=_make_test_result(passed=False),
        assertion_fn=lambda y, p: None,
        scanner_name="slice",
    )


# ---------------------------------------------------------------
# Hypothesis tests
# ---------------------------------------------------------------


class TestHypothesis:
    """Hypothesis stores fix, apply_fn, and description."""

    def test_creation_with_all_fields(self) -> None:
        # SCENARIO: Build a Hypothesis with explicit description.
        # WHY: Verify all three fields are stored correctly.
        # EXPECTED: Each attribute matches the constructor arg.
        fix = _make_fix(title="Scale features")
        apply_fn = MagicMock()
        h = Hypothesis(fix=fix, apply_fn=apply_fn, description="Normalize X")

        assert h.fix is fix
        assert h.apply_fn is apply_fn
        assert h.description == "Normalize X"

    def test_description_default_is_empty(self) -> None:
        # SCENARIO: Build a Hypothesis without description.
        # WHY: Default should be empty string, not None.
        # EXPECTED: description == "".
        h = _make_hypothesis()

        assert h.description == ""


# ---------------------------------------------------------------
# HypothesisResult tests
# ---------------------------------------------------------------


class TestHypothesisResultIsWinning:
    """is_winning: True iff fix turned failure into pass."""

    def test_winning_when_fix_passes_and_baseline_failed(self) -> None:
        # SCENARIO: Baseline failed, fix passed.
        # WHY: This is the canonical "winning" case.
        # EXPECTED: is_winning is True.
        hr = _make_hypothesis_result(
            baseline_passed=False,
            fixed_passed=True,
        )

        assert hr.is_winning is True

    def test_not_winning_when_both_pass(self) -> None:
        # SCENARIO: Both baseline and fix passed.
        # WHY: Fix did not change the outcome -- not a real fix.
        # EXPECTED: is_winning is False.
        hr = _make_hypothesis_result(
            baseline_passed=True,
            fixed_passed=True,
        )

        assert hr.is_winning is False

    def test_not_winning_when_fix_also_fails(self) -> None:
        # SCENARIO: Baseline failed and fix also failed.
        # WHY: Fix did not resolve the issue.
        # EXPECTED: is_winning is False.
        hr = _make_hypothesis_result(
            baseline_passed=False,
            fixed_passed=False,
        )

        assert hr.is_winning is False


class TestHypothesisResultMetricDelta:
    """metric_delta: extracts delta from details dicts."""

    def test_delta_with_actual_key(self) -> None:
        # SCENARIO: Both details have "actual" key.
        # WHY: "actual" is the first key checked in metric_delta.
        # EXPECTED: Returns fixed - baseline = 0.9 - 0.6 = 0.3.
        hr = _make_hypothesis_result(
            baseline_details={"actual": 0.6},
            fixed_details={"actual": 0.9},
        )

        assert hr.metric_delta == pytest.approx(0.3)

    def test_delta_with_statistic_key(self) -> None:
        # SCENARIO: Details use "statistic" key (no "actual").
        # WHY: "statistic" is the second fallback key.
        # EXPECTED: Returns fixed - baseline = 5.0 - 2.0 = 3.0.
        hr = _make_hypothesis_result(
            baseline_details={"statistic": 2.0},
            fixed_details={"statistic": 5.0},
        )

        assert hr.metric_delta == pytest.approx(3.0)

    def test_delta_returns_zero_when_no_metric_keys(self) -> None:
        # SCENARIO: Details contain no recognized metric keys.
        # WHY: Fallback should be 0.0, not an error.
        # EXPECTED: metric_delta == 0.0.
        hr = _make_hypothesis_result(
            baseline_details={"other": "abc"},
            fixed_details={"other": "def"},
        )

        assert hr.metric_delta == 0.0

    def test_delta_with_non_numeric_values_returns_zero(self) -> None:
        # SCENARIO: "actual" exists but holds non-numeric strings.
        # WHY: Conversion should fail gracefully, returning 0.0.
        # EXPECTED: metric_delta == 0.0.
        hr = _make_hypothesis_result(
            baseline_details={"actual": "not-a-number"},
            fixed_details={"actual": "also-not-a-number"},
        )

        assert hr.metric_delta == 0.0


# ---------------------------------------------------------------
# ExperimentResult tests
# ---------------------------------------------------------------


class TestExperimentResultAnyFixWorks:
    """any_fix_works: True when at least one hypothesis wins."""

    def test_true_when_one_hypothesis_wins(self) -> None:
        # SCENARIO: Two hypotheses -- one wins, one fails.
        # WHY: any_fix_works should be True if any single fix works.
        # EXPECTED: True.
        winning = _make_hypothesis_result(
            baseline_passed=False,
            fixed_passed=True,
        )
        losing = _make_hypothesis_result(
            baseline_passed=False,
            fixed_passed=False,
        )
        er = ExperimentResult(
            finding=_make_finding(),
            baseline_result=_make_test_result(passed=False),
            hypothesis_results=[losing, winning],
        )

        assert er.any_fix_works is True

    def test_false_when_all_fail(self) -> None:
        # SCENARIO: All hypotheses fail to fix the issue.
        # WHY: No winning hypothesis means no fix works.
        # EXPECTED: False.
        hr1 = _make_hypothesis_result(
            baseline_passed=False,
            fixed_passed=False,
        )
        hr2 = _make_hypothesis_result(
            baseline_passed=False,
            fixed_passed=False,
        )
        er = ExperimentResult(
            finding=_make_finding(),
            baseline_result=_make_test_result(passed=False),
            hypothesis_results=[hr1, hr2],
        )

        assert er.any_fix_works is False


class TestExperimentResultBestResult:
    """best_result: returns the rank-1 hypothesis."""

    def test_returns_rank_one_hypothesis(self) -> None:
        # SCENARIO: Three hypotheses with different ranks.
        # WHY: best_result should find rank==1 regardless of list order.
        # EXPECTED: Returns the hypothesis with rank=1.
        hr_rank2 = _make_hypothesis_result(rank=2)
        hr_rank1 = _make_hypothesis_result(rank=1)
        hr_rank3 = _make_hypothesis_result(rank=3)
        er = ExperimentResult(
            finding=_make_finding(),
            baseline_result=_make_test_result(passed=False),
            hypothesis_results=[hr_rank2, hr_rank1, hr_rank3],
        )

        assert er.best_result is hr_rank1


class TestExperimentResultCounts:
    """hypotheses_tested and winning_count aggregations."""

    def test_hypotheses_tested_count(self) -> None:
        # SCENARIO: Three hypothesis results in the list.
        # WHY: Should reflect len(hypothesis_results).
        # EXPECTED: 3.
        er = ExperimentResult(
            finding=_make_finding(),
            baseline_result=_make_test_result(passed=False),
            hypothesis_results=[
                _make_hypothesis_result(),
                _make_hypothesis_result(),
                _make_hypothesis_result(),
            ],
        )

        assert er.hypotheses_tested == 3

    def test_winning_count(self) -> None:
        # SCENARIO: Two winning hypotheses and one losing.
        # WHY: winning_count should count only is_winning==True.
        # EXPECTED: 2.
        winner1 = _make_hypothesis_result(
            baseline_passed=False,
            fixed_passed=True,
        )
        winner2 = _make_hypothesis_result(
            baseline_passed=False,
            fixed_passed=True,
        )
        loser = _make_hypothesis_result(
            baseline_passed=False,
            fixed_passed=False,
        )
        er = ExperimentResult(
            finding=_make_finding(),
            baseline_result=_make_test_result(passed=False),
            hypothesis_results=[winner1, loser, winner2],
        )

        assert er.winning_count == 2
