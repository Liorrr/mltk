"""Hypothesis dataclasses for the experiment runner.

A :class:`Hypothesis` pairs a :class:`~mltk.scan.finding.FixSuggestion`
with a callable that applies the fix to a model/data pipeline.  The
:class:`ExperimentRunner` executes each hypothesis, producing a
:class:`HypothesisResult` that captures the before/after comparison.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from mltk.core.result import TestResult
from mltk.scan.finding import FixSuggestion

__all__ = ["Hypothesis", "HypothesisResult"]


@dataclass
class Hypothesis:
    """A single fix hypothesis to test against a finding.

    Attributes:
        fix: The fix suggestion being tested.
        apply_fn: Callable that applies the fix and returns a
            :class:`~mltk.core.result.TestResult`.  Signature:
            ``() -> TestResult``.  The caller is responsible for
            setting up modified model/data internally.
        description: Human-readable description of what
            ``apply_fn`` does.  Defaults to empty string.
    """

    fix: FixSuggestion
    apply_fn: Callable
    description: str = ""


@dataclass
class HypothesisResult:
    """Result of testing one hypothesis.

    Attributes:
        hypothesis: The hypothesis that was tested.
        baseline_result: Original failing assertion result
            (before applying the fix).
        fixed_result: Assertion result after applying the fix.
        improvement: Binary score -- ``1.0`` if the fix turned
            a failure into a pass, ``0.0`` otherwise.
        rank: Position in the ranked list (1 = best).  Set by
            the ranking strategy after all hypotheses run.
    """

    hypothesis: Hypothesis
    baseline_result: TestResult
    fixed_result: TestResult
    improvement: float = 0.0
    rank: int = 0

    @property
    def is_winning(self) -> bool:
        """True if the fix turned a failure into a pass."""
        return self.fixed_result.passed and not self.baseline_result.passed

    @property
    def metric_delta(self) -> float:
        """Delta between fixed and baseline metric values from details.

        Looks for common metric keys (``actual``, ``statistic``,
        ``score``, ``value``) in both the baseline and fixed
        ``TestResult.details`` dicts.  Returns the first numeric
        delta found, or ``0.0`` if no matching keys exist.
        """
        for key in ("actual", "statistic", "score", "value"):
            baseline_val = self.baseline_result.details.get(key)
            fixed_val = self.fixed_result.details.get(key)
            if baseline_val is not None and fixed_val is not None:
                try:
                    return float(fixed_val) - float(baseline_val)
                except (TypeError, ValueError):
                    continue
        return 0.0
