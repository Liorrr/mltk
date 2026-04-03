"""Experiment result -- aggregated outcome for one finding.

An :class:`ExperimentResult` collects all :class:`HypothesisResult`
objects produced by the :class:`ExperimentRunner` for a single
:class:`~mltk.scan.finding.ScanFinding`.  It exposes convenience
properties for querying whether any fix worked, which fix is best,
and how many hypotheses were tested.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from mltk.core.result import TestResult
from mltk.experiment.hypothesis import HypothesisResult
from mltk.scan.finding import FixSuggestion, ScanFinding

__all__ = ["ExperimentResult"]


@dataclass
class ExperimentResult:
    """Aggregated result from testing all hypotheses for one finding.

    Attributes:
        finding: The scan finding that prompted the experiment.
        baseline_result: The original (pre-fix) assertion result.
        hypothesis_results: Ordered list of per-hypothesis outcomes.
        selected_fix: The fix chosen by the ranking strategy, or
            ``None`` if no fix resolved the finding.
        duration_ms: Wall-clock time for the entire experiment
            (all hypotheses combined), in milliseconds.
    """

    finding: ScanFinding
    baseline_result: TestResult
    hypothesis_results: list[HypothesisResult] = field(default_factory=list)
    selected_fix: FixSuggestion | None = None
    duration_ms: float = 0.0

    @property
    def any_fix_works(self) -> bool:
        """True if at least one hypothesis resolved the finding."""
        return any(hr.is_winning for hr in self.hypothesis_results)

    @property
    def best_result(self) -> HypothesisResult | None:
        """Return the top-ranked hypothesis (rank == 1).

        Falls back to the first hypothesis in the list if none
        has been explicitly ranked yet.  Returns ``None`` when
        no hypotheses were tested.
        """
        for hr in self.hypothesis_results:
            if hr.rank == 1:
                return hr
        return self.hypothesis_results[0] if self.hypothesis_results else None

    @property
    def hypotheses_tested(self) -> int:
        """Number of hypotheses that were evaluated."""
        return len(self.hypothesis_results)

    @property
    def winning_count(self) -> int:
        """Number of hypotheses that resolved the finding."""
        return sum(1 for hr in self.hypothesis_results if hr.is_winning)
