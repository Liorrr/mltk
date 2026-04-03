"""Experiment runner -- validate fix suggestions via hypothesis testing.

The experiment module bridges scan findings (what's wrong) and fix
suggestions (how to fix it) by actually testing each fix against the
model/data and ranking results.

Pipeline::

    ScanFinding + FixSuggestions
        -> ExperimentRunner.run(finding, model_fn, X, y, apply_fns)
        -> For each fix: apply -> re-run assertion -> collect result
        -> Rank by strategy (passed > delta > confidence)
        -> ExperimentResult with selected_fix

Usage::

    from mltk.experiment import ExperimentRunner

    runner = ExperimentRunner()
    result = runner.run(finding, model_fn, X, y, apply_fns={
        finding.suggested_fixes[0]: lambda m, X, y: retrained_model,
    })
    print(result.selected_fix.title)
"""
from __future__ import annotations

from mltk.experiment.hypothesis import Hypothesis, HypothesisResult
from mltk.experiment.ranking import rank_hypotheses
from mltk.experiment.result import ExperimentResult
from mltk.experiment.runner import ExperimentRunner
from mltk.experiment.sandbox import SandboxedExperimentRunner
from mltk.experiment.worktree import GitWorktree

__all__ = [
    "ExperimentResult",
    "ExperimentRunner",
    "GitWorktree",
    "Hypothesis",
    "HypothesisResult",
    "SandboxedExperimentRunner",
    "rank_hypotheses",
]
