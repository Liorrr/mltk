"""Composable evaluation pipelines for LLM testing.

The ``mltk.eval`` package provides a modular evaluation
framework inspired by UK AISI Inspect AI, adapted for
mltk's pytest-native, zero-dependency philosophy.

Architecture overview::

    Dataset (EvalSample[])
        |
        v
    Solver pipeline (prompt engineering)
        |
        v
    Scorer fan-out (independent scoring)
        |
        v
    EvalResult (aggregated metrics)

Core components:

- **Data types** (``_types``): EvalSample, EvalState,
  Score, EvalResult -- the contract between stages.
- **Solvers** (``solvers``): Transform state through a
  pipeline (CoT, few-shot, generate).
- **Scorers** (``scorers``): Score model output against
  expected answers (exact match, includes, LLM judge).
- **Task runner** (``task``): Orchestrates the full
  pipeline and aggregates results.

Quick start::

    from mltk.eval import (
        EvalSample,
        EvalTask,
        ExactMatchScorer,
        GenerateSolver,
    )

    task = EvalTask(
        name="qa",
        solver=GenerateSolver(),
        scorers=ExactMatchScorer(),
        dataset=[
            EvalSample("What is 2+2?", "4"),
            EvalSample("Capital of France?", "Paris"),
        ],
    )
    result = task.run(my_model_fn)
    assert result.metrics["ExactMatchScorer/accuracy"] >= 0.9

Pytest integration::

    def test_qa_eval():
        task = EvalTask(...)
        result = task.to_test_result(model_fn)
        assert result.passed
"""

from __future__ import annotations

# Data types -- shared contract between pipeline stages
from mltk.eval._types import (
    EvalResult,
    EvalSample,
    EvalState,
    Score,
)

# Scorers -- score model output against expected answers
from mltk.eval.scorers import (
    ExactMatchScorer,
    IncludesScorer,
    LLMJudgeScorer,
    PatternScorer,
    Scorer,
)

# Solvers -- transform state through a processing pipeline
from mltk.eval.solvers import (
    ChainOfThoughtSolver,
    FewShotSolver,
    GenerateSolver,
    Solver,
    chain,
)

# Task runner -- orchestrate the full evaluation pipeline
from mltk.eval.task import EvalTask, load_dataset

__all__ = [
    # Data types
    "EvalSample",
    "EvalState",
    "Score",
    "EvalResult",
    # Solvers
    "Solver",
    "GenerateSolver",
    "ChainOfThoughtSolver",
    "FewShotSolver",
    "chain",
    # Scorers
    "Scorer",
    "ExactMatchScorer",
    "IncludesScorer",
    "LLMJudgeScorer",
    "PatternScorer",
    # Task runner
    "EvalTask",
    "load_dataset",
]
