"""Composable evaluation pipelines for LLM testing.

The ``mltk.eval`` package provides a modular evaluation
framework inspired by UK AISI Inspect AI, adapted for
mltk's pytest-native, zero-dependency philosophy.

Architecture overview::

    Dataset (list[EvalSample] | EvalDataset)
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
- **Dataset** (``dataset``): EvalDataset with metadata,
  versioning, quality checks, registry, and diffs.
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

With an EvalDataset::

    from mltk.eval import EvalDataset, DatasetCard

    ds = EvalDataset(
        name="qa-v1",
        samples=[EvalSample("2+2?", "4")],
        card=DatasetCard(description="Math QA set"),
        version="1.0.0",
    )
    task = EvalTask(
        name="qa",
        solver=GenerateSolver(),
        scorers=ExactMatchScorer(),
        dataset=ds,
    )
    assert task.eval_dataset is ds

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

# Dataset -- structured datasets with metadata + quality
from mltk.eval.dataset import (
    DatasetCard,
    DatasetDiff,
    DatasetInfo,
    DatasetRegistry,
    EvalDataset,
    assert_dataset_quality,
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
    # Dataset
    "DatasetCard",
    "DatasetDiff",
    "DatasetInfo",
    "DatasetRegistry",
    "EvalDataset",
    "assert_dataset_quality",
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
