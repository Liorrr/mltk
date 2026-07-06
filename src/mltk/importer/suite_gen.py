"""Build in-memory mltk suites from imported evaluation datasets."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

from mltk.core.suite import MltkSuite
from mltk.eval.dataset import EvalDataset
from mltk.importer.classify import TaskType
from mltk.importer.schema import ColumnMapping

_BUCKET_SIZE = 0.05
_FLOAT_EPSILON = 1e-12


def _floor_to_bucket(value: float) -> float:
    """Floor *value* to the nearest readable baseline bucket."""
    bucket = math.floor((value + _FLOAT_EPSILON) / _BUCKET_SIZE)
    return round(bucket * _BUCKET_SIZE, 2)


def _ceil_to_bucket(value: float) -> float:
    """Ceil *value* to the nearest readable baseline bucket."""
    bucket = math.ceil((value - _FLOAT_EPSILON) / _BUCKET_SIZE)
    return round(bucket * _BUCKET_SIZE, 2)


def _duplicate_rate(dataset: EvalDataset) -> float:
    """Mirror assert_dataset_quality's duplicate-input calculation."""
    inputs = [sample.input for sample in dataset.samples]
    total = len(inputs)
    if total == 0:
        return 0.0
    return 1.0 - (len(set(inputs)) / total)


def _context_from_metadata(metadata: dict[str, Any]) -> str | list[str] | None:
    """Return normalized context metadata, or None when absent."""
    if "context" not in metadata:
        return None

    context = metadata.get("context")
    if context is None:
        return None

    if isinstance(context, list):
        present = [str(item) for item in context if item is not None]
        return present or None

    return context if isinstance(context, str) else str(context)


def compute_baseline_thresholds(dataset: EvalDataset) -> dict[str, Any]:
    """Compute self-passing dataset-quality thresholds for an EvalDataset.

    Baselines are bucketed to readable 0.05 increments rather than exact
    snapshots, while still ensuring the current dataset satisfies them.
    """
    duplicate_rate = _duplicate_rate(dataset)
    categories = dataset.categories

    return {
        "min_samples": dataset.sample_count,
        "min_target_coverage": _floor_to_bucket(dataset.target_coverage),
        "max_duplicate_rate": (
            0.01 if duplicate_rate == 0.0 else _ceil_to_bucket(duplicate_rate)
        ),
        "min_categories": len(categories) if categories else None,
    }


def build_suite(
    eval_dataset: EvalDataset,
    mapping: ColumnMapping,
    task_type: TaskType,
    *,
    judge_fn: Callable[[str, str], float] | None = None,
) -> MltkSuite:
    """Build a runnable in-memory MltkSuite for an imported dataset."""
    from mltk.eval.dataset import assert_dataset_quality

    suite = MltkSuite(f"import:{eval_dataset.name}")
    baseline = compute_baseline_thresholds(eval_dataset)
    quality_kwargs = {
        "min_samples": baseline["min_samples"],
        "min_target_coverage": baseline["min_target_coverage"],
        "max_duplicate_rate": baseline["max_duplicate_rate"],
    }
    if baseline["min_categories"] is not None:
        quality_kwargs["min_categories"] = baseline["min_categories"]

    suite.add(assert_dataset_quality, eval_dataset, **quality_kwargs)

    if judge_fn is None:
        return suite

    if task_type is TaskType.QA_RAG:
        from mltk.domains.llm.rag import (
            assert_answer_relevancy,
            assert_context_relevancy,
            assert_faithfulness,
        )

        for sample in eval_dataset.samples:
            context = _context_from_metadata(sample.metadata)
            if context is None:
                continue

            if sample.target is not None:
                suite.add(
                    assert_faithfulness,
                    answer=sample.target,
                    context=context,
                    method="llm",
                    judge_fn=judge_fn,
                )
                suite.add(
                    assert_answer_relevancy,
                    question=sample.input,
                    answer=sample.target,
                    method="llm",
                    judge_fn=judge_fn,
                )

            suite.add(
                assert_context_relevancy,
                question=sample.input,
                context=context,
                method="llm",
                judge_fn=judge_fn,
            )
    elif task_type is TaskType.RETRIEVAL:
        from mltk.domains.llm.rag import assert_context_relevancy

        for sample in eval_dataset.samples:
            context = _context_from_metadata(sample.metadata)
            if context is None:
                continue

            suite.add(
                assert_context_relevancy,
                question=sample.input,
                context=context,
                method="llm",
                judge_fn=judge_fn,
            )

    # Other S98 task types stay Tier 1: model-bound checks are emitted
    # into the generated pytest file by a separate worker.
    return suite


__all__ = [
    "build_suite",
    "compute_baseline_thresholds",
]
