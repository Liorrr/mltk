"""Evaluation task runner -- compose solvers + scorers + dataset.

The EvalTask is the central orchestrator for composable
evaluation pipelines. It runs a dataset through a solver
pipeline, scores each output with one or more scorers,
and aggregates the results.

Accepts both a plain ``list[EvalSample]`` and a structured
``EvalDataset`` (with metadata, versioning, quality checks).
When an EvalDataset is provided, its metadata is preserved
and accessible via the ``eval_dataset`` property.

Pipeline architecture::

    Dataset (list[EvalSample] | EvalDataset)
        |
        v
    [Solver 1] -> [Solver 2] -> ... -> [Solver N]
        |
        v
    [Scorer A]  [Scorer B]  ... [Scorer M]
        |            |               |
        v            v               v
    Score[]      Score[]         Score[]
        |            |               |
        +------+-----+------...-----+
               |
               v
          EvalResult (aggregated metrics)

Each solver runs in sequence (like a pipeline). Each scorer
runs independently on the same final state (like fan-out).

Inspired by UK AISI Inspect AI's task pattern, adapted for
mltk's pytest-native, zero-dependency philosophy.

Example::

    task = EvalTask(
        name="qa-eval",
        solver=[ChainOfThoughtSolver(), GenerateSolver()],
        scorers=[ExactMatchScorer(), LLMJudgeScorer(judge)],
        dataset=[
            EvalSample("What is 2+2?", "4"),
            EvalSample("Capital of France?", "Paris"),
        ],
    )
    result = task.run(model_fn)
    assert result.metrics["ExactMatchScorer/accuracy"] >= 0.9

With an EvalDataset::

    ds = EvalDataset(
        name="qa-v1",
        samples=[EvalSample("2+2?", "4")],
        version="1.0.0",
    )
    task = EvalTask(
        name="qa-eval",
        solver=GenerateSolver(),
        scorers=ExactMatchScorer(),
        dataset=ds,
    )
    assert task.eval_dataset is ds
"""

from __future__ import annotations

import csv
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from mltk.eval._types import (
    EvalResult,
    EvalSample,
    EvalState,
    Score,
)

if TYPE_CHECKING:
    from mltk.core.result import TestResult
    from mltk.eval.dataset import EvalDataset
    from mltk.eval.scorers import Scorer
    from mltk.eval.solvers import Solver

__all__ = ["EvalTask", "load_dataset"]


class EvalTask:
    """Composable evaluation task.

    Combines a solver pipeline, one or more scorers, and a
    dataset into a runnable evaluation. Follows the Inspect AI
    pattern adapted for mltk's pytest-native design.

    The task validates its inputs eagerly at construction time
    -- you get clear errors immediately, not midway through a
    long evaluation run.

    The ``dataset`` parameter accepts either a plain
    ``list[EvalSample]`` or a structured ``EvalDataset``.
    When an ``EvalDataset`` is provided, its samples are
    extracted for evaluation and the dataset metadata is
    accessible via the ``eval_dataset`` property.

    Args:
        name: Task name for logging/reporting.
        solver: Single solver or list of solvers (pipeline).
            If a list, they run in sequence (like chain()).
        scorers: Single scorer or list of scorers.
            Each scorer runs independently on the same state.
        dataset: List of EvalSample or an EvalDataset.

    Raises:
        ValueError: If solver, scorers, or dataset is empty.

    Example:
        >>> from mltk.eval.solvers import GenerateSolver
        >>> from mltk.eval.scorers import ExactMatchScorer
        >>> task = EvalTask(
        ...     name="math",
        ...     solver=GenerateSolver(),
        ...     scorers=ExactMatchScorer(),
        ...     dataset=[EvalSample("2+2?", "4")],
        ... )
        >>> result = task.run(lambda p: "4")
        >>> result.metrics["ExactMatchScorer/accuracy"]
        1.0
    """

    def __init__(
        self,
        name: str,
        solver: Solver | list[Solver],
        scorers: Scorer | list[Scorer],
        dataset: list[EvalSample] | EvalDataset,
    ) -> None:
        self.name = name
        self._solvers: list[Solver] = (
            solver if isinstance(solver, list) else [solver]
        )
        self._scorers: list[Scorer] = (
            scorers
            if isinstance(scorers, list)
            else [scorers]
        )

        # Accept EvalDataset or plain list[EvalSample]
        if hasattr(dataset, "samples"):
            self._eval_dataset: EvalDataset | None = (
                dataset  # type: ignore[assignment]
            )
            self._dataset = dataset.samples  # type: ignore[union-attr]
        else:
            self._eval_dataset = None
            self._dataset = dataset  # type: ignore[assignment]

        if not self._solvers:
            raise ValueError(
                "EvalTask requires at least one solver"
            )
        if not self._scorers:
            raise ValueError(
                "EvalTask requires at least one scorer"
            )
        if not self._dataset:
            raise ValueError(
                "EvalTask requires a non-empty dataset"
            )

    @property
    def eval_dataset(self) -> EvalDataset | None:
        """The EvalDataset if one was provided, else None.

        Returns:
            The original EvalDataset instance, or None if
            the task was constructed with a plain sample
            list.
        """
        return self._eval_dataset

    def run(
        self,
        model_fn: Callable[[str], str],
    ) -> EvalResult:
        """Execute the evaluation pipeline.

        For each sample:

        1. Create EvalState from sample.
        2. Run solver pipeline (each solver in sequence).
        3. Run all scorers on the final state.
        4. Collect scores.

        After all samples:

        5. Compute aggregated metrics per scorer.

        Scorer errors are caught and recorded as zero-score
        results with an error explanation -- a single bad
        scorer does not crash the entire evaluation.

        Args:
            model_fn: Callable that sends a prompt string
                to the model and returns the response string.

        Returns:
            EvalResult with per-sample scores and aggregated
            metrics (accuracy and mean per scorer).
        """
        start = time.perf_counter()

        states: list[EvalState] = []
        all_scores: dict[str, list[Score]] = {
            s.name: [] for s in self._scorers
        }

        for sample in self._dataset:
            # 1. Create state from sample
            state = EvalState(sample=sample)

            # 2. Run solver pipeline
            for solver in self._solvers:
                if state.completed:
                    break
                state = solver.solve(state, model_fn)

            states.append(state)

            # 3. Run all scorers on the final state
            for scorer in self._scorers:
                try:
                    score = scorer.score(state)
                except Exception as exc:
                    score = Score(
                        value=0.0,
                        explanation=(
                            f"Scorer error: {exc}"
                        ),
                        metadata={"error": str(exc)},
                    )
                all_scores[scorer.name].append(score)

        # 5. Aggregate metrics
        metrics = self._aggregate_metrics(all_scores)

        elapsed_ms = (time.perf_counter() - start) * 1000

        return EvalResult(
            task_name=self.name,
            samples=states,
            scores=all_scores,
            metrics=metrics,
            duration_ms=round(elapsed_ms, 2),
        )

    def _aggregate_metrics(
        self,
        all_scores: dict[str, list[Score]],
    ) -> dict[str, float]:
        """Compute metrics for each scorer.

        For each scorer, two metrics are computed:

        - **accuracy**: fraction of scores >= 0.5 (binary).
        - **mean**: arithmetic mean of score values.

        Metric keys follow the ``"{ScorerName}/{metric}"``
        convention. An empty score list for a scorer is
        silently skipped (no metrics produced).

        Args:
            all_scores: Mapping of scorer name to list of
                Score objects (one per evaluated sample).

        Returns:
            Flat dict of metric name to float value,
            rounded to 4 decimal places.
        """
        metrics: dict[str, float] = {}

        for scorer_name, scores in all_scores.items():
            if not scores:
                continue

            values = [s.value for s in scores]

            # accuracy: fraction scoring >= 0.5
            passing = sum(
                1 for v in values if v >= 0.5
            )
            accuracy = passing / len(values)
            metrics[f"{scorer_name}/accuracy"] = round(
                accuracy, 4
            )

            # mean score
            mean = sum(values) / len(values)
            metrics[f"{scorer_name}/mean"] = round(
                mean, 4
            )

        return metrics

    def to_test_result(
        self,
        model_fn: Callable[[str], str],
        min_accuracy: float = 0.8,
    ) -> TestResult:
        """Run evaluation and convert to mltk TestResult.

        Bridges the eval pipeline with pytest -- enables
        eval tasks to participate in standard mltk test
        suites and CI pipelines::

            result = task.to_test_result(model_fn)
            assert result.passed

        The assertion name follows the ``eval.task.<name>``
        convention. Severity is always CRITICAL so failures
        raise ``MltkAssertionError`` in pytest.

        Args:
            model_fn: Callable that sends a prompt to the
                model and returns the response string.
            min_accuracy: Minimum accuracy threshold for
                any scorer. Default 0.8 (80%).

        Returns:
            TestResult with pass/fail based on whether all
            accuracy metrics meet the threshold.

        Raises:
            MltkAssertionError: If any scorer accuracy is
                below ``min_accuracy`` (via assert_true).
        """
        from mltk.core.assertion import assert_true
        from mltk.core.result import Severity

        eval_result = self.run(model_fn)

        # Check if all accuracy metrics meet threshold
        accuracies = {
            k: v
            for k, v in eval_result.metrics.items()
            if k.endswith("/accuracy")
        }

        all_pass = all(
            v >= min_accuracy for v in accuracies.values()
        )

        if all_pass:
            msg = (
                f"Eval '{self.name}': all scorers above "
                f"{min_accuracy:.0%} accuracy"
            )
        else:
            failures = [
                f"{k}={v:.4f}"
                for k, v in accuracies.items()
                if v < min_accuracy
            ]
            msg = (
                f"Eval '{self.name}' below threshold: "
                + ", ".join(failures)
            )

        return assert_true(
            all_pass,
            name=f"eval.task.{self.name}",
            message=msg,
            severity=Severity.CRITICAL,
            metrics=eval_result.metrics,
            total_samples=eval_result.total_samples,
            duration_ms=eval_result.duration_ms,
        )

    def __repr__(self) -> str:
        solver_names = ", ".join(
            s.name for s in self._solvers
        )
        scorer_names = ", ".join(
            s.name for s in self._scorers
        )
        return (
            f"EvalTask(name={self.name!r}, "
            f"solvers=[{solver_names}], "
            f"scorers=[{scorer_names}], "
            f"samples={len(self._dataset)})"
        )


# ---------------------------------------------------------------
# Dataset loading utilities
# ---------------------------------------------------------------


def load_dataset(
    path: str | Path,
    input_column: str = "input",
    target_column: str = "target",
) -> list[EvalSample]:
    """Load evaluation dataset from CSV or JSON file.

    Supports two formats:

    - **CSV**: Must have a header row. The ``input_column``
      column is required; ``target_column`` is optional.
      All other columns become ``metadata``.
    - **JSON**: Must be an array of objects. Each object
      must have the ``input_column`` key. Same rules apply
      for target and metadata.

    No pandas dependency -- uses only stdlib ``csv`` and
    ``json`` modules.

    Args:
        path: Path to CSV or JSON file.
        input_column: Column/key name for inputs.
            Defaults to ``"input"``.
        target_column: Column/key name for targets.
            Defaults to ``"target"``. If the column is
            missing from a row, target is set to None.

    Returns:
        List of EvalSample objects, one per row/object.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file extension is not ``.csv``
            or ``.json``, or if the input column is missing.

    Example:
        Load a CSV dataset::

            # data.csv:
            # input,target,category
            # "What is 2+2?","4","math"
            # "Capital of France?","Paris","geography"

            samples = load_dataset("data.csv")
            assert len(samples) == 2
            assert samples[0].input == "What is 2+2?"
    """
    filepath = Path(path)

    if not filepath.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {filepath}"
        )

    suffix = filepath.suffix.lower()

    if suffix == ".csv":
        return _load_csv(
            filepath, input_column, target_column
        )
    elif suffix == ".json":
        return _load_json(
            filepath, input_column, target_column
        )
    else:
        raise ValueError(
            f"Unsupported file format '{suffix}'. "
            f"Use .csv or .json."
        )


def _load_csv(
    filepath: Path,
    input_column: str,
    target_column: str,
) -> list[EvalSample]:
    """Load samples from a CSV file.

    Args:
        filepath: Path to the CSV file.
        input_column: Column name for input text.
        target_column: Column name for target text.

    Returns:
        List of EvalSample objects.

    Raises:
        ValueError: If the input column is not found
            in the CSV header.
    """
    samples: list[EvalSample] = []

    with open(filepath, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        if reader.fieldnames is None:
            return samples

        if input_column not in reader.fieldnames:
            raise ValueError(
                f"CSV missing required column "
                f"'{input_column}'. "
                f"Found: {list(reader.fieldnames)}"
            )

        reserved = {input_column, target_column}

        for row in reader:
            input_val = row.get(input_column, "")
            if not input_val:
                continue

            target_val = row.get(target_column)
            if target_val == "":
                target_val = None

            metadata = {
                k: v
                for k, v in row.items()
                if k not in reserved and v
            }

            samples.append(
                EvalSample(
                    input=input_val,
                    target=target_val,
                    metadata=metadata,
                )
            )

    return samples


def _load_json(
    filepath: Path,
    input_column: str,
    target_column: str,
) -> list[EvalSample]:
    """Load samples from a JSON file.

    Expects a JSON array of objects at the top level.

    Args:
        filepath: Path to the JSON file.
        input_column: Key name for input text.
        target_column: Key name for target text.

    Returns:
        List of EvalSample objects.

    Raises:
        ValueError: If any object is missing the input key,
            or if the top-level JSON is not an array.
    """
    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(
            "JSON dataset must be an array of objects. "
            f"Got {type(data).__name__}."
        )

    samples: list[EvalSample] = []
    reserved = {input_column, target_column}

    for i, obj in enumerate(data):
        if not isinstance(obj, dict):
            raise ValueError(
                f"JSON item at index {i} is not an object. "
                f"Got {type(obj).__name__}."
            )

        if input_column not in obj:
            raise ValueError(
                f"JSON item at index {i} missing required "
                f"key '{input_column}'."
            )

        input_val = str(obj[input_column])
        if not input_val:
            continue

        target_val = obj.get(target_column)
        if target_val is not None:
            target_val = str(target_val)

        metadata = {
            k: v
            for k, v in obj.items()
            if k not in reserved
        }

        samples.append(
            EvalSample(
                input=input_val,
                target=target_val,
                metadata=metadata,
            )
        )

    return samples
