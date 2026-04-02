"""Tests for mltk.eval.task — EvalTask runner + load_dataset."""

from __future__ import annotations

import csv
import json

import pytest

from mltk.eval._types import (
    EvalResult,
    EvalSample,
    EvalState,
)
from mltk.eval.scorers import (
    ExactMatchScorer,
    IncludesScorer,
    PatternScorer,
)
from mltk.eval.solvers import (
    FewShotSolver,
    GenerateSolver,
    chain,
)
from mltk.eval.task import EvalTask, load_dataset

# ---------------------------------------------------------------
# Mock model functions
# ---------------------------------------------------------------


def fixed_model(prompt: str) -> str:
    """Always returns '4'."""
    return "4"


def echo_model(prompt: str) -> str:
    """Returns the prompt back."""
    return prompt


def answer_model(prompt: str) -> str:
    """Returns 'Answer: 4'."""
    return "Answer: 4"


def _simple_dataset(n: int = 5) -> list[EvalSample]:
    """Build a small dataset of math samples."""
    return [
        EvalSample(input=f"{i}+{i}?", target=str(i * 2))
        for i in range(1, n + 1)
    ]


def _matching_dataset() -> list[EvalSample]:
    """Dataset where fixed_model ('4') matches target."""
    return [
        EvalSample(input="2+2?", target="4"),
        EvalSample(input="1+3?", target="4"),
        EvalSample(input="0+4?", target="4"),
    ]


def _mixed_dataset() -> list[EvalSample]:
    """Dataset with some matches and some mismatches."""
    return [
        EvalSample(input="2+2?", target="4"),
        EvalSample(input="3+3?", target="6"),
        EvalSample(input="1+3?", target="4"),
        EvalSample(input="5+5?", target="10"),
    ]


# ===============================================================
# EvalTask.run()
# ===============================================================


class TestEvalTaskRun:
    """EvalTask: running solver + scorer on datasets."""

    def test_basic_run(self):
        # SCENARIO: run with single solver + scorer
        # WHY: core contract
        # EXPECTED: EvalResult returned with scores
        task = EvalTask(
            name="math",
            solver=GenerateSolver(),
            scorers=[ExactMatchScorer()],
            dataset=_matching_dataset(),
        )
        result = task.run(fixed_model)
        assert isinstance(result, EvalResult)
        assert result.task_name == "math"

    def test_single_solver_single_scorer(self):
        # SCENARIO: minimal config
        # WHY: simplest valid configuration
        # EXPECTED: scores for all samples
        dataset = _matching_dataset()
        task = EvalTask(
            name="basic",
            solver=GenerateSolver(),
            scorers=[ExactMatchScorer()],
            dataset=dataset,
        )
        result = task.run(fixed_model)
        assert result.total_samples == len(dataset)
        assert len(result.scores) >= 1

    def test_solver_pipeline(self):
        # SCENARIO: chain of solvers as solver arg
        # WHY: pipelines are a core use case
        # EXPECTED: pipeline runs successfully
        pipeline = chain(
            FewShotSolver(examples=[("1+1", "2")]),
            GenerateSolver(),
        )
        task = EvalTask(
            name="pipeline",
            solver=pipeline,
            scorers=[ExactMatchScorer()],
            dataset=_matching_dataset(),
        )
        result = task.run(fixed_model)
        assert result.total_samples == 3

    def test_multi_scorer(self):
        # SCENARIO: two scorers on same dataset
        # WHY: multi-dimensional evaluation
        # EXPECTED: separate score lists per scorer
        task = EvalTask(
            name="multi",
            solver=GenerateSolver(),
            scorers=[
                ExactMatchScorer(),
                IncludesScorer(),
            ],
            dataset=_matching_dataset(),
        )
        result = task.run(fixed_model)
        assert len(result.scores) >= 2

    def test_empty_dataset_raises(self):
        # SCENARIO: empty dataset
        # WHY: must validate input
        # EXPECTED: ValueError
        with pytest.raises(ValueError, match="dataset"):
            EvalTask(
                name="empty",
                solver=GenerateSolver(),
                scorers=[ExactMatchScorer()],
                dataset=[],
            )

    def test_no_solver_raises(self):
        # SCENARIO: empty solver list
        # WHY: must validate input
        # EXPECTED: ValueError
        with pytest.raises(ValueError, match="solver"):
            EvalTask(
                name="nosolver",
                solver=[],
                scorers=[ExactMatchScorer()],
                dataset=_simple_dataset(),
            )

    def test_no_scorer_raises(self):
        # SCENARIO: empty scorers list
        # WHY: must validate input
        # EXPECTED: ValueError
        with pytest.raises(ValueError, match="scorer"):
            EvalTask(
                name="noscorer",
                solver=GenerateSolver(),
                scorers=[],
                dataset=_simple_dataset(),
            )

    def test_metrics_have_accuracy(self):
        # SCENARIO: check metrics dict has accuracy
        # WHY: accuracy is a standard metric
        # EXPECTED: at least one key with "accuracy"
        task = EvalTask(
            name="metrics",
            solver=GenerateSolver(),
            scorers=[ExactMatchScorer()],
            dataset=_matching_dataset(),
        )
        result = task.run(fixed_model)
        accuracy_keys = [
            k for k in result.metrics
            if "accuracy" in k.lower()
        ]
        assert len(accuracy_keys) >= 1

    def test_metrics_have_mean(self):
        # SCENARIO: check metrics dict has mean
        # WHY: mean score is a standard metric
        # EXPECTED: at least one key with "mean"
        task = EvalTask(
            name="metrics",
            solver=GenerateSolver(),
            scorers=[ExactMatchScorer()],
            dataset=_matching_dataset(),
        )
        result = task.run(fixed_model)
        mean_keys = [
            k for k in result.metrics
            if "mean" in k.lower()
        ]
        assert len(mean_keys) >= 1

    def test_accuracy_calculation(self):
        # SCENARIO: 3/3 matches = 100% accuracy
        # WHY: accuracy math must be correct
        # EXPECTED: accuracy == 1.0
        task = EvalTask(
            name="perfect",
            solver=GenerateSolver(),
            scorers=[ExactMatchScorer()],
            dataset=_matching_dataset(),
        )
        result = task.run(fixed_model)
        accuracy_keys = [
            k for k in result.metrics
            if "accuracy" in k.lower()
        ]
        for k in accuracy_keys:
            assert result.metrics[k] == pytest.approx(1.0)

    def test_mean_calculation(self):
        # SCENARIO: all match = mean 1.0
        # WHY: mean math must be correct
        # EXPECTED: mean == 1.0
        task = EvalTask(
            name="perfect",
            solver=GenerateSolver(),
            scorers=[ExactMatchScorer()],
            dataset=_matching_dataset(),
        )
        result = task.run(fixed_model)
        mean_keys = [
            k for k in result.metrics
            if "mean" in k.lower()
        ]
        for k in mean_keys:
            assert result.metrics[k] == pytest.approx(1.0)

    def test_duration_ms_populated(self):
        # SCENARIO: duration is measured
        # WHY: timing is part of result contract
        # EXPECTED: duration_ms >= 0
        task = EvalTask(
            name="timed",
            solver=GenerateSolver(),
            scorers=[ExactMatchScorer()],
            dataset=_matching_dataset(),
        )
        result = task.run(fixed_model)
        assert result.duration_ms >= 0.0

    def test_total_samples_correct(self):
        # SCENARIO: 5 samples in dataset
        # WHY: total_samples must match dataset size
        # EXPECTED: total_samples == 5
        dataset = _simple_dataset(5)
        task = EvalTask(
            name="count",
            solver=GenerateSolver(),
            scorers=[ExactMatchScorer()],
            dataset=dataset,
        )
        result = task.run(fixed_model)
        assert result.total_samples == 5

    def test_scorer_exception_caught(self):
        # SCENARIO: scorer raises during scoring
        # WHY: one bad scorer must not crash the run
        # EXPECTED: run completes, error handled

        class BadScorer:
            """Scorer that raises on every call."""

            name = "BadScorer"

            def score(self, state):
                raise RuntimeError("scorer broke")

        task = EvalTask(
            name="robust",
            solver=GenerateSolver(),
            scorers=[BadScorer()],
            dataset=_matching_dataset(),
        )
        # Should not raise — error caught internally
        result = task.run(fixed_model)
        assert isinstance(result, EvalResult)


# ===============================================================
# EvalResult properties
# ===============================================================


class TestEvalResult:
    """EvalResult: aggregated result properties."""

    def test_passed_with_high_metrics(self):
        # SCENARIO: all metrics >= 0.5
        # WHY: passed convention is >= 0.5
        # EXPECTED: passed == True
        result = EvalResult(
            task_name="test",
            metrics={
                "ExactMatch/accuracy": 0.9,
                "ExactMatch/mean": 0.85,
            },
        )
        assert result.passed is True

    def test_passed_with_low_metrics(self):
        # SCENARIO: a metric < 0.5
        # WHY: any metric below 0.5 = failed
        # EXPECTED: passed == False
        result = EvalResult(
            task_name="test",
            metrics={
                "ExactMatch/accuracy": 0.3,
                "ExactMatch/mean": 0.2,
            },
        )
        assert result.passed is False

    def test_passed_empty_metrics(self):
        # SCENARIO: no metrics
        # WHY: empty = nothing to fail
        # EXPECTED: passed == True
        result = EvalResult(task_name="test")
        assert result.passed is True

    def test_passed_boundary_value(self):
        # SCENARIO: metric exactly 0.5
        # WHY: boundary at >= 0.5
        # EXPECTED: passed == True
        result = EvalResult(
            task_name="test",
            metrics={"metric": 0.5},
        )
        assert result.passed is True

    def test_total_samples_empty(self):
        # SCENARIO: no samples
        # WHY: empty result
        # EXPECTED: total_samples == 0
        result = EvalResult(task_name="test")
        assert result.total_samples == 0

    def test_total_samples_count(self):
        # SCENARIO: 3 samples
        # WHY: count must match
        # EXPECTED: total_samples == 3
        samples = [
            EvalState(sample=EvalSample(input=f"q{i}"))
            for i in range(3)
        ]
        result = EvalResult(
            task_name="test", samples=samples
        )
        assert result.total_samples == 3


# ===============================================================
# to_test_result
# ===============================================================


class TestToTestResult:
    """EvalTask.to_test_result: mltk TestResult bridge."""

    def test_returns_test_result(self):
        # SCENARIO: basic to_test_result call
        # WHY: must return mltk TestResult
        # EXPECTED: has passed, name, message attrs
        task = EvalTask(
            name="bridge",
            solver=GenerateSolver(),
            scorers=[ExactMatchScorer()],
            dataset=_matching_dataset(),
        )
        tr = task.to_test_result(fixed_model)
        assert hasattr(tr, "passed")
        assert hasattr(tr, "name")
        assert hasattr(tr, "message")

    def test_passes_when_above_threshold(self):
        # SCENARIO: 100% accuracy, threshold 0.8
        # WHY: above threshold = passed
        # EXPECTED: passed == True
        task = EvalTask(
            name="above",
            solver=GenerateSolver(),
            scorers=[ExactMatchScorer()],
            dataset=_matching_dataset(),
        )
        tr = task.to_test_result(
            fixed_model, min_accuracy=0.8
        )
        assert tr.passed is True

    def test_fails_when_below_threshold(self):
        # SCENARIO: mixed dataset, threshold 0.9
        # WHY: below threshold = raises MltkAssertionError
        # EXPECTED: exception raised, result.passed == False
        from mltk.core.assertion import MltkAssertionError

        task = EvalTask(
            name="below",
            solver=GenerateSolver(),
            scorers=[ExactMatchScorer()],
            dataset=_mixed_dataset(),
        )
        with pytest.raises(MltkAssertionError) as exc:
            task.to_test_result(
                fixed_model, min_accuracy=0.9
            )
        assert exc.value.result.passed is False

    def test_custom_min_accuracy(self):
        # SCENARIO: threshold 0.0 — always passes
        # WHY: custom threshold must be respected
        # EXPECTED: passed == True
        task = EvalTask(
            name="custom",
            solver=GenerateSolver(),
            scorers=[ExactMatchScorer()],
            dataset=_mixed_dataset(),
        )
        tr = task.to_test_result(
            fixed_model, min_accuracy=0.0
        )
        assert tr.passed is True


# ===============================================================
# load_dataset
# ===============================================================


class TestLoadDataset:
    """load_dataset: CSV and JSON file loading."""

    def test_load_csv(self, tmp_path):
        # SCENARIO: load from CSV file
        # WHY: CSV is a primary dataset format
        # EXPECTED: list of EvalSample
        path = tmp_path / "data.csv"
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["input", "target"])
            writer.writerow(["2+2?", "4"])
            writer.writerow(["3+3?", "6"])
        samples = load_dataset(str(path))
        assert len(samples) == 2
        assert isinstance(samples[0], EvalSample)
        assert samples[0].input == "2+2?"
        assert samples[0].target == "4"

    def test_load_json(self, tmp_path):
        # SCENARIO: load from JSON file
        # WHY: JSON is a primary dataset format
        # EXPECTED: list of EvalSample
        path = tmp_path / "data.json"
        data = [
            {"input": "2+2?", "target": "4"},
            {"input": "3+3?", "target": "6"},
        ]
        with open(path, "w") as f:
            json.dump(data, f)
        samples = load_dataset(str(path))
        assert len(samples) == 2
        assert samples[1].input == "3+3?"
        assert samples[1].target == "6"

    def test_custom_columns(self, tmp_path):
        # SCENARIO: non-default column names
        # WHY: column mapping is configurable
        # EXPECTED: correct mapping to EvalSample
        path = tmp_path / "custom.csv"
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["question", "answer"])
            writer.writerow(["What is 1+1?", "2"])
        samples = load_dataset(
            str(path),
            input_column="question",
            target_column="answer",
        )
        assert len(samples) == 1
        assert samples[0].input == "What is 1+1?"
        assert samples[0].target == "2"

    def test_missing_file_raises(self, tmp_path):
        # SCENARIO: file does not exist
        # WHY: must raise on missing file
        # EXPECTED: FileNotFoundError or similar
        bad_path = str(tmp_path / "nonexistent.csv")
        with pytest.raises(
            (FileNotFoundError, OSError)
        ):
            load_dataset(bad_path)

    def test_json_custom_columns(self, tmp_path):
        # SCENARIO: JSON with custom column names
        # WHY: custom columns work for JSON too
        # EXPECTED: correct mapping
        path = tmp_path / "custom.json"
        data = [
            {"q": "capital of FR?", "a": "Paris"},
        ]
        with open(path, "w") as f:
            json.dump(data, f)
        samples = load_dataset(
            str(path),
            input_column="q",
            target_column="a",
        )
        assert samples[0].input == "capital of FR?"
        assert samples[0].target == "Paris"


# ===============================================================
# Multi-scorer metrics
# ===============================================================


class TestMultiScorerMetrics:
    """Multi-scorer evaluation produces separate metrics."""

    def test_separate_metrics_per_scorer(self):
        # SCENARIO: two scorers produce metrics
        # WHY: each scorer must have own metrics
        # EXPECTED: metrics keys for both scorers
        task = EvalTask(
            name="multi",
            solver=GenerateSolver(),
            scorers=[
                ExactMatchScorer(),
                IncludesScorer(),
            ],
            dataset=_matching_dataset(),
        )
        result = task.run(fixed_model)
        keys = list(result.metrics.keys())
        # Should have metrics for both scorers
        assert len(keys) >= 2


# ===============================================================
# Full pipeline integration
# ===============================================================


class TestFullPipeline:
    """End-to-end: FewShot + Generate + multi-scorer."""

    def test_fewshot_generate_exact_pattern(self):
        # SCENARIO: full pipeline with multiple scorers
        # WHY: realistic evaluation setup
        # EXPECTED: EvalResult with all metrics

        def model_fn(prompt: str) -> str:
            return "Answer: 4"

        pipeline = chain(
            FewShotSolver(
                examples=[("1+1", "2"), ("2+1", "3")]
            ),
            GenerateSolver(),
        )
        task = EvalTask(
            name="full",
            solver=pipeline,
            scorers=[
                ExactMatchScorer(),
                PatternScorer(),
            ],
            dataset=[
                EvalSample(input="2+2?", target="4"),
                EvalSample(input="3+1?", target="4"),
            ],
        )
        result = task.run(model_fn)

        assert isinstance(result, EvalResult)
        assert result.total_samples == 2
        assert result.duration_ms >= 0.0
        assert len(result.metrics) >= 2


# ===============================================================
# Edge-case / hardening tests (appended)
# ===============================================================


class TestEvalTaskSingleSample:
    """EvalTask with single-sample dataset."""

    def test_single_sample_dataset(self):
        # SCENARIO: Dataset with exactly 1 sample.
        # WHY: Boundary — minimal valid dataset.
        # EXPECTED: result has 1 sample, metrics exist.
        task = EvalTask(
            name="single",
            solver=GenerateSolver(),
            scorers=[ExactMatchScorer()],
            dataset=[
                EvalSample(input="2+2?", target="4"),
            ],
        )
        result = task.run(fixed_model)
        assert result.total_samples == 1
        assert len(result.metrics) >= 1
        accuracy_keys = [
            k for k in result.metrics
            if "accuracy" in k.lower()
        ]
        assert len(accuracy_keys) >= 1


class TestSolverSetsCompleted:
    """EvalTask where solver sets completed=True."""

    def test_completed_flag_skips_later_solvers(
        self,
    ):
        # SCENARIO: First solver marks state completed.
        # WHY: Pipeline must stop early on completed.
        # EXPECTED: Output set by first solver only.
        from mltk.eval.solvers import Solver

        class EarlySolver(Solver):
            """Solver that completes immediately."""

            name = "EarlySolver"

            def solve(self, state, generate):
                state.output = "early"
                state.completed = True
                return state

        class LateSolver(Solver):
            """Solver that would overwrite output."""

            name = "LateSolver"

            def solve(self, state, generate):
                state.output = "late"
                return state

        task = EvalTask(
            name="completed",
            solver=[EarlySolver(), LateSolver()],
            scorers=[ExactMatchScorer()],
            dataset=[
                EvalSample(
                    input="x", target="early",
                ),
            ],
        )
        result = task.run(fixed_model)
        assert result.total_samples == 1
        # Verify the early solver's output stuck
        state = result.samples[0]
        assert state.output == "early"


class TestLoadDatasetUnicode:
    """load_dataset with unicode content in CSV."""

    def test_unicode_csv(self, tmp_path):
        # SCENARIO: CSV with CJK + emoji + accents.
        # WHY: Real datasets contain unicode.
        # EXPECTED: Round-trip preserves all chars.
        path = tmp_path / "unicode.csv"
        with open(
            path, "w", newline="", encoding="utf-8",
        ) as f:
            writer = csv.writer(f)
            writer.writerow(["input", "target"])
            writer.writerow(
                ["\u4f60\u597d\u4e16\u754c", "\u5730\u7403"]
            )
            writer.writerow(
                [
                    "caf\u00e9 cr\u00e8me",
                    "\u00e9clair",
                ]
            )
        samples = load_dataset(str(path))
        assert len(samples) == 2
        assert samples[0].input == "\u4f60\u597d\u4e16\u754c"
        assert samples[0].target == "\u5730\u7403"
        assert samples[1].input == "caf\u00e9 cr\u00e8me"
        assert samples[1].target == "\u00e9clair"


class TestEvalResultMetricsKeysFormat:
    """EvalResult metrics keys follow ScorerName/metric."""

    def test_metrics_keys_format(self):
        # SCENARIO: Run task, inspect metric key format.
        # WHY: Key convention must be consistent.
        # EXPECTED: Every key matches "Name/metric".
        task = EvalTask(
            name="fmt",
            solver=GenerateSolver(),
            scorers=[
                ExactMatchScorer(),
                IncludesScorer(),
            ],
            dataset=_matching_dataset(),
        )
        result = task.run(fixed_model)
        for key in result.metrics:
            parts = key.split("/")
            assert len(parts) == 2, (
                f"Key {key!r} not in Name/metric"
            )
            assert parts[0] != ""
            assert parts[1] in ("accuracy", "mean")


class TestEvalTaskChainSolver:
    """EvalTask with chain() solver pipeline."""

    def test_chain_pipeline_runs(self):
        # SCENARIO: chain() with 3 solvers.
        # WHY: Realistic multi-stage pipeline.
        # EXPECTED: All solvers execute; result valid.
        pipeline = chain(
            FewShotSolver(
                examples=[("1+1", "2")],
            ),
            FewShotSolver(
                examples=[("2+2", "4")],
            ),
            GenerateSolver(),
        )
        task = EvalTask(
            name="multi_chain",
            solver=pipeline,
            scorers=[ExactMatchScorer()],
            dataset=_matching_dataset(),
        )
        result = task.run(fixed_model)
        assert result.total_samples == 3
        assert result.duration_ms >= 0.0
        assert len(result.metrics) >= 1
