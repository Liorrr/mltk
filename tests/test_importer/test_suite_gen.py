"""Tests for mltk.importer.suite_gen -- in-memory suite generation."""

from __future__ import annotations

import pytest

from mltk.eval._types import EvalSample
from mltk.eval.dataset import EvalDataset
from mltk.importer.classify import TaskType
from mltk.importer.schema import ColumnMapping, ColumnRole, ImportResult
from mltk.importer.suite_gen import build_suite, compute_baseline_thresholds

# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------


def _rows() -> list[dict]:
    return [
        {
            "question": "Question one?",
            "answer": "Answer one",
            "passage": "Context one",
            "category": "alpha",
        },
        {
            "question": "Question two?",
            "answer": "Answer two",
            "passage": "Context two",
            "category": "beta",
        },
    ]


def _dtypes() -> dict[str, str]:
    return {
        "question": "string",
        "answer": "string",
        "passage": "string",
        "category": "string",
    }


def _mapping() -> ColumnMapping:
    return ColumnMapping(
        roles={
            "question": ColumnRole.INPUT,
            "answer": ColumnRole.GOLDEN,
            "passage": ColumnRole.CONTEXT,
            "category": ColumnRole.LABEL,
        },
        samples=_rows()[0],
    )


def _import_result(rows: list[dict] | None = None) -> ImportResult:
    return ImportResult(
        source="inline.csv",
        columns=list(_dtypes().keys()),
        dtypes=_dtypes(),
        rows=rows if rows is not None else _rows(),
        mapping=_mapping(),
    )


def _dataset(name: str = "tiny-rag", rows: list[dict] | None = None) -> EvalDataset:
    return _import_result(rows=rows).to_eval_dataset(name=name)


def _judge(score: float):
    def judge(_left: str, _right: str) -> float:
        return score

    return judge


def _result_signature(result):
    return [
        (
            item.name,
            item.passed,
            item.details.get("method"),
            item.details.get("score"),
        )
        for item in result.results
    ]


# ===============================================================
# Baselines
# ===============================================================


class TestComputeBaselineThresholds:
    """compute_baseline_thresholds() -- readable self-passing gates."""

    def test_counts_coverage_duplicates_and_categories(self):
        # SCENARIO: one missing target, duplicate inputs, two categories
        # WHY: baseline must mirror assert_dataset_quality internals
        # EXPECTED: thresholds bucket down/up without failing current data
        dataset = EvalDataset(
            name="baseline",
            version="0.1.0",
            samples=[
                EvalSample("same", "A", {"category": "alpha"}),
                EvalSample("same", None, {"category": "beta"}),
                EvalSample("unique", "C", {"category": "beta"}),
            ],
        )

        thresholds = compute_baseline_thresholds(dataset)

        assert thresholds == {
            "min_samples": 3,
            "min_target_coverage": 0.65,
            "max_duplicate_rate": 0.35,
            "min_categories": 2,
        }

    def test_zero_duplicate_rate_uses_one_percent_baseline(self):
        # SCENARIO: every input is unique
        # WHY: zero duplicate data gets a small readable tolerance
        # EXPECTED: max_duplicate_rate is exactly 0.01
        dataset = EvalDataset(
            name="unique",
            version="0.1.0",
            samples=[
                EvalSample("a", "A"),
                EvalSample("b", "B"),
            ],
        )

        thresholds = compute_baseline_thresholds(dataset)

        assert thresholds["max_duplicate_rate"] == 0.01

    def test_no_categories_returns_none(self):
        # SCENARIO: no sample has metadata["category"]
        # WHY: min_categories should only be passed when categories exist
        # EXPECTED: min_categories is None
        dataset = EvalDataset(
            name="no-categories",
            version="0.1.0",
            samples=[
                EvalSample("a", "A"),
                EvalSample("b", "B"),
            ],
        )

        thresholds = compute_baseline_thresholds(dataset)

        assert thresholds["min_categories"] is None


# ===============================================================
# Suite generation
# ===============================================================


class TestBuildSuiteTierOne:
    """build_suite() -- always adds dataset quality baseline."""

    def test_tier_one_only_without_judge(self):
        # SCENARIO: QA_RAG dataset without judge_fn
        # WHY: Tier 2 is gated strictly on judge availability
        # EXPECTED: exactly one pending assertion and run passes
        dataset = _dataset()

        suite = build_suite(
            dataset,
            _mapping(),
            TaskType.QA_RAG,
            judge_fn=None,
        )
        assert len(suite) == 1

        result = suite.run()

        assert suite.name == "import:tiny-rag"
        assert result.total == 1
        assert result.passed


class TestBuildSuiteQaRag:
    """build_suite() -- QA_RAG judge-backed dataset assertions."""

    def test_qa_rag_with_passing_judge_adds_all_golden_context_checks(self):
        # SCENARIO: two samples have target and context
        # WHY: QA_RAG Tier 2 adds faithfulness, answer, and context checks
        # EXPECTED: 1 Tier 1 + 2 faithfulness + 2 answer + 2 context
        dataset = _dataset()

        suite = build_suite(
            dataset,
            _mapping(),
            TaskType.QA_RAG,
            judge_fn=_judge(1.0),
        )
        assert len(suite) == 7

        result = suite.run()
        names = [item.name for item in result.results]

        assert result.total == 7
        assert result.passed
        assert names.count("eval.dataset.quality") == 1
        assert names.count("llm.rag.faithfulness") == 2
        assert names.count("llm.rag.answer_relevancy") == 2
        assert names.count("llm.rag.context_relevancy") == 2

    def test_qa_rag_with_failing_judge_captures_failures(self):
        # SCENARIO: judge returns a failing score for every RAG assertion
        # WHY: suite.run() must collect failures rather than raise
        # EXPECTED: Tier 1 passes; six Tier 2 checks fail as results
        dataset = _dataset()

        suite = build_suite(
            dataset,
            _mapping(),
            TaskType.QA_RAG,
            judge_fn=_judge(0.0),
        )
        result = suite.run()

        assert result.total == 7
        assert result.failed_count == 6
        assert result.passed is False

    def test_missing_target_skips_answer_dependent_checks(self):
        # SCENARIO: one sample has context but target is missing
        # WHY: no values are invented for golden/context checks
        # EXPECTED: missing-target sample only gets context relevancy
        rows = _rows()
        rows[1]["answer"] = None
        dataset = _dataset(rows=rows)

        suite = build_suite(
            dataset,
            _mapping(),
            TaskType.QA_RAG,
            judge_fn=_judge(1.0),
        )
        assert len(suite) == 5

        result = suite.run()
        names = [item.name for item in result.results]

        assert result.passed
        assert names.count("llm.rag.faithfulness") == 1
        assert names.count("llm.rag.answer_relevancy") == 1
        assert names.count("llm.rag.context_relevancy") == 2


class TestBuildSuiteRetrieval:
    """build_suite() -- RETRIEVAL judge-backed dataset assertions."""

    def test_retrieval_adds_only_context_relevancy(self):
        # SCENARIO: retrieval dataset with judge_fn
        # WHY: S98 only adds dataset-side context checks for retrieval
        # EXPECTED: 1 Tier 1 + one context check per contextual sample
        dataset = _dataset()

        suite = build_suite(
            dataset,
            _mapping(),
            TaskType.RETRIEVAL,
            judge_fn=_judge(1.0),
        )
        assert len(suite) == 3

        result = suite.run()
        names = [item.name for item in result.results]

        assert result.passed
        assert names.count("llm.rag.context_relevancy") == 2
        assert "llm.rag.faithfulness" not in names
        assert "llm.rag.answer_relevancy" not in names


class TestBuildSuiteOtherTasks:
    """build_suite() -- non-RAG tasks stay Tier 1 in S98."""

    @pytest.mark.parametrize(
        "task_type",
        [
            TaskType.CLASSIFICATION,
            TaskType.GENERATION,
            TaskType.SUMMARIZATION,
        ],
    )
    def test_other_task_types_ignore_judge_for_tier_two(self, task_type):
        # SCENARIO: non-RAG task receives a judge_fn
        # WHY: model-bound assertions are emitted into pytest elsewhere
        # EXPECTED: suite still contains only dataset quality
        dataset = _dataset()

        suite = build_suite(
            dataset,
            _mapping(),
            task_type,
            judge_fn=_judge(1.0),
        )
        assert len(suite) == 1

        result = suite.run()

        assert result.total == 1
        assert result.passed


class TestBuildSuiteDeterminism:
    """build_suite() -- deterministic pending counts and outcomes."""

    def test_same_inputs_have_same_count_and_run_outcome(self):
        # SCENARIO: build the same QA_RAG suite twice
        # WHY: generated suites should be stable for reproducible imports
        # EXPECTED: pending counts and result signatures match
        dataset = _dataset()

        first = build_suite(
            dataset,
            _mapping(),
            TaskType.QA_RAG,
            judge_fn=_judge(1.0),
        )
        second = build_suite(
            dataset,
            _mapping(),
            TaskType.QA_RAG,
            judge_fn=_judge(1.0),
        )

        assert len(first) == len(second) == 7

        first_result = first.run()
        second_result = second.run()

        assert first_result.total == second_result.total
        assert first_result.passed_count == second_result.passed_count
        assert first_result.failed_count == second_result.failed_count
        assert _result_signature(first_result) == _result_signature(second_result)
