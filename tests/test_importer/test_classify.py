"""Tests for mltk.importer.classify -- deterministic task-type classification.

Exercises the role-presence priority table used after column auto-mapping:
CONTEXT+GOLDEN -> QA_RAG; CONTEXT without GOLDEN -> RETRIEVAL; LABEL
without GOLDEN -> CLASSIFICATION; summary-named GOLDEN -> SUMMARIZATION;
other GOLDEN -> GENERATION; otherwise GENERATION.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mltk.importer.classify import TaskType, classify_task
from mltk.importer.loader import DatasetImporter
from mltk.importer.schema import ColumnMapping, ColumnRole

FIXTURES_DIR = Path(__file__).parent / "fixtures"
CSV_PATH = FIXTURES_DIR / "tiny.csv"


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------


def _mapping(roles: dict[str, ColumnRole]) -> ColumnMapping:
    return ColumnMapping(roles=roles)


# ===============================================================
# Contract guards
# ===============================================================


class TestTaskTypeContract:
    """Pinned enum values shared by importer suite/codegen/CLI workers."""

    def test_enum_values_are_pinned(self):
        assert [(task_type.name, task_type.value) for task_type in TaskType] == [
            ("CLASSIFICATION", "classification"),
            ("QA_RAG", "qa_rag"),
            ("SUMMARIZATION", "summarization"),
            ("GENERATION", "generation"),
            ("RETRIEVAL", "retrieval"),
        ]


# ===============================================================
# Decision table
# ===============================================================


class TestClassifyTaskDecisionTable:
    """First matching role-presence rule wins."""

    def test_context_and_golden_is_qa_rag(self):
        mapping = _mapping(
            {
                "question": ColumnRole.INPUT,
                "answer": ColumnRole.GOLDEN,
                "passage": ColumnRole.CONTEXT,
            }
        )
        assert classify_task(mapping) == TaskType.QA_RAG

    def test_context_without_golden_is_retrieval(self):
        mapping = _mapping(
            {
                "query": ColumnRole.INPUT,
                "document": ColumnRole.CONTEXT,
            }
        )
        assert classify_task(mapping) == TaskType.RETRIEVAL

    def test_label_without_golden_is_classification(self):
        mapping = _mapping(
            {
                "text": ColumnRole.INPUT,
                "category": ColumnRole.LABEL,
            }
        )
        assert classify_task(mapping) == TaskType.CLASSIFICATION

    @pytest.mark.parametrize("column", ["summary", "summary_text"])
    def test_summary_named_golden_is_summarization(self, column):
        mapping = _mapping(
            {
                "document": ColumnRole.INPUT,
                column: ColumnRole.GOLDEN,
            }
        )
        assert classify_task(mapping) == TaskType.SUMMARIZATION

    def test_non_summary_golden_is_generation(self):
        mapping = _mapping(
            {
                "prompt": ColumnRole.INPUT,
                "answer": ColumnRole.GOLDEN,
            }
        )
        assert classify_task(mapping) == TaskType.GENERATION

    def test_fallback_without_context_golden_or_label_is_generation(self):
        mapping = _mapping(
            {
                "prompt": ColumnRole.INPUT,
                "id": ColumnRole.METADATA,
            }
        )
        assert classify_task(mapping) == TaskType.GENERATION


# ===============================================================
# Priority and tokenizer behavior
# ===============================================================


class TestClassifyTaskPriority:
    """Earlier rules must beat later ones."""

    def test_context_golden_and_label_is_qa_rag(self):
        mapping = _mapping(
            {
                "question": ColumnRole.INPUT,
                "answer": ColumnRole.GOLDEN,
                "passage": ColumnRole.CONTEXT,
                "category": ColumnRole.LABEL,
            }
        )
        assert classify_task(mapping) == TaskType.QA_RAG

    def test_context_and_label_without_golden_is_retrieval(self):
        mapping = _mapping(
            {
                "query": ColumnRole.INPUT,
                "passage": ColumnRole.CONTEXT,
                "category": ColumnRole.LABEL,
            }
        )
        assert classify_task(mapping) == TaskType.RETRIEVAL

    def test_label_and_non_summary_golden_is_generation(self):
        mapping = _mapping(
            {
                "prompt": ColumnRole.INPUT,
                "answer": ColumnRole.GOLDEN,
                "category": ColumnRole.LABEL,
            }
        )
        assert classify_task(mapping) == TaskType.GENERATION


class TestSummarizationTokenMatching:
    """Summary matching is whole-token, reusing mapping._tokenize semantics."""

    def test_summary_text_matches_whole_summary_token(self):
        mapping = _mapping(
            {
                "article": ColumnRole.INPUT,
                "summary_text": ColumnRole.GOLDEN,
            }
        )
        assert classify_task(mapping) == TaskType.SUMMARIZATION

    def test_summarylike_does_not_match_as_substring(self):
        mapping = _mapping(
            {
                "article": ColumnRole.INPUT,
                "summarylike": ColumnRole.GOLDEN,
            }
        )
        assert classify_task(mapping) == TaskType.GENERATION


# ===============================================================
# End-to-end fixture and purity
# ===============================================================


class TestClassifyTaskEndToEnd:
    """Classifier behavior against real importer output."""

    def test_tiny_csv_auto_mapping_classifies_as_qa_rag(self):
        result = DatasetImporter.load(str(CSV_PATH))
        assert classify_task(result.mapping) == TaskType.QA_RAG

    def test_dataset_argument_is_reserved_and_does_not_change_result(self):
        result = DatasetImporter.load(str(CSV_PATH))
        dataset = result.to_eval_dataset(name="tiny")
        assert classify_task(result.mapping, dataset) == TaskType.QA_RAG


class TestClassifyTaskProperties:
    """Purity and determinism."""

    def test_deterministic_across_calls(self):
        mapping = _mapping(
            {
                "prompt": ColumnRole.INPUT,
                "answer": ColumnRole.GOLDEN,
            }
        )
        first = classify_task(mapping)
        second = classify_task(mapping)
        assert first == second == TaskType.GENERATION
