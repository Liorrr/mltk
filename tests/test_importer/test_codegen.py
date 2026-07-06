"""Tests for mltk.importer.codegen -- committable pytest emission."""

from __future__ import annotations

import ast
import importlib
import re
import sys
import types
from enum import Enum
from pathlib import Path

import pytest

from mltk.importer.codegen import generate_pytest
from mltk.importer.loader import DatasetImporter
from mltk.importer.schema import ColumnMapping, ColumnRole, ImportResult

FIXTURES_DIR = Path(__file__).parent / "fixtures"
CSV_PATH = FIXTURES_DIR / "tiny.csv"


class TaskType(Enum):
    """Local stand-in for the parallel classify.py contract."""

    CLASSIFICATION = "classification"
    QA_RAG = "qa_rag"
    SUMMARIZATION = "summarization"
    GENERATION = "generation"
    RETRIEVAL = "retrieval"


@pytest.fixture(autouse=True)
def suite_gen_stub(monkeypatch):
    """Provide the parallel suite_gen contract for local codegen tests."""

    try:
        importlib.import_module("mltk.importer.suite_gen")
        return
    except ModuleNotFoundError:
        pass

    module = types.ModuleType("mltk.importer.suite_gen")

    def compute_baseline_thresholds(dataset):
        inputs = [sample.input for sample in dataset.samples]
        duplicate_rate = 0.0
        if inputs:
            duplicate_rate = 1.0 - (len(set(inputs)) / len(inputs))

        categories = len(dataset.categories)
        return {
            "min_samples": dataset.sample_count,
            "min_target_coverage": dataset.target_coverage,
            "max_duplicate_rate": duplicate_rate,
            "min_categories": categories or None,
        }

    module.compute_baseline_thresholds = compute_baseline_thresholds
    monkeypatch.setitem(sys.modules, "mltk.importer.suite_gen", module)


def _basic_import_result(
    *,
    source: str = "qa.csv",
    rows: list[dict] | None = None,
) -> ImportResult:
    if rows is None:
        rows = [
            {
                "question": "Q1?",
                "answer": "A1",
                "passage": "Context one.",
                "category": "cat-a",
            },
            {
                "question": "Q2?",
                "answer": "A2",
                "passage": "Context two.",
                "category": "cat-b",
            },
        ]

    return ImportResult(
        source=source,
        columns=["question", "answer", "passage", "category"],
        dtypes={
            "question": "string",
            "answer": "string",
            "passage": "string",
            "category": "string",
        },
        rows=rows,
        mapping=ColumnMapping(
            roles={
                "question": ColumnRole.INPUT,
                "answer": ColumnRole.GOLDEN,
                "passage": ColumnRole.CONTEXT,
                "category": ColumnRole.LABEL,
            },
            samples=rows[0] if rows else {},
        ),
    )


def _dataset_name_from_content(content: str) -> str:
    match = re.search(r"name=(['\"])([^'\"]+)\1", content)
    assert match is not None
    return match.group(2)


class TestGeneratePytestTinyCsv:
    """End-to-end code emission from the tiny CSV fixture."""

    def test_tiny_csv_qa_rag_file_shape(self):
        # SCENARIO: load the real tiny CSV and emit a QA_RAG pytest file
        # WHY: this is the sprint acceptance fixture shape
        # EXPECTED: syntax-valid content with runnable Tier 1 and
        #   skipped Tier 2 scaffolding
        import_result = DatasetImporter.load(str(CSV_PATH))

        content = generate_pytest(import_result, TaskType.QA_RAG)

        ast.parse(content)
        assert "def test_schema" in content
        assert "def test_no_nulls" in content
        assert "def test_dataset_quality" in content
        assert "def predict_fn" in content
        assert "def test_faithfulness" in content
        assert "def test_answer_relevancy" in content
        assert "def test_context_relevancy" in content
        # The emitter renders the load call multi-line; assert the source
        # literal is embedded rather than pinning the call formatting.
        assert "DatasetImporter.load(" in content
        assert f"{str(CSV_PATH)!r}" in content
        assert "timestamp" not in content.lower()
        assert "run_id" not in content.lower()

    def test_deterministic_for_same_input(self):
        import_result = DatasetImporter.load(str(CSV_PATH))

        first = generate_pytest(import_result, TaskType.QA_RAG)
        second = generate_pytest(import_result, TaskType.QA_RAG)

        assert first == second

    def test_output_path_writes_returned_content(self, tmp_path):
        import_result = DatasetImporter.load(str(CSV_PATH))
        output_path = tmp_path / "nested" / "test_imported_tiny.py"

        content = generate_pytest(
            import_result,
            TaskType.QA_RAG,
            output_path=output_path,
        )

        assert output_path.read_text(encoding="utf-8") == content


class TestGeneratePytestTaskShapes:
    """Tier 2 tests vary by task type."""

    def test_classification_has_metric_test(self):
        content = generate_pytest(
            _basic_import_result(), TaskType.CLASSIFICATION
        )

        assert "def test_metric" in content
        assert "assert_metric(" in content

    def test_retrieval_has_only_context_relevancy_tier2(self):
        content = generate_pytest(_basic_import_result(), TaskType.RETRIEVAL)

        assert "def test_context_relevancy" in content
        assert "def test_answer_relevancy" not in content
        assert "def test_faithfulness" not in content

    def test_generation_has_output_format(self):
        content = generate_pytest(_basic_import_result(), TaskType.GENERATION)
        lines = content.splitlines()

        assert "def test_output_format" in content
        assert "assert_output_format(" in content
        assert "assert_json_schema" in content
        assert "    \\" not in lines
        assert not any(line.rstrip().endswith("\\") for line in lines)
        assert (
            "    def test_output_format(self, predict_fn, dataset):"
            in lines
        )
        assert 'pattern=r"\\S",' in content


class TestGeneratePytestBaselineHonesty:
    """Generated Tier 1 tests must not fail on day one."""

    def test_null_golden_omits_no_nulls_test_with_comment(self):
        rows = [
            {
                "question": "Q1?",
                "answer": None,
                "passage": "Context one.",
                "category": "cat-a",
            }
        ]
        import_result = _basic_import_result(rows=rows)

        content = generate_pytest(import_result, TaskType.QA_RAG)

        assert "def test_no_nulls" not in content
        assert "test_no_nulls omitted" in content


class TestGeneratePytestDatasetName:
    """Default dataset names are sanitized from the import source."""

    def test_file_source_stem_becomes_identifier(self):
        import_result = _basic_import_result(source="some/dir/my-data.csv")

        content = generate_pytest(import_result, TaskType.QA_RAG)

        assert _dataset_name_from_content(content) == "my_data"

    def test_hf_id_name_becomes_identifier(self):
        import_result = _basic_import_result(source="org/name")

        content = generate_pytest(import_result, TaskType.QA_RAG)
        dataset_name = _dataset_name_from_content(content)

        assert dataset_name.isidentifier()
        assert "/" not in dataset_name
