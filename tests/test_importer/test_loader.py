"""Tests for mltk.importer.loader -- multi-source dataset loading.

Covers local CSV/JSON/Parquet loading, cross-format equivalence, error
paths for bad/missing local files, and a fully-mocked HuggingFace Hub
path (no real network calls are made anywhere in this file).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mltk.importer.loader import DatasetImporter
from mltk.importer.schema import ColumnRole, ImportResult

try:
    import pyarrow  # noqa: F401

    _HAS_PYARROW = True
except ImportError:
    _HAS_PYARROW = False

requires_pyarrow = pytest.mark.skipif(
    not _HAS_PYARROW,
    reason="pyarrow not installed (required for the .parquet fixture; "
    "CI's base+dev+scipy install does not include it)",
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"
CSV_PATH = FIXTURES_DIR / "tiny.csv"
JSON_PATH = FIXTURES_DIR / "tiny.json"
PARQUET_PATH = FIXTURES_DIR / "tiny.parquet"

EXPECTED_QUESTIONS = [
    "What is the capital of France?",
    "What is 7 plus 5?",
    "What is the capital of Japan?",
    "What is 9 times 6?",
    "What is the capital of Australia?",
]
EXPECTED_ANSWERS = ["Paris", "12", "Tokyo", "54", "Canberra"]


# ===============================================================
# Local file loading: CSV / JSON / Parquet
# ===============================================================


class TestLoadLocalCsv:
    """DatasetImporter.load() against the tiny.csv fixture."""

    def test_returns_import_result(self):
        result = DatasetImporter.load(str(CSV_PATH))
        assert isinstance(result, ImportResult)

    def test_row_count(self):
        result = DatasetImporter.load(str(CSV_PATH))
        assert len(result.rows) == 5

    def test_columns(self):
        result = DatasetImporter.load(str(CSV_PATH))
        assert set(result.columns) == {
            "question", "answer", "passage", "category", "id",
        }

    def test_row_content(self):
        result = DatasetImporter.load(str(CSV_PATH))
        questions = [row["question"] for row in result.rows]
        answers = [row["answer"] for row in result.rows]
        assert questions == EXPECTED_QUESTIONS
        assert answers == EXPECTED_ANSWERS

    def test_auto_mapping_question_is_input(self):
        result = DatasetImporter.load(str(CSV_PATH))
        assert result.mapping.roles["question"] == ColumnRole.INPUT

    def test_auto_mapping_answer_is_golden(self):
        result = DatasetImporter.load(str(CSV_PATH))
        assert result.dtypes["answer"] == "string"
        assert result.mapping.roles["answer"] == ColumnRole.GOLDEN

    def test_auto_mapping_passage_is_context(self):
        result = DatasetImporter.load(str(CSV_PATH))
        assert result.mapping.roles["passage"] == ColumnRole.CONTEXT

    def test_auto_mapping_category_is_label(self):
        result = DatasetImporter.load(str(CSV_PATH))
        assert result.mapping.roles["category"] == ColumnRole.LABEL

    def test_auto_mapping_id_is_metadata(self):
        result = DatasetImporter.load(str(CSV_PATH))
        assert result.mapping.roles["id"] == ColumnRole.METADATA

    def test_source_recorded(self):
        result = DatasetImporter.load(str(CSV_PATH))
        assert str(CSV_PATH) in result.source or result.source == str(
            CSV_PATH
        )


class TestLoadLocalJson:
    """DatasetImporter.load() against the tiny.json fixture."""

    def test_returns_import_result(self):
        result = DatasetImporter.load(str(JSON_PATH))
        assert isinstance(result, ImportResult)

    def test_row_count(self):
        result = DatasetImporter.load(str(JSON_PATH))
        assert len(result.rows) == 5

    def test_columns(self):
        result = DatasetImporter.load(str(JSON_PATH))
        assert set(result.columns) == {
            "question", "answer", "passage", "category", "id",
        }

    def test_row_content(self):
        result = DatasetImporter.load(str(JSON_PATH))
        questions = [row["question"] for row in result.rows]
        answers = [row["answer"] for row in result.rows]
        assert questions == EXPECTED_QUESTIONS
        assert answers == EXPECTED_ANSWERS

    def test_auto_mapping_matches_csv(self):
        result = DatasetImporter.load(str(JSON_PATH))
        assert result.mapping.roles["question"] == ColumnRole.INPUT
        assert result.mapping.roles["passage"] == ColumnRole.CONTEXT
        assert result.mapping.roles["category"] == ColumnRole.LABEL
        assert result.mapping.roles["id"] == ColumnRole.METADATA


@requires_pyarrow
class TestLoadLocalParquet:
    """DatasetImporter.load() against the tiny.parquet fixture."""

    def test_returns_import_result(self):
        result = DatasetImporter.load(str(PARQUET_PATH))
        assert isinstance(result, ImportResult)

    def test_row_count(self):
        result = DatasetImporter.load(str(PARQUET_PATH))
        assert len(result.rows) == 5

    def test_columns(self):
        result = DatasetImporter.load(str(PARQUET_PATH))
        assert set(result.columns) == {
            "question", "answer", "passage", "category", "id",
        }

    def test_row_content(self):
        result = DatasetImporter.load(str(PARQUET_PATH))
        questions = [row["question"] for row in result.rows]
        answers = [row["answer"] for row in result.rows]
        assert questions == EXPECTED_QUESTIONS
        assert answers == EXPECTED_ANSWERS

    def test_auto_mapping_matches_csv(self):
        result = DatasetImporter.load(str(PARQUET_PATH))
        assert result.mapping.roles["question"] == ColumnRole.INPUT
        assert result.mapping.roles["passage"] == ColumnRole.CONTEXT
        assert result.mapping.roles["category"] == ColumnRole.LABEL
        assert result.mapping.roles["id"] == ColumnRole.METADATA


class TestCrossFormatEquivalenceCsvJson:
    """CSV and JSON fixtures encode the same 5 rows (no pyarrow needed)."""

    def test_same_row_count(self):
        csv_result = DatasetImporter.load(str(CSV_PATH))
        json_result = DatasetImporter.load(str(JSON_PATH))
        assert len(csv_result.rows) == len(json_result.rows) == 5

    def test_same_columns(self):
        csv_result = DatasetImporter.load(str(CSV_PATH))
        json_result = DatasetImporter.load(str(JSON_PATH))
        assert set(csv_result.columns) == set(json_result.columns)

    def test_same_question_answer_content(self):
        csv_result = DatasetImporter.load(str(CSV_PATH))
        json_result = DatasetImporter.load(str(JSON_PATH))
        for i in range(5):
            assert (
                csv_result.rows[i]["question"] == json_result.rows[i]["question"]
            )
            assert csv_result.rows[i]["answer"] == json_result.rows[i]["answer"]

    def test_same_mapping(self):
        csv_result = DatasetImporter.load(str(CSV_PATH))
        json_result = DatasetImporter.load(str(JSON_PATH))
        assert (
            csv_result.mapping.roles["question"]
            == json_result.mapping.roles["question"]
            == ColumnRole.INPUT
        )


@requires_pyarrow
class TestCrossFormatEquivalenceWithParquet:
    """CSV, JSON, and Parquet fixtures encode the same 5 rows."""

    def test_same_row_count_across_formats(self):
        csv_result = DatasetImporter.load(str(CSV_PATH))
        json_result = DatasetImporter.load(str(JSON_PATH))
        parquet_result = DatasetImporter.load(str(PARQUET_PATH))
        assert (
            len(csv_result.rows)
            == len(json_result.rows)
            == len(parquet_result.rows)
            == 5
        )

    def test_same_columns_across_formats(self):
        csv_result = DatasetImporter.load(str(CSV_PATH))
        json_result = DatasetImporter.load(str(JSON_PATH))
        parquet_result = DatasetImporter.load(str(PARQUET_PATH))
        assert (
            set(csv_result.columns)
            == set(json_result.columns)
            == set(parquet_result.columns)
        )

    def test_same_question_answer_content_across_formats(self):
        csv_result = DatasetImporter.load(str(CSV_PATH))
        json_result = DatasetImporter.load(str(JSON_PATH))
        parquet_result = DatasetImporter.load(str(PARQUET_PATH))
        for i in range(5):
            csv_q = csv_result.rows[i]["question"]
            json_q = json_result.rows[i]["question"]
            parquet_q = parquet_result.rows[i]["question"]
            assert csv_q == json_q == parquet_q

            csv_a = csv_result.rows[i]["answer"]
            json_a = json_result.rows[i]["answer"]
            parquet_a = parquet_result.rows[i]["answer"]
            assert csv_a == json_a == parquet_a

    def test_same_mapping_across_formats(self):
        csv_result = DatasetImporter.load(str(CSV_PATH))
        json_result = DatasetImporter.load(str(JSON_PATH))
        parquet_result = DatasetImporter.load(str(PARQUET_PATH))
        assert (
            csv_result.mapping.roles["question"]
            == json_result.mapping.roles["question"]
            == parquet_result.mapping.roles["question"]
            == ColumnRole.INPUT
        )


# ===============================================================
# Error paths: missing / unrecognized local files
# ===============================================================


class TestLoadLocalFileErrors:
    """Missing files and unsupported extensions."""

    def test_nonexistent_csv_raises_file_not_found(self):
        # SCENARIO: path has a recognized extension but doesn't exist
        # WHY: must fail fast and specifically, not silently no-op
        # EXPECTED: FileNotFoundError
        missing = FIXTURES_DIR / "does_not_exist.csv"
        assert not missing.exists()
        with pytest.raises(FileNotFoundError):
            DatasetImporter.load(str(missing))

    def test_nonexistent_parquet_raises_file_not_found(self):
        missing = FIXTURES_DIR / "does_not_exist.parquet"
        assert not missing.exists()
        with pytest.raises(FileNotFoundError):
            DatasetImporter.load(str(missing))

    def test_unrecognized_extension_local_file_raises_value_error(
        self, tmp_path
    ):
        # SCENARIO: local file exists but has an unsupported extension
        # WHY: must reject unknown local formats explicitly rather than
        #   silently misrouting to the HuggingFace Hub branch
        # EXPECTED: ValueError
        bad_file = tmp_path / "tiny.txt"
        bad_file.write_text("not a supported format", encoding="utf-8")
        with pytest.raises(ValueError):  # noqa: PT011 -- message text not part of contract
            DatasetImporter.load(str(bad_file))


# ===============================================================
# HuggingFace Hub path (fully mocked -- no network calls)
# ===============================================================


class TestLoadHuggingFaceHub:
    """DatasetImporter.load() for a bare dataset id -> datasets.load_dataset.

    The loader imports HF's ``load_dataset`` under a guarded try/except as
    ``mltk.importer.loader._hf_load_dataset``, gated by a module-level
    ``_DATASETS_AVAILABLE`` flag computed once at import time. Since the
    optional ``datasets`` package is not installed in this environment,
    both the availability flag and the loader function are patched so the
    HF branch can be exercised without a real install or any network call.
    """

    def _make_mock_dataset(self):
        mock_ds = MagicMock()
        mock_ds.column_names = ["question", "answer"]
        mock_ds.to_list.return_value = [
            {"question": "Q1?", "answer": "A1"},
            {"question": "Q2?", "answer": "A2"},
        ]
        return mock_ds

    def test_hf_load_uses_mocked_load_dataset(self):
        mock_ds = self._make_mock_dataset()
        with (
            patch("mltk.importer.loader._DATASETS_AVAILABLE", True),
            patch("mltk.importer.loader._hf_load_dataset", create=True) as mock_load,
        ):
            mock_load.return_value = mock_ds
            result = DatasetImporter.load("some/hf-dataset-id")
        mock_load.assert_called_once()
        assert isinstance(result, ImportResult)

    def test_hf_load_returns_correct_rows(self):
        mock_ds = self._make_mock_dataset()
        with (
            patch("mltk.importer.loader._DATASETS_AVAILABLE", True),
            patch("mltk.importer.loader._hf_load_dataset", create=True) as mock_load,
        ):
            mock_load.return_value = mock_ds
            result = DatasetImporter.load("some/hf-dataset-id")
        assert len(result.rows) == 2
        assert result.rows[0]["question"] == "Q1?"
        assert result.rows[1]["answer"] == "A2"

    def test_hf_load_default_split_is_train(self):
        mock_ds = self._make_mock_dataset()
        with (
            patch("mltk.importer.loader._DATASETS_AVAILABLE", True),
            patch("mltk.importer.loader._hf_load_dataset", create=True) as mock_load,
        ):
            mock_load.return_value = mock_ds
            DatasetImporter.load("some/hf-dataset-id")
        _, kwargs = mock_load.call_args
        assert kwargs.get("split") == "train"

    def test_hf_load_explicit_split_forwarded(self):
        mock_ds = self._make_mock_dataset()
        with (
            patch("mltk.importer.loader._DATASETS_AVAILABLE", True),
            patch("mltk.importer.loader._hf_load_dataset", create=True) as mock_load,
        ):
            mock_load.return_value = mock_ds
            DatasetImporter.load("some/hf-dataset-id", split="validation")
        _, kwargs = mock_load.call_args
        assert kwargs.get("split") == "validation"

    def test_hf_load_no_real_network_call(self):
        # SCENARIO: mocked load_dataset never touches the network
        # WHY: hard repo rule -- no live HF calls in CI
        # EXPECTED: mock called exactly once, no exception raised
        mock_ds = self._make_mock_dataset()
        with (
            patch("mltk.importer.loader._DATASETS_AVAILABLE", True),
            patch("mltk.importer.loader._hf_load_dataset", create=True) as mock_load,
        ):
            mock_load.return_value = mock_ds
            DatasetImporter.load("some/hf-dataset-id")
            assert mock_load.call_count == 1

    def test_missing_datasets_package_raises_import_error(self):
        # SCENARIO: `datasets` extra not installed, unmocked HF-id load
        # WHY: contract requires a clear install hint, not a bare
        #   ModuleNotFoundError
        # EXPECTED: ImportError mentioning "pip install mltk[importer]"
        with patch("mltk.importer.loader._DATASETS_AVAILABLE", False):
            with pytest.raises(ImportError) as exc:
                DatasetImporter.load("some/hf-dataset-id")
        assert "pip install mltk[importer]" in str(exc.value)


# ===============================================================
# input_column / target_column overrides
# ===============================================================


class TestColumnOverrides:
    """input_column / target_column force-override auto-inferred roles."""

    def test_input_column_override_forces_input_role(self):
        # SCENARIO: 'passage' auto-maps to CONTEXT; force it to INPUT
        # WHY: user knows better than the heuristic for this dataset
        # EXPECTED: mapping.roles["passage"] == INPUT
        result = DatasetImporter.load(
            str(CSV_PATH), input_column="passage"
        )
        assert result.mapping.roles["passage"] == ColumnRole.INPUT

    def test_target_column_override_forces_golden_role(self):
        # SCENARIO: 'category' auto-maps to LABEL; force it to GOLDEN
        # WHY: user-directed override for target column
        # EXPECTED: mapping.roles["category"] == GOLDEN
        result = DatasetImporter.load(
            str(CSV_PATH), target_column="category"
        )
        assert result.mapping.roles["category"] == ColumnRole.GOLDEN

    def test_both_overrides_applied_together(self):
        result = DatasetImporter.load(
            str(CSV_PATH),
            input_column="passage",
            target_column="category",
        )
        assert result.mapping.roles["passage"] == ColumnRole.INPUT
        assert result.mapping.roles["category"] == ColumnRole.GOLDEN

    def test_overrides_applied_after_auto_mapping(self):
        # SCENARIO: 'question' would auto-map to INPUT; overriding
        #   input_column to 'passage' means 'question' is no longer
        #   forced to stay INPUT by the override itself (auto-mapping
        #   ran first, override only touches the named column)
        # WHY: override() only reassigns the single named column
        # EXPECTED: 'passage' becomes INPUT; other roles untouched by
        #   the override mechanism
        result = DatasetImporter.load(
            str(CSV_PATH), input_column="passage"
        )
        assert result.mapping.roles["passage"] == ColumnRole.INPUT
        assert result.mapping.roles["answer"] == ColumnRole.GOLDEN
        assert result.mapping.roles["category"] == ColumnRole.LABEL
