"""Tests for mltk.importer.loader -- multi-source dataset loading.

Covers local CSV/JSON/Parquet loading, cross-format equivalence, error
paths for bad/missing local files, and a fully-mocked HuggingFace Hub
path (no real network calls are made anywhere in this file).
"""

from __future__ import annotations

import json
import shutil
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

    def test_url_source_raises_not_implemented(self):
        # SCENARIO: source string looks like an http(s) URL
        # WHY: URL sources are explicitly documented as unsupported --
        #   must fail loudly with the documented exception type, not
        #   fall through to local-file or HF dispatch
        # EXPECTED: NotImplementedError
        with pytest.raises(NotImplementedError):
            DatasetImporter.load("https://example.com/data.csv")

    @requires_pyarrow
    def test_pq_suffix_loads_like_parquet(self, tmp_path):
        # SCENARIO: a real parquet file saved under the '.pq' extension
        #   (rather than '.parquet')
        # WHY: '.pq' is a documented recognized local extension --
        #   must dispatch to the same parquet loader, not the
        #   unrecognized-extension ValueError path
        # EXPECTED: loads all 5 rows, same auto-mapping as tiny.csv
        pq_path = tmp_path / "tiny.pq"
        shutil.copy(PARQUET_PATH, pq_path)
        result = DatasetImporter.load(str(pq_path))
        assert len(result.rows) == 5
        assert result.mapping.roles["question"] == ColumnRole.INPUT

    def test_zero_row_header_only_csv_loads_without_raising(self, tmp_path):
        # SCENARIO: a CSV with a header row but zero data rows
        # WHY: must not raise -- an empty dataset is a valid (if
        #   degenerate) input, and to_eval_dataset() must produce zero
        #   samples rather than erroring
        # EXPECTED: result.rows == [], and to_eval_dataset() sample_count
        #   == 0
        csv_path = tmp_path / "empty.csv"
        csv_path.write_text("question,answer\n", encoding="utf-8")
        result = DatasetImporter.load(str(csv_path))
        assert result.rows == []
        ds = result.to_eval_dataset(name="empty")
        assert ds.sample_count == 0

    def test_existing_directory_falls_through_to_hf_branch(self, tmp_path):
        # SCENARIO: a local directory that happens to share a name with
        #   a HuggingFace Hub id (e.g. a cloned './squad/' checkout)
        # WHY: P2 finding -- Path.exists() is True for directories too,
        #   so the old dispatch treated any existing path as local and
        #   raised a bogus "unrecognized extension ''" ValueError instead
        #   of trying the HF branch
        # EXPECTED: HF loader is attempted (mocked here), not a local
        #   extension error
        hub_like_dir = tmp_path / "squad"
        hub_like_dir.mkdir()
        mock_ds = MagicMock()
        mock_ds.column_names = ["question", "answer"]
        mock_ds.to_list.return_value = [{"question": "Q1?", "answer": "A1"}]
        with (
            patch("mltk.importer.loader._DATASETS_AVAILABLE", True),
            patch("mltk.importer.loader._hf_load_dataset", create=True) as mock_load,
        ):
            mock_load.return_value = mock_ds
            result = DatasetImporter.load(str(hub_like_dir))
        mock_load.assert_called_once()
        assert isinstance(result, ImportResult)

    def test_existing_directory_without_datasets_raises_import_error(
        self, tmp_path
    ):
        # SCENARIO: same directory-shadowing case, but the optional
        #   `datasets` package is unavailable
        # WHY: confirms the directory reaches the HF branch's ImportError
        #   guard rather than the local-file ValueError path
        # EXPECTED: ImportError, not ValueError
        hub_like_dir = tmp_path / "squad"
        hub_like_dir.mkdir()
        with patch("mltk.importer.loader._DATASETS_AVAILABLE", False):
            with pytest.raises(ImportError):
                DatasetImporter.load(str(hub_like_dir))


class TestJsonShapeErrors:
    """JSON shapes other than a bare list or {"samples": [...]} reject
    loudly with the documented ValueError, not a raw pandas error."""

    def test_samples_key_not_a_list_raises_value_error(self, tmp_path):
        # SCENARIO: {"samples": {...}} -- a dict, not a list, under the
        #   'samples' key
        # WHY: P2 finding -- pandas.DataFrame(a_dict) succeeds with an
        #   "all scalar values" ValueError that leaks pandas internals
        #   instead of the documented shape error
        # EXPECTED: ValueError (the documented one, raised before pandas
        #   ever sees the data)
        bad = tmp_path / "bad_samples_dict.json"
        bad.write_text(
            json.dumps({"samples": {"question": "Q1?"}}), encoding="utf-8"
        )
        with pytest.raises(ValueError):  # noqa: PT011 -- message text not part of contract
            DatasetImporter.load(str(bad))

    def test_missing_samples_key_raises_value_error(self, tmp_path):
        # SCENARIO: a JSON object with neither a bare list nor a
        #   'samples' key
        # EXPECTED: ValueError
        bad = tmp_path / "bad_no_samples_key.json"
        bad.write_text(json.dumps({"foo": []}), encoding="utf-8")
        with pytest.raises(ValueError):  # noqa: PT011 -- message text not part of contract
            DatasetImporter.load(str(bad))


class TestMissingCellNormalizationEndToEnd:
    """Blank-cell normalization through the REAL pandas blank -> NaN
    round-trip (not an in-memory None/NaN injected directly into rows,
    which only proves _normalize()/_is_missing() work in isolation --
    this proves the loader's pandas-sourced NaN actually gets normalized
    by the time it reaches sample metadata)."""

    def test_blank_label_and_metadata_cells_normalize_to_none(self, tmp_path):
        # SCENARIO: a real CSV with a blank cell in a LABEL column
        #   (category) and a blank cell in a METADATA column (id, which
        #   forces the whole column to a float dtype with NaN once
        #   pandas sees a missing numeric value)
        # EXPECTED: metadata['label']/['category'] is None for the
        #   blank-category row; metadata['id'] is None for the
        #   blank-id row -- never NaN, never the literal string "None"
        csv_path = tmp_path / "blanks.csv"
        csv_path.write_text(
            "question,answer,category,id\n"
            "Q1?,A1,,1\n"
            "Q2?,A2,math,\n",
            encoding="utf-8",
        )
        result = DatasetImporter.load(str(csv_path))
        ds = result.to_eval_dataset(name="blanks")

        assert ds.samples[0].metadata["label"] is None
        assert ds.samples[0].metadata["category"] is None
        assert ds.samples[1].metadata["id"] is None


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

    def test_hf_load_computes_dtypes_and_mapping_roles(self):
        # SCENARIO: mocked HF dataset with a string column and a
        #   boolean column
        # WHY: exercises _infer_value_dtype() (the HF-path dtype
        #   inferrer, distinct from the pandas-based
        #   _infer_series_dtype() used for local files) and confirms
        #   auto-mapping runs correctly against HF-sourced dtypes
        # EXPECTED: dtypes classified per-column; roles follow the usual
        #   heuristics against those dtypes
        mock_ds = MagicMock()
        mock_ds.column_names = ["question", "verified"]
        mock_ds.to_list.return_value = [
            {"question": "Q1?", "verified": True},
            {"question": "Q2?", "verified": False},
        ]
        with (
            patch("mltk.importer.loader._DATASETS_AVAILABLE", True),
            patch("mltk.importer.loader._hf_load_dataset", create=True) as mock_load,
        ):
            mock_load.return_value = mock_ds
            result = DatasetImporter.load("some/hf-dataset-id")
        assert result.dtypes == {"question": "string", "verified": "boolean"}
        assert result.mapping.roles["question"] == ColumnRole.INPUT
        assert result.mapping.roles["verified"] == ColumnRole.UNKNOWN

    def test_boolean_column_via_real_json_file_infers_boolean_dtype(
        self, tmp_path
    ):
        # SCENARIO: a real (non-mocked) local JSON file with a boolean
        #   field, exercising the pandas-based _infer_series_dtype()
        #   path (distinct from the HF path's _infer_value_dtype())
        # EXPECTED: dtypes['flag'] == 'boolean'
        json_path = tmp_path / "bool_column.json"
        json_path.write_text(
            json.dumps(
                [
                    {"question": "Q1?", "flag": True},
                    {"question": "Q2?", "flag": False},
                ]
            ),
            encoding="utf-8",
        )
        result = DatasetImporter.load(str(json_path))
        assert result.dtypes["flag"] == "boolean"


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


class TestColumnOverridesActuallyTakeEffectEndToEnd:
    """P1 finding: overrides used to only touch the `roles` dict entry
    for the named column, without demoting the column that was ALREADY
    holding that role -- so the force-assigned column would silently
    lose the tie in `to_eval_dataset()`'s "first column with role X"
    pick. `load()` now applies overrides with `exclusive=True`, so the
    named column is guaranteed to become the effective sample field."""

    def test_input_column_override_actually_becomes_sample_input(self):
        result = DatasetImporter.load(str(CSV_PATH), input_column="passage")
        ds = result.to_eval_dataset(name="t")
        assert ds.samples[0].input == (
            "France is a country in Western Europe. Its capital city is "
            "Paris, known for the Eiffel Tower."
        )
        # the demoted auto-mapped INPUT column ('question') is not
        # silently dropped -- it surfaces as UNKNOWN and passes through
        assert result.mapping.roles["question"] == ColumnRole.UNKNOWN
        assert ds.samples[0].metadata["question"] == (
            "What is the capital of France?"
        )

    def test_target_column_override_actually_becomes_sample_target(self):
        result = DatasetImporter.load(str(CSV_PATH), target_column="category")
        ds = result.to_eval_dataset(name="t")
        assert ds.samples[0].target == "geography"
        # the demoted auto-mapped GOLDEN column ('answer') is not
        # silently dropped -- it surfaces as UNKNOWN and passes through
        assert result.mapping.roles["answer"] == ColumnRole.UNKNOWN
        assert ds.samples[0].metadata["answer"] == "Paris"
