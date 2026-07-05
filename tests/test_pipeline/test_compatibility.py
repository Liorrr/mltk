"""Tests for mltk.pipeline.compatibility -- inter-stage schema flow validation."""

from __future__ import annotations

import pytest

from mltk.core.assertion import MltkAssertionError
from mltk.core.result import Severity
from mltk.pipeline.compatibility import StageSpec, assert_pipeline_stages_compatible

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _ingest() -> StageSpec:
    return StageSpec("ingest", produces={"raw": "object", "timestamp": "datetime64"})


def _preprocess() -> StageSpec:
    return StageSpec(
        "preprocess",
        requires={"raw": "object"},
        produces={"x": "float64", "y": "float64"},
    )


def _model() -> StageSpec:
    return StageSpec(
        "model",
        requires={"x": "float64", "y": "float64"},
        produces={"prediction": "float64"},
    )


# ---------------------------------------------------------------------------
# Happy-path: compatible pipelines
# ---------------------------------------------------------------------------


class TestCleanPipeline:
    """Happy-path tests for fully compatible stage sequences."""

    def test_three_stage_pipeline_passes(self) -> None:
        """PASS: A clean 3-stage pipeline where all requirements are satisfied."""
        result = assert_pipeline_stages_compatible([_ingest(), _preprocess(), _model()])
        assert result.passed is True
        assert result.details["n_stages"] == 3
        assert result.details["incompatibilities"] == []

    def test_message_summarises_stage_count(self) -> None:
        """PASS: Result message mentions the stage count on success."""
        result = assert_pipeline_stages_compatible([_ingest(), _preprocess(), _model()])
        assert "3 stages compatible" in result.message

    def test_cumulative_schema_allows_skipping_stage(self) -> None:
        """PASS: Stage 3 may consume a column produced by stage 1 (skipping stage 2)."""
        stage1 = StageSpec("stage1", produces={"raw": "float64"})
        stage2 = StageSpec("stage2", produces={"intermediate": "int64"})
        # stage3 requires "raw" which came from stage1, not stage2
        stage3 = StageSpec("stage3", requires={"raw": "float64"})
        result = assert_pipeline_stages_compatible([stage1, stage2, stage3])
        assert result.passed is True

    def test_empty_stages_passes(self) -> None:
        """PASS: Empty stage list passes with n_stages=0 and no incompatibilities."""
        result = assert_pipeline_stages_compatible([])
        assert result.passed is True
        assert result.details["n_stages"] == 0
        assert result.details["incompatibilities"] == []

    def test_single_stage_passes(self) -> None:
        """PASS: Single stage always passes -- nothing to validate upstream."""
        stage = StageSpec("only", produces={"x": "float64"}, requires={"raw": "object"})
        result = assert_pipeline_stages_compatible([stage])
        assert result.passed is True
        assert result.details["n_stages"] == 1

    def test_two_stage_pipeline_passes(self) -> None:
        """PASS: Minimal 2-stage pipeline with matching schema."""
        stage1 = StageSpec("producer", produces={"feature": "float32"})
        stage2 = StageSpec("consumer", requires={"feature": "float32"})
        result = assert_pipeline_stages_compatible([stage1, stage2])
        assert result.passed is True


# ---------------------------------------------------------------------------
# Failure cases: missing columns
# ---------------------------------------------------------------------------


class TestMissingColumns:
    """Stage requires a column that was never produced upstream."""

    def test_missing_column_raises_critical(self) -> None:
        """FAIL: Stage requires column never produced upstream → AssertionError."""
        stage1 = StageSpec("stage1", produces={"x": "float64"})
        stage2 = StageSpec("stage2", requires={"embedding": "float64"})
        with pytest.raises(MltkAssertionError):
            assert_pipeline_stages_compatible([stage1, stage2])

    def test_missing_column_captured_in_incompatibilities(self) -> None:
        """FAIL: Missing column name appears in the incompatibilities detail list."""
        stage1 = StageSpec("stage1", produces={"x": "float64"})
        stage2 = StageSpec("stage2", requires={"embedding": "float64"})
        with pytest.raises(MltkAssertionError) as exc_info:
            assert_pipeline_stages_compatible([stage1, stage2])
        details = exc_info.value.result.details
        assert len(details["incompatibilities"]) == 1
        incomp = details["incompatibilities"][0]
        assert "embedding" in incomp["missing"]
        assert incomp["to"] == "stage2"


# ---------------------------------------------------------------------------
# Failure cases: dtype mismatches
# ---------------------------------------------------------------------------


class TestDtypeMismatches:
    """Produced dtype differs from the required dtype."""

    @pytest.mark.parametrize(
        ("produced_dtype", "required_dtype"),
        [
            ("int64", "Int64"),
            ("int64", "int"),
            ("float64", "float"),
        ],
    )
    def test_canonical_equivalent_dtypes_pass(
        self, produced_dtype: str, required_dtype: str
    ) -> None:
        """PASS: Equivalent dtype spellings are treated as compatible."""
        stage1 = StageSpec("stage1", produces={"x": produced_dtype})
        stage2 = StageSpec("stage2", requires={"x": required_dtype})
        result = assert_pipeline_stages_compatible(
            [stage1, stage2], check_dtypes=True
        )
        assert result.passed is True

    def test_canonical_dtypes_still_flag_real_mismatch(self) -> None:
        """FAIL: Canonicalized dtype comparison still rejects real mismatches."""
        stage1 = StageSpec("stage1", produces={"x": "int64"})
        stage2 = StageSpec("stage2", requires={"x": "float64"})
        with pytest.raises(MltkAssertionError):
            assert_pipeline_stages_compatible([stage1, stage2], check_dtypes=True)

    def test_dtype_mismatch_flagged_when_check_dtypes_true(self) -> None:
        """FAIL: Produced dtype differs from required dtype → MltkAssertionError."""
        stage1 = StageSpec("stage1", produces={"x": "int32"})
        stage2 = StageSpec("stage2", requires={"x": "float64"})
        with pytest.raises(MltkAssertionError):
            assert_pipeline_stages_compatible([stage1, stage2], check_dtypes=True)

    def test_dtype_mismatch_ignored_when_check_dtypes_false(self) -> None:
        """PASS: Dtype difference is silently ignored when check_dtypes=False."""
        stage1 = StageSpec("stage1", produces={"x": "int32"})
        stage2 = StageSpec("stage2", requires={"x": "float64"})
        result = assert_pipeline_stages_compatible([stage1, stage2], check_dtypes=False)
        assert result.passed is True

    def test_dtype_mismatch_detail_recorded(self) -> None:
        """FAIL: Dtype mismatch detail includes column, required dtype, available dtype."""
        stage1 = StageSpec("stage1", produces={"score": "int32"})
        stage2 = StageSpec("stage2", requires={"score": "float64"})
        with pytest.raises(MltkAssertionError) as exc_info:
            assert_pipeline_stages_compatible([stage1, stage2], check_dtypes=True)
        incomp = exc_info.value.result.details["incompatibilities"][0]
        mismatch = incomp["dtype_mismatch"][0]
        assert mismatch["column"] == "score"
        assert mismatch["required"] == "float64"
        assert mismatch["available"] == "int32"


# ---------------------------------------------------------------------------
# First-stage exemption
# ---------------------------------------------------------------------------


class TestFirstStageExemption:
    """First stage requirements are never validated against upstream stages."""

    def test_first_stage_requires_not_validated(self) -> None:
        """PASS: First stage may declare any requires without causing failure."""
        stage1 = StageSpec(
            "stage1",
            requires={"external_data": "object"},  # nothing upstream produces this
            produces={"x": "float64"},
        )
        stage2 = StageSpec("stage2", requires={"x": "float64"})
        result = assert_pipeline_stages_compatible([stage1, stage2])
        assert result.passed is True


# ---------------------------------------------------------------------------
# Severity parameter
# ---------------------------------------------------------------------------


class TestSeverity:
    """Severity controls raise vs. return on failure."""

    def test_warning_severity_returns_failed_result_without_raising(self) -> None:
        """FAIL+WARNING: Incompatible stages with WARNING return a failed result."""
        stage1 = StageSpec("stage1", produces={"x": "float64"})
        stage2 = StageSpec("stage2", requires={"missing_col": "float64"})
        result = assert_pipeline_stages_compatible(
            [stage1, stage2], severity=Severity.WARNING
        )
        assert result.passed is False

    def test_critical_severity_raises_on_failure(self) -> None:
        """FAIL+CRITICAL: Incompatible stages with CRITICAL raise MltkAssertionError."""
        stage1 = StageSpec("stage1", produces={"x": "float64"})
        stage2 = StageSpec("stage2", requires={"missing_col": "float64"})
        with pytest.raises(MltkAssertionError):
            assert_pipeline_stages_compatible(
                [stage1, stage2], severity=Severity.CRITICAL
            )


# ---------------------------------------------------------------------------
# Result metadata
# ---------------------------------------------------------------------------


class TestResultDetails:
    """Validate fixed fields in result.details and result.name."""

    def test_details_include_n_stages_and_check_dtypes_flag(self) -> None:
        """Details always carry n_stages and check_dtypes."""
        result = assert_pipeline_stages_compatible(
            [_ingest(), _preprocess()], check_dtypes=False
        )
        assert result.details["n_stages"] == 2
        assert result.details["check_dtypes"] is False

    def test_result_name_is_correct(self) -> None:
        """Result name is 'pipeline.stages_compatible'."""
        result = assert_pipeline_stages_compatible([_ingest()])
        assert result.name == "pipeline.stages_compatible"

    def test_duration_ms_is_populated(self) -> None:
        """timed_assertion decorator populates duration_ms."""
        result = assert_pipeline_stages_compatible([_ingest(), _preprocess()])
        assert result.duration_ms >= 0.0
