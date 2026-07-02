"""Tests for mltk.pipeline.resilience -- ML chaos engineering via fault injection."""

import numpy as np
import pandas as pd
import pytest

from mltk.core.assertion import MltkAssertionError
from mltk.core.result import Severity
from mltk.pipeline.resilience import DEFAULT_FAULTS, apply_fault, assert_pipeline_resilient

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_BASELINE = pd.DataFrame(
    {
        "a": [1.0, 2.0, 3.0, 4.0, 5.0],
        "b": [10.0, 20.0, 30.0, 40.0, 50.0],
        "c": [100.0, 200.0, 300.0, 400.0, 500.0],
    }
)
_SEED = 42


def _robust_pipeline(df: pd.DataFrame) -> float:
    """Survive all faults: select numeric cols, handle empty gracefully."""
    numeric = df.select_dtypes("number")
    if numeric.empty or len(numeric) == 0:
        return 0.0
    return float(numeric.sum().sum())


def _brittle_pipeline(df: pd.DataFrame) -> float:
    """Crashes when column 'c' is missing (e.g. after dropped_column fault)."""
    return float(df["c"].sum())


# ---------------------------------------------------------------------------
# assert_pipeline_resilient tests
# ---------------------------------------------------------------------------


class TestAssertPipelineResilient:
    """Integration tests for the main assertion."""

    def test_robust_pipeline_passes_all_faults(self) -> None:
        """PASS: Robust pipeline survives every default fault without crashing."""
        result = assert_pipeline_resilient(_robust_pipeline, _BASELINE, seed=_SEED)
        assert result.passed is True
        assert result.details["failure_rate"] == 0.0
        assert result.details["faults_run"] == len(DEFAULT_FAULTS)

    def test_brittle_pipeline_raises_assertion_error_at_default_tolerance(self) -> None:
        """FAIL: Brittle pipeline crashes on dropped_column → CRITICAL raises."""
        with pytest.raises(MltkAssertionError, match="failure_rate"):
            assert_pipeline_resilient(_brittle_pipeline, _BASELINE, seed=_SEED)

    def test_max_failure_rate_tolerates_one_crash(self) -> None:
        """PASS: One crash out of 7 faults (~0.14) is within max_failure_rate=0.2."""
        result = assert_pipeline_resilient(
            _brittle_pipeline,
            _BASELINE,
            max_failure_rate=0.2,
            seed=_SEED,
        )
        assert result.passed is True
        crashes = sum(1 for r in result.details["results"] if r["crashed"])
        assert crashes == 1

    def test_warning_severity_returns_result_no_raise(self) -> None:
        """WARNING severity: failure returns TestResult with passed=False, no exception."""
        result = assert_pipeline_resilient(
            _brittle_pipeline,
            _BASELINE,
            severity=Severity.WARNING,
            seed=_SEED,
        )
        assert result.passed is False
        assert result.severity == Severity.WARNING

    def test_subset_faults_runs_only_specified(self) -> None:
        """Passing faults=['null_injection'] runs exactly one fault."""
        result = assert_pipeline_resilient(
            _robust_pipeline,
            _BASELINE,
            faults=["null_injection"],
            seed=_SEED,
        )
        assert result.passed is True
        assert result.details["faults_run"] == 1
        assert len(result.details["results"]) == 1
        assert result.details["results"][0]["fault"] == "null_injection"

    def test_unknown_fault_raises_valueerror(self) -> None:
        """Unrecognised fault name raises ValueError before any pipeline call."""
        with pytest.raises(ValueError, match="Unknown"):
            assert_pipeline_resilient(
                _robust_pipeline, _BASELINE, faults=["nonexistent_fault"]
            )

    def test_baseline_input_unchanged_after_call(self) -> None:
        """The original baseline_input DataFrame must not be modified by the assertion."""
        original_copy = _BASELINE.copy(deep=True)
        assert_pipeline_resilient(_robust_pipeline, _BASELINE, seed=_SEED)
        pd.testing.assert_frame_equal(_BASELINE, original_copy)

    def test_result_detail_keys_present(self) -> None:
        """TestResult details must contain 'results', 'failure_rate', and 'faults_run'."""
        result = assert_pipeline_resilient(_robust_pipeline, _BASELINE, seed=_SEED)
        assert "results" in result.details
        assert "failure_rate" in result.details
        assert "faults_run" in result.details

    def test_per_fault_result_entry_structure(self) -> None:
        """Each entry in results list has 'fault', 'crashed', and 'error' keys."""
        result = assert_pipeline_resilient(_robust_pipeline, _BASELINE, seed=_SEED)
        for entry in result.details["results"]:
            assert "fault" in entry
            assert "crashed" in entry
            assert "error" in entry
            assert isinstance(entry["crashed"], bool)

    def test_crashed_entry_captures_error_message(self) -> None:
        """A crashed fault entry stores a non-None error string."""
        result = assert_pipeline_resilient(
            _brittle_pipeline,
            _BASELINE,
            max_failure_rate=1.0,
            seed=_SEED,
        )
        crashed_entries = [r for r in result.details["results"] if r["crashed"]]
        assert len(crashed_entries) >= 1
        for entry in crashed_entries:
            assert entry["error"] is not None
            assert isinstance(entry["error"], str)

    def test_zero_faults_list_gives_zero_failure_rate(self) -> None:
        """An empty faults list → 0 faults run, 0 crashes, failure_rate=0.0, passed=True."""
        result = assert_pipeline_resilient(
            _brittle_pipeline,
            _BASELINE,
            faults=[],
            seed=_SEED,
        )
        assert result.passed is True
        assert result.details["failure_rate"] == 0.0
        assert result.details["faults_run"] == 0

    def test_duration_ms_is_populated(self) -> None:
        """timed_assertion decorator populates duration_ms on the result."""
        result = assert_pipeline_resilient(_robust_pipeline, _BASELINE, seed=_SEED)
        assert result.duration_ms > 0.0


# ---------------------------------------------------------------------------
# apply_fault unit tests
# ---------------------------------------------------------------------------


class TestApplyFault:
    """Unit tests for the apply_fault helper."""

    def _rng(self) -> np.random.Generator:
        return np.random.default_rng(_SEED)

    def test_null_injection_preserves_shape(self) -> None:
        """null_injection output has same shape as input."""
        rng = self._rng()
        faulted = apply_fault(_BASELINE, "null_injection", rng)
        assert faulted.shape == _BASELINE.shape

    def test_null_injection_introduces_nans(self) -> None:
        """null_injection produces at least one NaN in the faulted frame."""
        rng = self._rng()
        faulted = apply_fault(_BASELINE, "null_injection", rng)
        assert faulted.isna().any().any()

    def test_null_injection_does_not_mutate_original(self) -> None:
        """null_injection never writes NaN back into the source DataFrame."""
        rng = self._rng()
        original_copy = _BASELINE.copy(deep=True)
        apply_fault(_BASELINE, "null_injection", rng)
        pd.testing.assert_frame_equal(_BASELINE, original_copy)

    def test_empty_input_returns_empty_frame_with_same_columns(self) -> None:
        """empty_input produces a DataFrame with zero rows but identical columns."""
        rng = self._rng()
        faulted = apply_fault(_BASELINE, "empty_input", rng)
        assert len(faulted) == 0
        assert list(faulted.columns) == list(_BASELINE.columns)

    def test_dropped_column_removes_last_column(self) -> None:
        """dropped_column reduces column count by 1, removing the last column."""
        rng = self._rng()
        faulted = apply_fault(_BASELINE, "dropped_column", rng)
        assert faulted.shape[1] == _BASELINE.shape[1] - 1
        assert list(faulted.columns) == list(_BASELINE.columns[:-1])

    def test_dtype_corruption_converts_first_numeric_to_str(self) -> None:
        """dtype_corruption casts the first numeric column to a string-backed dtype."""
        rng = self._rng()
        faulted = apply_fault(_BASELINE, "dtype_corruption", rng)
        first_numeric = _BASELINE.select_dtypes(include="number").columns[0]
        # pandas 3.x returns StringDtype; pandas 2.x returns object.
        # Check that values are actual strings, regardless of backing dtype.
        assert isinstance(faulted[first_numeric].iloc[0], str)

    def test_scale_shift_multiplies_numeric_by_1e6(self) -> None:
        """scale_shift makes numeric values 1e6 times larger."""
        rng = self._rng()
        faulted = apply_fault(_BASELINE, "scale_shift", rng)
        pd.testing.assert_frame_equal(
            faulted,
            _BASELINE * 1e6,
            check_dtype=False,
        )

    def test_duplicate_rows_doubles_row_count(self) -> None:
        """duplicate_rows produces a frame with exactly 2× the original rows."""
        rng = self._rng()
        faulted = apply_fault(_BASELINE, "duplicate_rows", rng)
        assert len(faulted) == 2 * len(_BASELINE)

    def test_single_row_keeps_exactly_one_row(self) -> None:
        """single_row keeps only the first row of the input."""
        rng = self._rng()
        faulted = apply_fault(_BASELINE, "single_row", rng)
        assert len(faulted) == 1
        # Values should match the first row of the original
        pd.testing.assert_series_equal(
            faulted.iloc[0].reset_index(drop=True),
            _BASELINE.iloc[0].reset_index(drop=True),
            check_names=False,
        )

    def test_unknown_fault_raises_valueerror(self) -> None:
        """apply_fault raises ValueError for any unrecognised fault name."""
        rng = self._rng()
        with pytest.raises(ValueError, match="Unknown fault"):
            apply_fault(_BASELINE, "bad_fault_xyz", rng)
