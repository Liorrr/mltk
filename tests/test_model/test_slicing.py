"""Tests for mltk.model.slicing -- subgroup performance and calibration.

These tests catch the most dangerous ML bias: a model that works well
on average but fails for specific demographics or data segments.
"""

import numpy as np
import pytest

from mltk.core.assertion import MltkAssertionError
from mltk.model.slicing import assert_calibration, assert_slice_performance


class TestAssertSlicePerformance:
    """Tests for subgroup performance validation."""

    def test_all_slices_pass(
        self, binary_classification: tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        """PASS: Every subgroup meets the minimum threshold.

        Scenario: Model performs equally well across age groups.
        No demographic bias detected.
        """
        y_true, y_pred, _ = binary_classification
        n = len(y_true)
        slices = {
            "first_half": np.arange(n) < n // 2,
            "second_half": np.arange(n) >= n // 2,
        }
        result = assert_slice_performance(
            y_true, y_pred, slices=slices, metric="accuracy", min_threshold=0.7
        )
        assert result.passed is True
        assert "slice_results" in result.details

    def test_one_slice_fails(self) -> None:
        """FAIL: One subgroup performs significantly worse.

        Scenario: Model is 90% accurate overall, but only 30% for
        the minority group. This is the classic fairness failure.
        """
        # Good predictions for slice A, terrible for slice B
        y_true = np.array([0, 0, 1, 1, 0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 0, 1, 1, 0, 1, 0, 0, 1, 0])  # slice B all wrong
        slices = {
            "slice_a": np.array([True, True, True, True, True, False, False, False, False, False]),
            "slice_b": np.array(
                [False, False, False, False, False, True, True, True, True, True]
            ),
        }
        with pytest.raises(MltkAssertionError) as exc:
            assert_slice_performance(
                y_true, y_pred, slices=slices, metric="accuracy", min_threshold=0.8
            )
        assert "slice" in str(exc.value).lower()

    def test_empty_slice(self) -> None:
        """FAIL: Empty slice (0 samples) is flagged.

        Scenario: Data filter produced a slice with no samples.
        This indicates a data pipeline issue, not a model issue.
        """
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        slices = {
            "has_data": np.array([True, True, True, True]),
            "empty": np.array([False, False, False, False]),
        }
        with pytest.raises(MltkAssertionError):
            assert_slice_performance(
                y_true, y_pred, slices=slices, metric="accuracy", min_threshold=0.5
            )


class TestAssertCalibration:
    """Tests for prediction calibration (ECE)."""

    def test_well_calibrated(self) -> None:
        """PASS: Model probabilities match actual outcomes.

        Scenario: When model says 80% confident, it's correct ~80%
        of the time. This is well-calibrated — trustworthy confidence.
        """
        rng = np.random.default_rng(42)
        n = 1000
        # Generate well-calibrated probabilities
        y_prob = rng.uniform(0, 1, n)
        y_true = (rng.random(n) < y_prob).astype(int)
        result = assert_calibration(y_true, y_prob, max_error=0.1)
        assert result.passed is True
        assert result.details["ece"] < 0.1

    def test_poorly_calibrated(self) -> None:
        """FAIL: Model is overconfident — says 90% but correct 50%.

        Scenario: Model always outputs high confidence (0.9) regardless
        of actual correctness. Dangerous for decision-making systems.
        """
        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_prob = np.array([0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9])
        with pytest.raises(MltkAssertionError) as exc:
            assert_calibration(y_true, y_prob, max_error=0.05)
        assert "poorly calibrated" in str(exc.value)

    def test_calibration_details(self) -> None:
        """Verify ECE value and bin data are in result details.

        Scenario: QA report needs calibration curve data for
        visualization — assert returns per-bin breakdown.
        """
        rng = np.random.default_rng(42)
        n = 500
        y_prob = rng.uniform(0, 1, n)
        y_true = (rng.random(n) < y_prob).astype(int)
        result = assert_calibration(y_true, y_prob, max_error=0.2)
        assert "ece" in result.details
        assert "bin_data" in result.details
        assert isinstance(result.details["bin_data"], list)

    def test_empty_arrays(self) -> None:
        """FAIL: Empty input handled gracefully."""
        with pytest.raises(MltkAssertionError):
            assert_calibration([], [], max_error=0.05)
