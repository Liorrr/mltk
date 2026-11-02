"""Tests for mltk.model.regression -- baseline comparison and regression detection.

These tests verify that models maintain quality over time. Each test simulates
a real CI/CD scenario: save baseline → train new model → compare.
"""

import json

import numpy as np
import pytest

from mltk.core.assertion import MltkAssertionError
from mltk.model.regression import assert_no_regression, save_baseline


class TestSaveBaseline:
    """Tests for save_baseline -- persist metrics for future comparison."""

    def test_save_load_roundtrip(
        self,
        tmp_path: object,
        binary_classification: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        """Save metrics, load file, verify contents.

        Scenario: End of training pipeline saves metrics. CI/CD later
        loads them to compare against new model.
        """
        y_true, y_pred, _ = binary_classification
        path = f"{tmp_path}/baseline.json"

        result = save_baseline(y_true, y_pred, metrics=["accuracy", "f1"], path=path)

        assert "accuracy" in result["metrics"]
        assert "f1" in result["metrics"]
        assert result["sample_count"] == len(y_true)

        # Verify file is valid JSON
        with open(path) as f:
            loaded = json.load(f)
        assert loaded["metrics"]["accuracy"] == result["metrics"]["accuracy"]


class TestAssertNoRegression:
    """Tests for assert_no_regression -- detect quality drops."""

    def test_no_regression(
        self, binary_classification: tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        """PASS: Current model meets baseline.

        Scenario: New model version is as good as the previous one.
        2% tolerance allows for normal variance.
        """
        y_true, y_pred, _ = binary_classification
        result = assert_no_regression(
            y_true, y_pred, baseline=0.80, metric="accuracy", tolerance=0.02
        )
        assert result.passed is True

    def test_regression_detected(self) -> None:
        """FAIL: Current model significantly worse than baseline.

        Scenario: Bug in feature pipeline caused model quality to drop
        from 95% to 60%. Caught before deployment.
        """
        y_true = np.array([0, 0, 0, 1, 1, 1, 0, 1, 0, 1])
        y_pred = np.array([1, 1, 0, 0, 0, 1, 1, 0, 0, 1])  # ~40% accuracy
        with pytest.raises(MltkAssertionError) as exc:
            assert_no_regression(y_true, y_pred, baseline=0.95, metric="accuracy", tolerance=0.02)
        assert "Regression" in str(exc.value)

    @pytest.mark.parametrize("metric", ["mse", "rmse", "mae"])
    def test_error_metric_improvement_passes(self, metric: str) -> None:
        result = assert_no_regression(
            [1.0, 2.0], [1.0, 2.0], baseline=0.02, metric=metric,
            tolerance=0.01,
        )
        assert result.passed is True
        assert result.details["direction"] == "lower_is_better"
        assert result.details["min_allowed"] is None
        assert result.details["max_allowed"] == pytest.approx(0.03)

    @pytest.mark.parametrize("metric", ["mse", "rmse", "mae"])
    def test_error_metric_within_tolerance_passes(self, metric: str) -> None:
        result = assert_no_regression(
            [0.0, 0.0], [0.0, 0.01], baseline=0.01, metric=metric,
            tolerance=0.01,
        )
        assert result.passed is True

    @pytest.mark.parametrize("metric", ["mse", "rmse", "mae"])
    def test_error_metric_rise_beyond_tolerance_fails(self, metric: str) -> None:
        with pytest.raises(MltkAssertionError, match=f"Regression: {metric} rose"):
            assert_no_regression(
                [0.0, 0.0], [0.0, 1.0], baseline=0.01, metric=metric,
                tolerance=0.01,
            )

    def test_accuracy_direction_details(self) -> None:
        with pytest.raises(MltkAssertionError) as exc:
            assert_no_regression(
                [0, 0, 1, 1], [0, 1, 0, 1], baseline=0.9,
                metric="accuracy", tolerance=0.01,
            )
        details = exc.value.result.details
        assert details["direction"] == "higher_is_better"
        assert details["min_allowed"] == pytest.approx(0.89)
        assert details["max_allowed"] is None

        result = assert_no_regression(
            [0, 0, 1, 1], [0, 0, 1, 1], baseline=0.9,
            metric="accuracy", tolerance=0.01,
        )
        assert result.passed is True

    def test_baseline_from_float(
        self, binary_classification: tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        """PASS: Baseline as direct float value.

        Scenario: Quick check against a known threshold.
        """
        y_true, y_pred, _ = binary_classification
        result = assert_no_regression(y_true, y_pred, baseline=0.75, metric="accuracy")
        assert result.passed is True

    def test_baseline_from_dict(
        self, binary_classification: tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        """PASS: Baseline from dict of metrics.

        Scenario: Metrics stored in a config dict, not a file.
        """
        y_true, y_pred, _ = binary_classification
        baseline_dict = {"accuracy": 0.80, "f1": 0.78}
        result = assert_no_regression(y_true, y_pred, baseline=baseline_dict, metric="accuracy")
        assert result.passed is True

    def test_baseline_from_file(
        self,
        tmp_path: object,
        binary_classification: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        """PASS: Baseline loaded from JSON file.

        Scenario: CI/CD pipeline loads baseline saved during last release.
        """
        y_true, y_pred, _ = binary_classification
        path = f"{tmp_path}/baseline.json"
        save_baseline(y_true, y_pred, metrics=["accuracy"], path=path)

        result = assert_no_regression(y_true, y_pred, baseline=path, metric="accuracy")
        assert result.passed is True

    def test_tolerance_boundary(
        self, binary_classification: tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        """PASS: Exactly at tolerance boundary passes.

        Scenario: Model dropped exactly 2% -- still within tolerance.
        """
        y_true, y_pred, _ = binary_classification
        from mltk.model.metrics import _compute_metric

        actual = _compute_metric(y_true, y_pred, "accuracy")
        # Set baseline so current is exactly at tolerance boundary
        result = assert_no_regression(
            y_true, y_pred, baseline=actual + 0.02, metric="accuracy", tolerance=0.02
        )
        assert result.passed is True

    def test_empty_predictions(self) -> None:
        """FAIL: Empty predictions handled gracefully.

        Scenario: Pipeline produced zero predictions — should fail clearly.
        """
        with pytest.raises(MltkAssertionError):
            assert_no_regression([], [], baseline=0.9, metric="accuracy")
