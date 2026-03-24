"""Tests for mltk.model.metrics -- model performance validation.

These tests verify that assert_metric correctly gates on metric thresholds.
Each test simulates a real ML evaluation scenario.
"""

import numpy as np
import pytest

from mltk.core.assertion import MltkAssertionError
from mltk.model.metrics import assert_metric


class TestClassificationMetrics:
    """Classification metric assertions."""

    def test_accuracy_above_threshold(
        self, binary_classification: tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        """PASS: Good classifier exceeds accuracy threshold.

        Scenario: Model achieves ~85% accuracy on balanced test set.
        Gate set at 80%. Should pass deployment check.
        """
        y_true, y_pred, _ = binary_classification
        result = assert_metric(y_true, y_pred, metric="accuracy", threshold=0.80)
        assert result.passed is True
        assert result.details["value"] >= 0.80

    def test_accuracy_below_threshold(self) -> None:
        """FAIL: Random predictions fail accuracy check.

        Scenario: Model outputs random labels. Catches completely
        broken models before they reach production.
        """
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        y_pred = np.array([1, 1, 1, 1, 0, 0, 0, 0])  # All wrong
        with pytest.raises(MltkAssertionError) as exc:
            assert_metric(y_true, y_pred, metric="accuracy", threshold=0.5)
        assert "does not meet" in str(exc.value)

    def test_f1_weighted(
        self, multiclass_classification: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """PASS: Weighted F1 for multiclass classification.

        Scenario: 3-class model evaluated with weighted F1 to account
        for class frequency differences.
        """
        y_true, y_pred = multiclass_classification
        result = assert_metric(y_true, y_pred, metric="f1", threshold=0.7, average="weighted")
        assert result.passed is True

    def test_auc_binary(
        self, binary_classification: tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        """PASS: AUC on binary classification probabilities.

        Scenario: AUC measures overall discrimination ability. A model
        with ~85% accuracy should have AUC well above 0.5 (random).
        """
        y_true, _, y_prob = binary_classification
        result = assert_metric(y_true, y_prob, metric="auc", threshold=0.5)
        assert result.passed is True

    def test_perfect_predictions(self) -> None:
        """PASS: Perfect predictions give metric=1.0.

        Scenario: Sanity check -- model that gets everything right
        should score 1.0 on accuracy, F1, precision, recall.
        """
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 0, 1, 1, 0, 1])
        result = assert_metric(y_true, y_pred, metric="accuracy", threshold=1.0)
        assert result.passed is True
        assert result.details["value"] == 1.0


class TestRegressionMetrics:
    """Regression metric assertions."""

    def test_mse_below_threshold(
        self, regression_results: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """PASS: MSE below threshold for regression model.

        Scenario: Prediction error is within acceptable bounds.
        MSE is a "lower is better" metric.
        """
        y_true, y_pred = regression_results
        result = assert_metric(y_true, y_pred, metric="mse", threshold=50.0)
        assert result.passed is True

    def test_r2_above_threshold(
        self, regression_results: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """PASS: R2 score above threshold.

        Scenario: Model explains >80% of variance in target variable.
        """
        y_true, y_pred = regression_results
        result = assert_metric(y_true, y_pred, metric="r2", threshold=0.8)
        assert result.passed is True


class TestMetricEdgeCases:
    """Edge cases and error handling."""

    def test_unknown_metric(self) -> None:
        """FAIL: Invalid metric name raises clear error.

        Scenario: Typo in metric name -- should fail fast.
        """
        with pytest.raises(MltkAssertionError) as exc:
            assert_metric([0, 1], [0, 1], metric="nonexistent")
        assert "Unknown metric" in str(exc.value)

    def test_empty_arrays(self) -> None:
        """FAIL: Empty arrays handled gracefully.

        Scenario: Pipeline bug produces empty predictions.
        """
        with pytest.raises(MltkAssertionError):
            assert_metric([], [], metric="accuracy")
