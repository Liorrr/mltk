"""Tests for mltk.model.overfitting -- overfitting detection and label drift.

These tests verify that the gap between train and test metrics stays bounded
and that label distributions remain stable across splits. Each test simulates
a real evaluation scenario in a CI/CD pipeline.
"""

from __future__ import annotations

import numpy as np
import pytest

from mltk.core.assertion import MltkAssertionError
from mltk.model.overfitting import assert_label_drift, assert_no_overfitting


class TestAssertNoOverfitting:
    """Tests for assert_no_overfitting -- train/test gap gating."""

    def test_no_overfitting_pass(self) -> None:
        # SCENARIO: Model trained well — small generalisation gap
        # WHY: train=0.92, test=0.89 → gap=0.03 is well inside default max_gap=0.10
        # EXPECTED: pass, gap stored in details
        result = assert_no_overfitting(train_score=0.92, test_score=0.89)
        assert result.passed is True
        assert result.details["gap"] == pytest.approx(0.03, abs=1e-9)
        assert result.details["train_score"] == pytest.approx(0.92)
        assert result.details["test_score"] == pytest.approx(0.89)

    def test_overfitting_detected(self) -> None:
        # SCENARIO: Model memorised training data — large gap
        # WHY: train=0.99, test=0.70 → gap=0.29 exceeds max_gap=0.10
        # EXPECTED: raises MltkAssertionError, message contains "Overfitting"
        with pytest.raises(MltkAssertionError) as exc:
            assert_no_overfitting(train_score=0.99, test_score=0.70, max_gap=0.10)
        assert "Overfitting" in str(exc.value)
        assert exc.value.result.passed is False

    def test_no_overfitting_exact_boundary(self) -> None:
        # SCENARIO: Gap is exactly equal to max_gap
        # WHY: gap == max_gap should be a PASS (condition is gap <= max_gap)
        # EXPECTED: pass — boundary is inclusive
        result = assert_no_overfitting(
            train_score=0.90, test_score=0.80, max_gap=0.10
        )
        assert result.passed is True
        assert result.details["gap"] == pytest.approx(0.10, abs=1e-9)

    def test_overfitting_negative_gap(self) -> None:
        # SCENARIO: Test score is higher than train score (underfitting / lucky test set)
        # WHY: gap = train - test = -0.05 < 0 < max_gap → should pass
        # EXPECTED: pass — negative gap means no overfitting
        result = assert_no_overfitting(train_score=0.80, test_score=0.85, max_gap=0.10)
        assert result.passed is True
        assert result.details["gap"] < 0.0

    def test_overfitting_custom_metric(self) -> None:
        # SCENARIO: Using a non-default metric name (F1 instead of accuracy)
        # WHY: metric_name should appear verbatim in the result details and message
        # EXPECTED: pass, details["metric_name"] == "f1"
        result = assert_no_overfitting(
            train_score=0.88,
            test_score=0.85,
            max_gap=0.10,
            metric_name="f1",
        )
        assert result.passed is True
        assert result.details["metric_name"] == "f1"
        assert "f1" in result.message

    def test_overfitting_details_complete(self) -> None:
        # SCENARIO: Verify all expected keys are present in details
        # WHY: Downstream reporters depend on these keys being populated
        # EXPECTED: train_score, test_score, gap, max_gap, metric_name all present
        result = assert_no_overfitting(train_score=0.91, test_score=0.88, max_gap=0.15)
        for key in ("train_score", "test_score", "gap", "max_gap", "metric_name"):
            assert key in result.details, f"Missing details key: {key}"

    def test_overfitting_timed(self) -> None:
        # SCENARIO: @timed_assertion decorator must populate duration_ms
        # WHY: All mltk assertions must carry timing for performance diagnostics
        # EXPECTED: duration_ms is a positive float
        result = assert_no_overfitting(train_score=0.90, test_score=0.85)
        assert result.duration_ms >= 0.0


class TestAssertLabelDrift:
    """Tests for assert_label_drift -- label distribution stability."""

    def test_label_drift_identical(self) -> None:
        # SCENARIO: Train and test share the exact same label distribution
        # WHY: TV distance = 0.0, well under any max_drift threshold
        # EXPECTED: pass, tv_distance == 0.0
        labels = [0, 0, 1, 1, 0, 1]
        result = assert_label_drift(labels, labels, max_drift=0.1)
        assert result.passed is True
        assert result.details["tv_distance"] == pytest.approx(0.0, abs=1e-9)

    def test_label_drift_shifted(self) -> None:
        # SCENARIO: Test set has a very different class ratio than train
        # WHY: train is 80% class-0, test is 80% class-1 → large TV distance
        # EXPECTED: raises MltkAssertionError, message contains "drift"
        train = [0] * 80 + [1] * 20
        test = [0] * 20 + [1] * 80
        with pytest.raises(MltkAssertionError) as exc:
            assert_label_drift(train, test, max_drift=0.1)
        assert exc.value.result.passed is False
        assert "drift" in exc.value.result.message.lower()

    def test_label_drift_multiclass(self) -> None:
        # SCENARIO: Three-class problem with a balanced split
        # WHY: Each class appears equally in both splits → TV ≈ 0 → pass
        # EXPECTED: pass, train_distribution has 3 keys
        train = [0, 1, 2, 0, 1, 2, 0, 1, 2]
        test = [0, 1, 2, 0, 1, 2, 0, 1, 2]
        result = assert_label_drift(train, test, max_drift=0.05)
        assert result.passed is True
        assert len(result.details["train_distribution"]) == 3

    def test_label_drift_within_tolerance(self) -> None:
        # SCENARIO: Minor imbalance that is within a relaxed tolerance
        # WHY: Small drift occurs naturally in random splits; we accept up to 20%
        # EXPECTED: pass when max_drift is generous enough
        train = [0, 0, 0, 1, 1, 1, 0, 1]  # 50/50
        test = [0, 0, 1, 1, 1, 1, 0, 1]   # 37.5% / 62.5% → TV = 0.125
        result = assert_label_drift(train, test, max_drift=0.2)
        assert result.passed is True

    def test_label_drift_distribution_keys_present(self) -> None:
        # SCENARIO: Verify details always contain distribution dicts
        # WHY: Downstream drift reports read these keys to render class tables
        # EXPECTED: train_distribution, test_distribution, tv_distance, max_drift present
        result = assert_label_drift([0, 1, 0, 1], [0, 1, 0, 1], max_drift=0.1)
        for key in ("tv_distance", "max_drift", "train_distribution", "test_distribution"):
            assert key in result.details, f"Missing details key: {key}"

    def test_label_drift_numpy_input(self) -> None:
        # SCENARIO: Caller passes numpy arrays instead of Python lists
        # WHY: Both list and ndarray inputs must be supported transparently
        # EXPECTED: pass, no TypeError
        train = np.array([0, 0, 1, 1, 0, 1])
        test = np.array([0, 0, 1, 1, 0, 1])
        result = assert_label_drift(train, test, max_drift=0.05)
        assert result.passed is True

    def test_label_drift_timed(self) -> None:
        # SCENARIO: @timed_assertion decorator must populate duration_ms
        # WHY: Consistent with all other mltk assertions — timing is mandatory
        # EXPECTED: duration_ms is a non-negative float
        result = assert_label_drift([0, 1, 0, 1], [0, 1, 0, 1])
        assert result.duration_ms >= 0.0
