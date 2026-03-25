"""Tests for mltk.training.skew — training-serving skew detection."""

import numpy as np
import pytest

from mltk.core.assertion import MltkAssertionError
from mltk.training.skew import assert_no_training_serving_skew


class TestNoTrainingServingSkew:
    """Training-serving output comparison assertions."""

    def test_no_skew_identical(self) -> None:
        # SCENARIO: train and serve produce byte-for-byte identical outputs
        # WHY: Smoke test — perfectly matching pipelines must always pass
        # EXPECTED: pass
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = assert_no_training_serving_skew(arr, arr.copy(), tolerance=0.01)
        assert result.passed is True
        assert result.details["max_diff"] == 0.0
        assert result.details["num_skewed"] == 0

    def test_skew_detected(self) -> None:
        # SCENARIO: serve pipeline applies different normalisation — outputs differ by 0.5+
        # WHY: Catches the most common source of silent accuracy degradation
        # EXPECTED: fail with "skew" in the message
        train_out = np.array([0.1, 0.5, 0.9, 0.3, 0.7])
        serve_out = np.array([0.6, 1.0, 1.4, 0.8, 1.2])  # shifted by +0.5
        with pytest.raises(MltkAssertionError) as exc:
            assert_no_training_serving_skew(train_out, serve_out, tolerance=0.01)
        assert "skew" in str(exc.value).lower()

    def test_skew_within_tolerance(self) -> None:
        # SCENARIO: outputs differ by at most 0.005, tolerance=0.01
        # WHY: Tiny floating-point differences from hardware/library versions
        #      should not trigger a false alarm
        # EXPECTED: pass
        train_out = np.array([1.0, 2.0, 3.0])
        serve_out = np.array([1.005, 2.003, 2.999])  # max diff = 0.005
        result = assert_no_training_serving_skew(train_out, serve_out, tolerance=0.01)
        assert result.passed is True
        assert result.details["max_diff"] <= 0.01

    def test_skew_list_input(self) -> None:
        # SCENARIO: caller passes plain Python lists instead of ndarray
        # WHY: Convenience — users should not need to pre-convert inputs
        # EXPECTED: pass (lists are converted correctly)
        result = assert_no_training_serving_skew(
            [0.0, 1.0, 2.0],
            [0.0, 1.0, 2.0],
            tolerance=0.001,
        )
        assert result.passed is True
        assert result.details["num_elements"] == 3

    def test_skew_shape_mismatch_fails(self) -> None:
        # SCENARIO: train output has 3 elements, serve has 4
        # WHY: Shape mismatch means pipelines are fundamentally different —
        #      must fail loudly rather than silently truncate
        # EXPECTED: fail with "shape mismatch" message
        with pytest.raises(MltkAssertionError) as exc:
            assert_no_training_serving_skew(
                np.array([1.0, 2.0, 3.0]),
                np.array([1.0, 2.0, 3.0, 4.0]),
                tolerance=0.01,
            )
        assert "shape" in str(exc.value).lower()

    def test_skew_tolerance_boundary(self) -> None:
        # SCENARIO: max diff exactly equals tolerance
        # WHY: Boundary condition — exact equality should pass (<= not <)
        # EXPECTED: pass
        train_out = np.array([0.0, 0.0, 0.0])
        serve_out = np.array([0.01, 0.0, 0.0])  # max diff = 0.01, tolerance = 0.01
        result = assert_no_training_serving_skew(train_out, serve_out, tolerance=0.01)
        assert result.passed is True

    def test_skew_metrics_populated(self) -> None:
        # SCENARIO: verify result carries all diagnostic details
        # WHY: Callers need max_diff, mean_diff, num_skewed for triage
        # EXPECTED: pass and all detail keys present
        a = np.linspace(0, 1, 20)
        b = a + 0.001
        result = assert_no_training_serving_skew(a, b, tolerance=0.01)
        assert result.passed is True
        for key in ("max_diff", "mean_diff", "tolerance", "num_skewed", "num_elements"):
            assert key in result.details
