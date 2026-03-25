"""Tests for mltk.training.numerical — NaN/Inf arrays, loss trends, softmax validity."""

import numpy as np
import pytest

from mltk.core.assertion import MltkAssertionError
from mltk.training.numerical import (
    assert_loss_decreasing,
    assert_no_loss_divergence,
    assert_no_nan_inf,
    assert_softmax_valid,
)


class TestNoNanInf:
    """assert_no_nan_inf — array-level NaN/Inf detection."""

    def test_no_nan_inf_clean(self) -> None:
        # SCENARIO: All weight and activation arrays have only finite values
        # WHY: Healthy model state — nothing to flag
        # EXPECTED: passed=True, problematic_arrays={}
        arrays = [
            np.array([0.1, -0.2, 0.3]),
            np.array([[1.0, 2.0], [3.0, 4.0]]),
            np.zeros(5),
        ]
        result = assert_no_nan_inf(arrays, names=["w1", "w2", "bias"])
        assert result.passed is True
        assert result.details["problematic_arrays"] == {}
        assert result.details["arrays_checked"] == 3

    def test_no_nan_inf_with_nan(self) -> None:
        # SCENARIO: Second weight matrix contains a single NaN value
        # WHY: A NaN in weights will silently corrupt all downstream outputs
        # EXPECTED: MltkAssertionError, "w2" in problematic_arrays
        arrays = [
            np.array([0.5, 0.5]),
            np.array([1.0, float("nan"), 2.0]),
        ]
        with pytest.raises(MltkAssertionError) as exc:
            assert_no_nan_inf(arrays, names=["w1", "w2"])
        result = exc.value.result
        assert result.passed is False
        assert "w2" in result.details["problematic_arrays"]

    def test_no_nan_inf_with_inf(self) -> None:
        # SCENARIO: Activation array overflowed to Inf during forward pass
        # WHY: Inf activations propagate and cause NaN in subsequent ops (Inf - Inf)
        # EXPECTED: MltkAssertionError, problematic entry notes Inf count
        arrays = [np.array([float("inf"), 0.0, -float("inf")])]
        with pytest.raises(MltkAssertionError) as exc:
            assert_no_nan_inf(arrays, names=["activations"])
        assert "activations" in exc.value.result.details["problematic_arrays"]

    def test_no_nan_inf_auto_names(self) -> None:
        # SCENARIO: No names provided — auto fallback names used
        # WHY: names parameter is optional; must not crash and must key correctly
        # EXPECTED: MltkAssertionError with "array_1" in problematic_arrays
        arrays = [np.array([1.0]), np.array([float("nan")])]
        with pytest.raises(MltkAssertionError) as exc:
            assert_no_nan_inf(arrays)
        assert "array_1" in exc.value.result.details["problematic_arrays"]

    def test_no_nan_inf_empty_list(self) -> None:
        # SCENARIO: Empty array list passed
        # WHY: Edge case — should pass cleanly with zero arrays checked
        # EXPECTED: passed=True, arrays_checked=0
        result = assert_no_nan_inf([])
        assert result.passed is True
        assert result.details["arrays_checked"] == 0

    def test_no_nan_inf_returns_duration(self) -> None:
        # SCENARIO: @timed_assertion decorator populates duration_ms
        # WHY: Timing metadata required for all assertions
        # EXPECTED: duration_ms >= 0
        result = assert_no_nan_inf([np.array([1.0])])
        assert result.duration_ms >= 0.0


class TestLossDecreasing:
    """assert_loss_decreasing — training progress check."""

    def test_loss_decreasing_ok(self) -> None:
        # SCENARIO: Smoothly decreasing loss curve (linear ramp down)
        # WHY: Model is learning — loss at end should be well below start
        # EXPECTED: passed=True, decrease > 0
        losses = np.linspace(3.0, 0.1, 100)
        result = assert_loss_decreasing(losses, window=10)
        assert result.passed is True
        assert result.details["decrease"] > 0
        assert result.details["end_mean"] < result.details["start_mean"]

    def test_loss_increasing_fails(self) -> None:
        # SCENARIO: Loss is monotonically increasing (catastrophic divergence trend)
        # WHY: Model is not learning — learning rate too high or wrong objective
        # EXPECTED: MltkAssertionError, decrease < 0
        losses = np.linspace(0.1, 3.0, 100)
        with pytest.raises(MltkAssertionError) as exc:
            assert_loss_decreasing(losses, window=10)
        result = exc.value.result
        assert result.passed is False
        assert result.details["decrease"] < 0

    def test_loss_decreasing_noisy_still_passes(self) -> None:
        # SCENARIO: Noisy but downward-trending loss (realistic training noise)
        # WHY: Real loss curves are not smooth; window averaging absorbs noise
        # EXPECTED: passed=True
        rng = np.random.default_rng(42)
        base = np.linspace(2.0, 0.4, 80)
        noise = rng.normal(0, 0.05, 80)
        losses = base + noise
        result = assert_loss_decreasing(losses, window=10)
        assert result.passed is True

    def test_loss_decreasing_too_few_steps(self) -> None:
        # SCENARIO: Only 5 loss values provided but window=10
        # WHY: Not enough data to evaluate — should warn, not fail hard
        # EXPECTED: passed=True with WARNING severity
        from mltk.core.result import Severity
        losses = np.array([1.0, 0.9, 0.8, 0.7, 0.6])
        result = assert_loss_decreasing(losses, window=10)
        assert result.passed is True
        assert result.severity == Severity.WARNING

    def test_loss_flat_with_min_decrease(self) -> None:
        # SCENARIO: Loss is perfectly flat but min_decrease=0.5 is required
        # WHY: User expects measurable improvement — flat curve should fail
        # EXPECTED: MltkAssertionError
        losses = np.ones(40)
        with pytest.raises(MltkAssertionError):
            assert_loss_decreasing(losses, window=5, min_decrease=0.5)


class TestNoLossDivergence:
    """assert_no_loss_divergence — catastrophic spike detection."""

    def test_no_loss_divergence_stable(self) -> None:
        # SCENARIO: Loss oscillates gently around a decreasing trend — ratio low
        # WHY: Normal training with some variance but no explosion
        # EXPECTED: passed=True, ratio <= max_increase_ratio
        rng = np.random.default_rng(7)
        losses = np.linspace(1.0, 0.5, 50) + rng.normal(0, 0.02, 50)
        losses = np.abs(losses)  # ensure positive
        result = assert_no_loss_divergence(losses, max_increase_ratio=10.0)
        assert result.passed is True
        assert result.details["ratio"] <= 10.0

    def test_loss_divergence_detected(self) -> None:
        # SCENARIO: Loss spikes from ~0.5 to 500 mid-training (gradient explosion)
        # WHY: Learning rate too large; loss explodes, ratio >> max_increase_ratio
        # EXPECTED: MltkAssertionError, ratio >> 10
        losses = np.array([0.5, 0.4, 0.45, 500.0, 0.3])
        with pytest.raises(MltkAssertionError) as exc:
            assert_no_loss_divergence(losses, max_increase_ratio=10.0)
        result = exc.value.result
        assert result.passed is False
        assert result.details["ratio"] > 10.0
        assert "diverged" in result.message.lower()

    def test_no_loss_divergence_all_non_finite_fails(self) -> None:
        # SCENARIO: Every loss value is NaN — no finite values at all
        # WHY: No ratio can be computed; must report failure clearly
        # EXPECTED: MltkAssertionError
        losses = np.array([float("nan"), float("nan")])
        with pytest.raises(MltkAssertionError) as exc:
            assert_no_loss_divergence(losses)
        assert "finite" in exc.value.result.message.lower()

    def test_no_loss_divergence_stores_min_max(self) -> None:
        # SCENARIO: Normal losses — details dict must contain max and min
        # WHY: Diagnostics require knowing the actual loss range
        # EXPECTED: max_loss and min_loss populated correctly
        losses = np.array([2.0, 1.5, 1.0, 0.8, 0.6])
        result = assert_no_loss_divergence(losses)
        assert result.details["max_loss"] == pytest.approx(2.0)
        assert result.details["min_loss"] == pytest.approx(0.6)


class TestSoftmaxValid:
    """assert_softmax_valid — probability distribution sanity check."""

    def test_softmax_valid_ok(self) -> None:
        # SCENARIO: Perfect softmax outputs — each row sums to exactly 1.0
        # WHY: Well-implemented softmax layer; all distributions valid
        # EXPECTED: passed=True, max_sum_error near 0, out_of_range_count=0
        probs = np.array([
            [0.2, 0.5, 0.3],
            [0.1, 0.6, 0.3],
            [0.0, 1.0, 0.0],
        ])
        result = assert_softmax_valid(probs)
        assert result.passed is True
        assert result.details["out_of_range_count"] == 0
        assert result.details["max_sum_error"] < 1e-5

    def test_softmax_invalid_row_sum(self) -> None:
        # SCENARIO: One row sums to 1.5 instead of 1.0 (broken normalization)
        # WHY: Missing or incorrect softmax — distributions are not valid
        # EXPECTED: MltkAssertionError, max_sum_error > tolerance
        probs = np.array([
            [0.5, 0.5, 0.5],   # sums to 1.5
            [0.3, 0.3, 0.4],   # sums to 1.0
        ])
        with pytest.raises(MltkAssertionError) as exc:
            assert_softmax_valid(probs)
        result = exc.value.result
        assert result.passed is False
        assert result.details["max_sum_error"] > 1e-5

    def test_softmax_invalid_negative_value(self) -> None:
        # SCENARIO: Probabilities contain a negative value (-0.1)
        # WHY: Softmax outputs must be in [0,1]; negative value means wrong layer
        # EXPECTED: MltkAssertionError, out_of_range_count >= 1
        probs = np.array([
            [-0.1, 0.6, 0.5],
            [0.2, 0.3, 0.5],
        ])
        with pytest.raises(MltkAssertionError) as exc:
            assert_softmax_valid(probs)
        assert exc.value.result.details["out_of_range_count"] >= 1

    def test_softmax_1d_input_treated_as_single_sample(self) -> None:
        # SCENARIO: 1D array passed instead of 2D — represents one sample
        # WHY: Convenience — users may pass a single prediction row directly
        # EXPECTED: passed=True without shape error
        probs = np.array([0.1, 0.3, 0.6])
        result = assert_softmax_valid(probs)
        assert result.passed is True
        assert result.details["num_samples"] == 1

    def test_softmax_records_shape(self) -> None:
        # SCENARIO: 4-sample, 5-class prediction batch
        # WHY: Shape metadata should be stored for diagnostics
        # EXPECTED: num_samples=4, num_classes=5
        rng = np.random.default_rng(99)
        raw = rng.dirichlet(np.ones(5), size=4)  # guaranteed valid softmax
        result = assert_softmax_valid(raw)
        assert result.details["num_samples"] == 4
        assert result.details["num_classes"] == 5

    def test_softmax_numerical_precision_ok(self) -> None:
        # SCENARIO: Floating-point softmax with tiny rounding errors (~1e-16)
        # WHY: Real softmax implementations accumulate fp rounding; must not flag
        # EXPECTED: passed=True (tolerance is 1e-5, far above fp noise)
        probs = np.array([[0.3333333333, 0.3333333333, 0.3333333334]])
        result = assert_softmax_valid(probs)
        assert result.passed is True
