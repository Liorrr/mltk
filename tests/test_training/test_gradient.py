"""Tests for mltk.training.gradient — gradient flow, vanishing, exploding, loss finite."""

import numpy as np
import pytest

from mltk.core.assertion import MltkAssertionError
from mltk.training.gradient import (
    assert_gradient_flow,
    assert_loss_finite,
    assert_no_exploding_gradient,
    assert_no_vanishing_gradient,
)


class TestGradientFlow:
    """assert_gradient_flow — dead layer detection."""

    def test_gradient_flow_healthy(self) -> None:
        # SCENARIO: All layers have non-trivial mean absolute gradients
        # WHY: Normal training — every layer receives a learning signal
        # EXPECTED: passed=True, dead_layers=[]
        gradients = [
            np.array([0.01, 0.02, -0.03]),
            np.array([0.1, -0.05, 0.08]),
            np.array([-0.005, 0.007, 0.012]),
        ]
        result = assert_gradient_flow(gradients, min_mean_grad=1e-7)
        assert result.passed is True
        assert result.details["dead_layers"] == []
        assert len(result.details["per_layer_means"]) == 3

    def test_gradient_flow_dead_layer(self) -> None:
        # SCENARIO: Middle layer has all-zero gradients
        # WHY: Dead ReLU or disconnected path — common training bug in deep nets
        # EXPECTED: MltkAssertionError raised, dead_layers=[1]
        gradients = [
            np.array([0.01, 0.02]),
            np.array([0.0, 0.0]),   # dead layer
            np.array([0.05, -0.03]),
        ]
        with pytest.raises(MltkAssertionError) as exc:
            assert_gradient_flow(gradients, min_mean_grad=1e-7)
        result = exc.value.result
        assert result.passed is False
        assert 1 in result.details["dead_layers"]
        assert "dead" in result.message.lower()

    def test_gradient_flow_single_layer_alive(self) -> None:
        # SCENARIO: Single-layer model with healthy gradient
        # WHY: Edge case — single element list should not break iteration
        # EXPECTED: passed=True
        result = assert_gradient_flow([np.array([0.5])], min_mean_grad=1e-7)
        assert result.passed is True

    def test_gradient_flow_all_dead(self) -> None:
        # SCENARIO: Every layer has near-zero gradients
        # WHY: Entire network collapsed — learning rate too small or bad init
        # EXPECTED: MltkAssertionError, all layer indices in dead_layers
        gradients = [np.zeros(4), np.zeros(4), np.zeros(4)]
        with pytest.raises(MltkAssertionError) as exc:
            assert_gradient_flow(gradients)
        assert len(exc.value.result.details["dead_layers"]) == 3

    def test_gradient_flow_custom_threshold(self) -> None:
        # SCENARIO: Gradient just above a strict custom threshold
        # WHY: Users may set tight thresholds for sensitive domains
        # EXPECTED: passed=True when mean_abs just exceeds custom min_mean_grad
        gradients = [np.array([1e-4, -1e-4])]
        result = assert_gradient_flow(gradients, min_mean_grad=1e-5)
        assert result.passed is True

    def test_gradient_flow_returns_duration(self) -> None:
        # SCENARIO: @timed_assertion decorator is active
        # WHY: Timing metadata must be populated for every assertion
        # EXPECTED: duration_ms >= 0
        result = assert_gradient_flow([np.array([0.1])])
        assert result.duration_ms >= 0.0


class TestNoVanishingGradient:
    """assert_no_vanishing_gradient — L2 norm too small."""

    def test_no_vanishing_gradient_ok(self) -> None:
        # SCENARIO: All layer gradient norms are comfortably above threshold
        # WHY: Healthy network — norms in a normal range, no vanishing issue
        # EXPECTED: passed=True, vanishing_layers=[]
        gradients = [
            np.ones(10) * 0.05,
            np.ones(10) * 0.1,
            np.ones(10) * 0.03,
        ]
        result = assert_no_vanishing_gradient(gradients, min_grad_norm=1e-8)
        assert result.passed is True
        assert result.details["vanishing_layers"] == []
        assert len(result.details["layer_norms"]) == 3

    def test_vanishing_gradient_detected(self) -> None:
        # SCENARIO: Last layer has an extremely small gradient norm (~1e-12)
        # WHY: Classic vanishing gradient in early layers of deep unclipped network
        # EXPECTED: MltkAssertionError, vanishing_layers contains the index
        gradients = [
            np.ones(10) * 0.1,
            np.ones(10) * 0.05,
            np.ones(10) * 1e-13,  # vanishing
        ]
        with pytest.raises(MltkAssertionError) as exc:
            assert_no_vanishing_gradient(gradients, min_grad_norm=1e-8)
        result = exc.value.result
        assert result.passed is False
        assert 2 in result.details["vanishing_layers"]
        assert "vanishing" in result.message.lower()

    def test_vanishing_norms_stored(self) -> None:
        # SCENARIO: Multiple layers, check that all norms are recorded
        # WHY: Details dict must carry all layer norms for diagnostics
        # EXPECTED: layer_norms length == number of input layers
        rng = np.random.default_rng(0)
        gradients = [rng.normal(0, 0.1, 5) for _ in range(4)]
        result = assert_no_vanishing_gradient(gradients)
        assert len(result.details["layer_norms"]) == 4


class TestNoExplodingGradient:
    """assert_no_exploding_gradient — L2 norm too large."""

    def test_no_exploding_gradient_ok(self) -> None:
        # SCENARIO: All layer gradient norms are below the maximum threshold
        # WHY: Well-clipped or naturally bounded gradients — no explosion risk
        # EXPECTED: passed=True, exploding_layers=[]
        gradients = [
            np.ones(10) * 5.0,
            np.ones(10) * 10.0,
            np.ones(10) * 1.0,
        ]
        result = assert_no_exploding_gradient(gradients, max_grad_norm=1000.0)
        assert result.passed is True
        assert result.details["exploding_layers"] == []

    def test_exploding_gradient_detected(self) -> None:
        # SCENARIO: First layer gradient norm is 1e7 — far above max_grad_norm
        # WHY: Unbounded RNN gradient accumulation or extreme LR — training will diverge
        # EXPECTED: MltkAssertionError, exploding_layers=[0]
        gradients = [
            np.ones(10) * 1e6,   # norm = 1e6 * sqrt(10) >> 1000
            np.ones(10) * 0.01,
        ]
        with pytest.raises(MltkAssertionError) as exc:
            assert_no_exploding_gradient(gradients, max_grad_norm=1000.0)
        result = exc.value.result
        assert result.passed is False
        assert 0 in result.details["exploding_layers"]
        assert "exploding" in result.message.lower()

    def test_exploding_boundary_exactly_at_max(self) -> None:
        # SCENARIO: Norm exactly equals max_grad_norm (boundary condition)
        # WHY: Boundary values must not trigger false positives
        # EXPECTED: passed=True (<=, not <)
        # norm of np.ones(1) * 100.0 is exactly 100.0
        result = assert_no_exploding_gradient([np.array([100.0])], max_grad_norm=100.0)
        assert result.passed is True


class TestLossFinite:
    """assert_loss_finite — NaN/Inf in loss array."""

    def test_loss_finite_all_valid(self) -> None:
        # SCENARIO: Normal decreasing loss curve with no special values
        # WHY: Standard training run — every step produced a valid loss
        # EXPECTED: passed=True, nan_count=0, inf_count=0
        losses = np.linspace(2.0, 0.3, 50)
        result = assert_loss_finite(losses)
        assert result.passed is True
        assert result.details["nan_count"] == 0
        assert result.details["inf_count"] == 0
        assert result.details["total"] == 50

    def test_loss_finite_nan(self) -> None:
        # SCENARIO: Loss produces NaN at step 5 (e.g., log(0) or 0/0)
        # WHY: NaN propagates through the entire computation graph — must fail fast
        # EXPECTED: MltkAssertionError, nan_count=1
        losses = np.array([1.0, 0.9, 0.8, 0.7, float("nan"), 0.6])
        with pytest.raises(MltkAssertionError) as exc:
            assert_loss_finite(losses)
        result = exc.value.result
        assert result.passed is False
        assert result.details["nan_count"] == 1
        assert "nan" in result.message.lower()

    def test_loss_finite_inf(self) -> None:
        # SCENARIO: Loss overflows to Inf (exploding logits, no gradient clipping)
        # WHY: Inf loss means model weights are already irrecoverable
        # EXPECTED: MltkAssertionError, inf_count >= 1
        losses = np.array([0.5, 0.4, float("inf"), 0.3])
        with pytest.raises(MltkAssertionError) as exc:
            assert_loss_finite(losses)
        assert exc.value.result.details["inf_count"] >= 1

    def test_loss_finite_mixed_nan_inf(self) -> None:
        # SCENARIO: Both NaN and Inf present in the loss sequence
        # WHY: Catastrophic divergence — both types must be counted
        # EXPECTED: nan_count > 0 AND inf_count > 0
        losses = np.array([1.0, float("nan"), float("inf"), 0.5, float("nan")])
        with pytest.raises(MltkAssertionError) as exc:
            assert_loss_finite(losses)
        result = exc.value.result
        assert result.details["nan_count"] == 2
        assert result.details["inf_count"] == 1

    def test_loss_finite_single_value(self) -> None:
        # SCENARIO: Single valid loss value (first step)
        # WHY: Edge case — must not break on length-1 arrays
        # EXPECTED: passed=True
        result = assert_loss_finite(np.array([0.693]))
        assert result.passed is True
        assert result.details["total"] == 1
