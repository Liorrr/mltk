"""Gradient health assertions — detect dead, vanishing, and exploding gradients.

Silent gradient failures are among the most insidious training bugs. A dead layer
produces no learning signal downstream; vanishing gradients cause lower layers to
stop updating; exploding gradients destabilize the entire network. These assertions
take plain numpy arrays — framework agnostic by design.
"""

from __future__ import annotations

import numpy as np

from mltk.core.assertion import assert_true, timed_assertion
from mltk.core.result import Severity, TestResult


@timed_assertion
def assert_gradient_flow(
    gradients: list[np.ndarray],
    min_mean_grad: float = 1e-7,
) -> TestResult:
    """Assert that gradients are flowing through all layers (no dead layers).

    A layer whose mean absolute gradient falls below ``min_mean_grad`` is
    considered dead — it receives no learning signal and will never improve.

    Args:
        gradients: List of numpy arrays, one per layer (e.g., from
            ``[p.grad.numpy() for p in model.parameters()]``).
        min_mean_grad: Minimum acceptable mean absolute gradient value.
            Layers below this threshold are flagged as dead.

    Returns:
        TestResult with ``per_layer_means`` and ``dead_layers`` details.

    Example:
        >>> grads = [np.array([0.01, 0.02]), np.array([0.0, 0.0])]
        >>> assert_gradient_flow(grads, min_mean_grad=1e-7)
    """
    per_layer_means: list[float] = []
    dead_layers: list[int] = []

    for i, grad in enumerate(gradients):
        arr = np.asarray(grad, dtype=float)
        mean_abs = float(np.mean(np.abs(arr)))
        per_layer_means.append(mean_abs)
        if mean_abs < min_mean_grad:
            dead_layers.append(i)

    passed = len(dead_layers) == 0
    message = (
        f"Gradient flow OK: all {len(gradients)} layers active "
        f"(min_mean_grad={min_mean_grad})"
        if passed
        else f"Dead layers detected: layer indices {dead_layers} have mean |grad| "
        f"< {min_mean_grad} (out of {len(gradients)} layers)"
    )

    return assert_true(
        passed,
        name="training.gradient_flow",
        message=message,
        severity=Severity.CRITICAL,
        per_layer_means=per_layer_means,
        dead_layers=dead_layers,
        min_mean_grad=min_mean_grad,
        num_layers=len(gradients),
    )


@timed_assertion
def assert_no_vanishing_gradient(
    gradients: list[np.ndarray],
    min_grad_norm: float = 1e-8,
) -> TestResult:
    """Assert that no layer has vanishing gradients (L2 norm too small).

    Vanishing gradients occur when the L2 norm of a layer's gradient shrinks
    towards zero, typically in early layers of deep networks. Training stalls
    silently — parameters update by effectively nothing.

    Args:
        gradients: List of numpy arrays, one per layer.
        min_grad_norm: Minimum acceptable L2 norm. Layers below this are
            flagged as vanishing.

    Returns:
        TestResult with ``layer_norms`` and ``vanishing_layers`` details.

    Example:
        >>> grads = [np.ones(10) * 0.01, np.ones(10) * 1e-12]
        >>> assert_no_vanishing_gradient(grads, min_grad_norm=1e-8)
    """
    layer_norms: list[float] = []
    vanishing_layers: list[int] = []

    for i, grad in enumerate(gradients):
        arr = np.asarray(grad, dtype=float)
        norm = float(np.linalg.norm(arr))
        layer_norms.append(norm)
        if norm < min_grad_norm:
            vanishing_layers.append(i)

    passed = len(vanishing_layers) == 0
    message = (
        f"No vanishing gradients: all {len(gradients)} layers have norm "
        f">= {min_grad_norm}"
        if passed
        else f"Vanishing gradients at layer indices {vanishing_layers}: "
        f"norm < {min_grad_norm}"
    )

    return assert_true(
        passed,
        name="training.no_vanishing_gradient",
        message=message,
        severity=Severity.CRITICAL,
        layer_norms=layer_norms,
        vanishing_layers=vanishing_layers,
        min_grad_norm=min_grad_norm,
    )


@timed_assertion
def assert_no_exploding_gradient(
    gradients: list[np.ndarray],
    max_grad_norm: float = 1000.0,
) -> TestResult:
    """Assert that no layer has exploding gradients (L2 norm too large).

    Exploding gradients cause parameter updates so large they overshoot any
    reasonable optimum — weights diverge and loss spikes to NaN/Inf. Common
    in RNNs and very deep networks without gradient clipping.

    Args:
        gradients: List of numpy arrays, one per layer.
        max_grad_norm: Maximum acceptable L2 norm. Layers above this are
            flagged as exploding.

    Returns:
        TestResult with ``layer_norms`` and ``exploding_layers`` details.

    Example:
        >>> grads = [np.ones(10) * 0.1, np.ones(10) * 1e6]
        >>> assert_no_exploding_gradient(grads, max_grad_norm=1000.0)
    """
    layer_norms: list[float] = []
    exploding_layers: list[int] = []

    for i, grad in enumerate(gradients):
        arr = np.asarray(grad, dtype=float)
        norm = float(np.linalg.norm(arr))
        layer_norms.append(norm)
        if norm > max_grad_norm:
            exploding_layers.append(i)

    passed = len(exploding_layers) == 0
    message = (
        f"No exploding gradients: all {len(gradients)} layers have norm "
        f"<= {max_grad_norm}"
        if passed
        else f"Exploding gradients at layer indices {exploding_layers}: "
        f"norm > {max_grad_norm}"
    )

    return assert_true(
        passed,
        name="training.no_exploding_gradient",
        message=message,
        severity=Severity.CRITICAL,
        layer_norms=layer_norms,
        exploding_layers=exploding_layers,
        max_grad_norm=max_grad_norm,
    )


@timed_assertion
def assert_loss_finite(
    losses: np.ndarray,
) -> TestResult:
    """Assert that all loss values are finite (no NaN or Inf).

    NaN loss is a terminal training condition — once it appears it typically
    propagates through the entire computation graph. Inf loss indicates a
    numerical overflow. Both indicate an unrecoverable training run.

    Args:
        losses: 1D numpy array of loss values over training steps.

    Returns:
        TestResult with ``nan_count``, ``inf_count``, and ``total`` details.

    Example:
        >>> losses = np.array([1.0, 0.9, 0.8, float('nan')])
        >>> assert_loss_finite(losses)
    """
    arr = np.asarray(losses, dtype=float).ravel()
    nan_count = int(np.sum(np.isnan(arr)))
    inf_count = int(np.sum(np.isinf(arr)))
    total = len(arr)

    passed = nan_count == 0 and inf_count == 0
    message = (
        f"All {total} loss values are finite"
        if passed
        else f"Non-finite losses: {nan_count} NaN + {inf_count} Inf "
        f"out of {total} steps"
    )

    return assert_true(
        passed,
        name="training.loss_finite",
        message=message,
        severity=Severity.CRITICAL,
        nan_count=nan_count,
        inf_count=inf_count,
        total=total,
    )
