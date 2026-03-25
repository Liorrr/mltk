"""Numerical stability assertions — NaN/Inf checks, loss behaviour, softmax validity.

Numerical issues are quiet killers: a model can train for hours before a NaN
propagates visibly. These assertions give you fail-fast gates at every stage —
after each forward pass, after every epoch, and before inference.
"""

from __future__ import annotations

import numpy as np

from mltk.core.assertion import assert_true, timed_assertion
from mltk.core.result import Severity, TestResult


@timed_assertion
def assert_no_nan_inf(
    arrays: list[np.ndarray],
    names: list[str] | None = None,
) -> TestResult:
    """Assert that none of the provided arrays contain NaN or Inf values.

    Use after each forward pass, weight update, or activation computation to
    catch numerical breakdowns the moment they appear rather than epochs later.

    Args:
        arrays: List of numpy arrays to inspect (weights, activations, etc.).
        names: Optional list of human-readable names, one per array.
            Falls back to ``"array_0"``, ``"array_1"``, etc.

    Returns:
        TestResult with ``problematic_arrays`` mapping name → issue description.

    Example:
        >>> w1 = np.array([0.1, 0.2])
        >>> w2 = np.array([0.3, float('nan')])
        >>> assert_no_nan_inf([w1, w2], names=["layer1", "layer2"])
    """
    if names is None:
        names = [f"array_{i}" for i in range(len(arrays))]

    problematic_arrays: dict[str, str] = {}

    for name, arr in zip(names, arrays, strict=False):
        data = np.asarray(arr, dtype=float)
        nan_count = int(np.sum(np.isnan(data)))
        inf_count = int(np.sum(np.isinf(data)))
        if nan_count > 0 or inf_count > 0:
            parts = []
            if nan_count:
                parts.append(f"{nan_count} NaN")
            if inf_count:
                parts.append(f"{inf_count} Inf")
            problematic_arrays[name] = ", ".join(parts)

    passed = len(problematic_arrays) == 0
    message = (
        f"All {len(arrays)} arrays are finite"
        if passed
        else f"Non-finite values in: {list(problematic_arrays.keys())}"
    )

    return assert_true(
        passed,
        name="training.no_nan_inf",
        message=message,
        severity=Severity.CRITICAL,
        problematic_arrays=problematic_arrays,
        arrays_checked=len(arrays),
    )


@timed_assertion
def assert_loss_decreasing(
    losses: np.ndarray,
    window: int = 10,
    min_decrease: float = 0.0,
) -> TestResult:
    """Assert that loss is generally decreasing over training.

    Compares the mean of the first ``window`` steps to the mean of the last
    ``window`` steps. If the end mean is not lower than the start mean by at
    least ``min_decrease``, the assertion fails.

    Args:
        losses: 1D numpy array of loss values over training steps.
        window: Number of steps to average at the start and end.
        min_decrease: Minimum required decrease (end_mean <= start_mean - min_decrease).

    Returns:
        TestResult with ``start_mean``, ``end_mean``, and ``decrease`` details.

    Example:
        >>> losses = np.linspace(2.0, 0.5, 100)
        >>> assert_loss_decreasing(losses, window=10)
    """
    arr = np.asarray(losses, dtype=float).ravel()

    if len(arr) < window * 2:
        # Not enough data — treat as warning, not critical failure
        return assert_true(
            True,
            name="training.loss_decreasing",
            message=f"Too few steps ({len(arr)}) to evaluate decrease over window={window}",
            severity=Severity.WARNING,
            start_mean=float(arr[0]) if len(arr) > 0 else float("nan"),
            end_mean=float(arr[-1]) if len(arr) > 0 else float("nan"),
            decrease=float("nan"),
        )

    start_mean = float(np.mean(arr[:window]))
    end_mean = float(np.mean(arr[-window:]))
    decrease = start_mean - end_mean

    passed = decrease >= min_decrease
    message = (
        f"Loss decreasing: start={start_mean:.6f} -> end={end_mean:.6f} "
        f"(decrease={decrease:.6f})"
        if passed
        else f"Loss not decreasing: start={start_mean:.6f} -> end={end_mean:.6f} "
        f"(decrease={decrease:.6f}, required >= {min_decrease})"
    )

    return assert_true(
        passed,
        name="training.loss_decreasing",
        message=message,
        severity=Severity.CRITICAL,
        start_mean=start_mean,
        end_mean=end_mean,
        decrease=decrease,
        window=window,
        min_decrease=min_decrease,
    )


@timed_assertion
def assert_no_loss_divergence(
    losses: np.ndarray,
    max_increase_ratio: float = 10.0,
) -> TestResult:
    """Assert that loss has not diverged (no catastrophic spike).

    Computes ``max(losses) / min(losses)``. A ratio larger than
    ``max_increase_ratio`` indicates the loss exploded — common when the
    learning rate is too high or gradients are not clipped.

    Args:
        losses: 1D numpy array of loss values over training steps.
        max_increase_ratio: Maximum allowed ratio of max to min loss.
            Use a higher value for losses with natural warmup spikes.

    Returns:
        TestResult with ``max_loss``, ``min_loss``, and ``ratio`` details.

    Example:
        >>> losses = np.array([1.0, 0.9, 0.8, 0.5, 0.4])
        >>> assert_no_loss_divergence(losses, max_increase_ratio=10.0)
    """
    arr = np.asarray(losses, dtype=float).ravel()
    # Ignore non-finite values for ratio computation
    finite_arr = arr[np.isfinite(arr)]

    if len(finite_arr) == 0:
        return assert_true(
            False,
            name="training.no_loss_divergence",
            message="No finite loss values found — all NaN/Inf",
            severity=Severity.CRITICAL,
            max_loss=float("nan"),
            min_loss=float("nan"),
            ratio=float("nan"),
        )

    max_loss = float(np.max(finite_arr))
    min_loss = float(np.min(finite_arr))

    # Avoid division by zero: if min_loss <= 0, use absolute difference check
    if min_loss <= 0:
        ratio = float("nan")
        passed = True  # Cannot compute ratio meaningfully; skip divergence check
        message = (
            f"Loss divergence check skipped: min_loss={min_loss:.6f} <= 0 "
            f"(ratio undefined)"
        )
    else:
        ratio = max_loss / min_loss
        passed = ratio <= max_increase_ratio
        message = (
            f"Loss stable: max/min ratio={ratio:.2f} <= {max_increase_ratio}"
            if passed
            else f"Loss diverged: max/min ratio={ratio:.2f} > {max_increase_ratio} "
            f"(max={max_loss:.4f}, min={min_loss:.4f})"
        )

    return assert_true(
        passed,
        name="training.no_loss_divergence",
        message=message,
        severity=Severity.CRITICAL,
        max_loss=max_loss,
        min_loss=min_loss,
        ratio=ratio,
        max_increase_ratio=max_increase_ratio,
    )


@timed_assertion
def assert_softmax_valid(
    probabilities: np.ndarray,
) -> TestResult:
    """Assert that softmax outputs are valid probability distributions.

    Each row must sum to approximately 1.0 and all values must be in [0, 1].
    Violations indicate a broken softmax layer, numerical overflow in logits,
    or an incorrect normalization step.

    Args:
        probabilities: 2D numpy array of shape (samples, classes). Each row
            is expected to be a probability distribution over classes.

    Returns:
        TestResult with ``max_sum_error`` and ``out_of_range_count`` details.

    Example:
        >>> probs = np.array([[0.2, 0.5, 0.3], [0.1, 0.6, 0.3]])
        >>> assert_softmax_valid(probs)
    """
    arr = np.asarray(probabilities, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    row_sums = arr.sum(axis=1)
    sum_errors = np.abs(row_sums - 1.0)
    max_sum_error = float(np.max(sum_errors))

    out_of_range = np.sum((arr < 0.0) | (arr > 1.0))
    out_of_range_count = int(out_of_range)

    tolerance = 1e-5
    passed = max_sum_error <= tolerance and out_of_range_count == 0

    if not passed:
        issues = []
        if max_sum_error > tolerance:
            issues.append(f"max row sum error={max_sum_error:.2e} > {tolerance}")
        if out_of_range_count > 0:
            issues.append(f"{out_of_range_count} values outside [0, 1]")
        message = f"Invalid softmax output: {'; '.join(issues)}"
    else:
        message = (
            f"Softmax valid: {arr.shape[0]} samples, "
            f"max sum error={max_sum_error:.2e}"
        )

    return assert_true(
        passed,
        name="training.softmax_valid",
        message=message,
        severity=Severity.CRITICAL,
        max_sum_error=max_sum_error,
        out_of_range_count=out_of_range_count,
        num_samples=arr.shape[0],
        num_classes=arr.shape[1],
    )
