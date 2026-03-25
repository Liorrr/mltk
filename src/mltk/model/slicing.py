"""Model slicing and calibration -- test subgroup performance and prediction confidence.

The most insidious ML bug: overall accuracy 92% but 52% for age<18.
Slice-based testing catches subgroup failures that aggregate metrics hide.
Calibration testing catches models that say 90% confidence but are correct 60%.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from mltk.core.assertion import assert_true, timed_assertion
from mltk.core.result import Severity, TestResult
from mltk.model.metrics import _compute_metric


@timed_assertion
def assert_slice_performance(
    y_true: Any,
    y_pred: Any,
    slices: dict[str, Any],
    metric: str = "accuracy",
    min_threshold: float = 0.7,
    average: str = "weighted",
) -> TestResult:
    """Assert model meets minimum performance on EVERY data slice.

    Args:
        y_true: Ground truth labels/values.
        y_pred: Model predictions.
        slices: Dict mapping slice name to boolean mask array.
        metric: Metric to compute per slice.
        min_threshold: Minimum required value for each slice.
        average: Averaging for multiclass metrics.

    Returns:
        TestResult with per-slice metrics.

    Example:
        >>> import numpy as np
        >>> y_true = np.array([1, 0, 1, 0])
        >>> y_pred = np.array([1, 0, 0, 0])
        >>> slices = {"young": [True, True, False, False], "old": [False, False, True, True]}
        >>> assert_slice_performance(y_true, y_pred, slices, min_threshold=0.5)
    """
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)

    slice_results: dict[str, float] = {}
    failing_slices: dict[str, float] = {}

    for name, mask in slices.items():
        mask_arr = np.asarray(mask, dtype=bool)
        slice_t = y_t[mask_arr]
        slice_p = y_p[mask_arr]

        if len(slice_t) == 0:
            slice_results[name] = 0.0
            failing_slices[name] = 0.0
            continue

        value = _compute_metric(slice_t, slice_p, metric, average)
        slice_results[name] = value
        if value < min_threshold:
            failing_slices[name] = value

    passed = len(failing_slices) == 0

    if passed:
        message = f"All {len(slices)} slices meet {metric}>={min_threshold}"
    else:
        fail_strs = [f"'{k}'={v:.4f}" for k, v in failing_slices.items()]
        message = f"{len(failing_slices)} slice(s) below {min_threshold}: {', '.join(fail_strs)}"

    return assert_true(
        passed,
        name="model.slice_performance",
        message=message,
        severity=Severity.CRITICAL,
        metric=metric,
        min_threshold=min_threshold,
        slice_results=slice_results,
        failing_slices=failing_slices,
    )


@timed_assertion
def assert_calibration(
    y_true: Any,
    y_prob: Any,
    max_error: float = 0.05,
    n_bins: int = 10,
) -> TestResult:
    """Assert prediction probabilities are well-calibrated (Expected Calibration Error).

    Args:
        y_true: Binary ground truth (0/1).
        y_prob: Predicted probabilities (0.0-1.0).
        max_error: Maximum allowed ECE.
        n_bins: Number of bins for calibration curve.

    Returns:
        TestResult with ECE and per-bin calibration data.

    Example:
        >>> import numpy as np
        >>> y_true = np.array([1, 0, 1, 0])
        >>> y_prob = np.array([0.9, 0.1, 0.8, 0.2])
        >>> assert_calibration(y_true, y_prob, max_error=0.1)
    """
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_prob, dtype=float)

    if len(y_t) == 0:
        return assert_true(
            False,
            name="model.calibration",
            message="Cannot compute calibration on empty arrays",
            severity=Severity.CRITICAL,
        )

    # Compute Expected Calibration Error (ECE):
    # ECE = weighted average of |avg_predicted - avg_actual| per probability bin,
    # where weight = bin_count / total_samples
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_data: list[dict[str, float]] = []
    ece = 0.0

    for i in range(n_bins):
        # Last bin includes the right boundary (probability = 1.0)
        mask = (y_p >= bin_edges[i]) & (y_p < bin_edges[i + 1])
        if i == n_bins - 1:
            mask = (y_p >= bin_edges[i]) & (y_p <= bin_edges[i + 1])

        bin_count = int(mask.sum())
        if bin_count == 0:
            continue

        avg_predicted = float(y_p[mask].mean())
        avg_actual = float(y_t[mask].mean())
        bin_weight = bin_count / len(y_t)
        bin_error = abs(avg_predicted - avg_actual)
        ece += bin_weight * bin_error

        bin_data.append({
            "bin_start": float(bin_edges[i]),
            "bin_end": float(bin_edges[i + 1]),
            "count": bin_count,
            "avg_predicted": avg_predicted,
            "avg_actual": avg_actual,
            "error": bin_error,
        })

    passed = ece <= max_error

    message = (
        f"ECE={ece:.4f} <= {max_error} (well-calibrated)"
        if passed
        else f"ECE={ece:.4f} > {max_error} (poorly calibrated)"
    )

    return assert_true(
        passed,
        name="model.calibration",
        message=message,
        severity=Severity.CRITICAL,
        ece=ece,
        max_error=max_error,
        n_bins=n_bins,
        bin_data=bin_data,
    )
