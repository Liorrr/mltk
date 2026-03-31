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

_VALID_CAL_METHODS = ("ece", "smooth_ece")


def _reflected_gaussian_kernel(
    p: np.ndarray,
    q: np.ndarray,
    sigma: float,
) -> np.ndarray:
    """Reflected Gaussian kernel for [0,1]-bounded data.

    Adds mirror images at boundaries 0 and 1 so the kernel
    density stays correct near the edges.

    Args:
        p: Evaluation points, shape (m,) or scalar.
        q: Data points, shape (n,).
        sigma: Bandwidth (std-dev of the Gaussian).

    Returns:
        Kernel matrix of shape (m, n).
    """
    p = np.atleast_1d(p)[:, None]
    q = np.atleast_1d(q)[None, :]
    inv = 1.0 / (sigma * np.sqrt(2.0 * np.pi))
    half_inv_sq = -0.5 / (sigma * sigma)
    return inv * (
        np.exp(half_inv_sq * (p - q) ** 2)
        + np.exp(half_inv_sq * (p + q) ** 2)
        + np.exp(half_inv_sq * (p - q - 2) ** 2)
        + np.exp(half_inv_sq * (p - q + 2) ** 2)
    )


def _nadaraya_watson(
    eval_pts: np.ndarray,
    f: np.ndarray,
    y: np.ndarray,
    sigma: float,
) -> np.ndarray:
    """Nadaraya-Watson kernel regression estimator.

    Args:
        eval_pts: Points at which to estimate mu_hat.
        f: Predicted probabilities (data points).
        y: Binary outcomes.
        sigma: Kernel bandwidth.

    Returns:
        Estimated mu_hat at each evaluation point.
    """
    k = _reflected_gaussian_kernel(eval_pts, f, sigma)
    denom = k.sum(axis=1)
    denom = np.where(denom == 0, 1.0, denom)
    return (k @ y) / denom


def _smooth_ece_sigma(
    f: np.ndarray,
    y: np.ndarray,
    sigma: float,
) -> float:
    """Compute SmoothECE at a fixed bandwidth sigma.

    smECE = (1/n) * sum_i |mu_hat(f_i) - f_i|

    Args:
        f: Predicted probabilities.
        y: Binary outcomes.
        sigma: Kernel bandwidth.

    Returns:
        SmoothECE value (float).
    """
    mu_hat = _nadaraya_watson(f, f, y, sigma)
    return float(np.mean(np.abs(mu_hat - f)))


def _smooth_ece_auto(
    f: np.ndarray,
    y: np.ndarray,
) -> tuple[float, float]:
    """Auto-bandwidth SmoothECE via binary search.

    Finds the smallest sigma where smECE(sigma) >= sigma,
    following the Blasiok et al. self-consistent estimator.

    Args:
        f: Predicted probabilities.
        y: Binary outcomes.

    Returns:
        Tuple of (smECE, sigma_used).
    """
    lo, hi = 1e-4, 1.0
    for _ in range(50):
        mid = (lo + hi) / 2.0
        val = _smooth_ece_sigma(f, y, mid)
        if val >= mid:
            lo = mid
        else:
            hi = mid
    sigma_star = (lo + hi) / 2.0
    return _smooth_ece_sigma(f, y, sigma_star), sigma_star


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
        TestResult with per-slice metrics. Empty slices (zero samples) are
        treated as failing with a score of 0.0, since an empty slice
        indicates a data pipeline issue.

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
    method: str = "ece",
) -> TestResult:
    """Assert prediction probabilities are well-calibrated.

    Supports two methods:

    * ``"ece"`` -- classic Expected Calibration Error with equal-width
      bins (default, backward-compatible).
    * ``"smooth_ece"`` -- kernel-smoothed ECE using a reflected
      Gaussian kernel with automatic bandwidth selection.
      SmoothECE is provably consistent -- it converges to zero
      iff the model is truly calibrated, unlike binned ECE which
      has bin-boundary artifacts
      (Blasiok et al., ICLR 2024).

    Args:
        y_true: Binary ground truth (0/1).
        y_prob: Predicted probabilities (0.0-1.0).
        max_error: Maximum allowed calibration error.
        n_bins: Number of bins (used only when method="ece").
        method: ``"ece"`` or ``"smooth_ece"``.

    Returns:
        TestResult with calibration error and method-specific
        details.

    Example:
        >>> import numpy as np
        >>> y_true = np.array([1, 0, 1, 0])
        >>> y_prob = np.array([0.9, 0.1, 0.8, 0.2])
        >>> assert_calibration(y_true, y_prob, max_error=0.1)
        >>> assert_calibration(
        ...     y_true, y_prob, method="smooth_ece",
        ... )
    """
    if method not in _VALID_CAL_METHODS:
        return assert_true(
            False,
            name="model.calibration",
            message=(
                f"Unknown method: '{method}'. "
                f"Supported: {list(_VALID_CAL_METHODS)}"
            ),
            severity=Severity.CRITICAL,
        )

    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_prob, dtype=float)

    if len(y_t) == 0:
        return assert_true(
            False,
            name="model.calibration",
            message="Cannot compute calibration on empty arrays",
            severity=Severity.CRITICAL,
        )

    if method == "smooth_ece":
        return _calibration_smooth_ece(y_t, y_p, max_error)

    return _calibration_binned_ece(y_t, y_p, max_error, n_bins)


def _calibration_binned_ece(
    y_t: np.ndarray,
    y_p: np.ndarray,
    max_error: float,
    n_bins: int,
) -> TestResult:
    """Binned ECE (original implementation).

    Args:
        y_t: Binary ground truth.
        y_p: Predicted probabilities.
        max_error: Maximum allowed ECE.
        n_bins: Number of equal-width bins.

    Returns:
        TestResult with ECE and per-bin data.
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_data: list[dict[str, float]] = []
    ece = 0.0

    for i in range(n_bins):
        mask = (y_p >= bin_edges[i]) & (y_p < bin_edges[i + 1])
        if i == n_bins - 1:
            mask = (
                (y_p >= bin_edges[i])
                & (y_p <= bin_edges[i + 1])
            )

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
        method="ece",
        bin_data=bin_data,
    )


def _calibration_smooth_ece(
    y_t: np.ndarray,
    y_p: np.ndarray,
    max_error: float,
) -> TestResult:
    """Kernel-smoothed ECE (Blasiok et al., ICLR 2024).

    Uses a reflected Gaussian kernel with automatic bandwidth
    selection via binary search for self-consistent sigma.

    Args:
        y_t: Binary ground truth.
        y_p: Predicted probabilities.
        max_error: Maximum allowed smECE.

    Returns:
        TestResult with smECE and selected bandwidth.
    """
    sm_ece, sigma = _smooth_ece_auto(y_p, y_t)
    passed = sm_ece <= max_error

    message = (
        f"smECE={sm_ece:.4f} <= {max_error} "
        f"(well-calibrated, sigma={sigma:.4f})"
        if passed
        else f"smECE={sm_ece:.4f} > {max_error} "
        f"(poorly calibrated, sigma={sigma:.4f})"
    )

    return assert_true(
        passed,
        name="model.calibration",
        message=message,
        severity=Severity.CRITICAL,
        smooth_ece=sm_ece,
        sigma=sigma,
        max_error=max_error,
        method="smooth_ece",
    )
