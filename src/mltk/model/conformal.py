"""Conformal prediction assertions -- validate prediction intervals and sets.

Prediction intervals and prediction sets are the output of uncertainty
quantification methods (conformal prediction, Bayesian inference,
bootstrap, quantile regression).  These assertions verify that intervals
achieve target coverage and that prediction sets are informatively sized.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from mltk.core.assertion import assert_true, timed_assertion
from mltk.core.result import Severity, TestResult


@timed_assertion
def assert_interval_coverage(
    y_true: np.ndarray,
    y_lower: np.ndarray,
    y_upper: np.ndarray,
    target_coverage: float = 0.9,
    tolerance: float = 0.05,
    severity: Severity = Severity.CRITICAL,
) -> TestResult:
    """Assert that prediction intervals achieve target coverage.

    Checks the fraction of true values falling inside [y_lower, y_upper].
    Works with any interval source: conformal, Bayesian, bootstrap, or
    quantile regression.

    Args:
        y_true: Ground truth values.
        y_lower: Lower bounds of prediction intervals.
        y_upper: Upper bounds of prediction intervals.
        target_coverage: Desired coverage probability (e.g. 0.9 for 90%).
        tolerance: Allowed shortfall below target_coverage.
        severity: Severity level for the assertion (default CRITICAL).

    Returns:
        TestResult with empirical_coverage, target_coverage, tolerance,
        n_covered, n_total, avg_width, and median_width in details.

    Example:
        >>> y = np.array([1.0, 2.0, 3.0])
        >>> lo = np.array([0.5, 1.5, 2.5])
        >>> hi = np.array([1.5, 2.5, 3.5])
        >>> assert_interval_coverage(y, lo, hi, target_coverage=0.9)
    """
    y_t = np.asarray(y_true, dtype=np.float64)
    y_lo = np.asarray(y_lower, dtype=np.float64)
    y_hi = np.asarray(y_upper, dtype=np.float64)

    n_total = len(y_t)

    if n_total == 0:
        return assert_true(
            False,
            name="model.interval_coverage",
            message="Cannot compute coverage on empty arrays",
            severity=severity,
        )

    if len(y_lo) != n_total or len(y_hi) != n_total:
        return assert_true(
            False,
            name="model.interval_coverage",
            message=(
                f"Array length mismatch: y_true={n_total}, "
                f"y_lower={len(y_lo)}, y_upper={len(y_hi)}"
            ),
            severity=severity,
        )

    covered = (y_t >= y_lo) & (y_t <= y_hi)
    n_covered = int(np.sum(covered))
    empirical_coverage = n_covered / n_total

    widths = y_hi - y_lo
    avg_width = float(np.mean(widths))
    median_width = float(np.median(widths))

    threshold = target_coverage - tolerance
    passed = empirical_coverage >= threshold

    if passed:
        message = (
            f"coverage={empirical_coverage:.4f} >= "
            f"{target_coverage} - {tolerance} (threshold {threshold:.4f})"
        )
    else:
        message = (
            f"coverage={empirical_coverage:.4f} < "
            f"{threshold:.4f} (target {target_coverage} - tolerance {tolerance})"
        )

    return assert_true(
        passed,
        name="model.interval_coverage",
        message=message,
        severity=severity,
        empirical_coverage=empirical_coverage,
        target_coverage=target_coverage,
        tolerance=tolerance,
        n_covered=n_covered,
        n_total=n_total,
        avg_width=avg_width,
        median_width=median_width,
    )


@timed_assertion
def assert_prediction_set_size(
    prediction_sets: list[list[Any]] | list[set[Any]] | np.ndarray,
    max_avg_size: float,
    max_empty_frac: float = 0.1,
    severity: Severity = Severity.CRITICAL,
) -> TestResult:
    """Assert that prediction sets are informatively sized.

    For classification, each prediction set is a list/set of predicted
    classes; size is the cardinality.  For regression, each entry is a
    float representing the interval width; size is that width.

    Checks two conditions:
      1. Average set size <= max_avg_size
      2. Fraction of empty sets <= max_empty_frac

    Args:
        prediction_sets: List of prediction sets (lists/sets for
            classification) or ndarray of floats (interval widths for
            regression).
        max_avg_size: Maximum allowed average set size / width.
        max_empty_frac: Maximum allowed fraction of empty sets (default 0.1).
        severity: Severity level for the assertion (default CRITICAL).

    Returns:
        TestResult with avg_size, max_size, min_size, empty_count,
        empty_frac, and n_sets in details.

    Example:
        >>> sets = [{"cat", "dog"}, {"cat"}, {"dog", "bird"}]
        >>> assert_prediction_set_size(sets, max_avg_size=3.0)
    """
    if isinstance(prediction_sets, np.ndarray):
        # Regression mode: each element is a float width
        arr = np.asarray(prediction_sets, dtype=np.float64).ravel()
        n_sets = len(arr)
        if n_sets == 0:
            return assert_true(
                False,
                name="model.prediction_set_size",
                message="Cannot compute set size on empty input",
                severity=severity,
            )
        sizes = arr
        empty_count = int(np.sum(sizes == 0.0))
    else:
        # Classification mode: each element is a list/set of classes
        n_sets = len(prediction_sets)
        if n_sets == 0:
            return assert_true(
                False,
                name="model.prediction_set_size",
                message="Cannot compute set size on empty input",
                severity=severity,
            )
        sizes = np.array([len(s) for s in prediction_sets], dtype=np.float64)
        empty_count = int(np.sum(sizes == 0))

    avg_size = float(np.mean(sizes))
    max_size = float(np.max(sizes))
    min_size = float(np.min(sizes))
    empty_frac = empty_count / n_sets

    avg_ok = avg_size <= max_avg_size
    empty_ok = empty_frac <= max_empty_frac
    passed = avg_ok and empty_ok

    parts: list[str] = []
    if not avg_ok:
        parts.append(
            f"avg_size={avg_size:.4f} > max_avg_size={max_avg_size}"
        )
    if not empty_ok:
        parts.append(
            f"empty_frac={empty_frac:.4f} > max_empty_frac={max_empty_frac}"
        )

    if passed:
        message = (
            f"avg_size={avg_size:.4f} <= {max_avg_size}, "
            f"empty_frac={empty_frac:.4f} <= {max_empty_frac}"
        )
    else:
        message = "; ".join(parts)

    return assert_true(
        passed,
        name="model.prediction_set_size",
        message=message,
        severity=severity,
        avg_size=avg_size,
        max_size=max_size,
        min_size=min_size,
        empty_count=empty_count,
        empty_frac=empty_frac,
        n_sets=n_sets,
    )
