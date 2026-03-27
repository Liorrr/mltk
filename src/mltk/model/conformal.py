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


@timed_assertion
def assert_conformal_calibration(
    y_true: np.ndarray,
    y_lower: np.ndarray,
    y_upper: np.ndarray,
    nominal_coverage: float = 0.9,
    tolerance: float = 0.02,
    severity: Severity = Severity.CRITICAL,
) -> TestResult:
    """Assert that empirical coverage is close to the promised nominal level.

    Conformal prediction methods promise a specific coverage level (e.g.,
    "90% of true values will fall inside the interval"). This assertion
    checks whether that promise is *actually* met -- in both directions.

    **Why two-sided?**  A conformal method that promises 90% coverage but
    delivers 99% is also miscalibrated: it is being overly conservative,
    producing intervals that are wider than necessary.  An under-covering
    method (e.g., 82% when 90% was promised) fails to deliver on the
    statistical guarantee.

    This is subtly different from ``assert_interval_coverage``:

    - ``assert_interval_coverage`` asks: "Is coverage good enough?"
      (one-sided, flexible threshold).
    - ``assert_conformal_calibration`` asks: "Is coverage close to what
      was *promised*?" (two-sided, tight around the nominal value).

    Common causes of miscalibration:

    - Calibration set too small (finite-sample correction not applied).
    - Data is non-exchangeable (e.g., time series without proper splitting).
    - Nonconformity score poorly chosen for the data distribution.
    - Distribution shift between calibration and test sets.

    Args:
        y_true: Ground truth values.
        y_lower: Lower bounds of prediction intervals.
        y_upper: Upper bounds of prediction intervals.
        nominal_coverage: The promised coverage level (default 0.9).
        tolerance: Maximum allowed absolute deviation from nominal
            (default 0.02).  The assertion passes when
            ``|empirical - nominal| <= tolerance``.
        severity: Severity level for the assertion (default CRITICAL).

    Returns:
        TestResult with empirical_coverage, nominal_coverage, tolerance,
        deviation, n_total, and direction ("over", "under", or
        "calibrated") in details.

    Example:
        >>> y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        >>> lo = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
        >>> hi = np.array([1.5, 2.5, 3.5, 4.5, 5.5])
        >>> assert_conformal_calibration(y, lo, hi, nominal_coverage=0.9)
    """
    y_t = np.asarray(y_true, dtype=np.float64)
    y_lo = np.asarray(y_lower, dtype=np.float64)
    y_hi = np.asarray(y_upper, dtype=np.float64)

    n_total = len(y_t)

    if n_total == 0:
        return assert_true(
            False,
            name="model.conformal_calibration",
            message="Cannot compute calibration on empty arrays",
            severity=severity,
        )

    covered = (y_t >= y_lo) & (y_t <= y_hi)
    n_covered = int(np.sum(covered))
    empirical_coverage = n_covered / n_total

    deviation = abs(empirical_coverage - nominal_coverage)

    if empirical_coverage > nominal_coverage:
        direction = "over"
    elif empirical_coverage < nominal_coverage:
        direction = "under"
    else:
        direction = "calibrated"

    passed = deviation <= tolerance

    if passed:
        message = (
            f"calibrated: |{empirical_coverage:.4f} - "
            f"{nominal_coverage}| = {deviation:.4f} <= {tolerance}"
        )
    else:
        message = (
            f"miscalibrated ({direction}): |{empirical_coverage:.4f} - "
            f"{nominal_coverage}| = {deviation:.4f} > {tolerance}"
        )

    return assert_true(
        passed,
        name="model.conformal_calibration",
        message=message,
        severity=severity,
        empirical_coverage=empirical_coverage,
        nominal_coverage=nominal_coverage,
        tolerance=tolerance,
        deviation=deviation,
        n_total=n_total,
        direction=direction,
    )


@timed_assertion
def assert_conditional_coverage(
    y_true: np.ndarray,
    y_lower: np.ndarray,
    y_upper: np.ndarray,
    groups: np.ndarray | list,
    nominal_coverage: float = 0.9,
    min_group_coverage: float = 0.8,
    min_group_size: int = 10,
    severity: Severity = Severity.CRITICAL,
) -> TestResult:
    """Assert that prediction intervals achieve adequate coverage per group.

    Marginal (overall) coverage can hide fairness problems.  A model may
    achieve 95% coverage on majority groups but only 60% on minority
    groups; the overall average looks fine while the minority group is
    badly served.  This is the "fairness-coverage gap" in conformal
    prediction.

    Conditional coverage (also called *Mondrian conformal prediction*)
    checks coverage **separately for each group**.  It is essential in
    regulated domains:

    - **Healthcare**: Coverage by disease subtype or patient demographic.
    - **Finance**: Coverage by income bracket or geographic region.
    - **Any fairness-sensitive application**: Per-group guarantees matter
      more than aggregate statistics.

    Groups with fewer than ``min_group_size`` samples are skipped because
    coverage estimates are unreliable at very small sample sizes.

    Args:
        y_true: Ground truth values.
        y_lower: Lower bounds of prediction intervals.
        y_upper: Upper bounds of prediction intervals.
        groups: Group labels (same length as y_true).  Each unique value
            defines a group for per-group coverage analysis.
        nominal_coverage: The nominal coverage level for reference
            (default 0.9).  Included in details but not used in the
            pass/fail decision (``min_group_coverage`` is the gate).
        min_group_coverage: Minimum required coverage for each group
            (default 0.8).  The assertion fails if any sufficiently-large
            group falls below this level.
        min_group_size: Minimum number of samples for a group to be
            evaluated (default 10).  Smaller groups are reported in
            ``groups_skipped`` but do not affect the pass/fail outcome.
        severity: Severity level for the assertion (default CRITICAL).

    Returns:
        TestResult with per_group (dict mapping group label to
        {coverage, n_covered, n_total}), worst_group, worst_coverage,
        groups_below_threshold (list), and groups_skipped (list) in
        details.

    Example:
        >>> y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
        >>> lo = y - 0.5
        >>> hi = y + 0.5
        >>> grp = np.array(["A"]*5 + ["B"]*5)
        >>> assert_conditional_coverage(y, lo, hi, grp, min_group_size=3)
    """
    y_t = np.asarray(y_true, dtype=np.float64)
    y_lo = np.asarray(y_lower, dtype=np.float64)
    y_hi = np.asarray(y_upper, dtype=np.float64)
    grp = np.asarray(groups)

    n_total = len(y_t)

    if n_total == 0:
        return assert_true(
            False,
            name="model.conditional_coverage",
            message="Cannot compute conditional coverage on empty arrays",
            severity=severity,
        )

    covered = (y_t >= y_lo) & (y_t <= y_hi)

    unique_groups = np.unique(grp)
    per_group: dict[str, dict[str, Any]] = {}
    groups_below: list[str] = []
    groups_skipped: list[str] = []

    worst_group: str | None = None
    worst_coverage: float = 1.0

    for g in unique_groups:
        g_label = str(g)
        mask = grp == g
        g_total = int(np.sum(mask))

        if g_total < min_group_size:
            groups_skipped.append(g_label)
            # Still record the group info for transparency
            g_covered = int(np.sum(covered[mask]))
            g_cov = g_covered / g_total if g_total > 0 else 0.0
            per_group[g_label] = {
                "coverage": g_cov,
                "n_covered": g_covered,
                "n_total": g_total,
            }
            continue

        g_covered = int(np.sum(covered[mask]))
        g_cov = g_covered / g_total

        per_group[g_label] = {
            "coverage": g_cov,
            "n_covered": g_covered,
            "n_total": g_total,
        }

        if g_cov <= worst_coverage:
            worst_coverage = g_cov
            worst_group = g_label

        if g_cov < min_group_coverage:
            groups_below.append(g_label)

    # If all groups were skipped (too small), there is nothing to evaluate.
    # We consider this a pass since no group violated the threshold.
    evaluated_groups = [g for g in unique_groups if str(g) not in groups_skipped]
    if len(evaluated_groups) == 0:
        # No group was large enough to evaluate
        worst_group = None
        worst_coverage = float("nan")

    passed = len(groups_below) == 0

    if passed:
        if worst_group is not None:
            message = (
                f"all groups above {min_group_coverage}: "
                f"worst={worst_group} at {worst_coverage:.4f}"
            )
        else:
            message = (
                f"no groups with >= {min_group_size} samples to evaluate"
            )
    else:
        message = (
            f"{len(groups_below)} group(s) below {min_group_coverage}: "
            f"{groups_below}; worst={worst_group} at {worst_coverage:.4f}"
        )

    return assert_true(
        passed,
        name="model.conditional_coverage",
        message=message,
        severity=severity,
        per_group=per_group,
        worst_group=worst_group,
        worst_coverage=worst_coverage,
        groups_below_threshold=groups_below,
        groups_skipped=groups_skipped,
    )
