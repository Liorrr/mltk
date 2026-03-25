"""Column-level statistical assertions — mean, median, stdev, quantiles.

Statistical assertions catch distribution shape problems that range checks miss:
a column can have all values in [0, 100] but still have a wildly wrong mean
(all values near 0 when they should cluster around 50), a skewed median,
or a stdev so small that the feature carries no signal.

These assertions are especially useful after feature engineering steps where
transformations can silently shift the distribution.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mltk.core.assertion import assert_true, timed_assertion
from mltk.core.result import Severity, TestResult


@timed_assertion
def assert_column_mean(
    df: pd.DataFrame,
    column: str,
    min_val: float | None = None,
    max_val: float | None = None,
) -> TestResult:
    """Assert column mean is within [min_val, max_val]. At least one bound required.

    Args:
        df: DataFrame to validate.
        column: Name of the column to check.
        min_val: Lower bound for the mean (inclusive). Optional if max_val is given.
        max_val: Upper bound for the mean (inclusive). Optional if min_val is given.

    Returns:
        TestResult with actual mean value.

    Raises:
        ValueError: If neither min_val nor max_val is provided.

    Example:
        >>> assert_column_mean(df, "age", min_val=18.0, max_val=65.0)
        >>> assert_column_mean(df, "score", min_val=0.4)  # at least 0.4 average
    """
    if min_val is None and max_val is None:
        raise ValueError(
            "assert_column_mean requires at least one bound (min_val or max_val)"
        )

    actual_mean = float(df[column].mean())
    name = f"data.column_mean[{column}]"

    too_low = min_val is not None and actual_mean < min_val
    too_high = max_val is not None and actual_mean > max_val
    passed = not too_low and not too_high

    bounds_str = _bounds_str(min_val, max_val)
    message = (
        f"mean={actual_mean:.4f} is within {bounds_str}"
        if passed
        else f"mean={actual_mean:.4f} is outside {bounds_str}"
    )

    return assert_true(
        passed,
        name=name,
        message=message,
        severity=Severity.CRITICAL,
        column=column,
        actual_mean=actual_mean,
        min_val=min_val,
        max_val=max_val,
    )


@timed_assertion
def assert_column_median(
    df: pd.DataFrame,
    column: str,
    min_val: float | None = None,
    max_val: float | None = None,
) -> TestResult:
    """Assert column median is within [min_val, max_val]. At least one bound required.

    The median is more robust to outliers than the mean. Use this when you care
    about the typical value rather than the average, or when outliers are expected.

    Args:
        df: DataFrame to validate.
        column: Name of the column to check.
        min_val: Lower bound for the median (inclusive). Optional if max_val is given.
        max_val: Upper bound for the median (inclusive). Optional if min_val is given.

    Returns:
        TestResult with actual median value.

    Raises:
        ValueError: If neither min_val nor max_val is provided.

    Example:
        >>> assert_column_median(df, "income", min_val=30000.0, max_val=80000.0)
    """
    if min_val is None and max_val is None:
        raise ValueError(
            "assert_column_median requires at least one bound (min_val or max_val)"
        )

    actual_median = float(df[column].median())
    name = f"data.column_median[{column}]"

    too_low = min_val is not None and actual_median < min_val
    too_high = max_val is not None and actual_median > max_val
    passed = not too_low and not too_high

    bounds_str = _bounds_str(min_val, max_val)
    message = (
        f"median={actual_median:.4f} is within {bounds_str}"
        if passed
        else f"median={actual_median:.4f} is outside {bounds_str}"
    )

    return assert_true(
        passed,
        name=name,
        message=message,
        severity=Severity.CRITICAL,
        column=column,
        actual_median=actual_median,
        min_val=min_val,
        max_val=max_val,
    )


@timed_assertion
def assert_column_stdev(
    df: pd.DataFrame,
    column: str,
    min_val: float | None = None,
    max_val: float | None = None,
) -> TestResult:
    """Assert column standard deviation is within [min_val, max_val].

    Standard deviation bounds catch two failure modes:
    - stdev too low: feature is nearly constant, carries no signal.
    - stdev too high: distribution is unusually wide, may contain outliers or
      mixed populations.

    At least one bound required.

    Args:
        df: DataFrame to validate.
        column: Name of the column to check.
        min_val: Minimum allowed stdev. Optional if max_val is given.
        max_val: Maximum allowed stdev. Optional if min_val is given.

    Returns:
        TestResult with actual stdev value.

    Raises:
        ValueError: If neither min_val nor max_val is provided.

    Example:
        >>> assert_column_stdev(df, "response_time_ms", min_val=1.0, max_val=500.0)
        >>> assert_column_stdev(df, "normalized_score", max_val=0.5)
    """
    if min_val is None and max_val is None:
        raise ValueError(
            "assert_column_stdev requires at least one bound (min_val or max_val)"
        )

    actual_stdev = float(df[column].std())
    name = f"data.column_stdev[{column}]"

    too_low = min_val is not None and actual_stdev < min_val
    too_high = max_val is not None and actual_stdev > max_val
    passed = not too_low and not too_high

    bounds_str = _bounds_str(min_val, max_val)
    message = (
        f"stdev={actual_stdev:.4f} is within {bounds_str}"
        if passed
        else f"stdev={actual_stdev:.4f} is outside {bounds_str}"
    )

    return assert_true(
        passed,
        name=name,
        message=message,
        severity=Severity.CRITICAL,
        column=column,
        actual_stdev=actual_stdev,
        min_val=min_val,
        max_val=max_val,
    )


@timed_assertion
def assert_quantiles(
    df: pd.DataFrame,
    column: str,
    quantiles: dict[float, tuple[float, float]],
) -> TestResult:
    """Assert column quantile values are within specified bounds.

    Quantile assertions are the most expressive statistical check: they let you
    specify the expected shape of the distribution at multiple points simultaneously.
    A distribution that passes mean and stdev checks can still be wildly wrong
    at the tails — quantile checks catch that.

    Args:
        df: DataFrame to validate.
        column: Name of the column to check.
        quantiles: Mapping of quantile level → (min_bound, max_bound).
            Example: {0.25: (10, 30), 0.75: (50, 80), 0.95: (80, 100)}
            Each quantile value must fall within [min_bound, max_bound].

    Returns:
        TestResult with per-quantile actual values and which ones failed.

    Example:
        >>> assert_quantiles(df, "age", {
        ...     0.25: (20, 35),
        ...     0.50: (30, 50),
        ...     0.75: (45, 65),
        ...     0.95: (60, 80),
        ... })
    """
    if df.empty:
        return assert_true(
            False,
            name=f"data.quantiles[{column}]",
            message=f"Cannot check quantiles on empty DataFrame (column={column})",
            severity=Severity.CRITICAL,
            column=column,
            row_count=0,
        )

    series = df[column].dropna()
    actual_values: dict[float, float] = {}
    failures: list[str] = []

    for q, (lo, hi) in quantiles.items():
        actual = float(np.quantile(series, q))
        actual_values[q] = actual
        if actual < lo or actual > hi:
            failures.append(
                f"Q{q:.2f}={actual:.4f} outside [{lo}, {hi}]"
            )

    passed = len(failures) == 0
    name = f"data.quantiles[{column}]"

    if passed:
        qs_checked = ", ".join(f"Q{q:.2f}" for q in quantiles)
        message = f"All {len(quantiles)} quantile checks passed ({qs_checked})"
    else:
        detail = "; ".join(failures)
        message = f"{len(failures)}/{len(quantiles)} quantile(s) out of bounds: {detail}"

    return assert_true(
        passed,
        name=name,
        message=message,
        severity=Severity.CRITICAL,
        column=column,
        quantiles_checked=len(quantiles),
        failures=len(failures),
        actual_values={f"Q{k:.2f}": v for k, v in actual_values.items()},
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _bounds_str(min_val: float | None, max_val: float | None) -> str:
    """Format a human-readable bounds string like '[min, max]' or '[min, ∞)'."""
    lo = str(min_val) if min_val is not None else "-∞"
    hi = str(max_val) if max_val is not None else "∞"
    return f"[{lo}, {hi}]"
