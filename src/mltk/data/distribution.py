"""Data distribution validation — verify statistical properties of features.

Distribution tests catch subtle data issues that schema tests miss:
a column can have the right name and type but contain wildly wrong values.
These assertions ensure numeric ranges, uniqueness, and outlier bounds
before data enters the training pipeline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mltk.core.assertion import assert_true, timed_assertion
from mltk.core.result import Severity, TestResult


@timed_assertion
def assert_range(
    series: pd.Series,
    min_val: float,
    max_val: float,
) -> TestResult:
    """Assert all values in a Series fall within [min_val, max_val].

    Args:
        series: Pandas Series to validate.
        min_val: Minimum allowed value (inclusive).
        max_val: Maximum allowed value (inclusive).

    Returns:
        TestResult with actual min/max and violation count.

    Example:
        >>> assert_range(df["age"], min_val=0, max_val=150)
        >>> assert_range(df["probability"], min_val=0.0, max_val=1.0)
    """
    actual_min = float(series.min())
    actual_max = float(series.max())
    below = int((series < min_val).sum())
    above = int((series > max_val).sum())
    violations = below + above

    passed = violations == 0
    name = f"data.range[{series.name}]" if series.name else "data.range"
    message = (
        f"All {len(series)} values in [{min_val}, {max_val}]"
        if passed
        else f"{violations} values outside [{min_val}, {max_val}] "
        f"(min={actual_min}, max={actual_max})"
    )

    return assert_true(
        passed,
        name=name,
        message=message,
        severity=Severity.CRITICAL,
        actual_min=actual_min,
        actual_max=actual_max,
        below_count=below,
        above_count=above,
    )


@timed_assertion
def assert_unique(
    df: pd.DataFrame,
    columns: list[str],
) -> TestResult:
    """Assert no duplicate values across specified columns.

    For a single column, checks that all values are unique.
    For multiple columns, checks that the combination is unique (composite key).

    Args:
        df: DataFrame to validate.
        columns: Column(s) to check for uniqueness.

    Returns:
        TestResult with duplicate count.

    Example:
        >>> assert_unique(df, columns=["user_id"])
        >>> assert_unique(df, columns=["date", "store_id"])  # composite key
    """
    duplicates = df.duplicated(subset=columns, keep=False)
    dup_count = int(duplicates.sum())

    passed = dup_count == 0
    col_str = ", ".join(columns)
    message = (
        f"All {len(df)} rows unique on [{col_str}]"
        if passed
        else f"{dup_count} duplicate rows on [{col_str}]"
    )

    return assert_true(
        passed,
        name="data.unique",
        message=message,
        severity=Severity.CRITICAL,
        columns=columns,
        duplicate_count=dup_count,
        total_rows=len(df),
    )


@timed_assertion
def assert_no_outliers(
    series: pd.Series,
    method: str = "iqr",
    threshold: float = 1.5,
) -> TestResult:
    """Assert no statistical outliers in a numeric Series.

    Uses the IQR (Interquartile Range) method by default:
    outlier if value < Q1 - threshold*IQR or value > Q3 + threshold*IQR.

    Args:
        series: Numeric Series to validate.
        method: Outlier detection method ("iqr" supported).
        threshold: IQR multiplier (1.5 = standard, 3.0 = extreme only).

    Returns:
        TestResult with outlier count and bounds.

    Example:
        >>> assert_no_outliers(df["salary"], threshold=1.5)
        >>> assert_no_outliers(df["temperature"], threshold=3.0)  # only flag extreme
    """
    if method != "iqr":
        return assert_true(
            False,
            name="data.no_outliers",
            message=f"Unknown method: {method}. Supported: 'iqr'",
            severity=Severity.CRITICAL,
        )

    q1 = float(np.percentile(series.dropna(), 25))
    q3 = float(np.percentile(series.dropna(), 75))
    iqr = q3 - q1
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr

    outliers = ((series < lower_bound) | (series > upper_bound)).sum()
    outlier_count = int(outliers)

    passed = outlier_count == 0
    name = f"data.no_outliers[{series.name}]" if series.name else "data.no_outliers"
    message = (
        f"No outliers (IQR bounds: [{lower_bound:.2f}, {upper_bound:.2f}])"
        if passed
        else f"{outlier_count} outlier(s) outside [{lower_bound:.2f}, {upper_bound:.2f}]"
    )

    return assert_true(
        passed,
        name=name,
        message=message,
        severity=Severity.WARNING,
        method=method,
        threshold=threshold,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        outlier_count=outlier_count,
        q1=q1,
        q3=q3,
        iqr=iqr,
    )
