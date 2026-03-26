"""Data freshness validation — verify data recency and size.

Stale data is a silent killer for ML models. A model trained on data from
6 months ago may perform well on the test set but fail in production because
the real world has changed (concept drift). Freshness assertions ensure
your training data and feature pipelines are up-to-date.

Row count validation catches data pipeline failures where a table that
should have millions of rows suddenly has zero (empty extract) or
unexpectedly few (partial load).
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from mltk.core.assertion import assert_true, timed_assertion
from mltk.core.result import Severity, TestResult


@timed_assertion
def assert_freshness(
    df: pd.DataFrame,
    date_column: str,
    max_age_days: int,
    reference_date: datetime | None = None,
) -> TestResult:
    """Assert the most recent date in a column is within max_age_days.

    Args:
        df: DataFrame to validate.
        date_column: Name of the datetime column.
        max_age_days: Maximum allowed age in days.
        reference_date: Compare against this date (default: now).

    Returns:
        TestResult with age details.

    Example:
        >>> assert_freshness(df, date_column="created_at", max_age_days=7)
    """
    if date_column not in df.columns:
        return assert_true(
            False,
            name="data.freshness",
            message=f"Column '{date_column}' not found",
            severity=Severity.CRITICAL,
        )

    dates = pd.to_datetime(df[date_column], errors="coerce")
    if dates.isnull().all():
        return assert_true(
            False,
            name="data.freshness",
            message=f"No valid dates in '{date_column}'",
            severity=Severity.CRITICAL,
        )

    most_recent = dates.max()
    ref = reference_date or datetime.now()
    # Handle timezone-aware dates: convert to UTC then strip tz so subtraction works
    if most_recent.tzinfo is not None and ref.tzinfo is None:
        most_recent = most_recent.tz_convert(None)

    age = ref - most_recent.to_pydatetime()
    age_days = age.days

    passed = age_days <= max_age_days
    message = (
        f"Data is {age_days} day(s) old (limit: {max_age_days})"
        if passed
        else f"Data is {age_days} day(s) old, exceeds limit of {max_age_days} days"
    )

    return assert_true(
        passed,
        name="data.freshness",
        message=message,
        severity=Severity.CRITICAL,
        age_days=age_days,
        max_age_days=max_age_days,
        most_recent=str(most_recent),
        reference_date=str(ref),
    )


@timed_assertion
def assert_row_count(
    df: pd.DataFrame,
    min_rows: int | None = None,
    max_rows: int | None = None,
) -> TestResult:
    """Assert DataFrame row count is within bounds.

    Args:
        df: DataFrame to validate.
        min_rows: Minimum expected rows (inclusive). None = no lower bound.
        max_rows: Maximum expected rows (inclusive). None = no upper bound.

    Returns:
        TestResult with actual count.

    Example:
        >>> assert_row_count(df, min_rows=1000)                # at least 1000
        >>> assert_row_count(df, min_rows=100, max_rows=50000) # between 100 and 50k
    """
    count = len(df)
    errors = []

    if min_rows is not None and count < min_rows:
        errors.append(f"Row count {count} below minimum {min_rows}")
    if max_rows is not None and count > max_rows:
        errors.append(f"Row count {count} exceeds maximum {max_rows}")

    passed = len(errors) == 0
    message = f"Row count: {count}" if passed else "; ".join(errors)

    return assert_true(
        passed,
        name="data.row_count",
        message=message,
        severity=Severity.CRITICAL,
        row_count=count,
        min_rows=min_rows,
        max_rows=max_rows,
    )
