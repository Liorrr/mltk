"""Data validation assertions — format, set membership, conflicting labels.

These assertions target semantic correctness: values can be the right type
and within range but still be logically wrong (dates in the wrong format,
categories outside the agreed vocabulary, or identical feature rows mapped to
different labels).

All three checks should run before any training pipeline ingests data.
"""

from __future__ import annotations

import pandas as pd

from mltk.core.assertion import assert_true, timed_assertion
from mltk.core.result import Severity, TestResult


@timed_assertion
def assert_datetime_format(
    df: pd.DataFrame,
    column: str,
    fmt: str = "%Y-%m-%d",
) -> TestResult:
    """Assert all values in column match the specified datetime format.

    Uses pd.to_datetime with the given format string. Values that cannot be
    parsed are counted as violations. Reports the count of invalid rows.

    Args:
        df: DataFrame to validate.
        column: Name of the column containing date/datetime strings.
        format: strftime format string (default: "%Y-%m-%d").

    Returns:
        TestResult with invalid_count and total_rows.

    Example:
        >>> assert_datetime_format(df, "created_at")
        >>> assert_datetime_format(df, "event_ts", format="%Y-%m-%d %H:%M:%S")
    """
    series = df[column]
    total = len(series)

    def _is_invalid(val: object) -> bool:
        try:
            pd.to_datetime(val, format=fmt)
            return False
        except (ValueError, TypeError):
            return True

    invalid_mask = series.apply(_is_invalid)
    invalid_count = int(invalid_mask.sum())
    passed = invalid_count == 0
    name = f"data.datetime_format[{column}]"

    message = (
        f"All {total} values match format '{fmt}'"
        if passed
        else f"{invalid_count}/{total} values do not match format '{fmt}'"
    )

    return assert_true(
        passed,
        name=name,
        message=message,
        severity=Severity.CRITICAL,
        column=column,
        fmt=fmt,
        invalid_count=invalid_count,
        total_rows=total,
    )


@timed_assertion
def assert_values_in_set(
    df: pd.DataFrame,
    column: str,
    allowed_values: set | list,
) -> TestResult:
    """Assert all values in column are members of the allowed set.

    Catches vocabulary drift: a model trained on ["cat", "dog"] will fail if
    production data starts sending "Cat" or "bird". This assertion enforces
    the agreed categorical vocabulary.

    Args:
        df: DataFrame to validate.
        column: Name of the column to check.
        allowed_values: Iterable of permitted values (converted to a set internally).

    Returns:
        TestResult with invalid_count and up to 10 sample invalid values.

    Example:
        >>> assert_values_in_set(df, "status", allowed_values={"active", "inactive", "pending"})
        >>> assert_values_in_set(df, "country_code", allowed_values=["US", "GB", "DE", "FR"])
    """
    allowed_set = set(allowed_values)
    series = df[column]
    total = len(series)

    invalid_mask = ~series.isin(allowed_set)
    invalid_count = int(invalid_mask.sum())
    passed = invalid_count == 0
    name = f"data.values_in_set[{column}]"

    invalid_samples = sorted(
        str(v) for v in series[invalid_mask].unique()[:10]
    )

    message = (
        f"All {total} values are in the allowed set ({len(allowed_set)} values)"
        if passed
        else (
            f"{invalid_count}/{total} values not in allowed set. "
            f"Invalid samples: {invalid_samples}"
        )
    )

    return assert_true(
        passed,
        name=name,
        message=message,
        severity=Severity.CRITICAL,
        column=column,
        allowed_count=len(allowed_set),
        invalid_count=invalid_count,
        total_rows=total,
        invalid_samples=invalid_samples,
    )


@timed_assertion
def assert_no_conflicting_labels(
    df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
) -> TestResult:
    """Assert no rows have identical features but different labels.

    Conflicting labels are a data quality bug: the same input is mapped to
    multiple different outputs. This is a hard contradiction that a supervised
    model cannot resolve — it will produce inconsistent predictions and inflate
    training loss on those examples.

    Groups by feature_cols, checks that each group has exactly one unique label.
    Any group with 2+ distinct labels is a conflict.

    Args:
        df: DataFrame to validate.
        feature_cols: List of feature column names that together define an input.
        label_col: Name of the label/target column.

    Returns:
        TestResult with conflict_count (number of conflicting feature groups)
        and total_groups checked.

    Example:
        >>> assert_no_conflicting_labels(
        ...     df,
        ...     feature_cols=["age", "income", "region"],
        ...     label_col="churn",
        ... )
    """
    if df.empty:
        return assert_true(
            True,
            name="data.no_conflicting_labels",
            message="No rows to check (empty DataFrame)",
            severity=Severity.CRITICAL,
            feature_cols=feature_cols,
            label_col=label_col,
            conflict_count=0,
            total_groups=0,
        )

    grouped = df.groupby(feature_cols, sort=False)[label_col].nunique()
    conflicting = grouped[grouped > 1]
    conflict_count = int(len(conflicting))
    total_groups = int(len(grouped))
    passed = conflict_count == 0
    name = "data.no_conflicting_labels"

    cols_str = ", ".join(feature_cols)
    message = (
        f"No label conflicts across {total_groups} unique feature group(s) "
        f"(features: [{cols_str}])"
        if passed
        else (
            f"{conflict_count}/{total_groups} feature group(s) have conflicting labels "
            f"in '{label_col}' (features: [{cols_str}])"
        )
    )

    return assert_true(
        passed,
        name=name,
        message=message,
        severity=Severity.CRITICAL,
        feature_cols=feature_cols,
        label_col=label_col,
        conflict_count=conflict_count,
        total_groups=total_groups,
    )
