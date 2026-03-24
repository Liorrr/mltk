"""Data schema validation — verify DataFrame structure before it reaches the model.

Schema validation is the FIRST line of defense in ML data quality. If your data
doesn't have the right columns, types, or completeness, everything downstream
(feature engineering, training, inference) will silently produce garbage.

These assertions catch problems at the earliest possible point.
"""

from __future__ import annotations

import pandas as pd

from mltk.core.assertion import assert_true, timed_assertion
from mltk.core.result import Severity, TestResult


@timed_assertion
def assert_schema(
    df: pd.DataFrame,
    expected: dict[str, str],
    allow_extra_columns: bool = True,
) -> TestResult:
    """Assert DataFrame columns and dtypes match expected schema.

    Args:
        df: DataFrame to validate.
        expected: Dict mapping column names to dtype strings (e.g., {"id": "int64"}).
        allow_extra_columns: If False, fail when df has columns not in expected.

    Returns:
        TestResult with details about mismatches.

    Example:
        >>> assert_schema(df, {"id": "int64", "label": "int64", "text": "object"})
    """
    missing = set(expected.keys()) - set(df.columns)
    dtype_mismatches = {}

    for col, expected_dtype in expected.items():
        if col in df.columns:
            actual = str(df[col].dtype)
            if actual != expected_dtype:
                dtype_mismatches[col] = {"expected": expected_dtype, "actual": actual}

    extra = set(df.columns) - set(expected.keys()) if not allow_extra_columns else set()

    errors = []
    if missing:
        errors.append(f"Missing columns: {sorted(missing)}")
    if dtype_mismatches:
        for col, info in dtype_mismatches.items():
            errors.append(f"Column '{col}': expected {info['expected']}, got {info['actual']}")
    if extra:
        errors.append(f"Unexpected columns: {sorted(extra)}")

    passed = len(errors) == 0
    message = "Schema valid" if passed else "; ".join(errors)

    return assert_true(
        passed,
        name="data.schema",
        message=message,
        severity=Severity.CRITICAL,
        missing_columns=sorted(missing),
        dtype_mismatches=dtype_mismatches,
        extra_columns=sorted(extra),
    )


@timed_assertion
def assert_no_nulls(
    df: pd.DataFrame,
    columns: list[str] | None = None,
) -> TestResult:
    """Assert no null/NaN values in specified columns (or all columns).

    Args:
        df: DataFrame to validate.
        columns: Columns to check. If None, checks all columns.

    Returns:
        TestResult with null counts per column.

    Example:
        >>> assert_no_nulls(df, columns=["label", "feature_a"])
    """
    cols = columns if columns is not None else list(df.columns)
    null_counts = {col: int(df[col].isnull().sum()) for col in cols if df[col].isnull().any()}
    total_nulls = sum(null_counts.values())

    passed = total_nulls == 0
    message = (
        f"No nulls detected in {len(cols)} columns"
        if passed
        else f"{total_nulls} null(s) in columns: {list(null_counts.keys())}"
    )

    return assert_true(
        passed,
        name="data.no_nulls",
        message=message,
        severity=Severity.CRITICAL,
        null_counts=null_counts,
        columns_checked=cols,
    )


@timed_assertion
def assert_dtypes(
    df: pd.DataFrame,
    expected: dict[str, str],
) -> TestResult:
    """Assert exact dtype match for specified columns.

    Unlike assert_schema, this only checks dtypes for the listed columns
    without requiring all columns to be present in the expected dict.

    Args:
        df: DataFrame to validate.
        expected: Dict mapping column names to dtype strings.

    Returns:
        TestResult with mismatch details.

    Example:
        >>> assert_dtypes(df, {"score": "float64", "count": "int64"})
    """
    mismatches = {}
    missing = []

    for col, expected_dtype in expected.items():
        if col not in df.columns:
            missing.append(col)
        elif str(df[col].dtype) != expected_dtype:
            mismatches[col] = {"expected": expected_dtype, "actual": str(df[col].dtype)}

    passed = len(mismatches) == 0 and len(missing) == 0
    errors = []
    if missing:
        errors.append(f"Columns not found: {missing}")
    for col, info in mismatches.items():
        errors.append(f"'{col}': expected {info['expected']}, got {info['actual']}")

    message = "All dtypes match" if passed else "; ".join(errors)

    return assert_true(
        passed,
        name="data.dtypes",
        message=message,
        severity=Severity.CRITICAL,
        mismatches=mismatches,
        missing_columns=missing,
    )
