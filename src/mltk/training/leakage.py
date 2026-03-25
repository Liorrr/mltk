"""Data and feature leakage detection — the #1 cause of "works in dev, fails in prod."

Data leakage produces artificially inflated metrics. A model that appears 99% accurate
in development may drop to 60% in production because it memorized test data or
used target-correlated features that aren't available at inference time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mltk.core.assertion import assert_true, timed_assertion
from mltk.core.result import Severity, TestResult


@timed_assertion
def assert_no_train_test_overlap(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    key_cols: list[str],
) -> TestResult:
    """Assert zero row overlap between train and test sets on key columns.

    Args:
        train_df: Training DataFrame.
        test_df: Test DataFrame.
        key_cols: Columns to check for overlap (e.g., ["user_id", "timestamp"]).

    Returns:
        TestResult with overlap count.

    Example:
        >>> assert_no_train_test_overlap(train, test, key_cols=["user_id"])
    """
    train_keys = set(train_df[key_cols].apply(tuple, axis=1))
    test_keys = set(test_df[key_cols].apply(tuple, axis=1))
    overlap = train_keys & test_keys
    overlap_count = len(overlap)

    passed = overlap_count == 0
    message = (
        f"No overlap: {len(train_keys)} train, {len(test_keys)} test rows on {key_cols}"
        if passed
        else f"Data leakage: {overlap_count} rows overlap on {key_cols}"
    )

    return assert_true(
        passed, name="training.no_overlap", message=message,
        severity=Severity.CRITICAL,
        overlap_count=overlap_count,
        train_rows=len(train_df), test_rows=len(test_df),
        key_cols=key_cols,
    )


@timed_assertion
def assert_temporal_split(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    time_col: str,
) -> TestResult:
    """Assert train data is strictly before test data (no temporal leakage).

    Args:
        train_df: Training DataFrame.
        test_df: Test DataFrame.
        time_col: Name of the datetime/timestamp column.

    Returns:
        TestResult with max train time and min test time.

    Example:
        >>> assert_temporal_split(train, test, time_col="created_at")
    """
    train_max = pd.to_datetime(train_df[time_col]).max()
    test_min = pd.to_datetime(test_df[time_col]).min()

    passed = train_max < test_min
    message = (
        f"Temporal split OK: train max={train_max} < test min={test_min}"
        if passed
        else f"Temporal leakage: train max={train_max} >= test min={test_min}"
    )

    return assert_true(
        passed, name="training.temporal_split", message=message,
        severity=Severity.CRITICAL,
        train_max=str(train_max), test_min=str(test_min),
    )


@timed_assertion
def assert_no_target_leakage(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: list[str] | None = None,
    corr_threshold: float = 0.95,
) -> TestResult:
    """Assert no feature is suspiciously correlated with the target.

    A feature with >0.95 correlation to the target is likely a proxy
    for the target itself (e.g., "treatment_outcome" used to predict "diagnosis").

    Args:
        df: Full DataFrame with features and target.
        target_col: Name of the target column.
        feature_cols: Feature columns to check. None = all numeric except target.
        corr_threshold: Maximum allowed absolute correlation.

    Returns:
        TestResult with high-correlation features.

    Example:
        >>> assert_no_target_leakage(df, target_col="label", corr_threshold=0.95)
    """
    if feature_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [c for c in numeric_cols if c != target_col]

    if target_col not in df.columns:
        return assert_true(
            False, name="training.target_leakage",
            message=f"Target column '{target_col}' not found",
            severity=Severity.CRITICAL,
        )

    leaky_features: dict[str, float] = {}
    for col in feature_cols:
        if col not in df.columns:
            continue
        corr = abs(float(df[col].corr(df[target_col])))
        if corr >= corr_threshold:
            leaky_features[col] = round(corr, 4)

    passed = len(leaky_features) == 0
    message = (
        f"No target leakage in {len(feature_cols)} features (threshold={corr_threshold})"
        if passed
        else f"Target leakage: {list(leaky_features.keys())} correlated "
        f">= {corr_threshold} with '{target_col}'"
    )

    return assert_true(
        passed, name="training.target_leakage", message=message,
        severity=Severity.CRITICAL,
        leaky_features=leaky_features,
        features_checked=len(feature_cols),
        corr_threshold=corr_threshold,
    )
