"""Data augmentation validation — ensure test data isn't augmented.

Augmentation bugs are subtle: developers often accidentally apply training-time
augmentation pipelines to test sets, or augment with class-imbalanced synthetic
data that shifts the label distribution. Both inflate eval metrics and cause
silent production regressions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mltk.core.assertion import assert_true, timed_assertion
from mltk.core.result import Severity, TestResult


@timed_assertion
def assert_no_augmentation_on_test(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    key_cols: list[str] | None = None,
    max_dup_ratio: float = 0.01,
) -> TestResult:
    """Assert test set was NOT augmented (no suspicious duplicate patterns).

    Checks the duplicate row ratio in the test set against a threshold. If the
    test set has more than ``max_dup_ratio`` duplicate rows (on ``key_cols`` if
    provided, otherwise on all columns), it is likely that an augmentation
    pipeline was mistakenly applied to it — a common ML bug.

    Args:
        train_df: Training DataFrame (used only for context in result details).
        test_df: Test DataFrame to inspect for duplicate rows.
        key_cols: Columns to check for duplicates. None = all columns.
        max_dup_ratio: Maximum allowed fraction of duplicate rows in test set.
            Default 0.01 (1%).

    Returns:
        TestResult with duplicate count and ratio.

    Example:
        >>> assert_no_augmentation_on_test(train, test, key_cols=["id"])
    """
    if len(test_df) == 0:
        return assert_true(
            True,
            name="training.no_augmentation_on_test",
            message="Test set is empty — no duplicates possible",
            severity=Severity.WARNING,
            test_rows=0,
            dup_count=0,
            dup_ratio=0.0,
        )

    cols = key_cols if key_cols is not None else test_df.columns.tolist()
    # Only check columns that actually exist in the dataframe
    cols = [c for c in cols if c in test_df.columns]

    if not cols:
        return assert_true(
            False,
            name="training.no_augmentation_on_test",
            message=f"None of key_cols {key_cols} found in test DataFrame",
            severity=Severity.CRITICAL,
        )

    dup_mask = test_df.duplicated(subset=cols, keep="first")
    dup_count = int(dup_mask.sum())
    dup_ratio = dup_count / len(test_df)

    passed = dup_ratio <= max_dup_ratio
    message = (
        f"Test set clean: {dup_count} duplicate rows "
        f"({dup_ratio:.2%} <= {max_dup_ratio:.2%} threshold)"
        if passed
        else f"Test set likely augmented: {dup_count} duplicate rows "
        f"({dup_ratio:.2%} > {max_dup_ratio:.2%} threshold) — "
        f"augmentation pipeline may have been applied to test data"
    )

    return assert_true(
        passed,
        name="training.no_augmentation_on_test",
        message=message,
        severity=Severity.CRITICAL,
        dup_count=dup_count,
        dup_ratio=round(dup_ratio, 4),
        max_dup_ratio=max_dup_ratio,
        test_rows=len(test_df),
        train_rows=len(train_df),
        key_cols=cols,
    )


@timed_assertion
def assert_augmentation_preserves_signal(
    original: pd.DataFrame,
    augmented: pd.DataFrame,
    label_col: str,
    max_distribution_shift: float = 0.1,
) -> TestResult:
    """Assert augmented data preserves label distribution.

    Compares the label distribution between the original and augmented DataFrames.
    A total variation distance greater than ``max_distribution_shift`` indicates
    that the augmentation strategy introduced class imbalance or bias — for
    example, oversampling only the minority class without proportional resampling.

    Total variation distance = 0.5 * sum(|p_orig - p_aug|) for each label.
    This is bounded [0, 1]: 0 means identical distributions, 1 means disjoint.

    Args:
        original: Original (pre-augmentation) DataFrame.
        augmented: Augmented DataFrame (should include original rows too, typically).
        label_col: Name of the label/target column to compare.
        max_distribution_shift: Maximum allowed total variation distance.
            Default 0.1 (10% shift).

    Returns:
        TestResult with per-label distribution details and shift magnitude.

    Example:
        >>> assert_augmentation_preserves_signal(original, augmented, label_col="y")
    """
    if label_col not in original.columns:
        return assert_true(
            False,
            name="training.augmentation_preserves_signal",
            message=f"Label column '{label_col}' not found in original DataFrame",
            severity=Severity.CRITICAL,
        )

    if label_col not in augmented.columns:
        return assert_true(
            False,
            name="training.augmentation_preserves_signal",
            message=f"Label column '{label_col}' not found in augmented DataFrame",
            severity=Severity.CRITICAL,
        )

    if len(original) == 0 or len(augmented) == 0:
        return assert_true(
            True,
            name="training.augmentation_preserves_signal",
            message="One or both DataFrames are empty — skipping distribution check",
            severity=Severity.WARNING,
            original_rows=len(original),
            augmented_rows=len(augmented),
        )

    # Compute normalized label frequencies for both sets
    orig_counts = original[label_col].value_counts(normalize=True)
    aug_counts = augmented[label_col].value_counts(normalize=True)

    # Union of all labels seen in either set
    all_labels = orig_counts.index.union(aug_counts.index)
    orig_dist = orig_counts.reindex(all_labels, fill_value=0.0)
    aug_dist = aug_counts.reindex(all_labels, fill_value=0.0)

    # Total variation distance: 0.5 * L1 distance between distributions
    tv_distance = float(0.5 * np.abs(orig_dist.values - aug_dist.values).sum())

    passed = tv_distance <= max_distribution_shift
    message = (
        f"Augmentation preserves label distribution: "
        f"TV distance={tv_distance:.4f} <= {max_distribution_shift} threshold"
        if passed
        else f"Augmentation shifted label distribution: "
        f"TV distance={tv_distance:.4f} > {max_distribution_shift} threshold — "
        f"augmentation may have introduced class imbalance"
    )

    # Build per-label shift details for diagnostics
    per_label_shift = {
        str(label): round(float(abs(orig_dist[label] - aug_dist[label])), 4)
        for label in all_labels
    }

    return assert_true(
        passed,
        name="training.augmentation_preserves_signal",
        message=message,
        severity=Severity.CRITICAL,
        tv_distance=round(tv_distance, 4),
        max_distribution_shift=max_distribution_shift,
        original_rows=len(original),
        augmented_rows=len(augmented),
        labels_checked=len(all_labels),
        per_label_shift=per_label_shift,
    )
