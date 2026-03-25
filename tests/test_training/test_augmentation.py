"""Tests for mltk.training.augmentation — augmentation validation assertions."""

import numpy as np
import pandas as pd
import pytest

from mltk.core.assertion import MltkAssertionError
from mltk.training.augmentation import (
    assert_augmentation_preserves_signal,
    assert_no_augmentation_on_test,
)


class TestNoAugmentationOnTest:
    """Assert test sets have not been augmented (no suspicious duplicates)."""

    def test_no_augmentation_clean(self) -> None:
        # SCENARIO: Test set with entirely unique rows, no duplicates at all.
        # WHY: A properly held-out test set should never contain duplicate rows
        #      introduced by augmentation pipelines.
        # EXPECTED: Passes — duplicate ratio is 0, well below 1% threshold.
        train = pd.DataFrame({"id": [1, 2, 3], "x": [0.1, 0.2, 0.3]})
        test = pd.DataFrame({"id": [4, 5, 6], "x": [0.4, 0.5, 0.6]})
        result = assert_no_augmentation_on_test(train, test, key_cols=["id"])
        assert result.passed is True
        assert result.details["dup_count"] == 0
        assert result.details["dup_ratio"] == 0.0

    def test_no_augmentation_augmented(self) -> None:
        # SCENARIO: Test set has many duplicate rows — simulating an augmentation
        #           pipeline that was accidentally run on test data.
        # WHY: Augmentation tools (flip, rotate, noise) create near/exact duplicate
        #      keys; a ratio above 1% signals the pipeline touched test data.
        # EXPECTED: Fails with MltkAssertionError — duplicate ratio exceeds threshold.
        train = pd.DataFrame({"id": range(20), "x": range(20)})
        # 8 out of 10 test rows are duplicates of id=99 → 80% dup ratio
        test = pd.DataFrame({
            "id": [99, 99, 99, 99, 99, 99, 99, 99, 100, 101],
            "x":  [1,  1,  1,  1,  1,  1,  1,  1,  2,   3],
        })
        with pytest.raises(MltkAssertionError) as exc:
            assert_no_augmentation_on_test(train, test, key_cols=["id"], max_dup_ratio=0.01)
        assert "augmented" in str(exc.value).lower()

    def test_no_augmentation_key_cols_none_uses_all_columns(self) -> None:
        # SCENARIO: key_cols=None — deduplication checks all columns.
        # WHY: Verifies the None path works; a test row repeated verbatim
        #      (all columns identical) should still be caught.
        # EXPECTED: Fails — 50% duplicate ratio across all columns.
        train = pd.DataFrame({"a": [1], "b": [2]})
        test = pd.DataFrame({"a": [1, 1, 1, 7, 8, 9, 10, 11, 12, 13],
                              "b": [2, 2, 2, 3, 4, 5,  6,  7,  8,  9]})
        with pytest.raises(MltkAssertionError):
            assert_no_augmentation_on_test(train, test, key_cols=None, max_dup_ratio=0.01)

    def test_no_augmentation_custom_threshold(self) -> None:
        # SCENARIO: 5% duplicates is acceptable when max_dup_ratio=0.10.
        # WHY: Some datasets legitimately have small duplicate rates; threshold
        #      should be configurable so callers can relax the check.
        # EXPECTED: Passes — 1 duplicate out of 20 rows = 5%, under 10% threshold.
        train = pd.DataFrame({"id": range(20)})
        ids = list(range(19)) + [0]   # id=0 appears twice → 1 dup / 20 rows
        test = pd.DataFrame({"id": ids})
        result = assert_no_augmentation_on_test(
            train, test, key_cols=["id"], max_dup_ratio=0.10
        )
        assert result.passed is True

    def test_empty_dataframes(self) -> None:
        # SCENARIO: Both train and test DataFrames are empty.
        # WHY: Edge case — augmentation check on an empty test set must not crash
        #      and should return a graceful pass (nothing to augment).
        # EXPECTED: Passes with WARNING severity, dup_count=0.
        train = pd.DataFrame({"id": [], "x": []})
        test = pd.DataFrame({"id": [], "x": []})
        result = assert_no_augmentation_on_test(train, test, key_cols=["id"])
        assert result.passed is True
        assert result.details["dup_count"] == 0


class TestAugmentationPreservesSignal:
    """Assert augmented dataset preserves label distribution."""

    def test_preserves_signal_ok(self) -> None:
        # SCENARIO: Augmentation doubles every class equally — distribution unchanged.
        # WHY: A correct augmentation strategy (e.g., random flips applied uniformly)
        #      should not change the class balance.
        # EXPECTED: Passes — TV distance is 0.0, well below 0.1 threshold.
        original = pd.DataFrame({"y": [0, 0, 0, 1, 1, 1]})
        # Augmented = original repeated twice → same 50/50 split
        augmented = pd.concat([original, original], ignore_index=True)
        result = assert_augmentation_preserves_signal(original, augmented, label_col="y")
        assert result.passed is True
        assert result.details["tv_distance"] == pytest.approx(0.0, abs=1e-6)

    def test_preserves_signal_shifted(self) -> None:
        # SCENARIO: Augmentation massively oversamples class 1 only — heavily skewed.
        # WHY: Minority-class oversampling without proportional majority sampling
        #      shifts the label distribution, corrupting training signal for the model.
        # EXPECTED: Fails — TV distance far exceeds 0.1 threshold.
        original = pd.DataFrame({"y": [0] * 50 + [1] * 50})   # 50/50
        # Augmented: 50 class-0, 450 class-1 → ~90% class-1
        augmented = pd.DataFrame({"y": [0] * 50 + [1] * 450})
        with pytest.raises(MltkAssertionError) as exc:
            assert_augmentation_preserves_signal(
                original, augmented, label_col="y", max_distribution_shift=0.1
            )
        assert "shifted" in str(exc.value).lower()

    def test_preserves_signal_multiclass(self) -> None:
        # SCENARIO: 4-class dataset augmented proportionally — all classes scale equally.
        # WHY: Multi-class problems are common; TV distance must correctly handle
        #      more than 2 labels.
        # EXPECTED: Passes — TV distance is 0.0.
        rng = np.random.default_rng(7)
        labels = rng.integers(0, 4, size=200).tolist()
        original = pd.DataFrame({"label": labels})
        augmented = pd.concat([original] * 3, ignore_index=True)
        result = assert_augmentation_preserves_signal(
            original, augmented, label_col="label", max_distribution_shift=0.05
        )
        assert result.passed is True

    def test_preserves_signal_custom_threshold(self) -> None:
        # SCENARIO: Slight imbalance (TV~0.06) is acceptable when threshold=0.15.
        # WHY: Looser threshold should allow modest skew while catching severe cases.
        # EXPECTED: Passes under threshold=0.15 but would fail at default 0.10.
        original = pd.DataFrame({"y": [0] * 50 + [1] * 50})
        # 47/53 split → TV = 0.5 * (|0.50-0.47| + |0.50-0.53|) = 0.03
        augmented = pd.DataFrame({"y": [0] * 47 + [1] * 53})
        result = assert_augmentation_preserves_signal(
            original, augmented, label_col="y", max_distribution_shift=0.15
        )
        assert result.passed is True

    def test_empty_dataframes(self) -> None:
        # SCENARIO: Both DataFrames are empty.
        # WHY: Empty-input edge case must not raise ZeroDivisionError or KeyError.
        # EXPECTED: Passes with WARNING severity.
        from mltk.core.result import Severity
        original = pd.DataFrame({"y": []})
        augmented = pd.DataFrame({"y": []})
        result = assert_augmentation_preserves_signal(original, augmented, label_col="y")
        assert result.passed is True
        assert result.severity == Severity.WARNING

    def test_missing_label_col_raises(self) -> None:
        # SCENARIO: label_col does not exist in the original DataFrame.
        # WHY: Callers should get a clear CRITICAL failure rather than a raw KeyError.
        # EXPECTED: Fails with MltkAssertionError mentioning the column name.
        original = pd.DataFrame({"x": [1, 2, 3]})
        augmented = pd.DataFrame({"x": [1, 2, 3, 4]})
        with pytest.raises(MltkAssertionError) as exc:
            assert_augmentation_preserves_signal(original, augmented, label_col="y")
        assert "y" in str(exc.value)
