"""Tests for mltk.data.labels -- label quality validation.

Label tests catch class imbalance and missing classes before they corrupt
model training. Each test simulates a realistic labeling scenario.
"""

import pandas as pd
import pytest

from mltk.core.assertion import MltkAssertionError
from mltk.data.labels import assert_label_balance, assert_label_coverage


class TestAssertLabelBalance:
    """Tests for assert_label_balance -- class distribution."""

    def test_balanced_labels(self) -> None:
        """PASS: Binary labels are roughly balanced.

        Scenario: Annotation team maintained 50/50 positive/negative split.
        Ratio = 1.0, well within max_ratio=10.
        """
        labels = pd.Series([0, 1] * 50)
        result = assert_label_balance(labels, max_ratio=10.0)
        assert result.passed is True
        assert result.details["ratio"] == 1.0

    def test_imbalanced_labels(self) -> None:
        """FAIL: Extreme class imbalance (100:1 ratio).

        Scenario: Anomaly detection dataset where anomalies are 1% of data.
        A model trained on this will just predict 'normal' for everything.
        """
        labels = pd.Series([0] * 1000 + [1] * 10)
        with pytest.raises(MltkAssertionError) as exc:
            assert_label_balance(labels, max_ratio=10.0)
        assert "imbalance" in str(exc.value).lower()

    def test_multiclass_balanced(self) -> None:
        """PASS: 3 classes all within acceptable ratio.

        Scenario: Image classification with cat/dog/bird -- roughly equal.
        """
        labels = pd.Series(["cat"] * 40 + ["dog"] * 35 + ["bird"] * 30)
        result = assert_label_balance(labels, max_ratio=2.0)
        assert result.passed is True

    def test_single_class(self) -> None:
        """PASS: Only one class -- ratio is 1.0.

        Scenario: Pre-filtered dataset with only positive examples.
        Balance check passes (ratio=1), but you'd want coverage check too.
        """
        labels = pd.Series([1, 1, 1, 1, 1])
        result = assert_label_balance(labels)
        assert result.passed is True


class TestAssertLabelCoverage:
    """Tests for assert_label_coverage -- class presence and sample count."""

    def test_all_labels_present(self) -> None:
        """PASS: All expected labels exist with sufficient samples.

        Scenario: Multi-class classifier needs cat, dog, bird. All present.
        """
        labels = pd.Series(["cat"] * 50 + ["dog"] * 40 + ["bird"] * 30)
        result = assert_label_coverage(
            labels, expected_labels={"cat", "dog", "bird"}, min_samples=10
        )
        assert result.passed is True

    def test_missing_label(self) -> None:
        """FAIL: Expected label 'bird' is completely absent.

        Scenario: Latest annotation batch only had cat and dog images.
        The model will be unable to recognize birds at all.
        """
        labels = pd.Series(["cat"] * 50 + ["dog"] * 50)
        with pytest.raises(MltkAssertionError) as exc:
            assert_label_coverage(
                labels, expected_labels={"cat", "dog", "bird"}
            )
        assert "Missing" in str(exc.value)

    def test_insufficient_samples(self) -> None:
        """FAIL: Label exists but with too few samples.

        Scenario: 'bird' class has only 3 examples -- not enough
        for the model to learn meaningful patterns.
        """
        labels = pd.Series(["cat"] * 100 + ["dog"] * 100 + ["bird"] * 3)
        with pytest.raises(MltkAssertionError) as exc:
            assert_label_coverage(
                labels,
                expected_labels={"cat", "dog", "bird"},
                min_samples=10,
            )
        assert "bird" in str(exc.value)

    def test_auto_detect_labels(self) -> None:
        """PASS: No expected_labels -- check all observed classes have min_samples.

        Scenario: You don't know the exact classes upfront. Just ensure
        every class that appears has at least N examples.
        """
        labels = pd.Series(["A"] * 20 + ["B"] * 15 + ["C"] * 10)
        result = assert_label_coverage(labels, min_samples=5)
        assert result.passed is True

    def test_auto_detect_insufficient(self) -> None:
        """FAIL: Auto-detected class has too few samples.

        Scenario: Rare class 'D' appeared only twice. Even without
        explicit expected_labels, this should be flagged.
        """
        labels = pd.Series(["A"] * 50 + ["B"] * 50 + ["D"] * 2)
        with pytest.raises(MltkAssertionError):
            assert_label_coverage(labels, min_samples=5)
