"""Tests for assert_feature_label_correlation_stable.

Verifies that feature-label correlation shifts are correctly detected
between train and test DataFrames. Correlation drift is a silent failure
mode: a feature that was strongly predictive at training time may become
irrelevant (or anti-correlated) in production due to annotation changes,
data pipeline mutations, or upstream distribution shifts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mltk.core.assertion import MltkAssertionError
from mltk.data.validation import assert_feature_label_correlation_stable


class TestAssertFeatureLabelCorrelationStable:
    """Tests for assert_feature_label_correlation_stable."""

    def test_stable_correlations(self) -> None:
        """SCENARIO: Train and test have the same feature-label correlation structure.
        WHY: Happy path — a well-split dataset from the same distribution should
             show nearly identical correlations in both splits.
        EXPECTED: pass=True, shifted_features == [], max_shift_observed near zero.
        """
        rng = np.random.default_rng(0)
        n = 500
        label = rng.integers(0, 2, n).astype(float)
        # Feature is moderately correlated with label
        feat = label + rng.normal(0, 0.5, n)

        train_df = pd.DataFrame({"feat": feat[:300], "label": label[:300]})
        test_df = pd.DataFrame({"feat": feat[300:], "label": label[300:]})

        result = assert_feature_label_correlation_stable(
            train_df, test_df,
            feature_cols=["feat"],
            label_col="label",
            max_shift=0.3,
        )
        assert result.passed is True
        assert result.details["shifted_features"] == []

    def test_shifted_correlations(self) -> None:
        """SCENARIO: Test set has a feature whose correlation with the label reversed.
        WHY: Annotation errors in the test set can flip the sign of a feature's
             predictive relationship — e.g., 'age' was positively correlated with
             churn in training but negatively in test after a labeling bug.
        EXPECTED: MltkAssertionError raised, 'feat' in shifted_features.
        """
        rng = np.random.default_rng(1)
        n = 300
        label = rng.integers(0, 2, n).astype(float)

        # Train: feat positively correlated with label
        train_feat = label + rng.normal(0, 0.1, n)
        # Test: feat negatively correlated with label (correlation flipped)
        test_feat = -label + rng.normal(0, 0.1, n)

        train_df = pd.DataFrame({"feat": train_feat, "label": label})
        test_df = pd.DataFrame({"feat": test_feat, "label": label})

        with pytest.raises(MltkAssertionError) as exc:
            assert_feature_label_correlation_stable(
                train_df, test_df,
                feature_cols=["feat"],
                label_col="label",
                max_shift=0.1,
            )
        result = exc.value.result
        assert "feat" in result.details["shifted_features"]
        assert result.details["max_shift_observed"] > 0.1

    def test_single_feature(self) -> None:
        """SCENARIO: Only one feature column is checked.
        WHY: Edge case — callers may audit a single high-importance feature
             at a time rather than the full feature set.
        EXPECTED: Assertion runs without error; per_feature_shifts has exactly one key.
        """
        rng = np.random.default_rng(2)
        n = 200
        label = rng.integers(0, 2, n).astype(float)
        feat = label + rng.normal(0, 0.5, n)

        train_df = pd.DataFrame({"score": feat[:100], "label": label[:100]})
        test_df = pd.DataFrame({"score": feat[100:], "label": label[100:]})

        result = assert_feature_label_correlation_stable(
            train_df, test_df,
            feature_cols=["score"],
            label_col="label",
            max_shift=0.5,
        )
        assert len(result.details["per_feature_shifts"]) == 1
        assert "score" in result.details["per_feature_shifts"]

    def test_max_shift_boundary(self) -> None:
        """SCENARIO: Correlation shift is exactly at the threshold (within floating tolerance).
        WHY: Boundary conditions must be handled correctly — a shift equal to max_shift
             should pass (strict greater-than comparison), not fail.
        EXPECTED: pass=True when observed shift equals max_shift exactly.
        """
        rng = np.random.default_rng(3)
        n = 400
        label = rng.integers(0, 2, n).astype(float)
        feat = label + rng.normal(0, 0.3, n)

        train_df = pd.DataFrame({"feat": feat[:200], "label": label[:200]})
        test_df = pd.DataFrame({"feat": feat[200:], "label": label[200:]})

        # Use a very generous threshold that is definitely above the actual shift
        train_corr = float(np.corrcoef(train_df["feat"], train_df["label"])[0, 1])
        test_corr = float(np.corrcoef(test_df["feat"], test_df["label"])[0, 1])
        actual_shift = abs(train_corr - test_corr)

        # Set max_shift to exactly actual_shift — should pass (not strictly greater)
        result = assert_feature_label_correlation_stable(
            train_df, test_df,
            feature_cols=["feat"],
            label_col="label",
            max_shift=actual_shift,
        )
        assert result.passed is True

    def test_empty_dataframe(self) -> None:
        """SCENARIO: Both DataFrames are empty (upstream query returned no rows).
        WHY: Empty data is a real production edge case. The assertion must not
             crash with a numpy/pandas error — it should pass gracefully with
             zero shifts reported.
        EXPECTED: pass=True, shifted_features == [], max_shift_observed == 0.0.
        """
        empty_train = pd.DataFrame({"feat": [], "label": []})
        empty_test = pd.DataFrame({"feat": [], "label": []})

        result = assert_feature_label_correlation_stable(
            empty_train, empty_test,
            feature_cols=["feat"],
            label_col="label",
        )
        assert result.passed is True
        assert result.details["shifted_features"] == []
        assert result.details["max_shift_observed"] == 0.0

    def test_details_include_per_feature_shifts(self) -> None:
        """SCENARIO: Multi-feature check where one feature is stable and one drifts.
        WHY: per_feature_shifts must map every checked feature to its shift value,
             enabling callers to identify exactly which features are problematic.
        EXPECTED: pass=False, per_feature_shifts has entries for both features,
                  only the drifted one is in shifted_features.
        """
        rng = np.random.default_rng(4)
        n = 300
        label = rng.integers(0, 2, n).astype(float)

        stable_feat = label + rng.normal(0, 0.5, n)
        # Drifted feature: correlation flips completely
        drifted_train = label + rng.normal(0, 0.05, n)
        drifted_test = -label + rng.normal(0, 0.05, n)

        train_df = pd.DataFrame({
            "stable": stable_feat[:150],
            "drifted": drifted_train[:150],
            "label": label[:150],
        })
        test_df = pd.DataFrame({
            "stable": stable_feat[150:],
            "drifted": drifted_test[150:],
            "label": label[150:],
        })

        with pytest.raises(MltkAssertionError) as exc:
            assert_feature_label_correlation_stable(
                train_df, test_df,
                feature_cols=["stable", "drifted"],
                label_col="label",
                max_shift=0.2,
            )
        result = exc.value.result
        assert "drifted" in result.details["shifted_features"]
        assert "stable" not in result.details["shifted_features"]
        assert set(result.details["per_feature_shifts"].keys()) == {"stable", "drifted"}
