"""Tests for mltk.domains.tabular -- tabular data assertions.

Tabular domain tests provide DataFrame-level convenience wrappers for
common ML data validation tasks:
1. Feature drift: checks every numeric column for distribution shift (PSI)
2. Feature importance stability: detects when SHAP rankings change between
   model versions (which may indicate data quality issues or concept drift)
3. Class balance: DataFrame-level wrapper around label balance checks
"""

import numpy as np
import pandas as pd
import pytest

from mltk.core.assertion import MltkAssertionError
from mltk.domains.tabular import (
    assert_class_balance,
    assert_feature_drift,
    assert_feature_importance_stable,
)


class TestFeatureDrift:
    """Per-column drift detection on DataFrames.

    Validates that assert_feature_drift runs drift tests on EVERY numeric
    column in a DataFrame and fails if ANY column exceeds the threshold.
    """

    def test_no_drift(self) -> None:
        """PASS: Both features drawn from the same distribution.

        WHY: Reference and current data are split from the same random
        sample. No distributional shift should be detected. This verifies
        the PSI method produces stable (near-zero) scores for identical
        distributions.
        Expected: result.passed is True.
        """
        rng = np.random.default_rng(42)
        data_a = rng.normal(0, 1, 500)
        data_b = rng.normal(5, 2, 500)
        ref = pd.DataFrame({"a": data_a[:250], "b": data_b[:250]})
        cur = pd.DataFrame({"a": data_a[250:], "b": data_b[250:]})
        result = assert_feature_drift(ref, cur, method="psi", threshold=0.3)
        assert result.passed is True

    def test_drift_detected(self) -> None:
        """FAIL: Feature 'b' shifted from mean=5 to mean=50.

        WHY: A 10x shift in a feature's mean is massive drift. If 'b' is
        income and it shifted from $5K to $50K, the model's learned patterns
        are completely wrong for the new data. Must trigger retraining.
        Expected: MltkAssertionError raised.
        """
        rng = np.random.default_rng(42)
        ref = pd.DataFrame({"a": rng.normal(0, 1, 500), "b": rng.normal(5, 2, 500)})
        cur = pd.DataFrame({"a": rng.normal(0, 1, 500), "b": rng.normal(50, 2, 500)})
        with pytest.raises(MltkAssertionError):
            assert_feature_drift(ref, cur, method="psi", threshold=0.1)


class TestFeatureImportance:
    """SHAP ranking stability tests.

    Validates that assert_feature_importance_stable detects when the most
    important features change rank between model versions. A ranking shift
    often indicates concept drift or data pipeline changes.
    """

    def test_stable_rankings(self) -> None:
        """PASS: Feature importance rankings barely changed.

        WHY: Small value changes (0.50->0.48, 0.30->0.32) that preserve
        ranking order are normal variance. The top feature is still 'age'
        in both versions. No action needed.
        Expected: result.passed is True.
        """
        ref = {"age": 0.5, "income": 0.3, "score": 0.2}
        cur = {"age": 0.48, "income": 0.32, "score": 0.20}
        result = assert_feature_importance_stable(ref, cur, max_rank_change=1)
        assert result.passed is True

    def test_rankings_shifted(self) -> None:
        """WARN: Feature rankings changed significantly (score jumped to #1).

        WHY: 'score' went from rank 3 to rank 1, and 'age' dropped from
        rank 1 to rank 2. This is a 2-position shift, exceeding max_rank_change=1.
        This often means the data distribution changed or a feature engineering
        bug altered the signal. Uses WARNING severity (does not raise).
        Expected: result.passed is False (warning, not exception).
        """
        ref = {"age": 0.5, "income": 0.3, "score": 0.2}
        cur = {"score": 0.5, "age": 0.3, "income": 0.2}
        result = assert_feature_importance_stable(ref, cur, max_rank_change=1)
        # WARNING severity -- doesn't raise, but passed=False
        assert result.passed is False


class TestClassBalance:
    """Tabular class balance convenience wrapper.

    DataFrame-level interface for label balance checks. Takes a DataFrame
    and label column name instead of a raw Series.
    """

    def test_balanced(self) -> None:
        """PASS: 50/50 class split is within max_ratio=2.0.

        WHY: Perfectly balanced binary classification data. Ratio is 1.0,
        well within the 2.0 threshold. This is the ideal training scenario.
        Expected: result.passed is True.
        """
        df = pd.DataFrame({"label": [0, 1] * 50, "feature": range(100)})
        result = assert_class_balance(df, label_col="label", max_ratio=2.0)
        assert result.passed is True

    def test_imbalanced(self) -> None:
        """FAIL: 50:1 class ratio exceeds max_ratio=10.0.

        WHY: 100 negative examples vs 2 positive examples. Even a generous
        10:1 threshold is exceeded. A model trained on this will almost
        certainly predict the majority class for everything.
        Expected: MltkAssertionError raised.
        """
        df = pd.DataFrame({"label": [0] * 100 + [1] * 2, "feature": range(102)})
        with pytest.raises(MltkAssertionError):
            assert_class_balance(df, label_col="label", max_ratio=10.0)
