"""Tests for mltk.data.distribution — statistical property validation.

Distribution tests catch data issues that schema tests miss:
- Values outside expected ranges (negative ages, probabilities > 1.0)
- Duplicate records (data pipeline ran twice)
- Statistical outliers (sensor glitches, data entry errors)

These are critical for ML because models learn from data distributions.
If the distribution is wrong, the model learns wrong patterns.
"""

import numpy as np
import pandas as pd
import pytest

from mltk.core.assertion import MltkAssertionError
from mltk.data.distribution import assert_no_outliers, assert_range, assert_unique

# --- assert_range tests ---


class TestAssertRange:
    """Tests for assert_range — numeric value bounds."""

    def test_all_in_range(self) -> None:
        """PASS: All values fall within the expected bounds.

        Scenario: Probability scores from a model should be in [0, 1].
        If they are, the model output is well-calibrated at a basic level.
        """
        s = pd.Series([0.0, 0.5, 1.0], name="probability")
        result = assert_range(s, min_val=0.0, max_val=1.0)
        assert result.passed is True

    def test_below_minimum(self) -> None:
        """FAIL: Values below the minimum threshold.

        Scenario: Age feature has negative values — impossible in real life.
        This means data corruption or a bug in preprocessing.
        """
        s = pd.Series([-5, 10, 25, 50], name="age")
        with pytest.raises(MltkAssertionError) as exc:
            assert_range(s, min_val=0, max_val=150)
        assert "outside" in str(exc.value)

    def test_above_maximum(self) -> None:
        """FAIL: Values above the maximum threshold.

        Scenario: A temperature sensor reports 999.9 degrees — clearly
        a sensor malfunction, not real data.
        """
        s = pd.Series([20.0, 25.0, 999.9], name="temperature")
        with pytest.raises(MltkAssertionError) as exc:
            assert_range(s, min_val=-50.0, max_val=60.0)
        assert "1 values outside" in str(exc.value)


# --- assert_unique tests ---


class TestAssertUnique:
    """Tests for assert_unique — duplicate detection."""

    def test_all_unique(self) -> None:
        """PASS: All values in the ID column are unique.

        Scenario: Primary key validation. Every record should have a
        unique identifier. Duplicates would cause double-counting.
        """
        df = pd.DataFrame({"user_id": [1, 2, 3, 4]})
        result = assert_unique(df, columns=["user_id"])
        assert result.passed is True

    def test_duplicates_found(self) -> None:
        """FAIL: Duplicate user_ids detected.

        Scenario: A data pipeline ran twice, inserting duplicate records.
        Training on duplicates biases the model toward repeated examples.
        """
        df = pd.DataFrame({"user_id": [1, 2, 2, 3]})
        with pytest.raises(MltkAssertionError) as exc:
            assert_unique(df, columns=["user_id"])
        assert "duplicate" in str(exc.value).lower()

    def test_composite_key_unique(self) -> None:
        """PASS: Composite key (date + store_id) is unique.

        Scenario: Daily sales data should have one row per store per day.
        Individual columns may repeat, but the combination must be unique.
        """
        df = pd.DataFrame({"date": ["2026-01-01", "2026-01-01", "2026-01-02"], "store": [1, 2, 1]})
        result = assert_unique(df, columns=["date", "store"])
        assert result.passed is True


# --- assert_no_outliers tests ---


class TestAssertNoOutliers:
    """Tests for assert_no_outliers — IQR-based outlier detection."""

    def test_no_outliers(self) -> None:
        """PASS: All values within IQR bounds.

        Scenario: Salary data follows a normal-ish distribution. No extreme
        values that could skew feature scaling or model weights.
        """
        rng = np.random.default_rng(42)
        s = pd.Series(rng.normal(50000, 10000, 100), name="salary")
        result = assert_no_outliers(s, threshold=3.0)
        assert result.passed is True

    def test_outliers_detected(self) -> None:
        """WARN: Extreme outlier detected (IQR method).

        Scenario: One salary value is 10 million — likely a data entry error
        or unit mismatch (cents vs dollars). This would heavily distort
        the mean and any model trained on it.
        """
        s = pd.Series([40000, 45000, 50000, 55000, 60000, 10_000_000], name="salary")
        result = assert_no_outliers(s, threshold=1.5)
        # Outliers cause WARNING severity, not CRITICAL — no exception raised
        assert result.passed is False
        assert result.details["outlier_count"] > 0

    def test_unsupported_method(self) -> None:
        """FAIL: Unknown outlier detection method.

        Scenario: User passes an invalid method name. Should fail clearly
        rather than silently doing nothing.
        """
        s = pd.Series([1, 2, 3])
        with pytest.raises(MltkAssertionError):
            assert_no_outliers(s, method="invalid")
