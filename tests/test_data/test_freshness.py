"""Tests for mltk.data.freshness — data recency and size validation.

Freshness tests catch a class of bugs that are invisible to schema and
distribution tests: the data is structurally correct but STALE.
A model trained on 6-month-old data may look fine on the test set
but fail in production due to concept drift.

Row count tests catch pipeline failures: a table that normally has
1M rows suddenly has 0 (failed extract) or 100 (partial load).
"""

from datetime import datetime, timedelta

import pandas as pd
import pytest

from mltk.core.assertion import MltkAssertionError
from mltk.data.freshness import assert_freshness, assert_row_count

# --- assert_freshness tests ---


class TestAssertFreshness:
    """Tests for assert_freshness — data recency."""

    def test_fresh_data(self) -> None:
        """PASS: Most recent date is within the allowed age.

        Scenario: Daily ETL pipeline ran successfully. The newest record
        is from today. Data is fresh enough for training.
        """
        now = datetime.now()
        df = pd.DataFrame({"created_at": [now - timedelta(hours=2), now - timedelta(hours=1), now]})
        result = assert_freshness(df, date_column="created_at", max_age_days=1)
        assert result.passed is True

    def test_stale_data(self) -> None:
        """FAIL: Data is older than the allowed age.

        Scenario: The ETL pipeline silently stopped 2 weeks ago. The data
        looks fine structurally, but it's dangerously outdated for a model
        that needs recent patterns (e.g., fraud detection, recommendations).
        """
        old = datetime.now() - timedelta(days=30)
        df = pd.DataFrame({"created_at": [old, old - timedelta(days=5)]})
        with pytest.raises(MltkAssertionError) as exc:
            assert_freshness(df, date_column="created_at", max_age_days=7)
        assert "exceeds limit" in str(exc.value)

    def test_missing_date_column(self) -> None:
        """FAIL: The specified date column doesn't exist.

        Scenario: Column was renamed from 'created_at' to 'timestamp'
        but the test config wasn't updated. Catch this early.
        """
        df = pd.DataFrame({"id": [1, 2, 3]})
        with pytest.raises(MltkAssertionError):
            assert_freshness(df, date_column="created_at", max_age_days=7)

    def test_custom_reference_date(self) -> None:
        """PASS: Freshness measured against a custom reference date.

        Scenario: Backfill pipeline generates historical data. You want to
        verify data was fresh AT THE TIME OF TRAINING, not right now.
        """
        ref = datetime(2026, 1, 15)
        df = pd.DataFrame({"created_at": [datetime(2026, 1, 14), datetime(2026, 1, 13)]})
        result = assert_freshness(df, date_column="created_at", max_age_days=7, reference_date=ref)
        assert result.passed is True


# --- assert_row_count tests ---


class TestAssertRowCount:
    """Tests for assert_row_count — data volume validation."""

    def test_within_bounds(self, sample_df: pd.DataFrame) -> None:
        """PASS: Row count within expected range.

        Scenario: Training dataset should have between 50 and 10,000 rows.
        Having 100 rows is normal for this dataset.
        """
        result = assert_row_count(sample_df, min_rows=50, max_rows=10000)
        assert result.passed is True

    def test_below_minimum(self) -> None:
        """FAIL: Too few rows — possible pipeline failure.

        Scenario: Your dataset normally has 100K+ rows but today's extract
        only has 5. The data source is likely broken or filtered too
        aggressively. Training on this would produce a terrible model.
        """
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
        with pytest.raises(MltkAssertionError) as exc:
            assert_row_count(df, min_rows=100)
        assert "below minimum" in str(exc.value)

    def test_above_maximum(self) -> None:
        """FAIL: Too many rows — possible data duplication.

        Scenario: A pipeline bug caused data to be appended multiple times.
        Instead of 10K rows, you have 100K. Training on duplicated data
        severely biases the model.
        """
        df = pd.DataFrame({"a": range(1000)})
        with pytest.raises(MltkAssertionError) as exc:
            assert_row_count(df, max_rows=500)
        assert "exceeds maximum" in str(exc.value)

    def test_no_bounds(self, sample_df: pd.DataFrame) -> None:
        """PASS: No bounds specified — just reports count.

        Scenario: Informational check. You want to know the row count
        in the test output without setting pass/fail thresholds.
        """
        result = assert_row_count(sample_df)
        assert result.passed is True
        assert result.details["row_count"] == 100
