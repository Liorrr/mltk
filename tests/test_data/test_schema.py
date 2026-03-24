"""Tests for mltk.data.schema — DataFrame structure validation.

Schema tests are the FIRST line of defense. They catch:
- Missing columns (upstream pipeline dropped a field)
- Wrong dtypes (string instead of numeric, silent casting issues)
- Null values (missing labels, broken joins, partial loads)

Each test below validates a specific failure mode you'll encounter in production.
"""

import pandas as pd
import pytest

from mltk.core.assertion import MltkAssertionError
from mltk.data.schema import assert_dtypes, assert_no_nulls, assert_schema

# --- assert_schema tests ---


class TestAssertSchema:
    """Tests for assert_schema — validates column names + dtypes."""

    def test_valid_schema(self, sample_df: pd.DataFrame) -> None:
        """PASS: DataFrame matches expected schema exactly.

        Scenario: Your training pipeline produces data with the expected
        columns and types. This is the happy path.
        """
        result = assert_schema(
            sample_df,
            {"id": "int64", "feature_a": "float64", "feature_b": "int64", "label": "int64"},
        )
        assert result.passed is True

    def test_missing_column(self, sample_df: pd.DataFrame) -> None:
        """FAIL: Expected column 'missing_col' not in DataFrame.

        Scenario: An upstream ETL job changed its output schema. The feature
        engineering pipeline expects a column that no longer exists.
        """
        with pytest.raises(MltkAssertionError) as exc:
            assert_schema(sample_df, {"missing_col": "float64"})
        assert "Missing columns" in str(exc.value)

    def test_wrong_dtype(self, sample_df: pd.DataFrame) -> None:
        """FAIL: Column exists but has wrong dtype.

        Scenario: A numeric column was accidentally cast to string during
        CSV serialization. The model would get garbage features.
        """
        with pytest.raises(MltkAssertionError) as exc:
            assert_schema(sample_df, {"feature_a": "int64"})
        assert "expected int64" in str(exc.value)

    def test_extra_columns_allowed_by_default(self, sample_df: pd.DataFrame) -> None:
        """PASS: Extra columns are OK when allow_extra_columns=True (default).

        Scenario: You only care about specific columns. The DataFrame
        may have additional columns from a wider query — that's fine.
        """
        result = assert_schema(sample_df, {"id": "int64"})
        assert result.passed is True

    def test_extra_columns_rejected(self, sample_df: pd.DataFrame) -> None:
        """FAIL: Extra columns rejected when allow_extra_columns=False.

        Scenario: Strict mode for security-sensitive pipelines. Unexpected
        columns might contain PII that shouldn't reach the model.
        """
        with pytest.raises(MltkAssertionError) as exc:
            assert_schema(sample_df, {"id": "int64"}, allow_extra_columns=False)
        assert "Unexpected columns" in str(exc.value)

    def test_empty_dataframe(self) -> None:
        """PASS: Empty DataFrame with correct columns passes schema check.

        Scenario: A fresh table with zero rows but correct DDL. Schema
        validation should pass — row count is a separate concern.
        """
        df = pd.DataFrame({"id": pd.Series(dtype="int64"), "name": pd.Series(dtype="object")})
        result = assert_schema(df, {"id": "int64", "name": "object"})
        assert result.passed is True


# --- assert_no_nulls tests ---


class TestAssertNoNulls:
    """Tests for assert_no_nulls — detects missing values."""

    def test_no_nulls(self, sample_df: pd.DataFrame) -> None:
        """PASS: No null values in any column.

        Scenario: Clean dataset from a well-tested pipeline. Every row has
        complete data for all features and labels.
        """
        result = assert_no_nulls(sample_df)
        assert result.passed is True

    def test_nulls_detected(self) -> None:
        """FAIL: Null values found in the 'label' column.

        Scenario: A labeling pipeline failed mid-batch. Some rows have
        features but no labels. Training on these would be wrong.
        """
        df = pd.DataFrame({"id": [1, 2, 3], "label": [0, None, 1]})
        with pytest.raises(MltkAssertionError) as exc:
            assert_no_nulls(df)
        assert "null" in str(exc.value).lower()

    def test_subset_columns(self) -> None:
        """PASS: Nulls in unchecked columns are ignored.

        Scenario: You only care about label completeness. Other columns
        (like optional metadata) may have nulls — that's acceptable.
        """
        df = pd.DataFrame({"id": [1, 2], "label": [0, 1], "notes": ["ok", None]})
        result = assert_no_nulls(df, columns=["id", "label"])
        assert result.passed is True


# --- assert_dtypes tests ---


class TestAssertDtypes:
    """Tests for assert_dtypes — strict type checking for specific columns."""

    def test_correct_dtypes(self, sample_df: pd.DataFrame) -> None:
        """PASS: Selected columns match expected dtypes.

        Scenario: Verify that numeric features are actually numeric
        after loading from CSV (which can silently produce 'object' dtype).
        """
        result = assert_dtypes(sample_df, {"feature_a": "float64", "label": "int64"})
        assert result.passed is True

    def test_dtype_mismatch(self) -> None:
        """FAIL: Column is string but expected numeric.

        Scenario: A CSV file had commas in numbers (e.g., "1,234") causing
        pandas to load the column as object instead of int64.
        """
        df = pd.DataFrame({"score": ["high", "low", "medium"]})
        with pytest.raises(MltkAssertionError) as exc:
            assert_dtypes(df, {"score": "float64"})
        assert "score" in str(exc.value)

    def test_missing_column_in_dtypes(self) -> None:
        """FAIL: Column specified in expected dict doesn't exist.

        Scenario: A column was renamed upstream but the test config
        wasn't updated. Catches configuration drift.
        """
        df = pd.DataFrame({"a": [1, 2]})
        with pytest.raises(MltkAssertionError):
            assert_dtypes(df, {"nonexistent": "int64"})
