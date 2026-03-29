from __future__ import annotations

"""Tests for mltk.scan.detect -- auto-detection utilities.

detect.py identifies feature types (numeric vs categorical),
model types (classifier vs regressor), and sensitive columns
(gender, age, race, etc.).  These tests verify the detection
heuristics produce correct results on known inputs.
"""

import numpy as np
import pandas as pd
import pytest

try:
    from mltk.scan.detect import (
        detect_feature_types,
        detect_model_type,
        detect_sensitive_columns,
    )
except ImportError:
    detect_feature_types = None  # type: ignore[assignment]
    detect_model_type = None  # type: ignore[assignment]
    detect_sensitive_columns = None  # type: ignore[assignment]

pytestmark = pytest.mark.skipif(
    detect_feature_types is None,
    reason="mltk.scan.detect not yet implemented",
)


# ---------------------------------------------------------------
# detect_feature_types
# ---------------------------------------------------------------


class TestDetectFeatureTypes:
    """Classify DataFrame columns as numeric or categorical."""

    def test_numeric_float_column(self) -> None:
        """Float64 column is classified as numeric."""
        rng = np.random.default_rng(42)
        df = pd.DataFrame({"val": rng.normal(0, 1, 50)})
        num, cat = detect_feature_types(df)
        assert "val" in num
        assert "val" not in cat

    def test_string_column_is_categorical(self) -> None:
        """Object-dtype column is classified as categorical."""
        df = pd.DataFrame({"color": ["red", "blue"] * 25})
        num, cat = detect_feature_types(df)
        assert "color" in cat
        assert "color" not in num

    def test_int_few_unique_is_categorical(self) -> None:
        """Integer column with <= threshold unique values
        is categorical."""
        df = pd.DataFrame({"level": [1, 2, 3] * 20})
        num, cat = detect_feature_types(
            df, categorical_threshold=5,
        )
        assert "level" in cat

    def test_int_many_unique_is_numeric(self) -> None:
        """Integer column with many unique values stays
        numeric."""
        df = pd.DataFrame({"id": list(range(100))})
        num, cat = detect_feature_types(
            df, categorical_threshold=20,
        )
        assert "id" in num

    def test_mixed_dataframe(self) -> None:
        """DataFrame with both types is split correctly."""
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "price": rng.uniform(1, 100, 50),
            "city": ["A", "B", "C", "D", "E"] * 10,
        })
        num, cat = detect_feature_types(df)
        assert "price" in num
        assert "city" in cat


# ---------------------------------------------------------------
# detect_model_type
# ---------------------------------------------------------------


class TestDetectModelType:
    """Identify whether a model is a classifier or regressor."""

    def test_binary_predictions(self) -> None:
        """Model returning {0, 1} => classifier."""
        X = pd.DataFrame({"a": range(10)})
        model_fn = lambda x: np.array([0, 1] * (len(x) // 2))
        assert detect_model_type(model_fn, X) == "classifier"

    def test_continuous_predictions(self) -> None:
        """Float predictions with many unique values =>
        regressor."""
        X = pd.DataFrame({"a": range(50)})
        rng = np.random.default_rng(42)
        model_fn = lambda x: rng.normal(100, 20, len(x))
        assert detect_model_type(model_fn, X) == "regressor"

    def test_multiclass_predictions(self) -> None:
        """Predictions of {0, 1, 2} => classifier."""
        X = pd.DataFrame({"a": range(12)})
        model_fn = lambda x: np.array([0, 1, 2] * (len(x) // 3))
        assert detect_model_type(model_fn, X) == "classifier"


# ---------------------------------------------------------------
# detect_sensitive_columns
# ---------------------------------------------------------------


class TestDetectSensitiveColumns:
    """Find columns whose names suggest demographic data."""

    def test_finds_gender(self) -> None:
        """Column named 'gender' is flagged."""
        df = pd.DataFrame({
            "gender": [0, 1],
            "score": [80, 90],
        })
        sensitive = detect_sensitive_columns(df)
        assert "gender" in sensitive

    def test_finds_age(self) -> None:
        """Column named 'age' is flagged."""
        df = pd.DataFrame({
            "age": [25, 55],
            "income": [50000, 80000],
        })
        sensitive = detect_sensitive_columns(df)
        assert "age" in sensitive

    def test_finds_race(self) -> None:
        """Column named 'race' is flagged."""
        df = pd.DataFrame({
            "race": ["A", "B"],
            "score": [1, 2],
        })
        sensitive = detect_sensitive_columns(df)
        assert "race" in sensitive

    def test_no_sensitive_columns(self) -> None:
        """Innocuous column names yield empty list."""
        df = pd.DataFrame({
            "temperature": [20.0, 25.0],
            "pressure": [1.0, 1.1],
        })
        sensitive = detect_sensitive_columns(df)
        assert len(sensitive) == 0

    def test_case_insensitive(self) -> None:
        """Detection is case-insensitive (Gender, GENDER)."""
        df = pd.DataFrame({
            "Gender": [0, 1],
            "Score": [80, 90],
        })
        sensitive = detect_sensitive_columns(df)
        assert len(sensitive) >= 1
