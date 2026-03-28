"""Tests for mltk.core.polars_bridge -- Polars-to-pandas conversion.

Polars is OPTIONAL. Every test that touches polars uses
pytest.importorskip so the suite runs cleanly without it installed.
Tests cover three layers:
1. to_pandas: pass-through for pandas/numpy/None, conversion for Polars
2. is_polars: type detection without importing polars
3. coerce_dataframe: decorator that transparently converts args
4. Integration: real mltk assertions receiving Polars data
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mltk.core.polars_bridge import coerce_dataframe, is_polars, to_pandas


# ---------------------------------------------------------------------------
# to_pandas tests
# ---------------------------------------------------------------------------


class TestToPandas:
    """Tests for to_pandas -- the core conversion function."""

    def test_pandas_dataframe_passthrough(self) -> None:
        """PASS: pandas DataFrame returns the exact same object (zero copy).

        WHY: If data is already pandas, to_pandas must not copy or
        convert. Identity check (is) proves zero overhead.
        """
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
        result = to_pandas(df)
        assert result is df

    def test_pandas_series_passthrough(self) -> None:
        """PASS: pandas Series returns the exact same object."""
        s = pd.Series([1, 2, 3], name="x")
        result = to_pandas(s)
        assert result is s

    def test_polars_dataframe_converts(self) -> None:
        """PASS: Polars DataFrame converts to pandas DataFrame.

        WHY: This is the primary use case. A team using Polars for
        ETL passes the result to mltk assertions that expect pandas.
        """
        pl = pytest.importorskip("polars")
        pl_df = pl.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
        result = to_pandas(pl_df)
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["a", "b"]
        assert result["a"].tolist() == [1, 2, 3]
        assert result["b"].tolist() == [4.0, 5.0, 6.0]

    def test_polars_series_converts(self) -> None:
        """PASS: Polars Series converts to pandas Series.

        WHY: Drift assertions accept pd.Series. A Polars Series
        from a single column must convert cleanly.
        """
        pl = pytest.importorskip("polars")
        pl_s = pl.Series("feature", [1.0, 2.0, 3.0])
        result = to_pandas(pl_s)
        assert isinstance(result, pd.Series)
        assert result.tolist() == [1.0, 2.0, 3.0]

    def test_numpy_array_passthrough(self) -> None:
        """PASS: numpy array passes through unchanged.

        WHY: Some mltk functions accept numpy arrays directly.
        to_pandas should not interfere with non-Polars types.
        """
        arr = np.array([1, 2, 3])
        result = to_pandas(arr)
        assert result is arr

    def test_none_passthrough(self) -> None:
        """PASS: None passes through unchanged.

        WHY: Edge case -- callers may pass None for optional args.
        to_pandas must not crash on None.
        """
        result = to_pandas(None)
        assert result is None

    def test_string_passthrough(self) -> None:
        """PASS: Arbitrary types pass through unchanged."""
        result = to_pandas("hello")
        assert result == "hello"

    def test_dict_passthrough(self) -> None:
        """PASS: dict passes through unchanged."""
        d = {"key": "value"}
        result = to_pandas(d)
        assert result is d


# ---------------------------------------------------------------------------
# is_polars tests
# ---------------------------------------------------------------------------


class TestIsPolars:
    """Tests for is_polars -- type detection without importing polars."""

    def test_polars_dataframe_detected(self) -> None:
        """PASS: Polars DataFrame is detected as Polars."""
        pl = pytest.importorskip("polars")
        assert is_polars(pl.DataFrame({"a": [1]})) is True

    def test_polars_series_detected(self) -> None:
        """PASS: Polars Series is detected as Polars."""
        pl = pytest.importorskip("polars")
        assert is_polars(pl.Series("x", [1, 2])) is True

    def test_pandas_dataframe_not_polars(self) -> None:
        """PASS: pandas DataFrame is NOT detected as Polars."""
        df = pd.DataFrame({"a": [1]})
        assert is_polars(df) is False

    def test_pandas_series_not_polars(self) -> None:
        """PASS: pandas Series is NOT detected as Polars."""
        s = pd.Series([1, 2])
        assert is_polars(s) is False

    def test_numpy_array_not_polars(self) -> None:
        """PASS: numpy array is NOT detected as Polars."""
        assert is_polars(np.array([1, 2])) is False

    def test_none_not_polars(self) -> None:
        """PASS: None is NOT detected as Polars."""
        assert is_polars(None) is False

    def test_string_not_polars(self) -> None:
        """PASS: string is NOT detected as Polars."""
        assert is_polars("hello") is False


# ---------------------------------------------------------------------------
# coerce_dataframe tests
# ---------------------------------------------------------------------------


class TestCoerceDataframe:
    """Tests for the coerce_dataframe decorator."""

    def test_polars_arg_converted(self) -> None:
        """PASS: Polars DataFrame arg is converted to pandas inside func.

        WHY: The decorator should transparently convert Polars inputs
        so the wrapped function always sees pandas.
        """
        pl = pytest.importorskip("polars")

        @coerce_dataframe
        def get_type(df):  # type: ignore[no-untyped-def]
            return type(df).__name__

        pl_df = pl.DataFrame({"a": [1, 2]})
        assert get_type(pl_df) == "DataFrame"
        # Verify it's pandas DataFrame, not polars
        assert get_type(pl_df) != "polars.DataFrame"

    def test_polars_kwarg_converted(self) -> None:
        """PASS: Polars DataFrame passed as kwarg is converted."""
        pl = pytest.importorskip("polars")

        @coerce_dataframe
        def check(data=None):  # type: ignore[no-untyped-def]
            return isinstance(data, pd.DataFrame)

        pl_df = pl.DataFrame({"x": [10, 20]})
        assert check(data=pl_df) is True

    def test_pandas_arg_unchanged(self) -> None:
        """PASS: pandas input passes through the decorator unchanged.

        WHY: The decorator must be zero-overhead for pandas users.
        Identity check proves no copy occurred.
        """

        @coerce_dataframe
        def identity(df):  # type: ignore[no-untyped-def]
            return df

        pd_df = pd.DataFrame({"a": [1, 2]})
        result = identity(pd_df)
        assert result is pd_df

    def test_mixed_args(self) -> None:
        """PASS: Mix of Polars and non-Polars args handled correctly."""
        pl = pytest.importorskip("polars")

        @coerce_dataframe
        def check_types(df, name, threshold=0.5):  # type: ignore[no-untyped-def]
            return (
                isinstance(df, pd.DataFrame),
                name,
                threshold,
            )

        pl_df = pl.DataFrame({"a": [1]})
        is_pandas, name, thresh = check_types(pl_df, "test", threshold=0.9)
        assert is_pandas is True
        assert name == "test"
        assert thresh == 0.9

    def test_preserves_function_name(self) -> None:
        """PASS: Decorated function retains original __name__.

        WHY: functools.wraps must be used so introspection,
        pytest discovery, and error messages show the real name.
        """

        @coerce_dataframe
        def my_assertion(df):  # type: ignore[no-untyped-def]
            return df

        assert my_assertion.__name__ == "my_assertion"


# ---------------------------------------------------------------------------
# Integration tests -- real mltk assertions with Polars input
# ---------------------------------------------------------------------------


class TestIntegration:
    """Integration tests: mltk assertions receiving Polars data."""

    def test_assert_schema_with_polars(self) -> None:
        """PASS: assert_schema works when given a Polars DataFrame.

        WHY: End-to-end proof that the bridge works with real
        assertions. A Polars DataFrame is converted and schema-checked.
        """
        pl = pytest.importorskip("polars")
        from mltk.data.schema import assert_schema

        pl_df = pl.DataFrame({
            "id": [1, 2, 3],
            "score": [0.9, 0.8, 0.7],
        })
        pd_df = to_pandas(pl_df)
        result = assert_schema(
            pd_df,
            {"id": "int64", "score": "float64"},
        )
        assert result.passed is True

    def test_assert_no_drift_with_polars(self) -> None:
        """PASS: assert_no_drift works when given Polars Series.

        WHY: Drift detection is the most common data assertion.
        A Polars Series from an ETL pipeline must work seamlessly.
        Uses fixed seed for reproducibility.
        """
        pl = pytest.importorskip("polars")
        from mltk.data.drift import assert_no_drift

        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 500).tolist()

        ref = to_pandas(pl.Series("feature", data))
        cur = to_pandas(pl.Series("feature", data))
        result = assert_no_drift(ref, cur, method="ks")
        assert result.passed is True

    def test_coerced_function_with_polars_dataframe(self) -> None:
        """PASS: A coerced function correctly processes Polars input.

        WHY: Validates the full decorator flow -- Polars in, pandas
        operations inside, correct result out.
        """
        pl = pytest.importorskip("polars")

        @coerce_dataframe
        def column_means(df):  # type: ignore[no-untyped-def]
            return df.mean()

        pl_df = pl.DataFrame({
            "a": [1.0, 2.0, 3.0],
            "b": [4.0, 5.0, 6.0],
        })
        result = column_means(pl_df)
        assert isinstance(result, pd.Series)
        assert result["a"] == pytest.approx(2.0)
        assert result["b"] == pytest.approx(5.0)
