"""Tests for mltk.data.preset — one-call data quality check and report."""

import pandas as pd

from mltk.core.result import TestSuite
from mltk.data.preset import assert_data_quality, data_quality_report

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _clean_df() -> pd.DataFrame:
    """A small, fully clean DataFrame — no nulls, no duplicates, varied columns."""
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "value": [1.1, 2.2, 3.3, 4.4, 5.5],
            "label": ["a", "b", "c", "d", "e"],
        }
    )


# ---------------------------------------------------------------------------
# assert_data_quality
# ---------------------------------------------------------------------------


class TestAssertDataQuality:
    """assert_data_quality — comprehensive one-call suite."""

    def test_data_quality_clean(self) -> None:
        # SCENARIO: Clean DataFrame with no nulls, no duplicates, varied values.
        # WHY: All checks should pass; suite.passed must be True.
        # EXPECTED: returns TestSuite, suite.passed is True.
        df = _clean_df()
        suite = assert_data_quality(df)
        assert isinstance(suite, TestSuite)
        assert suite.passed is True
        assert suite.total >= 2  # at least row_count + no_nulls

    def test_data_quality_with_nulls(self) -> None:
        # SCENARIO: DataFrame has NaN values in one column; max_null_pct=0 (default).
        # WHY: assert_no_nulls catches the nulls; the suite should contain a failed result.
        # EXPECTED: suite.passed is False (CRITICAL null check fails).
        df = pd.DataFrame({"a": [1, None, 3], "b": [4, 5, 6]})
        suite = assert_data_quality(df)
        assert suite.failed_count >= 1
        assert suite.passed is False

    def test_data_quality_returns_suite(self) -> None:
        # SCENARIO: Any DataFrame is passed to assert_data_quality.
        # WHY: Return type must always be TestSuite regardless of pass/fail.
        # EXPECTED: isinstance(result, TestSuite) is True.
        df = _clean_df()
        result = assert_data_quality(df)
        assert isinstance(result, TestSuite)

    def test_data_quality_too_few_rows(self) -> None:
        # SCENARIO: Empty DataFrame with min_rows=5 in config.
        # WHY: Row count check should fail; suite contains a failed CRITICAL result.
        # EXPECTED: suite.passed is False.
        df = pd.DataFrame({"a": [1, 2]})
        suite = assert_data_quality(df, config={"min_rows": 5})
        assert suite.passed is False

    def test_data_quality_with_duplicates(self) -> None:
        # SCENARIO: DataFrame contains duplicate rows.
        # WHY: Duplicate check runs at WARNING severity; suite passes (no CRITICAL failures)
        #      but the duplicate result is present.
        # EXPECTED: suite.passed is True (WARNING only); a result about duplicates exists.
        df = pd.DataFrame({"a": [1, 1, 2], "b": [3, 3, 4]})
        suite = assert_data_quality(df)
        dup_results = [r for r in suite.results if "duplicates" in r.name]
        assert len(dup_results) == 1
        assert dup_results[0].passed is False  # duplicates detected

    def test_data_quality_with_constant_column(self) -> None:
        # SCENARIO: One column contains the same value in every row.
        # WHY: Constant-column check flags it at WARNING severity.
        # EXPECTED: a result named "data.quality.constant_columns" is present and failed.
        df = pd.DataFrame({"a": [1, 2, 3], "const": [0, 0, 0]})
        suite = assert_data_quality(df)
        const_results = [r for r in suite.results if "constant_columns" in r.name]
        assert len(const_results) == 1
        assert const_results[0].passed is False
        assert "const" in const_results[0].details["constant_columns"]

    def test_data_quality_soft_null_check(self) -> None:
        # SCENARIO: max_null_pct=0.5 allows up to 50 % nulls per column; df has 33 %.
        # WHY: When max_null_pct > 0 the preset uses per-column WARNING checks.
        # EXPECTED: suite.passed is True (WARNING severity columns don't block suite).
        df = pd.DataFrame({"a": [1, None, 3], "b": [4, 5, 6]})
        suite = assert_data_quality(df, config={"max_null_pct": 0.5})
        assert suite.passed is True


# ---------------------------------------------------------------------------
# data_quality_report
# ---------------------------------------------------------------------------


class TestDataQualityReport:
    """data_quality_report — pure introspection dict."""

    def test_data_quality_report_keys(self) -> None:
        # SCENARIO: Run report on a clean DataFrame.
        # WHY: Callers depend on a stable return shape with all documented keys.
        # EXPECTED: all six keys present in the returned dict.
        df = _clean_df()
        report = data_quality_report(df)
        expected_keys = {
            "total_rows",
            "total_columns",
            "missing_rate",
            "duplicate_rows",
            "constant_columns",
            "numeric_summary",
        }
        assert expected_keys.issubset(report.keys())

    def test_data_quality_report_values(self) -> None:
        # SCENARIO: Clean 5-row, 3-column DataFrame.
        # WHY: Validate that scalar counts are correct.
        # EXPECTED: total_rows=5, total_columns=3, no missing, no duplicates.
        df = _clean_df()
        report = data_quality_report(df)
        assert report["total_rows"] == 5
        assert report["total_columns"] == 3
        assert all(v == 0.0 for v in report["missing_rate"].values())
        assert report["duplicate_rows"]["count"] == 0
        assert report["duplicate_rows"]["pct"] == 0.0

    def test_data_quality_report_duplicates(self) -> None:
        # SCENARIO: DataFrame where two rows are identical.
        # WHY: duplicate_rows["count"] should equal the number of duplicated rows.
        # EXPECTED: count=1, pct ≈ 0.333.
        df = pd.DataFrame({"a": [1, 1, 2], "b": [3, 3, 4]})
        report = data_quality_report(df)
        assert report["duplicate_rows"]["count"] == 1
        assert abs(report["duplicate_rows"]["pct"] - 1 / 3) < 1e-9

    def test_data_quality_report_constants(self) -> None:
        # SCENARIO: One column has all identical values.
        # WHY: constant_columns list should contain that column name.
        # EXPECTED: "const" in constant_columns.
        df = pd.DataFrame({"a": [1, 2, 3], "const": [7, 7, 7]})
        report = data_quality_report(df)
        assert "const" in report["constant_columns"]
        assert "a" not in report["constant_columns"]

    def test_data_quality_report_missing_rate(self) -> None:
        # SCENARIO: Column "x" has 2 out of 4 values missing.
        # WHY: missing_rate["x"] should be 0.5.
        # EXPECTED: missing_rate["x"] == 0.5.
        df = pd.DataFrame({"x": [1.0, None, 3.0, None], "y": [1, 2, 3, 4]})
        report = data_quality_report(df)
        assert abs(report["missing_rate"]["x"] - 0.5) < 1e-9
        assert report["missing_rate"]["y"] == 0.0

    def test_data_quality_report_numeric_summary(self) -> None:
        # SCENARIO: DataFrame with a known numeric column.
        # WHY: numeric_summary should contain mean/std/min/max for numeric columns.
        # EXPECTED: "value" key present with correct min and max.
        df = _clean_df()
        report = data_quality_report(df)
        assert "value" in report["numeric_summary"]
        summary = report["numeric_summary"]["value"]
        assert abs(summary["min"] - 1.1) < 1e-9
        assert abs(summary["max"] - 5.5) < 1e-9
        assert summary["mean"] is not None
        assert summary["std"] is not None
