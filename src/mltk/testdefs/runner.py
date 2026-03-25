"""YAML test suite runner — dispatch YAML-declared assertions against a data source.

The runner is the execution engine for :class:`~mltk.testdefs.schema.TestSuiteYaml`.
It:

1. Loads the DataFrame from ``suite.data_source`` (CSV or Parquet).
2. Maps each ``TestDef.assertion`` key to the corresponding ``assert_*`` function.
3. Calls the function with the params from the YAML.
4. Catches :class:`~mltk.core.assertion.MltkAssertionError` so a failing test
   records the result without aborting the remaining tests.
5. Returns a flat list of :class:`~mltk.core.result.TestResult` objects.

Supported assertion keys
------------------------
- ``schema``         → :func:`~mltk.data.schema.assert_schema`
- ``no_nulls``       → :func:`~mltk.data.schema.assert_no_nulls`
- ``dtypes``         → :func:`~mltk.data.schema.assert_dtypes`
- ``range``          → :func:`~mltk.data.distribution.assert_range`
- ``unique``         → :func:`~mltk.data.distribution.assert_unique`
- ``row_count``      → :func:`~mltk.data.freshness.assert_row_count`
- ``no_pii``         → :func:`~mltk.data.pii.assert_no_pii`
- ``no_drift``       → :func:`~mltk.data.drift.assert_no_drift`
- ``label_balance``  → :func:`~mltk.data.labels.assert_label_balance`
- ``freshness``      → :func:`~mltk.data.freshness.assert_freshness`
- ``no_outliers``    → :func:`~mltk.data.distribution.assert_no_outliers`
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from mltk.core.assertion import MltkAssertionError
from mltk.core.result import Severity, TestResult
from mltk.testdefs.schema import TestSuiteYaml

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_dataframe(data_source: str) -> pd.DataFrame:
    """Load a DataFrame from a CSV or Parquet path.

    Args:
        data_source: File path string. Extension determines the loader:
            ``.csv`` → :func:`pandas.read_csv`;
            ``.parquet`` or ``.pq`` → :func:`pandas.read_parquet`.

    Returns:
        Loaded DataFrame.

    Raises:
        ValueError: If the file extension is not recognised.
        FileNotFoundError: If the path does not exist.
    """
    p = Path(data_source)
    if not p.exists():
        raise FileNotFoundError(f"Data source not found: {p}")

    suffix = p.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(p)
    elif suffix in (".parquet", ".pq"):
        return pd.read_parquet(p)
    else:
        raise ValueError(
            f"Unsupported data source extension '{suffix}'. "
            f"Supported: .csv, .parquet, .pq"
        )


def _make_error_result(name: str, assertion: str, message: str) -> TestResult:
    """Create a failed TestResult for a dispatch/config error.

    Args:
        name: Human-readable test name from the YAML entry.
        assertion: Assertion key that caused the error.
        message: Error description.

    Returns:
        A failed :class:`~mltk.core.result.TestResult`.
    """
    return TestResult(
        name=f"testdef.{assertion}",
        passed=False,
        severity=Severity.CRITICAL,
        message=f"[{name}] {message}",
    )


# ---------------------------------------------------------------------------
# Assertion dispatcher
# ---------------------------------------------------------------------------

def _dispatch(df: pd.DataFrame, test_def: TestSuiteYaml, name: str,
              assertion: str, params: dict[str, Any]) -> TestResult:
    """Map an assertion key to the correct mltk function and call it.

    This function is intentionally verbose so that each branch is easy to
    read, test in isolation, and extend.  Import costs are deferred to the
    first call of each branch — the same pattern used in
    :mod:`mltk.contracts.validator`.

    Args:
        df: Loaded DataFrame from the suite's data source.
        test_def: The parent suite (unused here, kept for future context).
        name: Human-readable label from the YAML entry.
        assertion: Assertion key (e.g., ``"schema"``, ``"range"``).
        params: Parameters dict from the YAML ``params`` block.

    Returns:
        :class:`~mltk.core.result.TestResult` from the assertion.
    """
    # ------------------------------------------------------------------
    # schema — assert_schema(df, expected, allow_extra_columns=True)
    # ------------------------------------------------------------------
    if assertion == "schema":
        from mltk.data.schema import assert_schema

        expected = params.get("expected")
        if not isinstance(expected, dict):
            return _make_error_result(
                name, assertion,
                "'params.expected' must be a mapping of column -> dtype"
            )
        allow_extra = params.get("allow_extra_columns", True)
        return assert_schema(df, expected, allow_extra_columns=allow_extra)

    # ------------------------------------------------------------------
    # no_nulls — assert_no_nulls(df, columns=None)
    # ------------------------------------------------------------------
    elif assertion == "no_nulls":
        from mltk.data.schema import assert_no_nulls

        columns = params.get("columns")  # list[str] or None
        return assert_no_nulls(df, columns=columns)

    # ------------------------------------------------------------------
    # dtypes — assert_dtypes(df, expected)
    # ------------------------------------------------------------------
    elif assertion == "dtypes":
        from mltk.data.schema import assert_dtypes

        expected = params.get("expected")
        if not isinstance(expected, dict):
            return _make_error_result(
                name, assertion,
                "'params.expected' must be a mapping of column -> dtype"
            )
        return assert_dtypes(df, expected)

    # ------------------------------------------------------------------
    # range — assert_range(series, min_val, max_val)
    # ------------------------------------------------------------------
    elif assertion == "range":
        from mltk.data.distribution import assert_range

        column = params.get("column")
        if not column:
            return _make_error_result(
                name, assertion, "'params.column' is required for 'range'"
            )
        if column not in df.columns:
            return _make_error_result(
                name, assertion,
                f"Column '{column}' not found in data source"
            )
        min_val = params.get("min_val")
        max_val = params.get("max_val")
        if min_val is None or max_val is None:
            return _make_error_result(
                name, assertion,
                "'params.min_val' and 'params.max_val' are required for 'range'"
            )
        return assert_range(df[column], min_val=float(min_val), max_val=float(max_val))

    # ------------------------------------------------------------------
    # unique — assert_unique(df, columns)
    # ------------------------------------------------------------------
    elif assertion == "unique":
        from mltk.data.distribution import assert_unique

        column = params.get("column")
        columns = params.get("columns")

        # Accept either 'column' (str) or 'columns' (list[str])
        if column is not None:
            col_list = [column]
        elif columns is not None:
            col_list = list(columns)
        else:
            return _make_error_result(
                name, assertion,
                "'params.column' or 'params.columns' is required for 'unique'"
            )
        return assert_unique(df, columns=col_list)

    # ------------------------------------------------------------------
    # row_count — assert_row_count(df, min_rows=None, max_rows=None)
    # ------------------------------------------------------------------
    elif assertion == "row_count":
        from mltk.data.freshness import assert_row_count

        min_rows = params.get("min_rows")
        max_rows = params.get("max_rows")
        return assert_row_count(
            df,
            min_rows=int(min_rows) if min_rows is not None else None,
            max_rows=int(max_rows) if max_rows is not None else None,
        )

    # ------------------------------------------------------------------
    # no_pii — assert_no_pii(df, columns=None, patterns=None)
    # ------------------------------------------------------------------
    elif assertion == "no_pii":
        from mltk.data.pii import assert_no_pii

        columns = params.get("columns")
        patterns = params.get("patterns")
        return assert_no_pii(df, columns=columns, patterns=patterns)

    # ------------------------------------------------------------------
    # no_drift — assert_no_drift(reference, current, method, threshold)
    #   Loads reference series from a separate CSV file.
    # ------------------------------------------------------------------
    elif assertion == "no_drift":
        from mltk.data.drift import assert_no_drift

        column = params.get("column")
        reference_path = params.get("reference")
        if not column:
            return _make_error_result(
                name, assertion, "'params.column' is required for 'no_drift'"
            )
        if not reference_path:
            return _make_error_result(
                name, assertion,
                "'params.reference' (path to reference CSV) is required for 'no_drift'"
            )

        try:
            ref_df = _load_dataframe(str(reference_path))
        except (FileNotFoundError, ValueError) as exc:
            return _make_error_result(name, assertion, str(exc))

        if column not in ref_df.columns:
            return _make_error_result(
                name, assertion,
                f"Column '{column}' not found in reference data '{reference_path}'"
            )
        if column not in df.columns:
            return _make_error_result(
                name, assertion,
                f"Column '{column}' not found in data source"
            )

        method = str(params.get("method", "ks"))
        threshold = params.get("threshold")
        return assert_no_drift(
            ref_df[column],
            df[column],
            method=method,
            threshold=float(threshold) if threshold is not None else None,
        )

    # ------------------------------------------------------------------
    # label_balance — assert_label_balance(series, max_ratio=10.0)
    # ------------------------------------------------------------------
    elif assertion == "label_balance":
        from mltk.data.labels import assert_label_balance

        column = params.get("column")
        if not column:
            return _make_error_result(
                name, assertion,
                "'params.column' is required for 'label_balance'"
            )
        if column not in df.columns:
            return _make_error_result(
                name, assertion,
                f"Column '{column}' not found in data source"
            )
        max_ratio = params.get("max_ratio", 10.0)
        return assert_label_balance(df[column], max_ratio=float(max_ratio))

    # ------------------------------------------------------------------
    # freshness — assert_freshness(df, date_column, max_age_days)
    # ------------------------------------------------------------------
    elif assertion == "freshness":
        from mltk.data.freshness import assert_freshness

        column = params.get("column")
        max_age_days = params.get("max_age_days")
        # Also accept the legacy key name used in contracts
        if column is None:
            column = params.get("date_column")
        if not column:
            return _make_error_result(
                name, assertion,
                "'params.column' is required for 'freshness'"
            )
        if max_age_days is None:
            return _make_error_result(
                name, assertion,
                "'params.max_age_days' is required for 'freshness'"
            )
        return assert_freshness(df, date_column=column, max_age_days=int(max_age_days))

    # ------------------------------------------------------------------
    # no_outliers — assert_no_outliers(series, method="iqr", threshold=1.5)
    # ------------------------------------------------------------------
    elif assertion == "no_outliers":
        from mltk.data.distribution import assert_no_outliers

        column = params.get("column")
        if not column:
            return _make_error_result(
                name, assertion,
                "'params.column' is required for 'no_outliers'"
            )
        if column not in df.columns:
            return _make_error_result(
                name, assertion,
                f"Column '{column}' not found in data source"
            )
        method = str(params.get("method", "iqr"))
        threshold = params.get("threshold", 1.5)
        return assert_no_outliers(
            df[column], method=method, threshold=float(threshold)
        )

    # ------------------------------------------------------------------
    # Unknown assertion key
    # ------------------------------------------------------------------
    else:
        supported = (
            "schema, no_nulls, dtypes, range, unique, row_count, "
            "no_pii, no_drift, label_balance, freshness, no_outliers"
        )
        return _make_error_result(
            name, assertion,
            f"Unknown assertion '{assertion}'. Supported: {supported}"
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_test_suite(suite: TestSuiteYaml) -> list[TestResult]:
    """Run all tests from a YAML test suite against the declared data source.

    Loads the DataFrame once, then iterates through every
    :class:`~mltk.testdefs.schema.TestDef` in order.
    :class:`~mltk.core.assertion.MltkAssertionError` exceptions are caught so
    a single failing test does not abort the remaining tests.

    Args:
        suite: Parsed test suite from :func:`~mltk.testdefs.schema.load_test_suite`.

    Returns:
        List of :class:`~mltk.core.result.TestResult` objects in the same order
        as ``suite.tests``.  Every test produces exactly one result regardless
        of pass/fail.

    Raises:
        FileNotFoundError: If the data source file cannot be found.
        ValueError: If the data source extension is not supported.

    Example:
        >>> from mltk.testdefs import load_test_suite, run_test_suite
        >>> suite = load_test_suite("tests/suite.yaml")
        >>> results = run_test_suite(suite)
        >>> passed = sum(r.passed for r in results)
        >>> print(f"{passed}/{len(results)} tests passed")
    """
    df = _load_dataframe(suite.data_source)
    results: list[TestResult] = []

    for test_def in suite.tests:
        try:
            result = _dispatch(
                df,
                suite,
                name=test_def.name,
                assertion=test_def.assertion,
                params=test_def.params,
            )
        except MltkAssertionError as exc:
            # Assertion raised — the result is embedded in the exception
            result = exc.result
        except Exception as exc:  # pragma: no cover — unexpected errors
            result = _make_error_result(
                test_def.name,
                test_def.assertion,
                f"Unexpected error: {type(exc).__name__}: {exc}",
            )

        results.append(result)

    return results
