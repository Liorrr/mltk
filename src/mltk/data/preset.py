"""Data quality preset — one-call comprehensive data quality check."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mltk.core.result import TestSuite
from mltk.data.freshness import assert_row_count
from mltk.data.schema import assert_no_nulls

# ---------------------------------------------------------------------------
# Default config values
# ---------------------------------------------------------------------------
_DEFAULTS: dict = {
    "min_rows": 1,
    "max_null_pct": 0.0,   # 0 = no nulls allowed; use >0 to allow partial nulls
    "check_duplicates": True,
    "check_constants": True,
}


def assert_data_quality(
    df: pd.DataFrame,
    config: dict | None = None,
) -> TestSuite:
    """Run comprehensive data quality checks in one call.

    Runs the following checks and collects every result into a single
    :class:`~mltk.core.result.TestSuite`:

    1. **Row count** — at least ``min_rows`` rows (default: 1).
    2. **No nulls** — zero null values across all columns, unless
       ``max_null_pct > 0`` in which case a WARNING-severity result is added
       for columns that exceed the threshold.
    3. **Duplicate rows** — flags when the dataset contains duplicate rows
       (WARNING severity so the suite still ``passed`` if only dups exist).
    4. **Constant columns** — flags columns with a single unique value
       (WARNING severity).

    All CRITICAL failures (row count, nulls when ``max_null_pct == 0``) will
    be caught inside the suite rather than raised — the suite collects every
    result so callers can inspect the full picture.

    Args:
        df: DataFrame to validate.
        config: Optional override dict.  Recognised keys:

            - ``min_rows`` (int, default 1) — minimum required row count.
            - ``max_null_pct`` (float, default 0.0) — maximum allowed null
              fraction per column (0.0 = no nulls).
            - ``check_duplicates`` (bool, default True) — whether to flag
              duplicate rows.
            - ``check_constants`` (bool, default True) — whether to flag
              columns whose entire column is a single value.

    Returns:
        :class:`~mltk.core.result.TestSuite` containing one
        :class:`~mltk.core.result.TestResult` per check.

    Example:
        >>> suite = assert_data_quality(df)
        >>> assert suite.passed
        >>> suite  # rich HTML in Jupyter
    """
    cfg = {**_DEFAULTS, **(config or {})}
    suite = TestSuite()

    # 1. Row count
    try:
        result = assert_row_count(df, min_rows=cfg["min_rows"])
        suite.add(result)
    except Exception as exc:
        # assert_row_count raises MltkAssertionError on failure; catch and add
        suite.add(exc.result)  # type: ignore[attr-defined]

    # 2. Null check
    if cfg["max_null_pct"] == 0.0:
        try:
            result = assert_no_nulls(df)
            suite.add(result)
        except Exception as exc:
            suite.add(exc.result)  # type: ignore[attr-defined]
    else:
        # Soft null check — WARNING per column that exceeds threshold
        from mltk.core.assertion import assert_true
        from mltk.core.result import Severity

        for col in df.columns:
            null_pct = df[col].isnull().mean()
            passed = null_pct <= cfg["max_null_pct"]
            msg = (
                f"Column '{col}': null rate {null_pct:.2%} <= {cfg['max_null_pct']:.2%}"
                if passed
                else f"Column '{col}': null rate {null_pct:.2%} > {cfg['max_null_pct']:.2%}"
            )
            suite.add(
                assert_true(
                    passed,
                    name=f"data.quality.nulls.{col}",
                    message=msg,
                    severity=Severity.WARNING,
                    null_pct=null_pct,
                    max_null_pct=cfg["max_null_pct"],
                )
            )

    # 3. Duplicate rows
    if cfg["check_duplicates"]:
        from mltk.core.assertion import assert_true
        from mltk.core.result import Severity

        dup_count = int(df.duplicated().sum())
        passed = dup_count == 0
        msg = (
            "No duplicate rows detected"
            if passed
            else f"{dup_count} duplicate row(s) found ({dup_count / max(len(df), 1):.2%} of total)"
        )
        suite.add(
            assert_true(
                passed,
                name="data.quality.duplicates",
                message=msg,
                severity=Severity.WARNING,
                duplicate_count=dup_count,
                total_rows=len(df),
            )
        )

    # 4. Constant columns
    if cfg["check_constants"]:
        from mltk.core.assertion import assert_true
        from mltk.core.result import Severity

        constant_cols = [col for col in df.columns if df[col].nunique(dropna=False) <= 1]
        passed = len(constant_cols) == 0
        msg = (
            "No constant columns detected"
            if passed
            else f"Constant column(s) detected (single unique value): {constant_cols}"
        )
        suite.add(
            assert_true(
                passed,
                name="data.quality.constant_columns",
                message=msg,
                severity=Severity.WARNING,
                constant_columns=constant_cols,
                total_columns=len(df.columns),
            )
        )

    return suite


def data_quality_report(df: pd.DataFrame) -> dict:
    """Generate a data quality summary report without raising on failures.

    Returns a plain dict suitable for logging, dashboards, or downstream
    processing.  No assertions are raised — this is a pure introspection
    utility.

    Returns:
        dict with the following keys:

        - ``total_rows`` (int) — number of rows.
        - ``total_columns`` (int) — number of columns.
        - ``missing_rate`` (dict[str, float]) — per-column null rate (0.0–1.0).
        - ``duplicate_rows`` (dict) — ``{"count": int, "pct": float}``.
        - ``constant_columns`` (list[str]) — columns with a single unique value.
        - ``numeric_summary`` (dict[str, dict]) — per numeric column:
          ``{"mean": float, "std": float, "min": float, "max": float}``.

    Example:
        >>> report = data_quality_report(df)
        >>> print(report["duplicate_rows"])
        {'count': 0, 'pct': 0.0}
    """
    total_rows = len(df)
    total_columns = len(df.columns)

    # Missing rate per column
    missing_rate: dict[str, float] = {
        col: float(df[col].isnull().mean()) for col in df.columns
    }

    # Duplicate rows
    dup_count = int(df.duplicated().sum())
    duplicate_rows: dict = {
        "count": dup_count,
        "pct": dup_count / total_rows if total_rows > 0 else 0.0,
    }

    # Constant columns
    constant_columns: list[str] = [
        col for col in df.columns if df[col].nunique(dropna=False) <= 1
    ]

    # Numeric summary
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_summary: dict[str, dict] = {}
    for col in numeric_cols:
        series = df[col].dropna()
        if series.empty:
            numeric_summary[col] = {"mean": None, "std": None, "min": None, "max": None}
        else:
            numeric_summary[col] = {
                "mean": float(series.mean()),
                "std": float(series.std()),
                "min": float(series.min()),
                "max": float(series.max()),
            }

    return {
        "total_rows": total_rows,
        "total_columns": total_columns,
        "missing_rate": missing_rate,
        "duplicate_rows": duplicate_rows,
        "constant_columns": constant_columns,
        "numeric_summary": numeric_summary,
    }
