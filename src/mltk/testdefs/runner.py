"""YAML test suite runner — dispatch YAML-declared assertions against a data source.

The runner is the execution engine for :class:`~mltk.testdefs.schema.TestSuiteYaml`.
It:

1. Loads the DataFrame from ``suite.data_source`` (CSV or Parquet).
2. Maps each ``TestDef.assertion`` key to the corresponding ``assert_*`` function.
3. Calls the function with the params from the YAML.
4. Catches :class:`~mltk.core.assertion.MltkAssertionError` so a failing test
   records the result without aborting the remaining tests.
5. Returns a flat list of :class:`~mltk.core.result.TestResult` objects.
6. Falls back to the plugin registry (:func:`~mltk.core.plugin.get_registered_assertions`)
   for any assertion key not matched by the built-in branches.

Supported assertion keys
------------------------

**Data assertions (11 original):**

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

**Data statistics (4):**

- ``column_mean``    → :func:`~mltk.data.statistics.assert_column_mean`
- ``column_median``  → :func:`~mltk.data.statistics.assert_column_median`
- ``column_stdev``   → :func:`~mltk.data.statistics.assert_column_stdev`
- ``quantiles``      → :func:`~mltk.data.statistics.assert_quantiles`

**Data validation (4):**

- ``label_coverage``        → :func:`~mltk.data.labels.assert_label_coverage`
- ``values_in_set``         → :func:`~mltk.data.validation.assert_values_in_set`
- ``datetime_format``       → :func:`~mltk.data.validation.assert_datetime_format`
- ``no_conflicting_labels`` → :func:`~mltk.data.validation.assert_no_conflicting_labels`

**Data drift (1):**

- ``no_embedding_drift`` → :func:`~mltk.data.embedding_drift.assert_no_embedding_drift`

**Model assertions (4):**

- ``metric``         → :func:`~mltk.model.metrics.assert_metric`
- ``no_regression``  → :func:`~mltk.model.regression.assert_no_regression`
- ``no_bias``        → :func:`~mltk.model.bias.assert_no_bias`
- ``calibration``    → :func:`~mltk.model.slicing.assert_calibration`
- ``no_overfitting`` → :func:`~mltk.model.overfitting.assert_no_overfitting`

**Training assertions (3):**

- ``no_train_test_overlap`` → :func:`~mltk.training.leakage.assert_no_train_test_overlap`
- ``temporal_split``        → :func:`~mltk.training.leakage.assert_temporal_split`
- ``no_target_leakage``     → :func:`~mltk.training.leakage.assert_no_target_leakage`

**Monitor assertions (2):**

- ``no_degradation`` → :func:`~mltk.monitor.drift_monitor.assert_no_degradation`
- ``sla``            → :func:`~mltk.monitor.drift_monitor.assert_sla`
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

    # ==================================================================
    # DATA STATISTICS — column_mean, column_median, column_stdev, quantiles
    # ==================================================================

    # ------------------------------------------------------------------
    # column_mean — assert_column_mean(df, column, min_val, max_val)
    # ------------------------------------------------------------------
    elif assertion == "column_mean":
        from mltk.data.statistics import assert_column_mean

        column = params.get("column")
        if not column:
            return _make_error_result(
                name, assertion,
                "'params.column' is required for 'column_mean'"
            )
        if column not in df.columns:
            return _make_error_result(
                name, assertion,
                f"Column '{column}' not found in data source"
            )
        min_val = params.get("min_val")
        max_val = params.get("max_val")
        return assert_column_mean(
            df, column,
            min_val=float(min_val) if min_val is not None else None,
            max_val=float(max_val) if max_val is not None else None,
        )

    # ------------------------------------------------------------------
    # column_median — assert_column_median(df, column, min_val, max_val)
    # ------------------------------------------------------------------
    elif assertion == "column_median":
        from mltk.data.statistics import assert_column_median

        column = params.get("column")
        if not column:
            return _make_error_result(
                name, assertion,
                "'params.column' is required for 'column_median'"
            )
        if column not in df.columns:
            return _make_error_result(
                name, assertion,
                f"Column '{column}' not found in data source"
            )
        min_val = params.get("min_val")
        max_val = params.get("max_val")
        return assert_column_median(
            df, column,
            min_val=float(min_val) if min_val is not None else None,
            max_val=float(max_val) if max_val is not None else None,
        )

    # ------------------------------------------------------------------
    # column_stdev — assert_column_stdev(df, column, min_val, max_val)
    # ------------------------------------------------------------------
    elif assertion == "column_stdev":
        from mltk.data.statistics import assert_column_stdev

        column = params.get("column")
        if not column:
            return _make_error_result(
                name, assertion,
                "'params.column' is required for 'column_stdev'"
            )
        if column not in df.columns:
            return _make_error_result(
                name, assertion,
                f"Column '{column}' not found in data source"
            )
        min_val = params.get("min_val")
        max_val = params.get("max_val")
        return assert_column_stdev(
            df, column,
            min_val=float(min_val) if min_val is not None else None,
            max_val=float(max_val) if max_val is not None else None,
        )

    # ------------------------------------------------------------------
    # quantiles — assert_quantiles(df, column, quantiles)
    # ------------------------------------------------------------------
    elif assertion == "quantiles":
        from mltk.data.statistics import assert_quantiles

        column = params.get("column")
        if not column:
            return _make_error_result(
                name, assertion,
                "'params.column' is required for 'quantiles'"
            )
        if column not in df.columns:
            return _make_error_result(
                name, assertion,
                f"Column '{column}' not found in data source"
            )
        raw_quantiles = params.get("quantiles")
        if not isinstance(raw_quantiles, dict):
            return _make_error_result(
                name, assertion,
                "'params.quantiles' must be a mapping of "
                "quantile_level -> [min_bound, max_bound]"
            )
        # Convert YAML keys (may be strings) and values (lists) to proper types
        parsed: dict[float, tuple[float, float]] = {}
        for k, v in raw_quantiles.items():
            if not isinstance(v, (list, tuple)) or len(v) != 2:
                return _make_error_result(
                    name, assertion,
                    f"Quantile {k}: value must be a [min, max] pair"
                )
            parsed[float(k)] = (float(v[0]), float(v[1]))
        return assert_quantiles(df, column, parsed)

    # ==================================================================
    # DATA VALIDATION — label_coverage, values_in_set, datetime_format,
    #                    no_conflicting_labels
    # ==================================================================

    # ------------------------------------------------------------------
    # label_coverage — assert_label_coverage(labels, expected_labels, min_samples)
    # ------------------------------------------------------------------
    elif assertion == "label_coverage":
        from mltk.data.labels import assert_label_coverage

        column = params.get("column")
        if not column:
            return _make_error_result(
                name, assertion,
                "'params.column' is required for 'label_coverage'"
            )
        if column not in df.columns:
            return _make_error_result(
                name, assertion,
                f"Column '{column}' not found in data source"
            )
        expected_labels = params.get("expected_labels")
        if expected_labels is not None:
            expected_labels = set(expected_labels)
        min_samples = params.get("min_samples", 1)
        return assert_label_coverage(
            df[column],
            expected_labels=expected_labels,
            min_samples=int(min_samples),
        )

    # ------------------------------------------------------------------
    # values_in_set — assert_values_in_set(df, column, allowed_values)
    # ------------------------------------------------------------------
    elif assertion == "values_in_set":
        from mltk.data.validation import assert_values_in_set

        column = params.get("column")
        if not column:
            return _make_error_result(
                name, assertion,
                "'params.column' is required for 'values_in_set'"
            )
        if column not in df.columns:
            return _make_error_result(
                name, assertion,
                f"Column '{column}' not found in data source"
            )
        allowed = params.get("allowed_values")
        if allowed is None:
            return _make_error_result(
                name, assertion,
                "'params.allowed_values' is required for 'values_in_set'"
            )
        return assert_values_in_set(df, column, set(allowed))

    # ------------------------------------------------------------------
    # datetime_format — assert_datetime_format(df, column, fmt)
    # ------------------------------------------------------------------
    elif assertion == "datetime_format":
        from mltk.data.validation import assert_datetime_format

        column = params.get("column")
        if not column:
            return _make_error_result(
                name, assertion,
                "'params.column' is required for 'datetime_format'"
            )
        if column not in df.columns:
            return _make_error_result(
                name, assertion,
                f"Column '{column}' not found in data source"
            )
        fmt = params.get("fmt", "%Y-%m-%d")
        return assert_datetime_format(df, column, fmt=str(fmt))

    # ------------------------------------------------------------------
    # no_conflicting_labels — assert_no_conflicting_labels(df, feature_cols, label_col)
    # ------------------------------------------------------------------
    elif assertion == "no_conflicting_labels":
        from mltk.data.validation import assert_no_conflicting_labels

        feature_cols = params.get("feature_cols")
        label_col = params.get("label_col")
        if not feature_cols or not isinstance(feature_cols, list):
            return _make_error_result(
                name, assertion,
                "'params.feature_cols' (list) is required for 'no_conflicting_labels'"
            )
        if not label_col:
            return _make_error_result(
                name, assertion,
                "'params.label_col' is required for 'no_conflicting_labels'"
            )
        return assert_no_conflicting_labels(df, feature_cols, label_col)

    # ==================================================================
    # DATA DRIFT — no_embedding_drift
    # ==================================================================

    # ------------------------------------------------------------------
    # no_embedding_drift — assert_no_embedding_drift(reference, current, method, threshold)
    #   Reads embedding columns from current and reference CSVs.
    # ------------------------------------------------------------------
    elif assertion == "no_embedding_drift":
        from mltk.data.embedding_drift import assert_no_embedding_drift

        columns = params.get("columns")
        reference_path = params.get("reference")
        if not columns or not isinstance(columns, list):
            return _make_error_result(
                name, assertion,
                "'params.columns' (list of embedding columns) is required "
                "for 'no_embedding_drift'"
            )
        if not reference_path:
            return _make_error_result(
                name, assertion,
                "'params.reference' (path to reference CSV) is required "
                "for 'no_embedding_drift'"
            )
        try:
            ref_df = _load_dataframe(str(reference_path))
        except (FileNotFoundError, ValueError) as exc:
            return _make_error_result(name, assertion, str(exc))

        for col in columns:
            if col not in df.columns:
                return _make_error_result(
                    name, assertion,
                    f"Column '{col}' not found in data source"
                )
            if col not in ref_df.columns:
                return _make_error_result(
                    name, assertion,
                    f"Column '{col}' not found in reference data '{reference_path}'"
                )

        import numpy as np
        ref_arr = ref_df[columns].to_numpy(dtype=np.float64)
        cur_arr = df[columns].to_numpy(dtype=np.float64)
        method = str(params.get("method", "cosine"))
        threshold = params.get("threshold", 0.1)
        return assert_no_embedding_drift(
            ref_arr, cur_arr, method=method, threshold=float(threshold)
        )

    # ==================================================================
    # MODEL — metric, no_regression, no_bias, calibration, no_overfitting
    #   These read y_true / y_pred from named columns in the DataFrame.
    # ==================================================================

    # ------------------------------------------------------------------
    # metric — assert_metric(y_true, y_pred, metric, threshold, average)
    # ------------------------------------------------------------------
    elif assertion == "metric":
        from mltk.model.metrics import assert_metric

        y_true_col = params.get("y_true_col")
        y_pred_col = params.get("y_pred_col")
        if not y_true_col or not y_pred_col:
            return _make_error_result(
                name, assertion,
                "'params.y_true_col' and 'params.y_pred_col' are required "
                "for 'metric'"
            )
        for col in (y_true_col, y_pred_col):
            if col not in df.columns:
                return _make_error_result(
                    name, assertion,
                    f"Column '{col}' not found in data source"
                )
        metric_name = str(params.get("metric", "accuracy"))
        threshold = float(params.get("threshold", 0.8))
        average = str(params.get("average", "weighted"))
        return assert_metric(
            df[y_true_col].to_numpy(),
            df[y_pred_col].to_numpy(),
            metric=metric_name,
            threshold=threshold,
            average=average,
        )

    # ------------------------------------------------------------------
    # no_regression — assert_no_regression(y_true, y_pred, baseline, ...)
    # ------------------------------------------------------------------
    elif assertion == "no_regression":
        from mltk.model.regression import assert_no_regression

        y_true_col = params.get("y_true_col")
        y_pred_col = params.get("y_pred_col")
        baseline = params.get("baseline")
        if not y_true_col or not y_pred_col:
            return _make_error_result(
                name, assertion,
                "'params.y_true_col' and 'params.y_pred_col' are required "
                "for 'no_regression'"
            )
        if baseline is None:
            return _make_error_result(
                name, assertion,
                "'params.baseline' (float or path to JSON) is required "
                "for 'no_regression'"
            )
        for col in (y_true_col, y_pred_col):
            if col not in df.columns:
                return _make_error_result(
                    name, assertion,
                    f"Column '{col}' not found in data source"
                )
        metric_name = str(params.get("metric", "accuracy"))
        tolerance = float(params.get("tolerance", 0.02))
        average = str(params.get("average", "weighted"))
        # baseline can be float or string (path)
        try:
            baseline_val = float(baseline)
        except (TypeError, ValueError):
            baseline_val = str(baseline)  # type: ignore[assignment]
        return assert_no_regression(
            df[y_true_col].to_numpy(),
            df[y_pred_col].to_numpy(),
            baseline=baseline_val,
            metric=metric_name,
            tolerance=tolerance,
            average=average,
        )

    # ------------------------------------------------------------------
    # no_bias — assert_no_bias(y_true, y_pred, sensitive_feature, ...)
    # ------------------------------------------------------------------
    elif assertion == "no_bias":
        from mltk.model.bias import assert_no_bias

        y_true_col = params.get("y_true_col")
        y_pred_col = params.get("y_pred_col")
        sensitive_col = params.get("sensitive_col")
        if not y_true_col or not y_pred_col or not sensitive_col:
            return _make_error_result(
                name, assertion,
                "'params.y_true_col', 'params.y_pred_col', and "
                "'params.sensitive_col' are required for 'no_bias'"
            )
        for col in (y_true_col, y_pred_col, sensitive_col):
            if col not in df.columns:
                return _make_error_result(
                    name, assertion,
                    f"Column '{col}' not found in data source"
                )
        method = str(params.get("method", "demographic_parity"))
        threshold = params.get("threshold")
        return assert_no_bias(
            df[y_true_col].to_numpy(),
            df[y_pred_col].to_numpy(),
            df[sensitive_col].to_numpy(),
            method=method,
            threshold=float(threshold) if threshold is not None else None,
        )

    # ------------------------------------------------------------------
    # calibration — assert_calibration(y_true, y_prob, max_error, n_bins)
    # ------------------------------------------------------------------
    elif assertion == "calibration":
        from mltk.model.slicing import assert_calibration

        y_true_col = params.get("y_true_col")
        y_prob_col = params.get("y_prob_col")
        if not y_true_col or not y_prob_col:
            return _make_error_result(
                name, assertion,
                "'params.y_true_col' and 'params.y_prob_col' are required "
                "for 'calibration'"
            )
        for col in (y_true_col, y_prob_col):
            if col not in df.columns:
                return _make_error_result(
                    name, assertion,
                    f"Column '{col}' not found in data source"
                )
        max_error = float(params.get("max_error", 0.05))
        n_bins = int(params.get("n_bins", 10))
        method = str(params.get("method", "ece"))
        return assert_calibration(
            df[y_true_col].to_numpy(),
            df[y_prob_col].to_numpy(),
            max_error=max_error,
            n_bins=n_bins,
            method=method,
        )

    # ------------------------------------------------------------------
    # no_overfitting — assert_no_overfitting(train_score, test_score, ...)
    # ------------------------------------------------------------------
    elif assertion == "no_overfitting":
        from mltk.model.overfitting import assert_no_overfitting

        train_score = params.get("train_score")
        test_score = params.get("test_score")
        if train_score is None or test_score is None:
            return _make_error_result(
                name, assertion,
                "'params.train_score' and 'params.test_score' are required "
                "for 'no_overfitting'"
            )
        max_gap = float(params.get("max_gap", 0.1))
        metric_name = str(params.get("metric_name", "accuracy"))
        return assert_no_overfitting(
            train_score=float(train_score),
            test_score=float(test_score),
            max_gap=max_gap,
            metric_name=metric_name,
        )

    # ==================================================================
    # TRAINING — no_train_test_overlap, temporal_split, no_target_leakage
    #   These load a second DataFrame for comparisons.
    # ==================================================================

    # ------------------------------------------------------------------
    # no_train_test_overlap — assert_no_train_test_overlap(train_df, test_df, key_cols)
    #   data_source = train CSV; params.test_data = test CSV
    # ------------------------------------------------------------------
    elif assertion == "no_train_test_overlap":
        from mltk.training.leakage import assert_no_train_test_overlap

        test_data_path = params.get("test_data")
        key_cols = params.get("key_cols")
        if not test_data_path:
            return _make_error_result(
                name, assertion,
                "'params.test_data' (path to test CSV) is required "
                "for 'no_train_test_overlap'"
            )
        if not key_cols or not isinstance(key_cols, list):
            return _make_error_result(
                name, assertion,
                "'params.key_cols' (list) is required for 'no_train_test_overlap'"
            )
        try:
            test_df = _load_dataframe(str(test_data_path))
        except (FileNotFoundError, ValueError) as exc:
            return _make_error_result(name, assertion, str(exc))
        return assert_no_train_test_overlap(df, test_df, key_cols)

    # ------------------------------------------------------------------
    # temporal_split — assert_temporal_split(train_df, test_df, time_col)
    # ------------------------------------------------------------------
    elif assertion == "temporal_split":
        from mltk.training.leakage import assert_temporal_split

        test_data_path = params.get("test_data")
        time_col = params.get("time_col")
        if not test_data_path:
            return _make_error_result(
                name, assertion,
                "'params.test_data' (path to test CSV) is required "
                "for 'temporal_split'"
            )
        if not time_col:
            return _make_error_result(
                name, assertion,
                "'params.time_col' is required for 'temporal_split'"
            )
        try:
            test_df = _load_dataframe(str(test_data_path))
        except (FileNotFoundError, ValueError) as exc:
            return _make_error_result(name, assertion, str(exc))
        return assert_temporal_split(df, test_df, time_col)

    # ------------------------------------------------------------------
    # no_target_leakage — assert_no_target_leakage(df, target_col, ...)
    # ------------------------------------------------------------------
    elif assertion == "no_target_leakage":
        from mltk.training.leakage import assert_no_target_leakage

        target_col = params.get("target_col")
        if not target_col:
            return _make_error_result(
                name, assertion,
                "'params.target_col' is required for 'no_target_leakage'"
            )
        if target_col not in df.columns:
            return _make_error_result(
                name, assertion,
                f"Column '{target_col}' not found in data source"
            )
        feature_cols = params.get("feature_cols")  # list or None
        corr_threshold = float(params.get("corr_threshold", 0.95))
        return assert_no_target_leakage(
            df, target_col,
            feature_cols=feature_cols,
            corr_threshold=corr_threshold,
        )

    # ==================================================================
    # MONITOR — no_degradation, sla
    # ==================================================================

    # ------------------------------------------------------------------
    # no_degradation — assert_no_degradation(metric_history, window, max_decline)
    #   Reads a metric column as the history series.
    # ------------------------------------------------------------------
    elif assertion == "no_degradation":
        from mltk.monitor.drift_monitor import assert_no_degradation

        column = params.get("column")
        metric_history = params.get("metric_history")
        if column:
            if column not in df.columns:
                return _make_error_result(
                    name, assertion,
                    f"Column '{column}' not found in data source"
                )
            history = df[column].tolist()
        elif metric_history and isinstance(metric_history, list):
            history = [float(v) for v in metric_history]
        else:
            return _make_error_result(
                name, assertion,
                "'params.column' or 'params.metric_history' (list) is "
                "required for 'no_degradation'"
            )
        window = int(params.get("window", 7))
        max_decline = float(params.get("max_decline", 0.05))
        return assert_no_degradation(history, window=window, max_decline=max_decline)

    # ------------------------------------------------------------------
    # sla — assert_sla(latency_p99, error_rate, thresholds)
    # ------------------------------------------------------------------
    elif assertion == "sla":
        from mltk.monitor.drift_monitor import assert_sla

        latency_p99 = params.get("latency_p99")
        error_rate = params.get("error_rate")
        thresholds = params.get("thresholds")
        return assert_sla(
            latency_p99=float(latency_p99) if latency_p99 is not None else None,
            error_rate=float(error_rate) if error_rate is not None else None,
            thresholds=thresholds,
        )

    # ------------------------------------------------------------------
    # Plugin registry — check for third-party registered assertions
    # ------------------------------------------------------------------
    else:
        from mltk.core.plugin import get_registered_assertions

        registered = get_registered_assertions()
        if assertion in registered:
            func = registered[assertion]
            try:
                return func(df=df, **params)
            except TypeError:
                # The plugin function may not accept a 'df' kwarg — try
                # calling with just the user-supplied params.
                return func(**params)

        # ------------------------------------------------------------------
        # Truly unknown — no built-in match AND not in the plugin registry
        # ------------------------------------------------------------------
        builtin = (
            "schema, no_nulls, dtypes, range, unique, row_count, no_pii, "
            "no_drift, label_balance, freshness, no_outliers, "
            "column_mean, column_median, column_stdev, quantiles, "
            "label_coverage, values_in_set, datetime_format, no_conflicting_labels, "
            "no_embedding_drift, "
            "metric, no_regression, no_bias, calibration, no_overfitting, "
            "no_train_test_overlap, temporal_split, no_target_leakage, "
            "no_degradation, sla"
        )
        plugin_names = ", ".join(sorted(registered.keys()))
        supported = builtin
        if plugin_names:
            supported += f"; plugins: {plugin_names}"
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
