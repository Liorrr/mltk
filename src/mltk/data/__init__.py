"""Data quality testing — schema, distribution, drift, freshness, PII, labels, embeddings."""

from mltk.data.distribution import assert_no_outliers, assert_range, assert_unique
from mltk.data.drift import assert_no_drift
from mltk.data.embedding_drift import assert_no_embedding_drift
from mltk.data.freshness import assert_freshness, assert_row_count
from mltk.data.labels import assert_label_balance, assert_label_coverage
from mltk.data.pii import assert_no_pii, scan_pii
from mltk.data.preset import assert_data_quality, data_quality_report
from mltk.data.schema import assert_dtypes, assert_no_nulls, assert_schema
from mltk.data.statistics import (
    assert_column_mean,
    assert_column_median,
    assert_column_stdev,
    assert_quantiles,
)
from mltk.data.validation import (
    assert_datetime_format,
    assert_no_conflicting_labels,
    assert_values_in_set,
)

__all__ = [
    "assert_schema",
    "assert_no_nulls",
    "assert_dtypes",
    "assert_range",
    "assert_unique",
    "assert_no_outliers",
    "assert_freshness",
    "assert_row_count",
    "assert_no_drift",
    "assert_no_pii",
    "scan_pii",
    "assert_label_balance",
    "assert_label_coverage",
    "assert_no_embedding_drift",
    "assert_column_mean",
    "assert_column_median",
    "assert_column_stdev",
    "assert_quantiles",
    "assert_datetime_format",
    "assert_values_in_set",
    "assert_no_conflicting_labels",
    # preset — one-call quality check
    "assert_data_quality",
    "data_quality_report",
]
