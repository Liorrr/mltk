# data

## Overview
Data quality assertions: schema, distributions, statistics, drift, freshness, PII, labels, lineage and synthetic-data checks.

## Interface
- `schema.py`: `assert_schema`, `assert_no_nulls`, `assert_dtypes`
- `distribution.py`: `assert_range`, `assert_unique`, `assert_no_outliers`
- `statistics.py`: `assert_column_mean`, `assert_column_median`, `assert_column_stdev`, `assert_quantiles`
- `drift.py`: `assert_no_drift`, `assert_no_multivariate_drift`
- `embedding_drift.py`: `assert_no_embedding_drift`
- `freshness.py`: `assert_freshness`, `assert_row_count`
- `labels.py`: `assert_label_balance`, `assert_label_coverage`
- `validation.py`: `assert_datetime_format`, `assert_values_in_set`, `assert_no_conflicting_labels`, `assert_feature_label_correlation_stable`
- `pii.py`: `PiiMatch`, `scan_pii`, `scan_pii_dispatch`, `assert_no_pii`
- `pii_ner.py`: `scan_pii_ner`, `scan_pii_gliner`, `scan_pii_hybrid`
- `lineage.py`: `LineageNode`, `LineageGraph`, `track_lineage`, `assert_lineage_complete`
- `synthetic.py`: `assert_marginal_fidelity`, `assert_correlation_preserved`, `assert_synthetic_novelty`, `assert_dcr_safe`
- `preset.py`: `assert_data_quality`, `data_quality_report`

## Invariants
- `assert_*` functions return `TestResult`; a failed check at `Severity.CRITICAL` (the default) raises `MltkAssertionError` via `mltk.core.assert_true`.
- Inputs are not mutated.

## Side Effects
- NER-based PII detection may load optional models; otherwise computation only.

## Verification
- Run: `python -m pytest tests/test_data -q`
