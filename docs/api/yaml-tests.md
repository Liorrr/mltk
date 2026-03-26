# YAML Test Definitions

Define ML tests in YAML without writing Python. Ideal for QA teams that want to create test suites without code.

**Module:** `mltk.testdefs`

**Run:** `mltk test tests.yaml` or `pytest --mltk-yaml tests.yaml`

---

## Quick Start

```yaml
# mltk-tests.yaml
data_source: data/training.csv

tests:
  - name: "Schema matches spec"
    assert: schema
    expected:
      id: int64
      score: float64
      label: int64

  - name: "No nulls in critical columns"
    assert: no_nulls
    columns: [id, label]

  - name: "Score in valid range"
    assert: range
    column: score
    min: 0.0
    max: 1.0

  - name: "No PII in text fields"
    assert: no_pii
    columns: [notes]

  - name: "Minimum 1000 rows"
    assert: row_count
    min_rows: 1000
```

Run with CLI:
```bash
mltk test mltk-tests.yaml
```

Or integrate with pytest:
```bash
pytest --mltk-yaml mltk-tests.yaml
```

---

## YAML Schema

### Top-level fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `data_source` | `string` | yes | Path to CSV/Parquet, or `env:VAR_NAME` for environment variable |
| `tests` | `list[TestDef]` | yes | List of test definitions |

### Environment variable data sources

Use `env:` prefix to read the data path from an environment variable:

```yaml
data_source: env:TRAINING_DATA_PATH
```

### TestDef fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | `string` | yes | Human-readable test name |
| `assert` | `string` | yes | Assertion to run (see supported list) |
| *(varies)* | | | Additional params specific to the assertion |

---

## Supported Assertions

### Data assertions (original)

| `assert` value | Maps to | Extra params |
|----------------|---------|-------------|
| `schema` | `assert_schema` | `expected: {col: dtype}` |
| `no_nulls` | `assert_no_nulls` | `columns: [col1, col2]` (optional) |
| `dtypes` | `assert_dtypes` | `expected: {col: dtype}` |
| `range` | `assert_range` | `column`, `min_val`, `max_val` |
| `unique` | `assert_unique` | `column` or `columns: [col1, col2]` |
| `no_outliers` | `assert_no_outliers` | `column`, `method` (optional), `threshold` (optional) |
| `row_count` | `assert_row_count` | `min_rows`, `max_rows` (optional) |
| `freshness` | `assert_freshness` | `column`, `max_age_days` |
| `no_pii` | `assert_no_pii` | `columns: [col1]` (optional) |
| `no_drift` | `assert_no_drift` | `column`, `reference` (path), `method`, `threshold` |
| `label_balance` | `assert_label_balance` | `column`, `max_ratio` |

### Data statistics

| `assert` value | Maps to | Extra params |
|----------------|---------|-------------|
| `column_mean` | `assert_column_mean` | `column`, `min_val`, `max_val` |
| `column_median` | `assert_column_median` | `column`, `min_val`, `max_val` |
| `column_stdev` | `assert_column_stdev` | `column`, `min_val`, `max_val` |
| `quantiles` | `assert_quantiles` | `column`, `quantiles: {0.25: [min, max], ...}` |

### Data validation

| `assert` value | Maps to | Extra params |
|----------------|---------|-------------|
| `label_coverage` | `assert_label_coverage` | `column`, `expected_labels: [a, b]` (optional), `min_samples` |
| `values_in_set` | `assert_values_in_set` | `column`, `allowed_values: [a, b, c]` |
| `datetime_format` | `assert_datetime_format` | `column`, `fmt` (default: `%Y-%m-%d`) |
| `no_conflicting_labels` | `assert_no_conflicting_labels` | `feature_cols: [col1, col2]`, `label_col` |

### Data drift

| `assert` value | Maps to | Extra params |
|----------------|---------|-------------|
| `no_embedding_drift` | `assert_no_embedding_drift` | `columns: [emb_0, emb_1, ...]`, `reference` (path), `method`, `threshold` |

### Model assertions

| `assert` value | Maps to | Extra params |
|----------------|---------|-------------|
| `metric` | `assert_metric` | `y_true_col`, `y_pred_col`, `metric`, `threshold`, `average` |
| `no_regression` | `assert_no_regression` | `y_true_col`, `y_pred_col`, `baseline` (float or path), `metric`, `tolerance` |
| `no_bias` | `assert_no_bias` | `y_true_col`, `y_pred_col`, `sensitive_col`, `method`, `threshold` |
| `calibration` | `assert_calibration` | `y_true_col`, `y_prob_col`, `max_error`, `n_bins` |
| `no_overfitting` | `assert_no_overfitting` | `train_score`, `test_score`, `max_gap`, `metric_name` |

### Training assertions

| `assert` value | Maps to | Extra params |
|----------------|---------|-------------|
| `no_train_test_overlap` | `assert_no_train_test_overlap` | `test_data` (path), `key_cols: [col1, col2]` |
| `temporal_split` | `assert_temporal_split` | `test_data` (path), `time_col` |
| `no_target_leakage` | `assert_no_target_leakage` | `target_col`, `feature_cols` (optional), `corr_threshold` |

### Monitor assertions

| `assert` value | Maps to | Extra params |
|----------------|---------|-------------|
| `no_degradation` | `assert_no_degradation` | `column` or `metric_history: [0.9, 0.89, ...]`, `window`, `max_decline` |
| `sla` | `assert_sla` | `latency_p99`, `error_rate`, `thresholds: {latency_p99_ms: 500}` |

---

## Custom Assertions via Plugins

Any assertion registered through the plugin system (`@register_assertion`) is automatically available in YAML test definitions. The YAML runner checks the plugin registry when an assertion key does not match a built-in assertion, so no extra wiring is needed.

### Registering a custom assertion

```python
from mltk.core.plugin import register_assertion
from mltk.core.result import Severity, TestResult


@register_assertion("finance.sharpe_ratio")
def assert_sharpe_ratio(df=None, min_ratio=1.0, **kwargs):
    """Assert portfolio Sharpe ratio meets minimum."""
    import numpy as np

    returns = df["daily_return"].to_numpy()
    ratio = float(np.mean(returns) / max(np.std(returns), 1e-10))

    return TestResult(
        name="finance.sharpe_ratio",
        passed=ratio >= min_ratio,
        severity=Severity.CRITICAL,
        message=f"Sharpe ratio {ratio:.4f} >= {min_ratio}",
        details={"sharpe_ratio": round(ratio, 4), "threshold": min_ratio},
    )
```

### Using it in YAML

```yaml
data_source: data/portfolio.csv

tests:
  - name: "Sharpe ratio above 1.0"
    assertion: finance.sharpe_ratio
    params:
      min_ratio: 1.0
```

### Function signature

The runner calls plugin assertions with two patterns, trying in order:

1. `func(df=df, **params)` -- if the function accepts a `df` keyword argument, it receives the loaded DataFrame.
2. `func(**params)` -- if the first call raises `TypeError` (no `df` parameter), the runner retries with only the YAML params.

This means plugin assertions can work with the DataFrame directly or operate on standalone parameters (e.g., scalar thresholds passed in from YAML).

### Discovery

Plugin assertions are available as soon as they are registered. For auto-discovery of installed plugin packages, call `discover_plugins()` before loading the test suite:

```python
from mltk.core.plugin import discover_plugins
from mltk.testdefs import load_test_suite, run_test_suite

discover_plugins()  # imports mltk_plugin_* packages, triggering @register_assertion

suite = load_test_suite("tests.yaml")
results = run_test_suite(suite)
```

See the [Developer Guide](../guides/developer-guide.md) for full details on creating plugin packages.

---

## Python API

```python
from mltk.testdefs import load_test_suite, run_test_suite

suite = load_test_suite("mltk-tests.yaml")
results = run_test_suite(suite)
for r in results:
    print(f"{'PASS' if r.passed else 'FAIL'} {r.name}: {r.message}")
```

---
