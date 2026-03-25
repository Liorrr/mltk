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

| `assert` value | Maps to | Extra params |
|----------------|---------|-------------|
| `schema` | `assert_schema` | `expected: {col: dtype}` |
| `no_nulls` | `assert_no_nulls` | `columns: [col1, col2]` (optional) |
| `dtypes` | `assert_dtypes` | `expected: {col: dtype}` |
| `range` | `assert_range` | `column`, `min`, `max` |
| `unique` | `assert_unique` | `column` |
| `no_outliers` | `assert_no_outliers` | `column`, `method` (optional) |
| `row_count` | `assert_row_count` | `min_rows` |
| `freshness` | `assert_freshness` | `column`, `max_age_hours` |
| `no_pii` | `assert_no_pii` | `columns: [col1]` |
| `no_drift` | `assert_no_drift` | `reference`, `method`, `threshold` |
| `label_balance` | `assert_label_balance` | `column`, `max_ratio` |

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
