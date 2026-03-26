# QA Engineer Guide

You write test suites for a living. mltk gives you **118 ready-made assertions** that plug into pytest -- the tool you already know. No ML expertise required.

---

## What mltk does for QA

| Traditional QA | ML QA with mltk |
|----------------|-----------------|
| "Does the API return 200?" | "Does the model return predictions in < 100ms?" |
| "Is the response valid JSON?" | "Does the data match the expected schema?" |
| "Did the test pass?" | "Did the model accuracy regress from last week?" |
| Manual test scripts | YAML-defined test suites that anyone can edit |

---

## Quick start (5 minutes)

```bash
pip install mltk[cli,report]
mltk init
pytest --mltk-report -v
```

You now have a working test file, a config, and an HTML report. Read the [Getting Started](../getting-started.md) tutorial for a full walkthrough.

---

## No-code testing with YAML

You do not need to write Python. Define tests in YAML and let mltk generate the test code:

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

  - name: "At least 1000 rows"
    assert: row_count
    min_rows: 1000
```

Run it:

```bash
mltk test mltk-tests.yaml
```

Or run it through pytest for integration with your existing test suite:

```bash
pytest --mltk-yaml mltk-tests.yaml -v
```

:point_right: Full reference: [YAML Test Definitions](../api/yaml-tests.md)

---

## Data contracts

Define what "good data" looks like in a contract file. mltk auto-generates tests from it:

```yaml
# contract.yaml
name: training_data
version: "1.0"

columns:
  id:
    type: int64
    nullable: false
    unique: true
  score:
    type: float64
    min: 0.0
    max: 1.0
  label:
    type: int64
    nullable: false
    values: [0, 1]
```

When the data team changes the contract, your tests update automatically. No code changes needed.

:point_right: Full reference: [Data Contracts](../api/contracts.md)

---

## Key assertions for QA

### Data quality (catch bad inputs)

```python
from mltk.data import assert_schema, assert_no_nulls, assert_range, assert_no_pii

# Structure
assert_schema(df, {"id": "int64", "score": "float64"})

# Completeness
assert_no_nulls(df, columns=["label"])

# Value ranges
assert_range(df["score"], min_val=0.0, max_val=1.0)

# Privacy compliance
assert_no_pii(df, columns=["notes"])
```

### Model quality (catch bad outputs)

```python
from mltk.model import assert_metric, assert_no_regression

# Accuracy threshold
assert_metric(y_true, y_pred, metric="accuracy", threshold=0.90)

# No regression from baseline
assert_no_regression(y_true, y_pred, baseline=0.89, metric="accuracy", tolerance=0.02)
```

### Inference quality (catch performance issues)

```python
from mltk.inference import assert_latency

# P99 latency under 200ms
assert_latency(predict_fn, p99=200)
```

---

## HTML reports

Every test run can produce an interactive HTML report:

```bash
pytest --mltk-report -v
```

The report includes pass/fail counts, severity levels, timing, and module breakdowns. Share it with stakeholders who need evidence that the ML system works.

:point_right: Full reference: [HTML Reports](../api/report.md)

---

## Flaky test detection

ML tests can be nondeterministic. mltk helps you identify and handle flaky tests:

```python
from mltk.testing import detect_flaky

summary = detect_flaky(my_test_fn, runs=10, threshold=0.8)
if summary.is_flaky:
    print(f"Flaky! Pass rate: {summary.pass_rate:.0%}")
```

Mark nondeterministic tests so they can be filtered:

```python
@pytest.mark.ml_nondeterministic
def test_model_with_random_dropout():
    ...
```

:point_right: Full reference: [Testing Patterns](../api/testing-patterns.md)

---

## Next steps

- [YAML Test Definitions](../api/yaml-tests.md) -- full YAML syntax reference
- [Data Contracts](../api/contracts.md) -- auto-generate tests from contracts
- [CI/CD Integration](cicd-integration.md) -- run mltk tests in your pipeline
- [pytest Plugin](../api/pytest-plugin.md) -- markers, fixtures, and plugin options
- [Testing Patterns](../api/testing-patterns.md) -- flaky detection, golden baselines, retry
