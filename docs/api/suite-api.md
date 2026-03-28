# Composable Suite API

## Why a composable API?

mltk's assertion functions are designed for pytest -- they raise `MltkAssertionError` on failure, which pytest catches and reports. But not everyone uses pytest:

| Use case | Problem with pytest | MltkSuite solution |
|----------|--------------------|--------------------|
| **Jupyter notebooks** | Exceptions halt the cell; no structured results | Failures become `TestResult` objects you can inspect |
| **CI scripts** | Need programmatic pass/fail, not stdout parsing | `suite.passed` returns a boolean; `suite.to_junit()` writes CI-native XML |
| **Monitoring jobs** | Cron jobs validating models need data, not tracebacks | `suite.run()` returns a `SuiteResult` dataclass |
| **Non-Python frameworks** | Integration tests want JSON, not Python exceptions | `suite.to_json()` exports structured results |

`MltkSuite` wraps assertion functions so they **never raise**. Failures are captured as `TestResult` objects in a list. You run all assertions, then inspect results programmatically.

---

## Quick start

```python
from mltk.core.suite import MltkSuite
from mltk.core.assertion import assert_true

suite = MltkSuite("my-model-tests")
suite.add(assert_true, accuracy > 0.9, name="model.accuracy", message="Accuracy check")
suite.add(assert_true, len(df) > 100, name="data.row_count", message="Enough rows")

result = suite.run()
print(result.passed)        # True / False
print(result.pass_rate)     # 0.0 -- 1.0
print(result.passed_count)  # e.g. 2
print(result.total)         # e.g. 2
```

---

## API reference

### `MltkSuite`

```python
class MltkSuite:
    def __init__(self, name: str = "mltk") -> None
```

Create a new suite. The `name` appears in exported reports and in `SuiteResult.name`.

#### `add(assertion_fn, *args, **kwargs) -> MltkSuite`

Register an assertion to be executed when `run()` is called. The assertion is **not** executed immediately -- it is stored as a deferred call.

- `assertion_fn` -- any mltk assertion function that returns a `TestResult` (and may raise `MltkAssertionError`).
- `*args, **kwargs` -- forwarded to `assertion_fn` at execution time.
- Returns `self` for method chaining.

```python
suite.add(assert_no_drift, train_col, prod_col)
suite.add(assert_metric, y_true, y_pred, metric="f1", threshold=0.85)
```

#### `run() -> SuiteResult`

Execute every registered assertion and collect results. Each assertion is called in registration order.

- If the function raises `MltkAssertionError`, the embedded result is captured as a failed `TestResult`.
- Any other exception is converted into a CRITICAL failure result.
- The suite **never raises** during `run()`.
- Calling `run()` multiple times re-executes all assertions and produces a fresh `SuiteResult`.

```python
result = suite.run()
```

#### `to_json(path: str) -> str`

Export results as a JSON file. Returns the absolute path to the written file.

The JSON structure:

```json
{
  "suite": "my-suite",
  "total": 5,
  "passed": 4,
  "failed": 1,
  "results": [
    {
      "name": "model.accuracy",
      "passed": true,
      "severity": "critical",
      "message": "accuracy 0.92 >= 0.90",
      "details": {"score": 0.92, "threshold": 0.9},
      "duration_ms": 1.23,
      "timestamp": "2025-01-15T10:30:00"
    }
  ]
}
```

#### `to_html(path: str) -> str`

Export results as a self-contained HTML report. Delegates to `mltk.report.generator.generate_report`. Requires `jinja2` (`pip install mltk[report]`).

Returns the absolute path to the generated HTML file.

#### `to_junit(path: str) -> str`

Export results as JUnit XML for CI/CD dashboards (Jenkins, GitLab CI, Azure DevOps, CircleCI). Delegates to `mltk.report.junit.export_junit_xml`.

Returns the absolute path to the written XML file.

#### `summary() -> str`

Human-readable one-line summary:

```
my-suite: 8/10 passed (80.0%) in 42.5 ms
```

Returns `"my-suite: not yet run"` if `run()` has not been called.

#### `passed -> bool` (property)

`True` if the last `run()` had zero failures. Raises `RuntimeError` if `run()` has not been called.

#### `results -> list[TestResult]` (property)

List of `TestResult` objects from the last run. Returns an empty list if `run()` has not been called.

#### `name -> str` (property)

Suite name (read-only).

---

### `SuiteResult`

```python
@dataclass
class SuiteResult:
    name: str
    results: list[TestResult]
    total: int
    passed_count: int
    failed_count: int
    duration_ms: float
```

Returned by `MltkSuite.run()`. Holds every `TestResult` plus convenience aggregations.

| Property | Type | Description |
|----------|------|-------------|
| `passed` | `bool` | `True` when `failed_count == 0` |
| `pass_rate` | `float` | `passed_count / total` (0.0 for empty suites) |
| `total` | `int` | Number of assertions executed |
| `passed_count` | `int` | Number that passed |
| `failed_count` | `int` | Number that failed |
| `duration_ms` | `float` | Wall-clock time for the entire run |

---

## Usage examples

### Jupyter notebook workflow

```python
from mltk.core.suite import MltkSuite
from mltk.data.drift import assert_no_drift
from mltk.model.metrics import assert_metric

suite = MltkSuite("nightly-validation")

# Add assertions -- nothing runs yet
suite.add(assert_no_drift, train_df["age"], prod_df["age"])
suite.add(assert_no_drift, train_df["income"], prod_df["income"])
suite.add(assert_metric, y_true, y_pred, metric="f1", threshold=0.85)
suite.add(assert_metric, y_true, y_pred, metric="precision", threshold=0.80)

# Run all at once
result = suite.run()

# Inspect in notebook
print(suite.summary())
# nightly-validation: 3/4 passed (75.0%) in 120.3 ms

# Drill into failures
for r in result.results:
    if not r.passed:
        print(f"FAIL: {r.name} -- {r.message}")

# Export
suite.to_json("validation-results.json")
```

### CI script

```python
#!/usr/bin/env python3
"""CI validation script -- exit 1 on failure."""

import sys
from mltk.core.suite import MltkSuite
from mltk.core.assertion import assert_true

suite = MltkSuite("ci-gate")
suite.add(assert_true, model_accuracy > 0.90,
          name="model.accuracy", message=f"accuracy={model_accuracy:.3f}")
suite.add(assert_true, data_freshness_hours < 24,
          name="data.freshness", message=f"age={data_freshness_hours}h")

result = suite.run()

# Write JUnit XML for CI dashboard
suite.to_junit("test-results.xml")

# Exit with appropriate code
sys.exit(0 if result.passed else 1)
```

### Monitoring job

```python
"""Scheduled model monitoring -- runs hourly via cron."""

from mltk.core.suite import MltkSuite
from mltk.data.drift import assert_no_drift

def monitor():
    suite = MltkSuite("hourly-monitor")

    # Load fresh data
    train = load_training_data()
    prod = load_production_data(last_hours=1)

    for col in ["age", "income", "credit_score"]:
        suite.add(assert_no_drift, train[col], prod[col])

    result = suite.run()

    if not result.passed:
        suite.to_json("/var/log/mltk/drift-alert.json")
        send_alert(suite.summary())

    return result.passed
```

### Method chaining

```python
result = (
    MltkSuite("quick-check")
    .add(assert_true, len(df) > 0, name="data.nonempty", message="Has rows")
    .add(assert_true, df.isnull().sum().sum() == 0, name="data.no_nulls", message="Clean")
    .run()
)
print(f"All passed: {result.passed}")
```

---

## Export formats

### JSON

Structured output for programmatic consumption. Each result includes name, passed, severity, message, details, duration, and timestamp. Matches the schema from `TestResult.json_schema()`.

### HTML

Self-contained report with donut chart, module breakdown, and details table. Powered by Jinja2 templates. Suitable for stakeholder review, email attachments, and artifact storage.

### JUnit XML

Industry-standard format for CI/CD dashboards. Compatible with Jenkins, GitLab CI, Azure DevOps, CircleCI, and GitHub Actions. Each assertion maps to a `<testcase>` element; failures include the assertion message.

---

## Comparison with other tools

### vs Evidently TestSuite

| Feature | mltk MltkSuite | Evidently TestSuite |
|---------|----------------|---------------------|
| Scope | Any mltk assertion (121+ built-in) | Evidently's own test presets |
| Dependencies | Zero (pure Python core) | pandas, scipy, plotly |
| Export | JSON, HTML, JUnit XML | JSON, HTML |
| CI integration | JUnit XML native | Requires custom parsing |
| Custom assertions | Any function returning TestResult | Must use Evidently's test classes |
| Method chaining | Yes | Yes |

mltk's approach is lighter: any function that returns `TestResult` (or raises `MltkAssertionError`) works. No need to learn a test-class hierarchy.

### vs Great Expectations Suite

| Feature | mltk MltkSuite | Great Expectations Suite |
|---------|----------------|--------------------------|
| Setup | `MltkSuite("name")` | Requires data context, datasource, checkpoint |
| Learning curve | 3 methods: add, run, to_json | Dozens of concepts (expectations, validators, stores) |
| Assertion count | 121+ mltk assertions | 300+ expectations |
| Export | JSON, HTML, JUnit XML | JSON (Data Docs for HTML) |
| Overhead | Milliseconds | Seconds (heavier validation engine) |

mltk prioritizes simplicity. If you already use mltk assertions in pytest, migrating to `MltkSuite` is a one-line change per assertion.

---

## Migration guide: from pytest to MltkSuite

### Before (pytest)

```python
# test_model.py -- requires pytest to run
from mltk.data.drift import assert_no_drift

def test_no_age_drift():
    assert_no_drift(train_df["age"], prod_df["age"])

def test_no_income_drift():
    assert_no_drift(train_df["income"], prod_df["income"])
```

### After (MltkSuite)

```python
# validate.py -- runs anywhere (notebook, script, cron)
from mltk.core.suite import MltkSuite
from mltk.data.drift import assert_no_drift

suite = MltkSuite("drift-checks")
suite.add(assert_no_drift, train_df["age"], prod_df["age"])
suite.add(assert_no_drift, train_df["income"], prod_df["income"])

result = suite.run()
# result.passed, result.pass_rate, result.results, etc.
```

### Key differences

1. **No test runner needed** -- `suite.run()` executes everything.
2. **Failures are data** -- no exceptions, no tracebacks. Results are `TestResult` objects.
3. **Export built-in** -- JSON, HTML, JUnit XML without plugins.
4. **Same assertions** -- the exact same assertion functions work in both pytest and MltkSuite.
