# Scanning Models

Every ML model has blind spots. It might score 92% overall
but collapse to 55% for users over age 60. It might look
calibrated on average but wildly over-confident on edge
cases. It might leak future data through a feature that
correlates 0.98 with the target.

`mltk scan` finds these problems **and writes the tests
for you**. Give it a model and a dataset. It runs eight
specialized scanners, reports every issue with severity
levels, and generates a self-contained pytest file you can
commit to your repo. No other tool does both.

**Module:** `mltk.scan`

**ML Lifecycle Stage:** Post-training evaluation / CI gate

**ML Bugs caught:**

- Subgroup performance drops invisible in aggregate metrics
- Fairness violations across demographic groups
- Miscalibrated confidence scores
- Prediction instability under small perturbations
- Data leakage from suspiciously correlated features
- PII exposure, null concentrations, outlier spikes
- Feature distribution anomalies
- Overfitting (train-test performance gap)

---

## How Scanning Works

### What you give it

```
You have:
+-- A trained model (model.pkl, or a predict function)
+-- A test dataset (X = features, y = labels)
+-- Optionally: sensitive column names
```

### What it does

The scan takes your **live model + live data** and probes
from every angle:

```
scan() receives:
|
+-- model_fn(X) -> predictions    (your trained model)
+-- X (DataFrame)                (your test features)
+-- y (array)                    (your ground truth)
|
+-- SliceScanner:
|   "What if I only look at age > 55?"
|   -> accuracy drops from 91% to 58%
|
+-- BiasScanner:
|   "Does the model treat genders differently?"
|   -> approval rate 80% for M, 51% for F
|
+-- CalibrationScanner:
|   "When it says 90% confident, is it right 90%?"
|   -> says 90% but only right 65%
|
+-- RobustnessScanner:
|   "What if I add tiny noise to the features?"
|   -> 4% of predictions flip with 1% noise
|
+-- LeakageScanner:
|   "Are any features correlated with the target?"
|   -> future_revenue has 0.98 correlation
|
+-- DataScanner:
|   "Is the data itself healthy?"
|   -> 12% nulls in income, PII in email column
|
+-- DriftScanner:
|   "Do features look unusual?"
|   -> statistical tests per column
|
+-- OverfitScanner (if training data provided):
    "Is there a train-test gap?"
    -> train accuracy 99%, test accuracy 81%
```

### What it gives you back

```
ScanReport:
+-- findings: list[ScanFinding]     # each issue
|   +-- .result (TestResult)        # what happened
|   +-- .assertion_fn               # how to reproduce
|   +-- .assertion_args/kwargs      # exact arguments
|   +-- .suggested_test             # pytest code string
|
+-- to_test_file("tests/...")       # write pytest file
+-- to_suite().run()                # run as MltkSuite
+-- to_html("report.html")         # visual report
+-- to_junit("results.xml")        # CI/CD integration
+-- summary()                       # console text
```

**Key insight**: scanners are **assertion orchestrators**.
They figure out WHICH existing mltk assertions to call
with WHICH data slices, run them, and collect the
failures. They do not reinvent ML logic.

---

## Quick Start

### Python API

```python
from mltk.scan import scan

report = scan(
    model.predict,
    X_test,
    y_test,
    sensitive_columns=["age", "gender"],
)

# See what was found
print(report.summary())

# Generate a pytest file you can commit
report.to_test_file("tests/test_scan_results.py")

# Run findings as an MltkSuite
report.to_suite().run()

# Export for CI/CD
report.to_junit("scan-results.xml")
```

### CLI

```bash
mltk scan-model \
    --model model.pkl \
    --data test.csv \
    --target label \
    --sensitive age,gender \
    --output tests/test_scan_results.py \
    --junit-xml scan-results.xml
```

---

## Built-in Scanners

Eight scanners ship with mltk. They run in a fixed order
designed to maximize value per second -- data-only
scanners first (fast, no model needed), then model
scanners by ascending computational cost.

**MVP (v0.8.0):** SliceScanner, BiasScanner, LeakageScanner. **Planned:** DataScanner, DriftScanner, CalibrationScanner, RobustnessScanner, OverfitScanner.

| # | Scanner | Requires | Detects |
|---|---------|----------|---------|
| 1 | **DataScanner** | X | Nulls, PII, outliers |
| 2 | **DriftScanner** | X | Feature distribution anomalies |
| 3 | **LeakageScanner** | X, y | Feature-target correlations |
| 4 | **SliceScanner** | model, X, y | Performance drops in subgroups |
| 5 | **BiasScanner** | model, X, y, sensitive | Fairness violations |
| 6 | **CalibrationScanner** | predict_proba, X, y | Miscalibrated confidence |
| 7 | **RobustnessScanner** | model, X | Prediction instability |
| 8 | **OverfitScanner** | model, X, y, X_train | Train-test gap |

Scanners that cannot run (missing requirements) are
skipped automatically and listed in the report.

---

## ScanReport API

The `ScanReport` returned by `scan()` gives you multiple
ways to consume the findings.

### Properties and Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `summary()` | `str` | Console-formatted text with severity markers |
| `to_test_file(path)` | `str` | Write self-contained pytest file |
| `to_suite()` | `MltkSuite` | Runnable suite from findings |
| `to_html(path)` | `str` | Visual HTML report |
| `to_junit(path)` | `str` | JUnit XML for CI/CD |
| `exit_code` | `int` | 0=clean, 1=findings, 2=error |

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `findings` | `list[ScanFinding]` | Every issue discovered |
| `scanners_run` | `list[str]` | Scanners that executed |
| `scanners_skipped` | `list[str]` | Scanners skipped (missing requirements) |
| `scanners_errored` | `dict[str, str]` | Scanners that crashed (name -> error) |
| `duration_ms` | `float` | Total scan time in milliseconds |
| `model_type` | `str` | Detected: "classifier", "regressor", "unknown" |
| `n_samples` | `int` | Number of rows scanned |
| `n_features` | `int` | Number of feature columns |
| `config` | `ScanConfig` | Configuration used |

---

## ScanConfig

All fields have sensible defaults. You can call
`scan(model, X, y)` with zero config and get useful
results.

```python
from mltk.scan.config import ScanConfig

config = ScanConfig(
    max_scan_rows=5_000,
    time_budget_seconds=30.0,
    critical_drop=0.15,
    scanner_config={
        "slice": {"metric": "f1"},
        "bias": {"threshold": 0.05},
    },
)

report = scan(model.predict, X, y, config=config)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_scan_rows` | `int` | `10,000` | Max rows to scan (larger sets are sampled) |
| `sample_strategy` | `str` | `"stratified"` | How to sample: preserves class proportions |
| `time_budget_seconds` | `float` | `60.0` | Total wall-clock budget for the scan |
| `per_scanner_timeout` | `float` | `30.0` | Max seconds per individual scanner |
| `seed` | `int` | `42` | Random seed for reproducibility |
| `categorical_threshold` | `int` | `20` | Columns with <= this many unique values are categorical |
| `max_slices_per_column` | `int` | `50` | Max unique values to test per categorical column |
| `min_slice_samples` | `int` | `30` | Min samples in a slice for it to be tested |
| `critical_drop` | `float` | `0.20` | Absolute performance drop that triggers CRITICAL |
| `warning_drop` | `float` | `0.10` | Absolute performance drop that triggers WARNING |
| `scanner_config` | `dict` | `{}` | Per-scanner parameter overrides |
| `enabled_scanners` | `list[str]` | `None` | If set, ONLY these scanners run |
| `disabled_scanners` | `list[str]` | `None` | If set, these scanners are skipped |

---

## ScanFinding

Each finding carries **evidence** and a **reproduction
recipe**:

```python
@dataclass
class ScanFinding:
    result: TestResult        # what happened
    assertion_fn: Callable    # which assertion caught it
    assertion_args: tuple     # positional args
    assertion_kwargs: dict    # keyword args
    suggested_test: str       # pytest code string
    scanner_name: str         # which scanner
```

This dual nature is what makes `mltk scan` unique. Other
tools report issues. mltk reports issues AND gives you
runnable code to reproduce them.

### Replaying a finding

```python
# Add to a suite
fn, args, kwargs = finding.to_pending()
suite.add(fn, *args, **kwargs)

# Or run the assertion directly
finding.assertion_fn(
    *finding.assertion_args,
    **finding.assertion_kwargs,
)
```

---

## Generated Test Files

When you call `report.to_test_file("tests/test_scan.py")`,
mltk writes a **self-contained** pytest file. It includes
all imports, model loading, data loading, and one test
function per finding.

### Example output

```python
"""Auto-generated by mltk scan. Do not edit."""

import os

import numpy as np
import pandas as pd
import pytest

from mltk.model.metrics import assert_metric
from mltk.scan.loader import load_model

# Paths: override via environment variables
MODEL_PATH = os.environ.get(
    "MLTK_MODEL_PATH", "model.pkl"
)
DATA_PATH = os.environ.get(
    "MLTK_DATA_PATH", "test.csv"
)


@pytest.fixture(scope="module")
def model():
    loaded = load_model(MODEL_PATH)
    return loaded.predict_fn


@pytest.fixture(scope="module")
def data():
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["label"])
    y = df["label"].values
    return X, y


def test_slice_age_performance(model, data):
    """Accuracy for age > 55 must meet threshold."""
    X, y = data
    mask = X["age"] > 55
    y_pred = model(X[mask])
    assert_metric(
        y[mask], y_pred,
        metric="accuracy",
        threshold=0.7100,
    )
```

Environment variable fallbacks (`MLTK_MODEL_PATH`,
`MLTK_DATA_PATH`) let you run the same test file in
CI/CD without hardcoded paths.

---

## Console Output

The `summary()` method (and the CLI) prints a
human-readable report:

```
+-- mltk scan ----------------------------------------+
| Model: classifier (sklearn) | 10,000 samples        |
| Features: 15 numeric, 5 categorical                 |
| Scanners: 7/8 run (OverfitScanner skipped: no train)|
| Duration: 12.3s                                      |
+------------------------------------------------------+

  X CRITICAL  Accuracy drops to 0.58 for age > 55
              (overall: 0.91)           [SliceScanner]
  X CRITICAL  Demographic parity violation on gender
              (ratio: 0.64)              [BiasScanner]
  ! WARNING   Model uncalibrated (ECE: 0.15)
                                  [CalibrationScanner]
  ! WARNING   3 features have > 0.8 correlation
              with target             [LeakageScanner]
  i INFO      Predictions unstable under 1% noise
              (4% flip rate)       [RobustnessScanner]

Summary: 2 critical, 2 warnings, 1 info
-> Run: pytest tests/test_scan_results.py
```

---

## Exit Codes

| Code | Meaning | When |
|------|---------|------|
| `0` | Clean | No findings at any severity |
| `1` | Findings | One or more issues detected |
| `2` | Error | Model/data could not be loaded, or scan crashed |

In CI/CD pipelines, use exit code 1 to fail the build
when the scan finds issues:

```yaml
- name: Scan model
  run: |
    mltk scan-model \
        --model model.pkl \
        --data test.csv \
        --target label
```

---

## Custom Scanners

Write your own scanner by subclassing `Scanner` and
registering it:

```python
from mltk.scan import register_scanner
from mltk.scan.scanners.base import Scanner
from mltk.scan.config import ScanContext
from mltk.scan.finding import ScanFinding


class LatencyScanner(Scanner):
    """Check prediction latency stays under budget."""

    name = "latency"
    category = "performance"
    requires = {"model_fn", "X"}

    def scan(
        self, ctx: ScanContext,
    ) -> list[ScanFinding]:
        import time

        start = time.perf_counter()
        ctx.model_fn(ctx.X.iloc[:100])
        elapsed_ms = (
            (time.perf_counter() - start) * 1000
        )
        # Build findings if latency exceeds budget...
        return []


# Register so scan() picks it up automatically
register_scanner(LatencyScanner)
```

### Scanner contract

1. Set `name` (short lowercase identifier)
2. Set `category` (grouping label for reports)
3. Set `requires` (set of ScanContext field names)
4. Implement `scan(ctx) -> list[ScanFinding]`

The engine handles timeout enforcement, error isolation,
and requirement checking. If your scanner raises an
exception, the engine catches it and continues.

---

## CLI Reference

```
mltk scan-model [OPTIONS]
```

### Options

| Option | Type | Required | Description |
|--------|------|----------|-------------|
| `--model` | PATH | Yes | Path to serialized model file |
| `--data` | PATH | Yes | Path to CSV or parquet dataset |
| `--target` | TEXT | Yes | Name of the target column |
| `--sensitive` | TEXT | No | Comma-separated sensitive column names |
| `--output` | PATH | No | Write generated pytest file |
| `--junit-xml` | PATH | No | Write JUnit XML report |

### Supported model formats

| Extension | Format | Required package |
|-----------|--------|------------------|
| `.pkl`, `.pickle` | pickle | `joblib` |
| `.joblib` | joblib | `joblib` |
| `.onnx` | ONNX | `onnxruntime` |
| `.pt`, `.pth` | PyTorch | `torch` |
| `.h5`, `.hdf5`, `.keras` | Keras | `tensorflow` or `keras` |

### Examples

```bash
# Basic scan
mltk scan-model \
    --model model.pkl \
    --data test.csv \
    --target label

# With sensitive columns and test generation
mltk scan-model \
    --model model.pkl \
    --data test.csv \
    --target label \
    --sensitive age,gender,race \
    --output tests/test_scan_results.py

# Full pipeline: scan + JUnit for CI
mltk scan-model \
    --model model.pkl \
    --data test.csv \
    --target label \
    --output tests/test_scan_results.py \
    --junit-xml scan-results.xml
```

---

## FAQ

### What if my model is not scikit-learn?

`mltk scan` works with any model that has a `predict()`
method or is callable. The model loader supports pickle,
joblib, ONNX, PyTorch, and Keras formats. For custom
frameworks, pass the predict function directly via the
Python API:

```python
report = scan(my_custom_model.predict, X, y)
```

### What about large datasets?

By default, datasets larger than 10,000 rows are sampled
down using stratified sampling (preserving class
proportions for classifiers, quantile-binned for
regressors). Control this with `ScanConfig`:

```python
config = ScanConfig(max_scan_rows=50_000)
report = scan(model.predict, X, y, config=config)
```

### Can I run only specific scanners?

Yes. Use `enabled_scanners` to run only what you need,
or `disabled_scanners` to skip specific ones:

```python
config = ScanConfig(
    enabled_scanners=["slice", "bias"],
)
```

### How do I use this in CI/CD?

The CLI returns exit code 1 when findings are detected.
Use it as a build gate:

```yaml
- run: |
    mltk scan-model \
        --model model.pkl \
        --data test.csv \
        --target label \
        --junit-xml scan-results.xml
```

The JUnit XML output integrates with most CI platforms
for inline test reporting.

### Does scanning modify my model or data?

No. Scanning is read-only. The model is called with
subsets of data and small perturbations (for robustness
testing), but the original model and data are never
modified.
