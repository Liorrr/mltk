# Getting Started

This hands-on tutorial takes you from zero to running ML tests with reports in about 5 minutes.

---

## Step 1: Install (30 seconds)

### Basic install

```bash
pip install mltk
```

This installs mltk with core dependencies: `numpy` and `pandas`.

### Recommended install

For the CLI and HTML reports -- what most users want:

```bash
pip install mltk[cli,report]
```

### All optional extras

```bash
# Statistical tests (scipy-based drift detection)
pip install "mltk[scipy]"

# Domain kits
pip install "mltk[cv]"      # Computer vision (OpenCV)
pip install "mltk[nlp]"     # NLP (NLTK, rouge-score)
pip install "mltk[speech]"  # Speech (jiwer)

# Embedding-based evaluation (sentence-transformers)
pip install "mltk[embedding]"

# NLI-based evaluation (bidirectional entailment)
pip install "mltk[nli]"

# Toxicity classifier (transformer-based)
pip install "mltk[classifier]"

# Server platform (dashboard + REST API)
pip install "mltk[server]"

# All optional dependencies
pip install "mltk[all]"

# Development (pytest, ruff, mypy)
pip install "mltk[dev]"
```

### Requirements

- Python 3.10 or later
- numpy >= 1.24
- pandas >= 2.0

### Verify your install

```bash
mltk doctor
```

Expected output:

```text
mltk doctor
  [OK]   Python version: 3.12.1 (>= 3.10)
  [OK]   Core deps: numpy 1.26.4, pandas 2.2.0
  [WARN] Optional dep: scipy not installed (pip install mltk[scipy])
  [OK]   Config: mltk.yaml found
  [OK]   Report dir: ./mltk-reports exists
  [FAIL] Baseline dir: ./mltk-baselines not found
  [OK]   Rust extension: loaded
  [OK]   pytest plugin: registered
  [OK]   Config valid: no misconfigurations
```

`mltk doctor` checks 9 things: Python version, core dependencies, optional dependencies, config files, report directory, baseline directory, Rust extension, pytest plugin registration, and config validity. Every `WARN` or `FAIL` includes a fix hint.

---

## Step 2: Scaffold (1 minute)

```bash
mltk init
```

This creates two files:

### `mltk.yaml` -- project configuration

```yaml
# mltk configuration
drift_method: ks
drift_threshold: 0.05
report_dir: ./mltk-reports
seed: 42
```

| Key | What it does |
|-----|-------------|
| `drift_method` | Statistical test for drift detection. Options: `ks` (Kolmogorov-Smirnov), `psi` (Population Stability Index), `kl` (KL divergence), `chi2` (chi-squared). |
| `drift_threshold` | Values below this threshold trigger a drift alert. |
| `report_dir` | Where HTML reports are saved. |
| `seed` | Random seed for reproducible nondeterministic tests. |

### `tests/test_mltk_example.py` -- starter test

```python
"""Example mltk test file."""

import pandas as pd
import pytest

from mltk.data import assert_no_nulls, assert_row_count, assert_schema


@pytest.mark.ml_data
def test_data_quality():
    # Replace with your actual data path
    df = pd.DataFrame({"id": [1, 2, 3], "value": [1.0, 2.0, 3.0]})
    assert_schema(df, {"id": "int64", "value": "float64"})
    assert_no_nulls(df)
    assert_row_count(df, min_rows=1)
```

The `@pytest.mark.ml_data` marker lets you run only data tests with `pytest -m ml_data`. mltk registers this marker automatically -- no `conftest.py` needed.

---

## Step 3: Write Your First Test (3 minutes)

Replace the example file or create a new one. Here is a complete test that covers data quality, model validation, and inference performance:

Create `tests/test_ml_pipeline.py`:

```python
"""End-to-end ML pipeline tests."""

import numpy as np
import pandas as pd
import pytest
import time

from mltk.data import assert_schema, assert_no_nulls, assert_range, assert_row_count
from mltk.model import assert_metric, assert_no_regression
from mltk.inference import assert_latency


# ── Data quality ────────────────────────────────────────

@pytest.mark.ml_data
def test_training_data_schema():
    """Every column must match the expected type.

    Why: A schema mismatch (e.g., string where int64 is expected)
    causes silent model degradation -- the model trains on garbage.
    """
    df = pd.DataFrame({
        "user_id": [1, 2, 3, 4, 5],
        "age": [25, 30, 35, 40, 45],
        "score": [0.8, 0.9, 0.7, 0.95, 0.6],
        "label": [1, 0, 1, 1, 0],
    })

    assert_schema(df, {
        "user_id": "int64",
        "age": "int64",
        "score": "float64",
        "label": "int64",
    })


@pytest.mark.ml_data
def test_no_missing_labels():
    """Every row must have a label.

    Why: Missing labels are silently dropped during training,
    reducing your effective dataset size and biasing the model.
    """
    df = pd.DataFrame({
        "feature_a": [1.0, 2.0, 3.0, 4.0],
        "feature_b": [0.5, 0.6, 0.7, 0.8],
        "label": [0, 1, 1, 0],
    })

    assert_no_nulls(df, columns=["label"])


@pytest.mark.ml_data
def test_feature_ranges():
    """Probability scores must be in [0, 1].

    Why: Out-of-range values indicate a preprocessing bug
    or data corruption upstream.
    """
    df = pd.DataFrame({
        "score": [0.1, 0.5, 0.9, 0.3, 0.7],
    })

    assert_range(df["score"], min_val=0.0, max_val=1.0)


@pytest.mark.ml_data
def test_minimum_dataset_size():
    """Training needs at least 1000 rows.

    Why: Small datasets lead to overfitting. This catches
    truncated exports or failed data pipeline runs.
    """
    df = pd.DataFrame({"x": range(1500), "y": range(1500)})

    assert_row_count(df, min_rows=1000)


# ── Model quality ───────────────────────────────────────

@pytest.mark.ml_model
def test_model_accuracy():
    """Model accuracy must exceed 85%.

    Why: This is the minimum bar for production deployment.
    A drop below this means the model is not useful.
    """
    y_true = [1, 0, 1, 1, 0, 1, 0, 0, 1, 1]
    y_pred = [1, 0, 1, 1, 0, 1, 1, 0, 1, 1]

    assert_metric(y_true, y_pred, metric="accuracy", threshold=0.85)


@pytest.mark.ml_model
def test_no_accuracy_regression():
    """New model must not regress vs. baseline.

    Why: Retraining on new data can silently degrade performance.
    This catches regressions before they ship.
    """
    y_true_reg = [1, 0, 1, 1, 0, 1, 0, 0, 1, 1]
    y_pred_reg = [1, 0, 1, 1, 0, 1, 1, 0, 1, 1]

    assert_no_regression(
        y_true=y_true_reg,
        y_pred=y_pred_reg,
        baseline=0.89,
        metric="accuracy",
        tolerance=0.02,
    )


# ── Inference performance ───────────────────────────────

@pytest.mark.ml_inference
def test_prediction_latency():
    """Single prediction must complete in < 100ms.

    Why: Slow predictions cause timeouts in production APIs.
    This catches performance regressions before deployment.
    """
    def predict():
        time.sleep(0.01)  # Simulate a 10ms prediction
        return [0.95]

    assert_latency(predict, p99=100)
```

### What each assertion does

| Assertion | Plain-language meaning |
|-----------|----------------------|
| `assert_schema(df, spec)` | "Every column must have this exact dtype" |
| `assert_no_nulls(df, columns)` | "These columns must have zero missing values" |
| `assert_range(series, min, max)` | "Every value must be within this range" |
| `assert_row_count(df, min_rows)` | "The dataset must have at least this many rows" |
| `assert_metric(y_true, y_pred, metric, threshold)` | "Model metric must meet the threshold" |
| `assert_no_regression(y_true, y_pred, baseline)` | "The new score must not drop below the baseline minus tolerance" |
| `assert_latency(fn, p99=ms)` | "99th percentile latency must be under this limit" |

---

## Step 4: Run Tests (30 seconds)

```bash
pytest tests/test_ml_pipeline.py -v --mltk-report
```

Expected output:

```text
tests/test_ml_pipeline.py::test_training_data_schema PASSED
tests/test_ml_pipeline.py::test_no_missing_labels PASSED
tests/test_ml_pipeline.py::test_feature_ranges PASSED
tests/test_ml_pipeline.py::test_minimum_dataset_size PASSED
tests/test_ml_pipeline.py::test_model_accuracy PASSED
tests/test_ml_pipeline.py::test_no_accuracy_regression PASSED
tests/test_ml_pipeline.py::test_prediction_latency PASSED

7 passed in 0.34s

MLTK Report: ./mltk-reports/report-20260326-120000.html
```

### Run subsets with markers

```bash
# Only data quality tests
pytest -m ml_data -v

# Only model tests
pytest -m ml_model -v

# Only inference tests
pytest -m ml_inference -v

# Everything except slow tests
pytest -m "not ml_slow" -v
```

### Available markers

| Marker | Purpose |
|--------|---------|
| `@pytest.mark.ml_data` | Data quality tests |
| `@pytest.mark.ml_model` | Model quality tests |
| `@pytest.mark.ml_drift` | Drift detection tests |
| `@pytest.mark.ml_inference` | Inference performance tests |
| `@pytest.mark.ml_slow` | Tests that take longer than 30 seconds |
| `@pytest.mark.ml_nondeterministic` | Tests with inherent randomness |

---

## Step 5: View the Report (1 minute)

Open the generated HTML file in your browser:

```bash
# The path is printed in the pytest output
open ./mltk-reports/report-20260326-120000.html
```

The report includes:

1. **Summary bar** -- total pass/fail/warning counts at a glance
2. **Module breakdown** -- results grouped by test module (data, model, inference)
3. **Test details table** -- each assertion with its severity, duration in milliseconds, and pass/fail message
4. **Dark theme** -- slate background with purple accent, easy on the eyes

Reports are self-contained single HTML files with embedded Plotly charts. Share them as email attachments, upload to Confluence, or archive in your artifact store.

---

## Step 6: Understanding TestResult

Every mltk assertion returns a `TestResult` object with structured data:

```python
from mltk.data import assert_schema

result = assert_schema(df, {"id": "int64", "name": "object"})

print(result.passed)       # True or False
print(result.name)         # "data.schema"
print(result.message)      # "Schema valid" or error description
print(result.severity)     # Severity.CRITICAL, WARNING, or INFO
print(result.details)      # Dict with specifics (missing columns, type mismatches)
print(result.duration_ms)  # Execution time in milliseconds
```

**Severity levels:**

| Severity | Behavior |
|----------|----------|
| `CRITICAL` | Raises `MltkAssertionError` (subclass of `AssertionError`) -- pytest catches it as a test failure |
| `WARNING` | Returns the result without raising -- useful for logging warnings without blocking the test run |
| `INFO` | Informational only -- always passes |

---

## Alternative: MltkSuite (No pytest required)

Not every environment has pytest. Jupyter notebooks, standalone scripts, CI pipelines with custom runners, and educational tutorials all benefit from a self-contained test runner. `MltkSuite` is a composable API that collects assertion results, computes a pass/fail summary, and renders rich output in notebooks -- no test framework needed.

### Basic usage

```python
from mltk.core import MltkSuite
from mltk.data import assert_schema, assert_no_nulls, assert_range

import pandas as pd

df = pd.DataFrame({
    "user_id": [1, 2, 3],
    "score": [0.8, 0.9, 0.7],
    "label": [1, 0, 1],
})

suite = MltkSuite()
suite.add(assert_schema(df, {"user_id": "int64", "score": "float64", "label": "int64"}))
suite.add(assert_no_nulls(df, columns=["label"]))
suite.add(assert_range(df["score"], min_val=0.0, max_val=1.0))

print(f"{suite.passed_count}/{suite.total} passed ({suite.score:.1f}%)")
assert suite.passed, f"Suite failed: {suite.failed_count} failures"
```

In Jupyter notebooks, `MltkSuite` renders an interactive HTML table automatically -- just put `suite` as the last expression in a cell.

### Comparison: pytest vs. MltkSuite

| | pytest approach | MltkSuite approach |
|---|---|---|
| **Setup** | `pip install mltk[dev]`, write `test_*.py` files | `pip install mltk`, use in any `.py` or `.ipynb` |
| **Run** | `pytest --mltk-report -v` | `suite.passed` / `suite.score` |
| **Output** | Terminal + HTML report file | Inline notebook display or print summary |
| **CI/CD** | Native: exit code, JUnit XML | Use `assert suite.passed` for exit code |
| **Best for** | Test suites, regression gates, team CI | Notebooks, scripts, demos, quick checks |

Both approaches use the same `assert_*` functions. Every assertion returns a `TestResult` object, so you can pass the result to `suite.add()` or let pytest catch the `MltkAssertionError` -- your choice.

### Notebook example

```python
# In a Jupyter cell:
from mltk.core import MltkSuite
from mltk.model import assert_metric, assert_no_regression

suite = MltkSuite()

y_true = [1, 0, 1, 1, 0, 1, 0, 0, 1, 1]
y_pred = [1, 0, 1, 1, 0, 1, 1, 0, 1, 1]

suite.add(assert_metric(y_true, y_pred, metric="accuracy", threshold=0.85))
suite.add(assert_no_regression(y_true, y_pred, baseline=0.89, metric="accuracy", tolerance=0.02))

suite  # renders a rich HTML table in Jupyter
```

For the full API reference (all properties, methods, HTML display, and report export), see **[MltkSuite API](api/suite-api.md)**.

---

## Next Steps

Now that you have mltk running, pick the path that matches your role:

### "I want no-code tests"

Define tests in YAML without writing Python. Ideal for QA teams.

:point_right: [YAML Test Definitions](api/yaml-tests.md)

### "I want data contracts"

Define expected data quality in YAML and auto-generate tests from contracts.

:point_right: [Data Contracts](api/contracts.md)

### "I need tests in CI/CD"

Add mltk to GitHub Actions, GitLab CI, or Jenkins. Block merges on ML test failures.

:point_right: [CI/CD Integration](guides/cicd-integration.md)

### "I need a test dashboard for my team"

Self-hosted server with persistent storage, live dashboard, trend tracking, and webhooks.

:point_right: [Server Platform](api/server-platform.md)

### "I need compliance evidence"

Generate EU AI Act compliance reports or FDA 21 CFR Part 11 audit trails from your test results.

:point_right: [EU AI Act Compliance](api/eu-ai-act.md) | [FDA Audit Trail](api/fda-audit.md)

### "I want to test LLMs and RAG"

Evaluate faithfulness, relevance, coherence, safety, and agentic tool use.

:point_right: [LLM Evaluation](api/llm.md) | [RAG & Agentic](api/rag-evaluation.md)

### "I want to scan data from the command line"

Quick one-off data quality checks without writing any code.

:point_right: [CLI Reference](api/cli.md)

---

## Full Reference

- [Configuration](configuration.md) -- customize mltk via `mltk.yaml`, `pyproject.toml`, or environment variables
- [pytest Plugin](api/pytest-plugin.md) -- markers, fixtures, and `--mltk-report` flag
- [CLI Reference](api/cli.md) -- all 14 commands
- [Testing Patterns](api/testing-patterns.md) -- flaky detection, golden baselines, smart test selection
- [All API modules](api/core.md) -- complete API reference
