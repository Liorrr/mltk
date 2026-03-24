# Getting Started

This guide walks you through installing mltk, writing your first ML data test, and running it with pytest.

## Installation

### Basic install

```bash
pip install mltk
```

This installs mltk with its core dependencies: `numpy` and `pandas`.

### Optional extras

mltk ships optional extras for specific use cases:

```bash
# Statistical tests (scipy-based drift detection)
pip install "mltk[scipy]"

# CLI tools (typer + rich)
pip install "mltk[cli]"

# HTML report generation (plotly + jinja2)
pip install "mltk[report]"

# Domain kits
pip install "mltk[cv]"      # Computer vision (OpenCV)
pip install "mltk[nlp]"     # NLP (NLTK, rouge-score)
pip install "mltk[speech]"  # Speech (jiwer)

# Everything
pip install "mltk[all]"

# Development (pytest, ruff, mypy)
pip install "mltk[dev]"

# Documentation (mkdocs-material, mkdocstrings)
pip install "mltk[docs]"
```

### Requirements

- Python 3.10 or later
- numpy >= 1.24
- pandas >= 2.0

## Your First Test

Create a file called `test_data_quality.py`:

```python
import pandas as pd
import numpy as np
from mltk.data import (
    assert_schema,
    assert_no_nulls,
    assert_range,
    assert_row_count,
)


def test_training_data_schema():
    """Verify the training dataset has the expected structure."""
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


def test_no_missing_labels():
    """Ensure every row has a label -- partial labels corrupt training."""
    df = pd.DataFrame({
        "feature": [1.0, 2.0, 3.0],
        "label": [0, 1, 1],
    })

    assert_no_nulls(df, columns=["label"])


def test_feature_ranges():
    """Verify probability scores are in [0, 1]."""
    df = pd.DataFrame({
        "score": [0.1, 0.5, 0.9, 0.3, 0.7],
    })

    assert_range(df["score"], min_val=0.0, max_val=1.0)


def test_dataset_size():
    """Ensure the dataset has a minimum number of rows for training."""
    df = pd.DataFrame({"x": range(500)})

    assert_row_count(df, min_rows=100, max_rows=10000)
```

## Running Tests

Run all tests:

```bash
pytest test_data_quality.py -v
```

Run only ML data tests using the built-in marker:

```bash
pytest -m ml_data -v
```

Expected output:

```
test_data_quality.py::test_training_data_schema PASSED
test_data_quality.py::test_no_missing_labels PASSED
test_data_quality.py::test_feature_ranges PASSED
test_data_quality.py::test_dataset_size PASSED

4 passed in 0.12s
```

## Understanding Test Results

Every mltk assertion returns a `TestResult` object with structured data:

```python
from mltk.data import assert_schema

result = assert_schema(df, {"id": "int64", "name": "object"})

print(result.passed)      # True or False
print(result.name)        # "data.schema"
print(result.message)     # "Schema valid" or error description
print(result.severity)    # Severity.CRITICAL, WARNING, or INFO
print(result.details)     # Dict with specifics (missing columns, etc.)
print(result.duration_ms) # Execution time in milliseconds
```

When a `CRITICAL` severity assertion fails, it raises `MltkAssertionError` -- which is a subclass of `AssertionError`, so pytest catches it naturally.

`WARNING` severity assertions return the result without raising, so you can log warnings without blocking the test run.

## pytest Markers

mltk registers these markers automatically via its pytest plugin:

| Marker | Purpose |
|--------|---------|
| `@pytest.mark.ml_data` | Data quality tests |
| `@pytest.mark.ml_model` | Model quality tests |
| `@pytest.mark.ml_drift` | Drift detection tests |
| `@pytest.mark.ml_inference` | Inference performance tests |
| `@pytest.mark.ml_slow` | Tests that take longer than 30 seconds |
| `@pytest.mark.ml_nondeterministic` | Tests with inherent randomness |

Use markers to run subsets:

```bash
# Only data quality
pytest -m ml_data

# Everything except slow tests
pytest -m "not ml_slow"

# Data + drift but not nondeterministic
pytest -m "(ml_data or ml_drift) and not ml_nondeterministic"
```

## Next Steps

- [Configuration](configuration.md) -- customize mltk via `pyproject.toml` or `mltk.yaml`
- [API Reference: Core](api/core.md) -- `TestResult`, `TestSuite`, `MltkConfig`, `assert_true`
- [API Reference: Data Schema](api/data-schema.md) -- `assert_schema`, `assert_no_nulls`, `assert_dtypes`
- [API Reference: Data Distribution](api/data-distribution.md) -- `assert_range`, `assert_unique`, `assert_no_outliers`
- [API Reference: Data Freshness](api/data-freshness.md) -- `assert_freshness`, `assert_row_count`
