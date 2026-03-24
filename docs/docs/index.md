# mltk -- ML Test Kit

**pytest for ML** -- unified testing across the entire ML lifecycle.

mltk is a Python testing toolkit that brings the simplicity of `assert` statements to ML system testing. One package covers data quality, model validation, drift detection, fairness testing, inference benchmarking, and production monitoring.

## Why mltk?

ML systems fail silently. A model can train successfully on corrupt data, produce confident predictions from stale features, and pass every unit test while being completely wrong in production. Traditional software testing does not catch these failure modes.

mltk provides purpose-built assertions for every stage of the ML lifecycle -- from raw data ingestion through production monitoring -- all runnable with `pytest`.

## Features

- **Data Testing** -- schema validation, distribution checks, drift detection, PII scanning, label quality
- **Model Testing** -- metrics, regression detection, slicing analysis, bias auditing, adversarial robustness
- **Inference Testing** -- latency benchmarks (P50/P95/P99), throughput measurement, API contract validation
- **Domain Kits** -- specialized assertions for CV, NLP, Speech, and Tabular data (optional extras)
- **pytest Plugin** -- ML-specific markers, fixtures, and auto-registration via entry points
- **Rust Acceleration** -- optional 10-100x speedup for drift detection via PyO3/Maturin
- **HTML Reports** -- interactive Plotly charts for test results
- **CLI** -- `mltk scan`, `mltk drift`, `mltk score` commands
- **YAML Config** -- no-code test definitions via `mltk.yaml` or `pyproject.toml`

## Quick Start

Install mltk:

```bash
pip install mltk
```

Write your first ML data test:

```python
import pandas as pd
from mltk.data import assert_schema, assert_no_nulls, assert_freshness

def test_training_data():
    df = pd.read_parquet("data/training.parquet")

    # Verify structure
    assert_schema(df, {
        "user_id": "int64",
        "feature_a": "float64",
        "feature_b": "float64",
        "label": "int64",
    })

    # Verify completeness
    assert_no_nulls(df, columns=["label", "feature_a", "feature_b"])

    # Verify recency
    assert_freshness(df, date_column="created_at", max_age_days=7)
```

Run with pytest:

```bash
pytest -m ml_data --tb=short
```

## Project Status

mltk is in **v0.1.0** (alpha). The core data quality assertions are stable and tested. Model, inference, and domain modules are under active development.

## License

Apache-2.0
