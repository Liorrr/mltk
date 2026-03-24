# mltk - ML Test Kit

**pytest for ML** -- unified testing across the entire ML lifecycle.

```bash
pip install mltk
```

## What is mltk?

mltk is a Python testing toolkit that brings the simplicity of `assert` statements to ML system testing. One package covers data quality, model validation, drift detection, fairness testing, inference benchmarking, and production monitoring.

## Quick Start

```python
from mltk.data import assert_schema, assert_no_drift, assert_no_pii
from mltk.model import assert_metric, assert_no_regression, assert_no_bias

# Data quality
def test_training_data(df):
    assert_schema(df, {"id": "int64", "text": "object", "label": "int64"})
    assert_no_pii(df, columns=["text"])

# Model quality
def test_model(y_true, y_pred):
    assert_metric(y_true, y_pred, metric="f1", threshold=0.85)
    assert_no_bias(y_true, y_pred, protected_attribute=gender)
```

Run with pytest:
```bash
pytest -m ml_data --mltk-report
```

## Features

- **Data Testing** -- schema, distribution, drift, PII, label quality
- **Model Testing** -- metrics, regression, slicing, bias, adversarial robustness
- **Inference Testing** -- latency (P50/P95/P99), throughput, API contracts
- **Domain Kits** -- CV, NLP, Speech, Tabular (optional extras)
- **pytest Plugin** -- markers, fixtures, auto-registered
- **Rust Acceleration** -- optional 10-100x speedup for drift detection
- **HTML Reports** -- interactive Plotly charts
- **CLI** -- `mltk scan`, `mltk drift`, `mltk score`
- **YAML Config** -- no-code test definitions

## License

Apache-2.0
