# mltk - ML Test Kit

**pytest for ML** -- unified testing across the entire ML lifecycle.

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://python.org)
[![Tests](https://img.shields.io/badge/tests-204%20passed-green.svg)]()
[![Rust](https://img.shields.io/badge/rust-accelerated-orange.svg)]()

```bash
pip install mltk
```

## Why mltk?

ML systems fail silently. A model can train on corrupt data, produce confident predictions from stale features, and pass every unit test while being completely wrong in production. Traditional testing doesn't catch these failures.

**mltk** is one toolkit that covers the entire ML lifecycle: data quality, model validation, drift detection, fairness testing, inference benchmarking, and production monitoring. No more gluing together 5 different tools.

## Quick Start

```python
import pytest
from mltk.data import assert_schema, assert_no_drift, assert_no_pii
from mltk.model import assert_metric, assert_no_regression, assert_no_bias

@pytest.mark.ml_data
def test_training_data():
    df = pd.read_csv("data/training.csv")
    assert_schema(df, {"id": "int64", "text": "object", "label": "int64"})
    assert_no_nulls(df, columns=["label"])
    assert_no_pii(df, columns=["text"])

@pytest.mark.ml_model
def test_model_quality(y_true, y_pred):
    assert_metric(y_true, y_pred, metric="f1", threshold=0.85)
    assert_no_regression(y_true, y_pred, baseline=0.90, tolerance=0.02)
    assert_no_bias(y_true, y_pred, sensitive_feature=gender,
                   method="demographic_parity")

@pytest.mark.ml_inference
def test_inference_performance():
    assert_latency(model.predict, X_test, p95=50.0, warmup=5)
    assert_throughput(model.predict, X_single, min_rps=100)
```

Run with HTML report:
```bash
pytest --mltk-report
```

## Feature Matrix (47 assertions)

| Module | Assertions | Purpose |
|--------|-----------|---------|
| **Data Schema** | `assert_schema`, `assert_no_nulls`, `assert_dtypes` | Validate DataFrame structure |
| **Data Distribution** | `assert_range`, `assert_unique`, `assert_no_outliers` | Verify statistical properties |
| **Data Drift** | `assert_no_drift` (KS, PSI, KL, Chi2) | Detect distribution shifts |
| **Data PII** | `assert_no_pii`, `scan_pii` (14 patterns) | Find leaked personal data |
| **Data Freshness** | `assert_freshness`, `assert_row_count` | Verify data recency and size |
| **Data Labels** | `assert_label_balance`, `assert_label_coverage` | Validate label quality |
| **Model Metrics** | `assert_metric` (9 metrics: accuracy, F1, AUC, MSE, R2...) | Quality gates |
| **Model Regression** | `assert_no_regression`, `save_baseline` | Catch silent degradation |
| **Model Slicing** | `assert_slice_performance`, `assert_calibration` | Subgroup fairness + ECE |
| **Model Bias** | `assert_no_bias` (5 methods) | EU AI Act / four-fifths rule |
| **Model Adversarial** | `assert_robust` | Perturbation stability |
| **Inference** | `assert_latency`, `assert_cold_start`, `assert_throughput`, `assert_api_contract` | Performance SLAs |
| **Pipeline** | `assert_reproducible`, `assert_checksum`, `assert_pipeline` | Determinism + E2E |
| **Monitor** | `assert_no_degradation`, `assert_sla` | Production health |
| **CV** | `assert_iou`, `assert_map`, `assert_frame_accuracy`, `assert_temporal_consistency`, `assert_topk_accuracy` | Video analytics |
| **NLP** | `assert_bleu`, `assert_rouge`, `assert_ner_f1`, `assert_no_prompt_injection` | Text + security |
| **Speech** | `assert_wer`, `assert_cer`, `assert_rtf`, `assert_accent_coverage` | Recognition + fairness |
| **Tabular** | `assert_feature_drift`, `assert_feature_importance_stable`, `assert_class_balance` | Feature validation |

## Installation

```bash
# Core (data + model assertions)
pip install mltk

# With CLI
pip install mltk[cli]

# With HTML reports
pip install mltk[report]

# Domain kits
pip install mltk[cv]       # Computer Vision
pip install mltk[nlp]      # NLP
pip install mltk[speech]   # Speech Recognition

# Everything
pip install mltk[all]
```

## CLI

```bash
mltk version                           # Show version
mltk init                              # Scaffold config + example tests
mltk scan data/training.csv            # Quick data quality scan
mltk drift data/ref.csv data/cur.csv   # Drift analysis
mltk score                             # ML Test Score (Google 28-test rubric)
```

## pytest Plugin

Auto-registered on install. Provides ML-specific markers and `--mltk-report`:

```bash
pytest -m ml_data                 # Run only data quality tests
pytest -m ml_model                # Run only model quality tests
pytest -m "not ml_slow"           # Skip long-running tests
pytest --mltk-report              # Generate HTML report
```

Markers: `ml_data`, `ml_model`, `ml_drift`, `ml_inference`, `ml_smoke`, `ml_slow`, `ml_gpu`, `ml_nondeterministic`

## Rust Acceleration

Optional Rust backend for 10-100x speedup on drift detection (KS test, PSI). Falls back to scipy/numpy automatically.

## Comparison

| Feature | Great Expectations | Deepchecks | Evidently | **mltk** |
|---|---|---|---|---|
| Data validation | Yes | Yes | Limited | **Yes** |
| Model testing | No | Yes | Limited | **Yes** |
| Drift detection | No | Yes | Yes | **Yes (Rust)** |
| Bias/fairness | No | No | No | **Yes** |
| Inference testing | No | No | No | **Yes** |
| pytest native | No | No | No | **Yes** |
| CLI | Yes | No | No | **Yes** |
| Domain kits | No | Partial | No | **Yes** |
| ML Test Score | No | No | No | **Yes** |
| Rust acceleration | No | No | No | **Yes** |
| License | Elastic | AGPL | Apache 2.0 | **Apache 2.0** |
| Core deps | Heavy | Heavy | Medium | **2 (numpy, pandas)** |

## Documentation

- [Getting Started](docs/getting-started.md)
- [Configuration](docs/configuration.md)
- [API Reference](docs/api/)
- [Examples](examples/)

## License

Apache-2.0
