# mltk - ML Test Kit

**pytest for ML** -- unified testing across the entire ML lifecycle.

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://python.org)
[![Tests](https://img.shields.io/badge/tests-700%2B%20passed-green.svg)]()
[![Rust](https://img.shields.io/badge/rust-accelerated-orange.svg)]()

```bash
pip install mltk
```

## Why mltk?

ML systems fail silently. A model can train on corrupt data, produce confident predictions from stale features, and pass every unit test while being completely wrong in production. Traditional testing doesn't catch these failures.

**mltk** is one toolkit that covers the entire ML lifecycle: data quality, model validation, drift detection, fairness testing, inference benchmarking, training bug detection, LLM evaluation, and production monitoring. No more gluing together 5 different tools.

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

## New in v0.2.0

**YAML test definitions** — write ML tests in YAML, no Python required. Run with `mltk test tests.yaml`. Supports 11 data assertions with `env:VAR_NAME` data source for CI/CD.

**EU AI Act compliance reports** — `mltk compliance results.json` auto-maps test results to EU AI Act articles, classifies risk level (unacceptable/high/limited/minimal), and generates an HTML evidence table with gap analysis.

**`mltk doctor`** — 9 diagnostic checks that inspect your Python version, dependencies, config, Rust extension, and pytest plugin, with actionable fix hints for each failure.

**Data contracts** — define column types, ranges, and uniqueness rules in YAML. Use `mltk contract generate-tests` to auto-generate a pytest file from the contract — zero boilerplate.

**LLM evaluation** — `assert_semantic_similarity`, `assert_no_toxicity`, `assert_no_hallucination`, `assert_ttft`, and `assert_itl` for GenAI systems.

**Training bug detection** — catch data leakage (`assert_no_train_test_overlap`, `assert_temporal_split`, `assert_no_target_leakage`), gradient failures (`assert_gradient_flow`, `assert_no_vanishing_gradient`, `assert_no_exploding_gradient`, `assert_loss_finite`), and numerical instability (`assert_no_nan_inf`, `assert_loss_decreasing`, `assert_no_loss_divergence`, `assert_softmax_valid`).

**Environment variable config** — all config keys available as `MLTK_*` env vars, highest priority in the cascade. Works out of the box in CI/CD pipelines.

**JSON export** — `--mltk-export-json` flag exports full test results to JSON for downstream tooling.

## Feature Matrix (119+ assertions)

| Module | Assertions | Purpose |
|--------|-----------|---------|
| **Data Schema** | `assert_schema`, `assert_no_nulls`, `assert_dtypes` | Validate DataFrame structure |
| **Data Distribution** | `assert_range`, `assert_unique`, `assert_no_outliers` | Verify statistical properties |
| **Data Drift** | `assert_no_drift` (KS, PSI, KL, Chi2) | Detect distribution shifts |
| **Data Drift (Advanced)** | `assert_no_embedding_drift` + 7 methods (KS, PSI, KL, Chi2, JS, Wasserstein, auto) | Embedding + tabular drift |
| **Data PII** | `assert_no_pii`, `scan_pii` (24 patterns + Luhn) | Find leaked personal data |
| **Data Freshness** | `assert_freshness`, `assert_row_count` | Verify data recency and size |
| **Data Labels** | `assert_label_balance`, `assert_label_coverage` | Validate label quality |
| **Data Contracts** | `validate_data`, `generate_tests_from_contract` | YAML → auto-test generation |
| **Model Metrics** | `assert_metric` (9 metrics: accuracy, F1, AUC, MSE, R2...) | Quality gates |
| **Model Regression** | `assert_no_regression`, `save_baseline` | Catch silent degradation |
| **Model Slicing** | `assert_slice_performance`, `assert_calibration` | Subgroup fairness + ECE |
| **Model Bias** | `assert_no_bias` (5 methods) | EU AI Act / four-fifths rule |
| **Model Adversarial** | `assert_robust` | Perturbation stability |
| **Training Bugs (P0)** | `assert_no_train_test_overlap`, `assert_temporal_split`, `assert_no_target_leakage` | Data leakage |
| **Training Bugs (P1)** | `assert_gradient_flow`, `assert_no_vanishing_gradient`, `assert_no_exploding_gradient`, `assert_loss_finite` | Gradient health |
| **Training Numerical** | `assert_no_nan_inf`, `assert_loss_decreasing`, `assert_no_loss_divergence`, `assert_softmax_valid` | Numerical stability |
| **Inference** | `assert_latency`, `assert_cold_start`, `assert_throughput`, `assert_api_contract` | Performance SLAs |
| **Pipeline** | `assert_reproducible`, `assert_checksum`, `assert_pipeline` | Determinism + E2E |
| **Monitor** | `assert_no_degradation`, `assert_sla` | Production health |
| **LLM Evaluation** | `assert_semantic_similarity`, `assert_no_toxicity`, `assert_no_hallucination`, `assert_ttft`, `assert_itl` | GenAI testing |
| **CV** | `assert_iou`, `assert_map`, `assert_frame_accuracy`, `assert_temporal_consistency`, `assert_topk_accuracy` | Video analytics |
| **CV Tracking** | `assert_mota`, `assert_motp`, `assert_idf1` | Multi-object tracking (CLEAR-MOT) |
| **CV Face** | `assert_face_far` | False accept rate (NIST FRVT) |
| **NLP** | `assert_bleu`, `assert_rouge`, `assert_ner_f1`, `assert_no_prompt_injection` | Text + security |
| **Speech** | `assert_wer`, `assert_cer`, `assert_rtf`, `assert_accent_coverage` | Recognition + fairness |
| **Tabular** | `assert_feature_drift`, `assert_feature_importance_stable`, `assert_class_balance` | Feature validation |
| **Integrations** | `JiraAdapter`, `TicketDecisionEngine` | Jira ticket creation |

## Installation

```bash
# Core (data + model assertions)
pip install mltk

# With CLI
pip install mltk[cli]

# With HTML reports
pip install mltk[report]

# Domain kits
pip install mltk[cv]          # Computer Vision
pip install mltk[nlp]         # NLP
pip install mltk[speech]      # Speech Recognition
pip install mltk[contracts]   # Data contracts (YAML)
pip install mltk[torch]       # Gradient inspection
pip install mltk[llm]         # LLM evaluation

# Everything
pip install mltk[all]
```

## CLI

```bash
mltk version                    # Show version
mltk init                       # Scaffold config + tests
mltk scan data.csv              # Quick data quality scan
mltk drift ref.csv cur.csv      # Drift analysis
mltk score                      # ML Test Score
mltk doctor                     # Diagnose environment
mltk test tests.yaml            # Run YAML-defined tests
mltk compliance results.json    # EU AI Act compliance report
mltk contract init              # Scaffold contract YAML
mltk contract validate data.csv # Validate against contract
mltk contract generate-tests    # Generate pytest from contract
```

## pytest Plugin

Auto-registered on install. Provides ML-specific markers and `--mltk-report`:

```bash
pytest -m ml_data                 # Run only data quality tests
pytest -m ml_model                # Run only model quality tests
pytest -m "not ml_slow"           # Skip long-running tests
pytest --mltk-report              # Generate HTML report
pytest --mltk-export-json         # Export results to JSON
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
| YAML test defs | No | No | No | **Yes** |
| EU AI Act compliance | No | No | No | **Yes** |
| Training bug detection | No | No | No | **Yes** |
| LLM evaluation | No | Yes (LLM-only) | No | **Yes** |
| License | Elastic | AGPL | Apache 2.0 | **Apache 2.0** |
| Core deps | Heavy | Heavy | Medium | **2 (numpy, pandas)** |

## Documentation

- [Getting Started](docs/getting-started.md)
- [Configuration](docs/configuration.md)
- [API Reference](docs/api/)
- [Examples](examples/)

## License

Apache-2.0
