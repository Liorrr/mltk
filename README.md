# mltk - ML Test Kit

**pytest for ML** -- unified testing across the entire ML lifecycle.

[![License](https://img.shields.io/badge/license-Elastic%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://python.org)
[![Tests](https://img.shields.io/badge/tests-4291%2B%20passed-green.svg)]()
[![Rust](https://img.shields.io/badge/rust-accelerated-orange.svg)]()

<!-- TEMP-PYPI-CLAIM: remove this block once mltk PyPI name is resolved -->
> **Installation note (temporary):** `pip install mltk` is not yet available — the PyPI name is pending a transfer claim. Until resolved, install directly from GitHub:
> ```bash
> pip install git+https://github.com/Liorrr/mltk
> ```
> All imports (`import mltk`), CLI commands (`mltk scan`), and functionality are identical.
<!-- END TEMP-PYPI-CLAIM -->

```bash
pip install mltk
```

## Why mltk?

ML systems fail silently. A model can train on corrupt data, produce confident predictions from stale features, and pass every unit test while being completely wrong in production. Traditional testing does not catch these failures.

**mltk** gives you **232 assertions** covering the entire ML lifecycle -- data quality, model validation, drift detection, fairness testing, inference benchmarking, training bug detection, LLM evaluation, behavioral consistency, NER-based PII detection, red team security, multimodal evaluation, observability, and production monitoring. One toolkit, one `pip install`, native pytest integration. No more gluing together 5 different tools.

## Quick Start

```python
import pandas as pd
import pytest
from mltk.data import (
    assert_schema, assert_no_nulls, assert_no_pii,
)
from mltk.model import (
    assert_metric, assert_no_regression, assert_no_bias,
)
from mltk.inference import assert_latency, assert_throughput

@pytest.mark.ml_data
def test_training_data():
    df = pd.read_csv("data/training.csv")
    assert_schema(
        df,
        {"id": "int64", "text": "object", "label": "int64"},
    )
    assert_no_nulls(df, columns=["label"])
    assert_no_pii(df, columns=["text"])

@pytest.mark.ml_model
def test_model_quality(y_true, y_pred):
    assert_metric(
        y_true, y_pred, metric="f1", threshold=0.85,
    )
    assert_no_regression(
        y_true, y_pred, baseline=0.90, tolerance=0.02,
    )
    assert_no_bias(
        y_true, y_pred,
        sensitive_feature=gender,
        method="demographic_parity",
    )

@pytest.mark.ml_inference
def test_inference_performance():
    assert_latency(model.predict, X_test, p95=50.0)
    assert_throughput(model.predict, X_single, min_rps=100)
```

Run with HTML report:
```bash
pytest --mltk-report
```

## What's Included (v0.12.4)

**Behavioral consistency testing** — 7 assertions (paraphrase invariance, format invariance, output stability, semantic equivalence, directional expectation, retrieval consistency, ParaphraseGenerator) that catch models memorizing phrasing instead of learning concepts. Multi-method evaluation: lexical (token F1), embedding, NLI, LLM-as-Judge. No other ML testing tool ships these as pytest assertions.

**NER PII detection** — `assert_no_pii(method="ner"|"gliner"|"hybrid")` adds Named Entity Recognition to PII scanning. Presidio + spaCy catches names, organizations, and locations that regex cannot find. GLiNER provides zero-shot detection for domain-specific entities (healthcare MRN, legal case numbers). Hybrid mode runs regex + NER with intelligent deduplication. Install with `pip install mltk[ner]`.

**YAML test definitions** — write ML tests in YAML, no Python required. Run with `mltk test tests.yaml`. Supports 11 data assertions with `env:VAR_NAME` data source for CI/CD.

**EU AI Act compliance reports** — `mltk compliance results.json` auto-maps test results to EU AI Act articles, classifies risk level (unacceptable/high/limited/minimal), and generates an HTML evidence table with gap analysis.

**`mltk doctor`** — 9 diagnostic checks that inspect your Python version, dependencies, config, Rust extension, and pytest plugin, with actionable fix hints for each failure.

**Data contracts** — define column types, ranges, and uniqueness rules in YAML. Use `mltk contract generate-tests` to auto-generate a pytest file from the contract — zero boilerplate.

**LLM evaluation** — `assert_semantic_similarity`, `assert_no_toxicity`, `assert_no_hallucination`, `assert_ttft`, `assert_itl`, RAG evaluation (faithfulness, context precision/recall, RAGAS composite), agentic testing (task completion, tool selection), multi-turn conversation, coherence, BERTScore, and text quality assertions for GenAI systems.

**Training bug detection** — catch data leakage (P0), gradient failures (P1), numerical instability, plus P2 bugs: augmentation corruption, checkpoint integrity, distributed training sync, memory leaks, and training-serving skew detection.

**Server platform** — `mltk server` starts a self-hosted result-tracking server with REST API, live dashboard, API key auth, webhooks, run comparison, and GitHub CI integration (PR comments + check runs).

**Cloud monitoring** — AWS SageMaker/CloudWatch, GCP Vertex AI, Azure ML, and Prometheus/Triton adapters for production endpoint health, latency, and error rate assertions.

**FDA audit trail** — `mltk fda-audit` generates a device-grade audit trail from test results for regulatory submissions.

**Compliance PDF** — `mltk compliance-pdf` exports any HTML compliance report (EU AI Act, OWASP) to PDF for auditors.

**Model Card Generator** — `mltk model-card results.json` generates a Google-format Model Card in Markdown from test results.

**Chat interface** — `mltk chat` provides interactive Q&A about test results with no LLM or external API required.

**Environment variable config** — all config keys available as `MLTK_*` env vars, highest priority in the cascade. Works out of the box in CI/CD pipelines.

**JSON export** — `--mltk-export-json` flag exports full test results to JSON for downstream tooling.

## Feature Matrix (232 assertions)

| Module | Assertions | Purpose |
|--------|-----------|---------|
| **Data Schema** | `assert_schema`, `assert_no_nulls`, `assert_dtypes` | Validate DataFrame structure |
| **Data Distribution** | `assert_range`, `assert_unique`, `assert_no_outliers` | Verify statistical properties |
| **Data Statistics** | `assert_column_mean`, `assert_column_median`, `assert_column_stdev`, `assert_quantiles` | Statistical bounds checking |
| **Data Validation** | `assert_datetime_format`, `assert_values_in_set`, `assert_no_conflicting_labels`, `assert_feature_label_correlation_stable` | Value-level data checks |
| **Data Drift** | `assert_no_drift` (KS, PSI, KL, Chi2) | Detect distribution shifts |
| **Data Drift (Advanced)** | `assert_no_embedding_drift` + 7 methods (KS, PSI, KL, Chi2, JS, Wasserstein, auto) | Embedding + tabular drift |
| **Data PII** | `assert_no_pii`, `scan_pii` (40+ regex patterns + Presidio NER + GLiNER zero-shot + hybrid) | Find leaked personal data |
| **Data Freshness** | `assert_freshness`, `assert_row_count` | Verify data recency and size |
| **Data Labels** | `assert_label_balance`, `assert_label_coverage` | Validate label quality |
| **Data Contracts** | `validate_data`, `generate_tests_from_contract` | YAML → auto-test generation |
| **Data Quality Preset** | `assert_data_quality`, `data_quality_report` | One-call full data audit |
| **Data Lineage** | `assert_lineage_complete` | Track data provenance |
| **Model Metrics** | `assert_metric` (9 metrics: accuracy, F1, AUC, MSE, R2...) | Quality gates |
| **Model Regression** | `assert_no_regression`, `save_baseline` | Catch silent degradation |
| **Model Slicing** | `assert_slice_performance`, `assert_calibration` | Subgroup fairness + ECE |
| **Model Bias** | `assert_no_bias` (5 methods) | EU AI Act / four-fifths rule |
| **Model Adversarial** | `assert_robust` | Perturbation stability |
| **Model Overfitting** | `assert_no_overfitting`, `assert_label_drift` | Training vs. validation gaps |
| **Training Bugs (P0)** | `assert_no_train_test_overlap`, `assert_temporal_split`, `assert_no_target_leakage` | Data leakage |
| **Training Bugs (P1)** | `assert_gradient_flow`, `assert_no_vanishing_gradient`, `assert_no_exploding_gradient`, `assert_loss_finite` | Gradient health |
| **Training Numerical** | `assert_no_nan_inf`, `assert_loss_decreasing`, `assert_no_loss_divergence`, `assert_softmax_valid` | Numerical stability |
| **Training Bugs (P2)** | `assert_augmentation_preserves_signal`, `assert_no_augmentation_on_test`, `assert_checkpoint_complete`, `assert_resume_loss_continuous`, `assert_effective_batch_size`, `assert_gradient_sync`, `assert_loss_is_detached`, `assert_no_memory_leak` | Augmentation, checkpoint, distributed, memory |
| **Training Skew** | `assert_no_training_serving_skew` | Training-serving distribution mismatch |
| **Inference** | `assert_latency`, `assert_cold_start`, `assert_throughput`, `assert_api_contract` | Performance SLAs |
| **Pipeline** | `assert_reproducible`, `assert_checksum`, `assert_pipeline` | Determinism + E2E |
| **Monitor** | `assert_no_degradation`, `assert_sla`, `assert_no_output_drift` | Production health |
| **Cloud Monitor** | AWS (`assert_endpoint_healthy/latency/error_rate`), GCP (`assert_endpoint_healthy`, `assert_prediction_latency`), Azure (`assert_endpoint_healthy/latency`), Prometheus/Triton | Multi-cloud monitoring |
| **LLM Evaluation** | `assert_semantic_similarity`, `assert_no_toxicity`, `assert_no_hallucination`, `assert_ttft`, `assert_itl` | GenAI testing |
| **LLM RAG** | `assert_faithfulness`, `assert_context_relevancy`, `assert_context_precision`, `assert_context_recall`, `assert_answer_relevancy`, `assert_ragas_score` | RAG pipeline evaluation (RAGAS) |
| **LLM Agentic** | `assert_task_completion`, `assert_tool_selection`, `assert_tool_call_correctness` | Agent/tool-use testing |
| **LLM Text Quality** | `assert_text_length`, `assert_output_format`, `assert_readability` | Output quality gates |
| **LLM Multi-turn** | `assert_knowledge_retention`, `assert_turn_relevancy`, `assert_conversation_completeness` | Conversation evaluation |
| **LLM Coherence** | `assert_coherence` | Logical consistency |
| **LLM BERTScore** | `assert_bertscore` (Rust-accelerated token matching) | Semantic similarity via BERT |
| **NLP** | `assert_bleu`, `assert_rouge`, `assert_ner_f1`, `assert_no_prompt_injection` | Text + security |
| **NLP Sentiment** | `assert_sentiment_positive`, `assert_no_sentiment_drift` | Sentiment analysis + drift |
| **Speech** | `assert_wer`, `assert_cer`, `assert_rtf`, `assert_accent_coverage` | Recognition + fairness |
| **CV** | `assert_iou`, `assert_map`, `assert_frame_accuracy`, `assert_temporal_consistency`, `assert_topk_accuracy` | Video analytics |
| **CV Tracking** | `assert_mota`, `assert_motp`, `assert_idf1` | Multi-object tracking (CLEAR-MOT) |
| **CV Face** | `assert_face_far` | False accept rate (NIST FRVT) |
| **Tabular** | `assert_feature_drift`, `assert_feature_importance_stable`, `assert_class_balance` | Feature validation |
| **Compliance** | EU AI Act report, OWASP LLM Top 10, FDA audit trail, compliance PDF export | Regulatory evidence |
| **Retrieval** | `assert_ndcg`, `assert_mrr`, `assert_recall_at_k`, `assert_map_at_k` | Ranking metrics (RAG) |
| **LLM-as-Judge** | `assert_llm_judge_score`, `assert_llm_judge_pairwise` | Vendor-neutral LLM eval |
| **Summarization** | `assert_summary_coverage`, `assert_summary_compression`, `assert_summary_faithfulness` | Summary quality |
| **Recommendation** | `assert_hit_rate`, `assert_diversity`, `assert_novelty`, `assert_coverage`, `assert_serendipity` | RecSys testing |
| **Long-Context LLM** | `assert_needle_in_haystack`, `assert_context_utilization`, `assert_no_lost_in_middle` | Long-context eval |
| **Composable Suite** | `MltkSuite`, `SuiteResult`, export to JSON/HTML/JUnit | Run without pytest |
| **Code Generation** | `assert_code_executes`, `assert_code_passes_tests`, `assert_no_code_vulnerabilities`, `assert_code_complexity` | LLM codegen testing |
| **Healthcare** | HIPAA compliance mapping, `assert_hipaa_coverage` | Regulated industry |
| **Finance** | SR 11-7 model risk, `assert_sr117_coverage` | Bank model governance |
| **Enterprise** | RBAC, audit log, custom compliance frameworks | SOC 2 + governance |
| **Observability** | `TestScheduler`, anomaly detection, impact analysis, Grafana dashboards | Production monitoring |
| **Testing Patterns** | `assert_matches_golden`, flaky detection, retry, test selection | Advanced test strategies |
| **Reports** | Visual diff, test history summarizer, bias report, model card | Analysis + reporting |
| **Integrations** | Jira, MLflow, GitHub App, W&B, DVC, Kubeflow, SageMaker, Slack, webhooks | Ecosystem connectors |

## Installation

```bash
# Core (data + model assertions)
pip install mltk

# With CLI
pip install mltk[cli]

# With HTML reports
pip install mltk[report]

# With server platform
pip install mltk[server]

# NER PII detection
pip install mltk[ner]         # Presidio + spaCy NER

# Domain kits
pip install mltk[cv]          # Computer Vision
pip install mltk[nlp]         # NLP
pip install mltk[speech]      # Speech Recognition
pip install mltk[contracts]   # Data contracts (YAML)
pip install mltk[torch]       # Gradient inspection
pip install mltk[llm]         # LLM evaluation

# Integrations
pip install mltk[integrations]  # Jira adapter
pip install mltk[mlflow]        # MLflow logging
pip install mltk[aws]           # AWS SageMaker/CloudWatch
pip install mltk[gcp]           # GCP Vertex AI
pip install mltk[azure]         # Azure ML monitoring
pip install mltk[pdf]           # Compliance PDF export
pip install mltk[server]        # Self-hosted server platform

# Everything
pip install mltk[all]
```

## CLI

```bash
# Core
mltk version                              # Show version
mltk init                                 # Scaffold config + tests
mltk scan data.csv                        # Quick data quality scan
mltk drift ref.csv cur.csv                # Drift analysis
mltk score                                # ML Test Score
mltk doctor                               # Diagnose environment

# Test execution & reporting
mltk test tests.yaml                      # Run YAML-defined tests
mltk model-card results.json              # Generate Google Model Card
mltk compliance results.json              # EU AI Act compliance report
mltk fda-audit results.json              # FDA device audit trail export
mltk compliance-pdf report.html          # Export compliance report to PDF

# Data contracts
mltk contract init                        # Scaffold contract YAML
mltk contract validate data.csv          # Validate against contract
mltk contract generate-tests             # Generate pytest from contract

# Documentation
mltk docs serve                           # Serve docs locally (hot reload)
mltk docs build                           # Build static HTML docs
mltk docs open                            # Build, serve, and open in browser

# Test registry
mltk registry push my_fixtures            # Save test files to registry
mltk registry pull my_fixtures            # Restore from registry
mltk registry list                        # List saved collections

# Notifications
mltk notify slack --results-json r.json  # Send results to Slack

# Server platform
mltk server                               # Start result-tracking server
mltk server-create-key --project myproj  # Generate API key for server

# Interactive
mltk chat --results-json results.json    # Q&A about test results
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

| Feature | Great Expectations | Deepchecks | Evidently | DeepEval | Giskard | Arize Phoenix | **mltk** |
|---|---|---|---|---|---|---|---|
| Data validation | Yes | Yes | Limited | No | Yes | No | **Yes** |
| Model testing | No | Yes | Limited | No | Yes (tabular) | No | **Yes** |
| Drift detection | No | Yes | Yes | No | No | No | **Yes (Rust)** |
| Streaming drift | No | No | No | No | No | No | **Yes (ADWIN/CUSUM)** |
| Bias/fairness | No | No | No | No | Yes | No | **Yes** |
| Inference testing | No | No | No | No | No | No | **Yes** |
| pytest native | No | No | No | Yes | Yes | No | **Yes** |
| CLI | Yes | No | No | Yes | Yes | Yes | **Yes** |
| Domain kits | No | Partial | No | No | No | No | **Yes** |
| ML Test Score | No | No | No | No | No | No | **Yes** |
| Rust acceleration | No | No | No | No | No | No | **Yes** |
| YAML test defs | No | No | No | No | No | No | **Yes** |
| Compliance frameworks | No | No | No | No | No | No | **Yes (8 frameworks)** |
| Training bug detection | No | No | No | No | No | No | **Yes** |
| Conformal prediction | No | No | No | No | No | No | **Yes** |
| Composable TestSuite | No | No | No | No | No | No | **Yes** |
| Code Generation | No | No | No | No | No | No | **Yes (4 assertions)** |
| LLM evaluation | No | Yes (LLM-only) | No | **Yes (50+ metrics)** | Yes (OWASP scanner) | **Yes (tracing)** | **Yes (232 assertions)** |
| Behavioral consistency | No | No | No | No | No | No | **Yes (7 assertions)** |
| NER PII detection | No | No | No | No | No | No | **Yes (4 methods)** |
| Agent trace testing | No | No | No | Yes (basic) | No | Yes (tracing) | **Yes (9 assertions)** |
| Multi-agent testing | No | No | No | No | No | No | **Yes** |
| Synthetic data validation | No | No | No | No | No | No | **Yes (4 assertions)** |
| Safety taxonomy | No | No | No | Yes (red-teaming) | Yes (OWASP) | No | **Yes (per-category)** |
| LLM observability | No | No | No | No | No | **Yes** | No |
| OWASP LLM scanning | No | No | No | No | **Yes** | No | **Yes** |
| License | Elastic | AGPL | Apache 2.0 | Apache 2.0 | Apache 2.0 | Elastic | **Elastic 2.0** |
| Core deps | Heavy | Heavy | Medium | Heavy (LLM) | Heavy | Heavy | **2 (numpy, pandas)** |

## Documentation

- [Getting Started](docs/getting-started.md)
- [Configuration](docs/configuration.md)
- [API Reference](docs/api/)
- [Examples](examples/)

## License

[Elastic License 2.0](LICENSE) — free for internal use; commercial licensing
available at lior1cc@gmail.com. See [LICENSE-COMMERCIAL](LICENSE-COMMERCIAL).
