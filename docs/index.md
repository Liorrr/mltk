# mltk -- ML Test Kit

**pytest for ML** -- catch silent failures across data, models, training, inference, LLMs, and production monitoring.

mltk is a Python testing toolkit that gives you `assert_*` functions for every stage of the ML lifecycle. Write tests in plain Python or YAML, run them with `pytest`, and get interactive HTML reports -- all from a single `pip install`.

---

## Install in 30 seconds

```bash
pip install mltk[cli,report]
mltk doctor
```

Then write your first test:

```python
import pandas as pd
from mltk.data import assert_schema, assert_no_nulls, assert_range

def test_training_data():
    df = pd.read_parquet("data/training.parquet")

    assert_schema(df, {"user_id": "int64", "score": "float64", "label": "int64"})
    assert_no_nulls(df, columns=["label", "score"])
    assert_range(df["score"], min_val=0.0, max_val=1.0)
```

```bash
pytest --mltk-report -v
```

:point_right: **[Full getting-started tutorial](getting-started.md)** -- install, scaffold, write, run, view reports in 5 minutes.

---

## Why mltk?

ML systems fail silently. A model can train on corrupt data, produce confident predictions from stale features, and pass every unit test while being completely wrong in production. Traditional software testing does not catch these failures.

mltk provides **195+ purpose-built assertions** spanning the full ML lifecycle -- from raw data ingestion through production monitoring, including specialized kits for recommendation systems and long-context LLM evaluation -- all runnable with `pytest`.

| What you get | Without mltk |
|---|---|
| `assert_no_nulls(df)` catches missing labels before training | Silent model degradation |
| `assert_drift(reference, current)` detects feature shift | Stale predictions in production |
| `assert_latency(fn, p99=200)` benchmarks inference | Surprise timeouts under load |
| `assert_bias(predictions, groups)` audits fairness | Regulatory violations |
| `assert_rag_faithfulness(answer, context)` validates RAG | Hallucinated responses |

---

## Feature Highlights

### :mag: Data Testing
Schema validation, null detection, range checks, distribution analysis, PII scanning, drift detection, label quality, data lineage, embedding drift.

:point_right: [Data Schema](api/data-schema.md) | [Distribution](api/data-distribution.md) | [Drift](api/data-drift.md) | [PII](api/data-pii.md) | [Labels](api/data-labels.md)

### :dart: Model Testing
Metric thresholds, regression detection, slice analysis, bias auditing, adversarial robustness, overfitting detection.

:point_right: [Metrics](api/model-metrics.md) | [Regression](api/model-regression.md) | [Slicing](api/model-slicing.md) | [Bias](api/model-bias.md) | [Adversarial](api/model-adversarial.md)

### :rocket: Inference Testing
Latency benchmarks (P50/P95/P99), throughput measurement, API contract validation.

:point_right: [Latency](api/inference-latency.md) | [Throughput](api/inference-throughput.md) | [Contract](api/inference-contract.md)

### :brain: LLM & RAG Evaluation
Faithfulness, relevance, coherence, conversation quality, BERTScore, agentic tool use, text safety, RAGAS metrics, long-context window testing (needle-in-haystack, utilization, lost-in-middle detection).

:point_right: [LLM Evaluation](api/llm.md) | [RAG & Agentic](api/rag-evaluation.md) | [Long-Context Testing](api/long-context.md)

### :star: Recommendation Systems
Hit rate, nDCG, coverage, diversity, novelty -- validate that your recommender surfaces relevant, varied, non-obvious items across all user segments.

:point_right: [Recommendation Systems](api/recommendation.md)

### :shield: Compliance & Audit
EU AI Act evidence reports, FDA 21 CFR Part 11 audit trails, OWASP LLM Top 10 checks, compliance PDF export.

:point_right: [EU AI Act](api/eu-ai-act.md) | [FDA Audit](api/fda-audit.md) | [Compliance PDF](api/compliance-pdf.md)

### :bar_chart: Server Platform
Self-hosted test result tracking -- persistent storage, live dashboard, REST API, webhooks, GitHub CI integration.

:point_right: [Server Platform](api/server-platform.md)

### :wrench: Training & Pipeline
Gradient health, data leakage detection, checkpoint validation, numerical stability, augmentation verification, distributed training checks, skew detection, end-to-end pipeline reproducibility.

:point_right: [Training Bugs](api/training-bugs.md) | [Pipeline](api/pipeline.md)

### :cloud: Production Monitoring
AWS CloudWatch, Azure Monitor, GCP monitoring, Prometheus metrics export, output drift detection.

:point_right: [Cloud Monitoring](api/cloud-monitoring.md)

---

## Who is mltk for?

=== "QA Engineer"

    You write test suites for a living. mltk gives you **195+ ready-made assertions** that plug into pytest -- the tool you already know. No ML expertise required: `assert_no_nulls`, `assert_range`, `assert_latency` read like plain English.

    :point_right: Start with [Getting Started](getting-started.md), then explore [YAML Test Definitions](api/yaml-tests.md) for no-code tests.

=== "DevOps / MLOps"

    You need ML tests in CI/CD that block bad deployments. mltk runs as a standard `pytest` step with exit codes, JUnit XML, and HTML reports. Add drift detection and latency gates to your pipeline in minutes.

    :point_right: Start with [CI/CD Integration](guides/cicd-integration.md), then explore [Server Platform](api/server-platform.md) for trend tracking.

=== "ML Developer"

    You build models and need to validate them systematically. mltk covers the full lifecycle: data quality, training health, model metrics, inference performance, and production monitoring.

    :point_right: Start with [Getting Started](getting-started.md), then explore [Model Metrics](api/model-metrics.md) and [Training Bugs](api/training-bugs.md).

=== "Product Manager"

    You need evidence that the ML system works. mltk generates interactive HTML reports and compliance documents (EU AI Act, FDA) that non-technical stakeholders can read.

    :point_right: Start with [HTML Reports](api/report.md), then explore [EU AI Act Compliance](api/eu-ai-act.md) and [ML Test Score](api/ml-test-score.md).

---

## By the Numbers

| Metric | Count |
|--------|-------|
| Assertions | 195+ across 70+ modules |
| Tests | 1,476 |
| CLI commands | 24 |
| Domain kits | 12 (CV, NLP, Speech, Tabular, LLM, Multimodal, RL, Recommendation, Healthcare, Code Generation, and more) |
| Compliance frameworks | 5 (EU AI Act, FDA, OWASP LLM, NIST AI RMF, ISO 42001) |
| Cloud providers | 3 (AWS, Azure, GCP) |
| Integrations | 12 (GitHub, Slack, Jira, MLflow, Linear, Asana, Prometheus, W&B, DVC, Kubeflow, SageMaker, Grafana) |

---

## Architecture

| Layer | Modules | What |
|-------|---------|------|
| **pytest plugin** | auto-registered | markers, fixtures, `--mltk-report` |
| **Assertions** | data, model, inference, llm | 195+ `assert_*` functions |
| **Training** | training, pipeline, monitor | gradient, leakage, drift, cloud |
| **Domains** | cv, nlp, speech, tabular, multimodal, rl, recommendation, healthcare, code gen | specialized domain assertions |
| **Compliance** | compliance, contracts, testdefs | EU AI Act, FDA, YAML, data contracts |
| **Platform** | report, server, integrations | HTML, dashboard, Jira/Slack/GitHub |
| **Foundation** | cli, core, rust extension | 24 commands, TestResult, PyO3 |

Every assertion returns a `TestResult` with `.passed`, `.message`, `.severity`, `.details`, and `.duration_ms`. Critical failures raise `MltkAssertionError` (a subclass of `AssertionError`), so pytest catches them naturally.

---

## Project Status

mltk is at **v0.8.0** (beta). Core modules are stable and tested with 195+ assertions and 1,476+ tests. The latest release adds recommendation system testing, long-context LLM evaluation, and healthcare/code generation domain kits. See the [CHANGELOG](https://github.com/Liorrr/mltk/blob/main/CHANGELOG.md) for release notes and the [Domain Overview](api/domain-overview.md) for a complete map of all testing capabilities.

## License

Apache-2.0
