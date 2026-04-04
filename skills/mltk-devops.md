---
description: >
  mltk DevOps persona — CI/CD integration, server setup, monitoring,
  quality gates, and infrastructure for ML testing pipelines.
---

# mltk DevOps Engineer Skill

## Role Summary

A DevOps engineer using mltk embeds ML quality gates into CI/CD pipelines, operates the mltk
FastAPI server and MCP server, configures continuous monitoring for drift and SLA violations,
and manages test artifact registries. The goal is automated enforcement of ML quality without
manual intervention.

---

## CI/CD Integration

### GitHub Actions

```yaml
- name: ML Quality Gate
  run: |
    pip install mltk[all]
    mltk scan --path ./data --model model.pkl --fail-on critical
    mltk test --fail-on-error
    mltk score --min-score 70
    mltk compliance --framework nist --fail-on-gap
```

Exit codes: `0` = pass, `1` = blocked. `--fail-on critical` blocks only on critical findings.

### GitLab CI

```yaml
ml_quality:
  stage: test
  script:
    - pip install mltk[all]
    - mltk scan --path ./data --fail-on critical
    - mltk compliance --framework eu_ai_act --fail-on-gap
  artifacts:
    paths: [mltk-reports/]
    reports:
      junit: mltk-reports/results.xml
```

### JUnit XML and selective runs

```bash
mltk test --junit mltk-reports/results.xml          # for CI test panels
python -m pytest tests/ -m ml_data                  # data quality only
python -m pytest tests/ -m "not ml_slow"            # skip slow tests
```

Available markers: `ml_data`, `ml_model`, `ml_drift`, `ml_inference`, `ml_slow`.

---

## Quality Gate Configuration

```toml
# pyproject.toml
[tool.mltk]
drift_method    = "ks"
drift_threshold = 0.05
report_dir      = "./mltk-reports"

[tool.coverage.report]
fail_under = 80
```

Threshold precedence: CLI flags > `pyproject.toml` > defaults.

---

## Server Setup

Source: `src/mltk/server/`

```bash
mltk server             # FastAPI server on :8000
mltk server-create-key  # generate API key
```

Dashboard: `http://localhost:8000/dashboard` | API docs: `http://localhost:8000/docs`

**Production:** `uvicorn mltk.server.app:app --host 0.0.0.0 --port 8000 --workers 4`

**Docker:**
```dockerfile
FROM python:3.11-slim
RUN pip install mltk[server,all]
EXPOSE 8000
CMD ["uvicorn", "mltk.server.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

Key endpoints: `POST /scan` (findings JSON), `POST /test` (pass/fail), `GET /report/{id}`, `GET /dashboard`.

---

## MCP Server Configuration

Enables AI agents (Claude Code, Cursor, VS Code) to call mltk tools directly.

```json
{
  "mcpServers": {
    "mltk": {
      "command": "python",
      "args": ["-m", "mltk.mcp"]
    }
  }
}
```

Save as `.mcp.json` in the project root. Source: `src/mltk/mcp/`.

11 tools: `mltk_scan` `mltk_test` `mltk_list` `mltk_eval` `mltk_dataset` `mltk_report` `mltk_suggest` `mltk_experiment` `mltk_workflow` `mltk_create_pr` `mltk_create_issue`.

---

## Monitoring Setup

Source: `src/mltk/monitor/`

```python
from __future__ import annotations

from mltk.monitor import (
    assert_no_degradation, assert_no_streaming_drift, assert_sla,
    assert_gpu_memory_local, assert_endpoint_healthy, assert_endpoint_latency,
    assert_prometheus_metric, assert_triton_healthy,
)

# Drift / degradation
assert_no_degradation(model, X_ref, X_current, metric="accuracy", threshold=0.03)
assert_no_streaming_drift(stream, window=1000, method="ks", threshold=0.05)
assert_sla(endpoint_url, max_p99_ms=200, duration_s=60)

# GPU
assert_gpu_memory_local(max_used_gb=10.0)

# Cloud endpoints — provider = "aws" (SageMaker) | "gcp" (Vertex AI) | "azure" (Azure ML)
assert_endpoint_healthy("my-endpoint", provider="aws")
assert_endpoint_latency("my-endpoint", provider="gcp", max_latency_ms=300)

# Prometheus / Triton
assert_prometheus_metric(url="http://prometheus:9090",
                         query='rate(http_requests_total{status="500"}[5m])',
                         max_value=0.01)
assert_triton_healthy("http://triton:8000")
```

```bash
mltk grafana-export --url http://grafana:3000 --output dashboards/
```

---

## Notifications

```bash
mltk notify slack --webhook-url "$SLACK_WEBHOOK" --channel "#ml-alerts" --on-severity critical
```

Via MCP: `mltk_create_issue(finding_id=..., tracker="github")` or `tracker="jira"`.
Source: `src/mltk/integrations/` (Slack, Jira, GitHub, MLflow).

---

## Registry — Artifact Management

Source: `src/mltk/registry/`

```bash
mltk registry push --artifact model.pkl --tag v1.2.3
mltk registry pull --artifact train.csv --tag baseline
mltk registry list
```

Use in CI to version-pin reference datasets and models for gate comparisons.

---

## Observability

Extras: `mltk[otel]` (OTLP), `mltk[phoenix]` (Arize Phoenix), `mltk[langfuse]` (Langfuse).

```python
from __future__ import annotations

from mltk.domains.llm import assert_trace_quality

def test_trace_quality():
    result = assert_trace_quality(traces, min_faithfulness=0.85, max_latency_ms=500)
    assert result.passed
```

Source: `src/mltk/eval/` (span evaluation, trace scoring).

---

## Installation Matrix

| Use Case | Install Extra |
|----------|--------------|
| CLI tools | `mltk[cli]` |
| HTML/JSON reports | `mltk[report]` |
| PDF export | `mltk[pdf]` |
| LLM eval (embeddings) | `mltk[embedding]` |
| NLI eval | `mltk[nli]` |
| PII detection | `mltk[ner]` |
| Toxicity classifier | `mltk[classifier]` |
| MCP server | `mltk[mcp]` |
| FastAPI server | `mltk[server]` |
| AWS monitoring | `mltk[aws]` |
| GCP monitoring | `mltk[gcp]` |
| Azure monitoring | `mltk[azure]` |
| Everything | `mltk[all]` |
