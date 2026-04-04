# OpenTelemetry Integration

Trace mltk assertion execution with OpenTelemetry -- per-assertion timing, failure attributes, and distributed tracing.

**Module:** `mltk.integrations.otel`

**Install:** `pip install mltk[otel]`

---

## Why Observability for ML Testing?

Consider a concrete scenario. You run 156 mltk assertions every night at 2 AM against your production feature store. The nightly run covers data quality checks, drift detection, model regression tests, and bias audits.

**Monday.** All 156 assertions pass. Total execution time: 45 seconds. The Slack notification says "all green" and nobody thinks about it.

**Thursday.** Three assertions fail. Total execution time: 180 seconds. The failures are `assert_dcr_safe` on the `transactions` table, `assert_no_drift` on the `user_age` feature, and `assert_latency` on the scoring endpoint. The Slack message says "3 failures" with the test names and messages.

Without observability, your debugging process looks like this:

1. SSH into the CI runner.
2. Read the pytest log output (500+ lines).
3. Ctrl+F for "FAILED" to find the three failures.
4. Notice the total time jumped from 45s to 180s but have no idea which assertion got slower.
5. Guess that `assert_dcr_safe` is the slow one because the transactions table grew.
6. Re-run locally with `--durations=0` to confirm.
7. Discover that `assert_dcr_safe` took 120 seconds because the dataset grew 10x overnight and the DCR computation is O(n^2).

With OpenTelemetry tracing, you open Grafana Tempo (or Jaeger), search for the nightly run trace, and see:

```
nightly_ml_tests (180.2s)
  +-- data.schema                    0.8s   PASS
  +-- data.schema[transactions]      0.3s   PASS
  +-- data.null_ratio[user_age]      0.5s   PASS
  +-- data.dcr_safe[transactions]  120.1s   FAIL  <-- immediately visible
  +-- data.drift[user_age]           2.3s   FAIL
  +-- model.accuracy                 1.2s   PASS
  +-- model.regression               0.9s   PASS
  +-- inference.latency              3.8s   FAIL
  +-- ...  (148 more spans)
```

Each span carries structured attributes: the assertion name, severity, pass/fail status, threshold values, actual values, and the full `TestResult.details` dictionary. You do not grep logs. You do not guess. You click on `data.dcr_safe[transactions]`, see `dataset_rows: 2,400,000` in the span attributes, compare it to Monday's span where `dataset_rows: 240,000`, and immediately understand the root cause.

Over time, the trace data reveals trends that are invisible in pass/fail dashboards:

- **Creeping latency.** An assertion that passed for 90 days but gradually increased from 2s to 8s. It has not failed yet, but it will -- or it is signaling that the underlying data is growing in a way that needs attention.
- **Correlated failures.** Two assertions that always fail together, suggesting a shared upstream cause rather than two independent issues.
- **Flaky tests.** An assertion that fails on Tuesdays and Thursdays. The trace attributes show that `dataset_rows` is different on those days, revealing a batch ingestion schedule issue.

---

## Design Decisions

### Why OpenTelemetry (not Prometheus, Datadog, or custom logging)?

We evaluated several observability approaches before choosing OpenTelemetry:

| Option | Considered? | Why not? |
|--------|:-----------:|----------|
| **OpenTelemetry** | **Chosen** | Vendor-neutral standard, works with ANY backend (Jaeger, Grafana, Datadog), no lock-in |
| Prometheus | Rejected | Metrics-only (counters, histograms) — no per-assertion trace spans or parent/child hierarchy |
| Datadog SDK | Rejected | Vendor lock-in, paid service required, heavy dependency |
| Custom JSON logging | Rejected | Reinvents the wheel — no standard schema, no visualization backends, no trace correlation |
| StatsD | Rejected | UDP-based metrics — no trace context, no structured attributes |

OpenTelemetry is the **CNCF standard** for observability. By emitting OTEL spans, mltk integrates with whatever backend you already use — Jaeger, Grafana Tempo, Zipkin, Datadog, New Relic, Honeycomb — without mltk caring which one. Zero vendor lock-in.

### Why graceful no-op (not a hard dependency)?

ML testing runs in diverse environments: CI pipelines, local laptops, air-gapped servers, Jupyter notebooks. Requiring `opentelemetry-sdk` as a hard dependency would break installs in constrained environments. Instead:

- **With OTEL installed** (`pip install mltk[otel]`): full tracing with real spans
- **Without OTEL**: all methods silently no-op, `export_json` still works (pure dict → JSON)
- **Your code stays the same** — instrument once, activate when ready

This follows the same pattern as `mltk[embedding]` (sentence-transformers) and `mltk[aws]` (boto3): optional extras that enhance functionality without burdening the core install.

---

## Quick Start

```python
from mltk.integrations.otel import MltkTracer
from mltk.core.result import TestSuite, TestResult, Severity

# Create a tracer (real OTEL if installed, no-op otherwise)
tracer = MltkTracer(service_name="ml-tests")

# Build a test suite from your mltk assertions
suite = TestSuite()
suite.add(TestResult(
    name="data.schema",
    passed=True,
    severity=Severity.CRITICAL,
    message="Schema valid",
    duration_ms=12.5,
))
suite.add(TestResult(
    name="data.drift[age]",
    passed=False,
    severity=Severity.CRITICAL,
    message="PSI 0.35 exceeds threshold 0.2",
    duration_ms=340.8,
    details={"psi_score": 0.35, "threshold": 0.2, "feature": "age"},
))

# Trace the entire suite (creates parent span + child spans)
tracer.trace_suite(suite, run_name="nightly-2024-03-15")
```

To view the traces, run Jaeger locally:

```bash
docker run -d --name jaeger \
  -p 16686:16686 \
  -p 4317:4317 \
  jaegertracing/all-in-one:latest
```

Then open `http://localhost:16686`, search for service `ml-tests`, and find the `nightly-2024-03-15` trace.

---

## MltkTracer

```python
class MltkTracer:
    def __init__(
        self,
        service_name: str = "mltk",
        endpoint: str | None = None,
    ) -> None: ...
```

The central class for all tracing operations. It wraps OpenTelemetry's tracer API and provides mltk-specific methods for tracing test results, suites, and exporting trace data.

### Constructor parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `service_name` | `str` | `"mltk"` | Service name that appears in the tracing backend. Use a descriptive name like `"ml-tests"` or `"nightly-validation"`. |
| `endpoint` | `str \| None` | `None` | OTLP endpoint URL (e.g., `"http://localhost:4317"`). When `None`, uses the `OTEL_EXPORTER_OTLP_ENDPOINT` environment variable, falling back to `http://localhost:4317`. |

### Modes

MltkTracer operates in one of two modes, determined automatically at initialization:

**Real mode** (OpenTelemetry SDK is installed). The tracer creates actual OTEL spans that are exported to your tracing backend. Spans carry structured attributes, timing data, and parent-child relationships. This mode requires `opentelemetry-api`, `opentelemetry-sdk`, and `opentelemetry-exporter-otlp` to be installed.

```bash
pip install mltk[otel]
# Installs: opentelemetry-api, opentelemetry-sdk, opentelemetry-exporter-otlp-proto-grpc
```

**No-op mode** (OpenTelemetry SDK is not installed). All tracing methods are silently no-ops. `trace_result` and `trace_suite` return immediately without error. `export_json` still works (it returns the trace data as a dict without requiring OTEL). This design means you can instrument your test code unconditionally -- tracing activates when the `otel` extra is installed and a backend is available, and does nothing otherwise.

```python
# This code works whether or not OTEL is installed
tracer = MltkTracer(service_name="ml-tests")
tracer.trace_result(my_result)  # Real span if OTEL installed, no-op if not
```

No `try/except ImportError` in your test code. No feature flags. No conditional imports. The tracer handles it internally.

---

### trace_result

```python
def trace_result(self, result: dict[str, Any]) -> None
```

Trace a single test result as an OpenTelemetry span. The span name is the assertion name (e.g., `data.drift[age]`), and structured attributes are attached for filtering and analysis in the tracing backend. In no-op mode (OTEL not installed) this returns immediately.

#### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `result` | `dict[str, Any]` | *(required)* | A dict with keys: `name` (str), `passed` (bool), `severity` (str), `message` (str), and optionally `duration_ms` (float). |

#### Span attributes

Every span created by `trace_result` carries these attributes:

| Attribute | Type | Example | Description |
|-----------|------|---------|-------------|
| `mltk.assertion.name` | `str` | `"data.drift[age]"` | Assertion name |
| `mltk.assertion.passed` | `bool` | `false` | Pass/fail status |
| `mltk.assertion.severity` | `str` | `"critical"` | Severity level |
| `mltk.assertion.message` | `str` | `"PSI 0.35 > 0.2"` | Human-readable result message |
| `mltk.assertion.duration_ms` | `float` | `340.8` | Assertion execution time in milliseconds |
| `openinference.span.kind` | `str` | `"EVALUATION"` | Phoenix Evaluations tab support |
| `eval.name` | `str` | `"data.drift[age]"` | OpenInference evaluation name |
| `eval.score` | `float` | `0.0` | 1.0 (pass) or 0.0 (fail) |
| `eval.label` | `str` | `"fail"` | "pass" or "fail" |

If the result failed, the span's status is set to `ERROR` with the failure message as the status description.

#### Example

```python
from mltk.integrations.otel import MltkTracer

tracer = MltkTracer(service_name="ml-tests")

tracer.trace_result({
    "name": "data.drift[age]",
    "passed": False,
    "severity": "critical",
    "message": "PSI 0.35 exceeds threshold 0.2",
    "duration_ms": 340.8,
})
```

---

### trace_suite

```python
def trace_suite(self, results: list[dict[str, Any]]) -> None
```

Trace an entire list of results as a parent span (`mltk.test_suite`) with one child span per result. Creates a tree structure in the tracing backend. In no-op mode this returns immediately.

#### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `results` | `list[dict[str, Any]]` | *(required)* | List of result dicts (same format as `trace_result`). |

#### Parent span attributes

| Attribute | Type | Example | Description |
|-----------|------|---------|-------------|
| `mltk.suite.total` | `int` | `156` | Total number of assertions |
| `mltk.suite.passed` | `int` | `153` | Number of passing assertions |
| `mltk.suite.failed` | `int` | `3` | Number of failing assertions |

#### Span hierarchy

```
mltk-run (parent)                     -- 45.2s total, 153/156 passed
  |-- data.schema                     -- 0.8s, PASS
  |-- data.null_ratio[user_age]       -- 0.5s, PASS
  |-- data.dcr_safe[transactions]     -- 120.1s, FAIL (ERROR status)
  |-- data.drift[user_age]            -- 2.3s, FAIL (ERROR status)
  |-- model.accuracy                  -- 1.2s, PASS
  |-- model.bias[gender]              -- 0.9s, PASS
  |-- inference.latency               -- 3.8s, FAIL (ERROR status)
  |-- ...  (149 more child spans)
```

#### Example

```python
from mltk.integrations.otel import MltkTracer

tracer = MltkTracer(service_name="ml-tests", endpoint="http://jaeger:4317")

results = [
    {"name": "data.schema", "passed": True, "severity": "critical",
     "message": "ok", "duration_ms": 800.0},
    {"name": "data.drift[age]", "passed": False, "severity": "critical",
     "message": "PSI 0.35 > 0.2", "duration_ms": 2300.0},
    {"name": "model.accuracy", "passed": True, "severity": "critical",
     "message": "0.94 >= 0.90", "duration_ms": 1200.0},
]

tracer.trace_suite(results)
```

---

### export_json

```python
def export_json(self, results: list[dict[str, Any]], output_path: str) -> str
```

Export results as OTLP-compatible JSON. This method does **not** require OpenTelemetry to be installed — it performs pure dict-to-JSON serialization using only the standard library. The output follows the OTLP `ResourceSpans` schema for import into Jaeger, Grafana Tempo, or custom tools.

Use `export_json` for:

- **Offline analysis.** Save trace data when no OTEL collector is running.
- **CI artifacts.** Attach as a build artifact for post-run analysis.
- **Custom dashboards.** Feed structured data into your own tools.

#### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `results` | `list[dict[str, Any]]` | *(required)* | List of result dicts (same format as `trace_result`). |
| `output_path` | `str` | *(required)* | File path where the JSON will be written. Parent dirs are created automatically. |

#### Returns

The absolute path of the written file (str). The JSON structure follows the OTLP `ResourceSpans` schema:

```python
{
    "run_name": "nightly-2024-03-15",
    "total": 3,
    "passed": 2,
    "failed": 1,
    "score": 66.7,
    "total_duration_ms": 4300.0,
    "spans": [
        {
            "name": "data.schema",
            "passed": True,
            "severity": "critical",
            "message": "ok",
            "duration_ms": 800.0,
            "details": {},
        },
        {
            "name": "data.drift[age]",
            "passed": False,
            "severity": "critical",
            "message": "PSI 0.35 > 0.2",
            "duration_ms": 2300.0,
            "details": {"psi_score": 0.35, "threshold": 0.2},
        },
        {
            "name": "model.accuracy",
            "passed": True,
            "severity": "critical",
            "message": "0.94 >= 0.90",
            "duration_ms": 1200.0,
            "details": {"accuracy": 0.94, "threshold": 0.90},
        },
    ],
}
```

#### Example

```python
from mltk.integrations.otel import MltkTracer
import json

tracer = MltkTracer(service_name="ml-tests")

# Export to JSON (works even without OTEL installed)
trace_data = tracer.export_json(suite, run_name="nightly-2024-03-15")

# Save as a CI artifact
with open("mltk-trace.json", "w") as f:
    json.dump(trace_data, f, indent=2)

# Or post to your own analytics endpoint
import urllib.request
req = urllib.request.Request(
    "https://analytics.internal/api/traces",
    data=json.dumps(trace_data).encode(),
    headers={"Content-Type": "application/json"},
)
urllib.request.urlopen(req)
```

---

## Visualization Backends

MltkTracer exports traces using the OTLP (OpenTelemetry Protocol) gRPC exporter. Any backend that accepts OTLP can receive mltk traces. Here are the most common options:

| Backend | Type | Setup | Best For |
|---------|------|-------|----------|
| **Jaeger** | Tracing | Single Docker container | Local development, quick exploration |
| **Grafana Tempo** | Tracing | Grafana Cloud or self-hosted | Production, long-term storage, team dashboards |
| **Zipkin** | Tracing | Single Docker container | Lightweight tracing, Zipkin-native ecosystems |
| **Prometheus + Grafana** | Metrics | Docker Compose (two containers) | Aggregated metrics dashboards (not per-trace) |

### Jaeger (recommended for local development)

```bash
# Start Jaeger all-in-one (UI + collector + storage in one container)
docker run -d --name jaeger \
  -p 16686:16686 \
  -p 4317:4317 \
  jaegertracing/all-in-one:latest
```

- **UI:** `http://localhost:16686`
- **OTLP endpoint:** `http://localhost:4317` (gRPC)
- **Storage:** In-memory (traces are lost on restart -- use Elasticsearch or Cassandra for persistence)

No configuration needed on the mltk side -- `MltkTracer()` defaults to `localhost:4317`.

### Grafana Tempo (recommended for production)

For Grafana Cloud, the endpoint and authentication are provided in your Grafana Cloud dashboard. Set them as environment variables:

```bash
export OTEL_EXPORTER_OTLP_ENDPOINT=https://tempo-us-central1.grafana.net:443
export OTEL_EXPORTER_OTLP_HEADERS="Authorization=Basic <base64_encoded_credentials>"
```

For self-hosted Tempo with Docker Compose:

```yaml
# docker-compose.yml
services:
  tempo:
    image: grafana/tempo:latest
    command: ["-config.file=/etc/tempo.yaml"]
    ports:
      - "4317:4317"   # OTLP gRPC
      - "3200:3200"   # Tempo query API
    volumes:
      - ./tempo.yaml:/etc/tempo.yaml

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_AUTH_ANONYMOUS_ENABLED=true
```

Add Tempo as a data source in Grafana, then search for mltk traces by service name.

### Zipkin

```bash
docker run -d --name zipkin \
  -p 9411:9411 \
  openzipkin/zipkin:latest
```

Zipkin uses a different protocol, so configure the Zipkin exporter instead of OTLP:

```bash
pip install opentelemetry-exporter-zipkin
export OTEL_EXPORTER_ZIPKIN_ENDPOINT=http://localhost:9411/api/v2/spans
```

### Prometheus + Grafana (metrics only)

OpenTelemetry can export metrics (not traces) to Prometheus. This is useful for building dashboards that track assertion pass rates, durations, and failure counts over time -- but without the per-run trace drill-down.

```bash
# docker-compose.yml
services:
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
```

This requires additional configuration with the OpenTelemetry metrics SDK and is beyond the scope of the tracing integration documented here.

---

## pytest Integration

The most common use case is tracing every mltk assertion that runs during a pytest session. This is done through a `conftest.py` fixture that creates an `MltkTracer` and hooks into the mltk pytest plugin's result collection.

### Basic conftest.py

```python
# conftest.py
import pytest
from mltk.integrations.otel import MltkTracer
from mltk.core.result import TestSuite


@pytest.fixture(scope="session")
def otel_tracer():
    """Session-scoped OTEL tracer for mltk results."""
    return MltkTracer(service_name="ml-tests")


@pytest.fixture(scope="session")
def traced_suite():
    """Collect all mltk results during the session for tracing."""
    return TestSuite()


@pytest.fixture(autouse=True)
def _trace_mltk_results(otel_tracer, traced_suite, request):
    """After each test, trace any mltk TestResults attached to the test."""
    yield
    # The mltk pytest plugin attaches results to the test node
    for result in getattr(request.node, "mltk_results", []):
        traced_suite.add(result)
        otel_tracer.trace_result(result)


def pytest_sessionfinish(session, exitstatus):
    """At session end, trace the full suite as a parent span."""
    tracer = MltkTracer(service_name="ml-tests")
    suite = TestSuite()

    for item in session.items:
        for result in getattr(item, "mltk_results", []):
            suite.add(result)

    if suite.results:
        tracer.trace_suite(suite, run_name=f"pytest-session-{exitstatus}")
```

### Usage in tests

With the conftest above, your tests do not need any tracing-specific code. Write normal mltk assertions and the traces are emitted automatically:

```python
import pandas as pd
from mltk.data import assert_no_nulls, assert_range, assert_no_drift

def test_feature_quality(ml_config):
    df = pd.read_parquet("features/latest.parquet")

    assert_no_nulls(df["user_age"])
    assert_range(df["user_age"], min_val=0, max_val=120)

def test_no_drift(ml_config):
    reference = pd.read_parquet("features/reference.parquet")
    current = pd.read_parquet("features/latest.parquet")

    assert_no_drift(reference["user_age"], current["user_age"], method="psi", threshold=0.2)
```

When you run `pytest`, each assertion automatically generates a span in Jaeger/Tempo. No test code changes needed.

### Combining with JSON export for CI

In CI environments where an OTEL collector might not be available, use `export_json` to save trace data as a build artifact:

```python
# conftest.py addition
import json

def pytest_sessionfinish(session, exitstatus):
    tracer = MltkTracer(service_name="ml-tests")
    suite = TestSuite()

    for item in session.items:
        for result in getattr(item, "mltk_results", []):
            suite.add(result)

    if suite.results:
        # Export as JSON artifact (always works, no OTEL needed)
        trace_data = tracer.export_json(suite, run_name=f"ci-run-{exitstatus}")
        with open("mltk-reports/trace.json", "w") as f:
            json.dump(trace_data, f, indent=2)

        # Also send to OTEL if available
        tracer.trace_suite(suite, run_name=f"ci-run-{exitstatus}")
```

### Combining with mltk server

If you are running the mltk server platform (see [Server Platform](server-platform.md)), traces and server submissions are complementary:

- The **mltk server** stores pass/fail history, trends, and comparison data in SQLite. It provides the dashboard and REST API for querying results.
- **OpenTelemetry** provides per-assertion timing breakdown, distributed trace context, and integration with your existing observability stack (Grafana, Datadog, etc.).

Use both: submit results to the server for the dashboard, and emit traces for deep debugging when something fails.

```bash
pytest --mltk-report \
       --mltk-server http://localhost:8080 \
       --mltk-server-key mltk_your_key_here \
       --mltk-server-project my-project
# Traces are emitted automatically via conftest.py
# Results are submitted to the mltk server via the --mltk-server flag
```

---

## Environment Variables

mltk respects standard OpenTelemetry environment variables. No mltk-specific env vars are needed.

| Variable | Default | Description |
|----------|---------|-------------|
| `OTEL_EXPORTER_OTLP_ENDPOINT` | `http://localhost:4317` | OTLP collector endpoint (gRPC). Used when no `endpoint` is passed to `MltkTracer()`. |
| `OTEL_EXPORTER_OTLP_PROTOCOL` | `grpc` | Transport protocol. `MltkTracer` uses gRPC; `register_phoenix` uses HTTP. |
| `OTEL_SERVICE_NAME` | — | Service name (alternative to `service_name` constructor arg). |
| `OTEL_RESOURCE_ATTRIBUTES` | — | Additional resource attributes as `key=value,key=value`. |

Example `.env` for local development:

```bash
# Point MltkTracer at a local Jaeger instance
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317

# Or point at Phoenix
# OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:6006/v1/traces
```

Example GitHub Actions secrets:

```yaml
env:
  OTEL_EXPORTER_OTLP_ENDPOINT: ${{ secrets.OTLP_ENDPOINT }}
```

---

## OpenInference Attributes (Phoenix Native UI)

When exporting spans, mltk automatically includes [OpenInference](https://github.com/Arize-ai/openinference) semantic attributes on every assertion span:

| Attribute | Value | Purpose |
|-----------|-------|---------|
| `openinference.span.kind` | `"EVALUATION"` | Phoenix displays span in Evaluations tab |
| `eval.name` | assertion name | Evaluation name in Phoenix UI |
| `eval.score` | `1.0` (pass) / `0.0` (fail) | Numeric score for aggregation |
| `eval.label` | `"pass"` / `"fail"` | Human-readable label |

This means mltk assertions appear as first-class evaluations in Phoenix -- not generic spans. No extra configuration needed.

---

## End-to-End Workflow Examples

### Workflow 1: Local Phoenix + mltk Assertions

```bash
# 1. Start Phoenix locally
pip install arize-phoenix
phoenix serve  # Runs on http://localhost:6006

# 2. Install mltk with Phoenix support
pip install mltk[phoenix]
```

```python
# 3. Configure and run
from mltk.integrations.phoenix import register_phoenix
from mltk.integrations.otel import MltkTracer
from mltk.data import assert_no_nulls, assert_no_drift

# One-line setup — all spans go to Phoenix
register_phoenix()

# Run assertions (they emit OTEL spans automatically via MltkTracer)
tracer = MltkTracer()
results = []

r1 = assert_no_nulls(df, columns=["age", "income"])
results.append(r1.to_dict())

r2 = assert_no_drift(train_df, serve_df)
results.append(r2.to_dict())

# Send to Phoenix
tracer.trace_suite(results)

# Open http://localhost:6006 — assertions appear in Evaluations tab
```

### Workflow 2: CI/CD with JSON Export (No Collector)

```yaml
# GitHub Actions — export traces as CI artifact
- name: Run ML tests with tracing
  run: |
    python -c "
    from mltk.integrations.otel import MltkTracer
    from mltk.data import assert_no_nulls
    import pandas as pd

    df = pd.read_csv('data/features.csv')
    r = assert_no_nulls(df)
    tracer = MltkTracer(service_name='ci-ml-tests')
    tracer.export_json([r.to_dict()], 'mltk-reports/spans.json')
    "

- name: Upload trace artifact
  uses: actions/upload-artifact@v4
  with:
    name: ml-test-traces
    path: mltk-reports/spans.json
```

### Workflow 3: Langfuse Production Scoring

```python
from mltk.integrations.langfuse import LangfuseAdapter
from mltk.domains.llm import assert_faithfulness

# Wrap assertion as Langfuse scorer
scorer = LangfuseAdapter(
    assert_faithfulness,
    name="rag_faithfulness",
)

# Score a production trace (posts result to Langfuse dashboard)
scorer.score(
    trace_id="trace-abc-123",
    answer="Paris is the capital of France.",
    context="France is in Europe. Its capital is Paris.",
)
```

---
