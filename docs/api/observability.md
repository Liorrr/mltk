# Observability Adapters

Wire mltk assertions into Phoenix and Langfuse so LLM call traces and quality
evaluations appear in one dashboard -- not two separate tools.

**Modules:**
- `mltk.integrations.phoenix` -- Phoenix adapter
- `mltk.integrations.langfuse` -- Langfuse adapter

**Install:**

```bash
pip install mltk[phoenix]    # arize-phoenix-evals + arize-phoenix-otel
pip install mltk[langfuse]   # langfuse >= 4.0
```

---

## Why Observability Integration?

### Two tools, two dashboards, one team

A typical ML/LLM team already runs an observability platform -- Phoenix or Langfuse --
to watch production LLM calls: latency, cost, token counts, error rates. That platform
shows *what the model did*. mltk assertions show *whether the model did it well*.
Without integration, those two signals live in separate dashboards:

```
Phoenix dashboard:
  trace abc123  llm.call  1,240ms  2,100 tokens
  trace abc124  llm.call    890ms  1,800 tokens
  [no quality signal here]

pytest output:
  PASSED  test_faithfulness -- 156ms
  FAILED  test_coherence -- score 0.61 < threshold 0.75
  [no link to which production trace caused the failure]
```

With the observability adapters, the same Phoenix or Langfuse dashboard shows both:

```
Phoenix evaluations panel:
  trace abc123  faithfulness  PASS  score=0.92  "lexical+NLI dispatch"
  trace abc123  coherence     FAIL  score=0.61  "below 0.75 threshold"
  trace abc124  faithfulness  PASS  score=0.87
```

The ML engineer browsing traces can click "Run faithfulness" directly inside Phoenix
and see the mltk assertion result attached to the span. No context switching.

### What the adapters provide

**Phoenix adapter (Pattern A + Pattern B):**
- Pattern A: Re-point `MltkTracer` at Phoenix with one helper call (`register_phoenix`).
  Any existing OTEL-instrumented test suite starts sending spans to Phoenix immediately.
- Pattern B: Wrap any mltk assertion as a Phoenix callable evaluator (`PhoenixAdapter`).
  Phoenix can invoke `assert_faithfulness` directly from its Evaluations tab.

**Langfuse adapter (Pattern A + Pattern C):**
- Pattern A: `MltkTracer` spans are forwarded to Langfuse via OTLP HTTP (v4+).
- Pattern C: After assertions run, `LangfuseAdapter` posts results as numeric scores
  attached to existing Langfuse trace IDs.

**Both platforms:** `assert_trace_quality` provides a single CI/CD quality gate
assertion that accepts a trace dict and checks latency, cost, and quality metrics
in one `assert`.

---

## Phoenix Adapter

### `PhoenixAdapter`

Wrap any mltk assertion as a Phoenix callable evaluator. The wrapped assertion
appears as a named evaluator in the Phoenix Evaluations tab and can be applied to
any stored span.

```python
class PhoenixAdapter:
    def __init__(
        self,
        assertion_fn: Callable[..., TestResult],
        name: str = "mltk",
        score_key: str = "passed",
    ) -> None: ...

    def __call__(
        self,
        output: Any = None,
        expected: Any = None,
        input: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]: ...
```

**Module:** `mltk.integrations.phoenix`

#### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `assertion_fn` | `Callable[..., TestResult]` | *(required)* | Any mltk assertion function that returns a `TestResult`. Must accept keyword arguments matching span attribute names (typically `output`, `expected`, `input`, `metadata`). |
| `name` | `str` | `"mltk"` | Human-readable evaluator name shown in the Phoenix Evaluations tab. |
| `score_key` | `str` | `"passed"` | Key name for the numeric score in the returned dict. Phoenix expects `"score"` by default, but some custom dashboards use different keys. |

#### How it works

Phoenix's callable evaluator pattern passes span attributes as keyword arguments
(`output`, `expected`, `input`, `metadata`). `PhoenixAdapter.__call__` forwards
the non-None arguments to the wrapped mltk assertion and returns a plain dict:

```python
{
    self.score_key: 1.0 if result.passed else 0.0,  # "passed" by default
    "label": "pass" or "fail",
    "explanation": result.message,
}
```

If the assertion raises `MltkAssertionError`, the adapter catches it and returns
a score of `0.0` with the error message as the explanation. Any other exception
is also caught and mapped to `0.0` so Phoenix never sees an unhandled error.

#### Example: Faithfulness evaluator

```python
from mltk.integrations.phoenix import PhoenixAdapter
from mltk.domains.llm import assert_faithfulness
from phoenix.client import Client

# Wrap the assertion
faithfulness_eval = PhoenixAdapter(assert_faithfulness, name="faithfulness")

# Apply to stored spans
client = Client()
spans = client.get_spans(project_name="my-llm-app")
for span in spans:
    score = faithfulness_eval(span.attributes)
    client.log_evaluations([score], span_id=span.span_id)
```

In the Phoenix UI, the Spans table gains a `faithfulness` column with pass/fail
labels and explanation text from mltk's assertion message.

#### Example: Multiple assertions as Phoenix evaluators

```python
from mltk.integrations.phoenix import PhoenixAdapter
from mltk.domains.llm import assert_faithfulness, assert_coherence
from mltk.domains.llm import assert_no_toxicity
from phoenix.client import Client

evaluators = [
    PhoenixAdapter(assert_faithfulness, name="faithfulness"),
    PhoenixAdapter(assert_coherence, name="coherence"),
    PhoenixAdapter(assert_no_toxicity, name="toxicity"),
]

client = Client()
spans = client.get_spans(project_name="my-llm-app", limit=100)
for span in spans:
    for ev in evaluators:
        score = ev(span.attributes)
        client.log_evaluations([score], span_id=span.span_id)
```

---

### `register_phoenix`

Configure `MltkTracer` to send OTLP spans to a running Phoenix instance.
This is a one-line setup helper for teams already running Phoenix.

```python
def register_phoenix(
    endpoint: str = "http://localhost:6006/v1/traces",
    project_name: str = "mltk",
) -> "TracerProvider":
    ...
```

**Module:** `mltk.integrations.phoenix`

#### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `endpoint` | `str` | `"http://localhost:6006/v1/traces"` | Phoenix OTLP HTTP endpoint. Change if Phoenix runs on a non-default port or remote host. |
| `project_name` | `str` | `"mltk"` | Project name shown in Phoenix UI. All traces from this session appear under this project. |

#### Returns

`TracerProvider` -- the registered OTEL provider. Can be passed to `MltkTracer`
for advanced configuration, or ignored for simple setups.

#### Example: Local Phoenix

```python
import phoenix as px
from mltk.integrations.phoenix import register_phoenix
from mltk.integrations.otel import MltkTracer

# Start a local Phoenix server (opens http://localhost:6006)
px.launch_app()

# Point MltkTracer at Phoenix
provider = register_phoenix(
    endpoint="http://localhost:6006/v1/traces",
    project_name="rag-eval",
)

# All MltkTracer spans now go to Phoenix
tracer = MltkTracer(
    service_name="ml-tests",
    endpoint="http://localhost:6006/v1/traces",
)
tracer.trace_suite(suite, run_name="nightly-2026-03-31")
```

#### Example: Remote Phoenix (self-hosted or cloud)

```python
import os
from mltk.integrations.phoenix import register_phoenix

provider = register_phoenix(
    endpoint=os.environ["PHOENIX_OTLP_ENDPOINT"],
    project_name=os.environ.get("PHOENIX_PROJECT", "mltk"),
)
```

---

### OpenInference Span Enrichment

When the Phoenix adapter is active, `MltkTracer` automatically adds
[OpenInference](https://github.com/Arize-ai/openinference) semantic attributes to
every span. These make mltk spans show up in Phoenix's "Evaluations" tab rather
than as generic OTEL spans:

| Attribute added | Value | Why |
|-----------------|-------|-----|
| `openinference.span.kind` | `"EVALUATION"` | Phoenix routes span to Evaluations view |
| `eval.name` | Assertion name | Shows evaluator name in Phoenix table |
| `eval.score` | `1.0` or `0.0` | Numeric pass/fail for Phoenix scoring UI |
| `eval.label` | `"pass"` or `"fail"` | Categorical label for filter/sort |

These attributes are additive -- they do not replace existing `mltk.test.*`
attributes. Jaeger and Grafana Tempo continue to work unchanged.

---

## Langfuse Adapter

### `LangfuseAdapter`

Wrap an mltk assertion as a Langfuse score function. Each call to `.score()`
runs the wrapped assertion and posts the result as a numeric score to a Langfuse
trace.

```python
class LangfuseAdapter:
    def __init__(
        self,
        assertion_fn: Callable[..., TestResult],
        name: str = "mltk",
    ) -> None: ...

    def score(
        self,
        trace_id: str,
        observation_id: str | None = None,
        **kwargs: Any,
    ) -> TestResult: ...
```

**Module:** `mltk.integrations.langfuse`

#### Constructor Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `assertion_fn` | `Callable[..., TestResult]` | *(required)* | Any mltk assertion function that returns a `TestResult`. Must accept keyword arguments matching the data you pass to `.score()`. |
| `name` | `str` | `"mltk"` | Score name shown in the Langfuse dashboard. Langfuse prefixes this in its UI, so a name like `"faithfulness"` becomes clearly visible. |

#### Raises

`ImportError` -- if `langfuse` is not installed. The error message includes
`pip install langfuse`. The import is lazy (happens at first `.score()` call,
not at module load time).

---

### `score`

Run the assertion and post the result as a Langfuse score.

```python
def score(
    self,
    trace_id: str,
    observation_id: str | None = None,
    **kwargs: Any,
) -> TestResult
```

#### Parameters

| Name | Type | Description |
|------|------|-------------|
| `trace_id` | `str` | Langfuse trace ID to attach the score to. Obtained from the Langfuse SDK or decorator. |
| `observation_id` | `str \| None` | Optional observation (span) ID for span-level scoring. When `None`, the score is attached to the trace itself. |
| `**kwargs` | `Any` | Arguments forwarded to the assertion function (e.g., `output`, `expected`, `context`). |

#### What gets posted

```python
client.score(
    trace_id=trace_id,
    name=self.name,
    value=1.0 if result.passed else 0.0,
    comment=result.message,
    data_type="NUMERIC",
)
```

If `observation_id` is provided, it is included in the score kwargs for
span-level granularity. Scores appear in the Langfuse trace detail view under
the "Scores" section.

#### Returns

The `TestResult` from the assertion function, so callers can use it for local
assertions as well.

#### Error handling

If the assertion raises `MltkAssertionError` (critical failure), the adapter
posts a score of `0.0` with the error message, then re-raises the exception
so pytest still sees the failure.

---

### Example: Run assertions, post scores to Langfuse

```python
from mltk.integrations.langfuse import LangfuseAdapter
from mltk.domains.llm import assert_faithfulness

# Wrap any assertion
adapter = LangfuseAdapter(assert_faithfulness, name="faithfulness")

# trace_id comes from the Langfuse trace created when the LLM call ran
trace_id = "clue-xyz-123"

# Run assertion + post score in one call
result = adapter.score(
    trace_id=trace_id,
    output="Q3 revenue grew 12%.",
    expected="Revenue increased 12% year-over-year in Q3.",
)
assert result.passed

# With observation-level granularity
result = adapter.score(
    trace_id=trace_id,
    observation_id="span-456",
    output="Paris is in France",
    expected="France",
)
```

---

### pytest + Langfuse conftest.py

```python
# conftest.py
import pytest
from mltk.integrations.langfuse import LangfuseAdapter
from mltk.core.assertion import assert_true


def check_length(output, **kwargs):
    return assert_true(
        len(output) > 10,
        name="min_length",
        message=f"Output has {len(output)} chars",
    )


@pytest.fixture(scope="session")
def langfuse_adapter():
    return LangfuseAdapter(check_length, name="min_length")


def test_output_quality(langfuse_adapter):
    result = langfuse_adapter.score(
        trace_id="trace-abc",
        output="Hello, world! This is a test.",
    )
    assert result.passed
```

---

## `assert_trace_quality`

A single assertion that bundles the most common production trace quality checks.
Accepts a trace dict and verifies latency, cost, and quality score in one call --
designed as a CI/CD deployment gate.

```python
from mltk.integrations.trace_quality import assert_trace_quality

def assert_trace_quality(
    trace: dict[str, Any],
    *,
    max_latency_ms: float | None = None,
    max_cost_usd: float | None = None,
    min_score: float | None = None,
    judge_fn: Callable[[dict[str, Any]], float] | None = None,
) -> TestResult:
    ...
```

**Module:** `mltk.integrations.trace_quality`

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `trace` | `dict[str, Any]` | *(required)* | Dict containing trace data. Expected keys: `latency_ms` (float), `cost_usd` (float), `score` (float), `output` (str), `input` (str). Any key can be absent -- the gate only checks thresholds you explicitly set. |
| `max_latency_ms` | `float \| None` | `None` | Maximum allowed total trace duration in milliseconds. Reads `trace["latency_ms"]`. If the key is missing, the check is skipped (not failed). |
| `max_cost_usd` | `float \| None` | `None` | Maximum allowed cost in US dollars. Reads `trace["cost_usd"]`. Same skip-if-missing behavior as latency. |
| `min_score` | `float \| None` | `None` | Minimum acceptable quality score (0.0--1.0). If `judge_fn` is provided, calls it with the trace dict. Otherwise reads `trace["score"]`. |
| `judge_fn` | `Callable[[dict[str, Any]], float] \| None` | `None` | Optional callable that takes the trace dict and returns a quality score (0.0--1.0). When provided, overrides `trace["score"]` for the quality check. |

### Trace format

The `trace` dict should contain keys matching what observability platforms export:

```python
trace = {
    "latency_ms": 450.0,        # End-to-end latency in milliseconds
    "cost_usd": 0.003,          # Total cost in US dollars
    "score": 0.92,              # Quality score from 0.0 to 1.0
    "output": "Paris is ...",   # LLM output text (used by judge_fn)
    "input": "What is ...",     # Original prompt (used by judge_fn)
}
```

### Return value

`TestResult` with `name="integrations.trace_quality"` and `.passed` reflecting
whether ALL requested checks passed. On failure, `.message` lists every violated
threshold. The details dict contains actual values for each checked dimension:

```python
# When all checks pass:
{
    "latency_ms": 450.0,
    "max_latency_ms": 2000.0,
    "cost_usd": 0.003,
    "max_cost_usd": 0.01,
    "score": 0.92,
    "min_score": 0.8,
}
```

Only the dimensions you actually check appear in the details dict. If you only
check `max_latency_ms`, the details will only contain `latency_ms` and
`max_latency_ms`.

### Example: Deployment gate in CI/CD

```python
import pytest
from mltk.integrations.trace_quality import assert_trace_quality

def test_production_trace_meets_sla(production_trace):
    """Gate: every production trace must clear all quality thresholds."""
    result = assert_trace_quality(
        production_trace,
        max_latency_ms=2000,
        max_cost_usd=0.01,
        min_score=0.8,
    )
    assert result.passed, (
        f"Trace quality gate failed: {result.message}\n"
        f"Details: {result.details}"
    )
```

### Example: With a custom judge function

```python
from mltk.integrations.trace_quality import assert_trace_quality

def my_judge(trace):
    return 0.95 if "correct" in trace.get("output", "") else 0.1

result = assert_trace_quality(
    {"output": "The correct answer is 42", "latency_ms": 100},
    max_latency_ms=500,
    judge_fn=my_judge,
    min_score=0.5,
)
assert result.passed
```

### Example: Latency-only gate (no judge needed)

```python
from mltk.integrations.trace_quality import assert_trace_quality

result = assert_trace_quality(
    trace,
    max_latency_ms=1500,
    # No min_score -- no judge required
)
assert result.passed
```

### Example: Batch-assert a sample of production traces

```python
from mltk.integrations.trace_quality import assert_trace_quality

traces = [...]  # List of trace dicts from your observability platform

failures = []
for i, trace in enumerate(traces):
    result = assert_trace_quality(
        trace,
        max_latency_ms=3000,
        min_score=0.80,
    )
    if not result.passed:
        failures.append((i, result.message))

assert not failures, f"{len(failures)} traces failed quality gate:\n" + \
    "\n".join(f"  trace {idx}: {msg}" for idx, msg in failures)
```

---

## Installation Reference

| Integration | Install command | Adds |
|-------------|----------------|------|
| Phoenix adapter | `pip install mltk[phoenix]` | `arize-phoenix-otel`, `arize-phoenix-evals` |
| Langfuse adapter | `pip install mltk[langfuse]` | `langfuse>=4.0` |
| Both | `pip install mltk[phoenix,langfuse]` | All of the above |
| OTEL only (Jaeger/Tempo) | `pip install mltk[otel]` | `opentelemetry-*` packages |

All adapters follow the graceful degradation pattern from `MltkTracer`: if the
optional package is not installed, importing the adapter raises a clear `ImportError`
with the correct install command rather than a cryptic module-not-found error.

---

## Competitive Comparison

mltk assertions as evaluators vs native platform evaluators:

| Dimension | Phoenix native evals | Langfuse scoring | mltk adapters |
|-----------|---------------------|-----------------|---------------|
| **LLM dependency** | Built-in LLM classifiers (OpenAI) | Manual score posting (any method) | `judge_fn` callable -- any backend |
| **Offline / air-gapped** | Requires LLM API | Manual (no API needed for scoring) | Lexical/embedding fallbacks work offline |
| **Assertion breadth** | LLM quality metrics (relevance, hallucination) | None built-in -- you post scores | 200+ assertions: drift, bias, schema, latency, LLM quality |
| **pytest native** | Not pytest-aware | Not pytest-aware | First-class pytest plugin |
| **CI/CD gate** | Custom post-processing needed | Custom post-processing needed | `assert_trace_quality` is a single pytest assert |
| **Vendor neutral** | Phoenix-specific | Langfuse-specific | Same assertion code works with any backend |
| **Self-hosted** | YES (local `px.launch_app()`) | YES (Docker) | YES (no SaaS dependency) |
| **Cost control** | Platform controls LLM judge calls | No built-in LLM calls | You own the `judge_fn` |

**When to use Phoenix native evaluators:** You want the evaluator UI built into Phoenix's
span browser and are already paying for an LLM API in your stack. The Phoenix UI
for browsing evaluation results is excellent.

**When to use mltk adapters:** You need more than LLM quality metrics (data drift,
schema, bias, latency) or you need offline/air-gapped operation, pytest integration,
or vendor neutrality. The adapters let mltk assertions appear *inside* Phoenix or
Langfuse rather than competing with them.

---

## Related Documentation

- [OpenTelemetry Integration](otel.md) -- `MltkTracer`, `trace_suite`, `export_json`
- [LLM-as-Judge Evaluation](llm-judge.md) -- `assert_llm_judge_score`, pairwise comparison
- [Judge Defaults](judge-defaults.md) -- `configure_default_judge`, auto-fallback chain
- [RAG Evaluation](rag-evaluation.md) -- `assert_faithfulness`, `assert_relevancy`
- [W&B Integration](wandb.md) -- `WandbLogger` for experiment tracking
