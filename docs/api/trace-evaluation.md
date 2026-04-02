# Span-Level Trace Evaluation

Assert quality, latency, cost, and structure at the individual span level
inside any pytest test — no observability platform required.

**Since:** v0.9.0

**Modules:**

- `mltk.domains.llm.span` — `SpanKind`, `Span`, `SpanTrace`
- `mltk.domains.llm.span_eval` — the four span assertions

---

## Why Span-Level Evaluation?

### Trace-level is a blunt instrument

A trace-level assertion tells you *whether* an agent run passed or failed.
It cannot tell you *where*. When a RAG pipeline fails, the failure could
live in the retriever (slow, wrong documents), the reranker (bad ordering),
the LLM call (hallucination, cost overrun), or a guardrail (false positive
blocking a valid answer). Trace-level metrics hide this entirely.

Consider a pipeline with five spans:

```
AGENT: research_query              [total: 8 200 ms, $0.041]
  ├─ RETRIEVER: fetch_docs         [  420 ms, 12 docs]
  ├─ RERANKER: rerank_docs         [  130 ms,  5 docs]
  ├─ LLM: draft_answer             [4 900 ms, 1 840 tokens, $0.037]
  ├─ GUARDRAIL: safety_check       [  110 ms, passed]
  └─ LLM: finalize_answer          [2 640 ms,  520 tokens, $0.004]
```

A trace-level assertion sees 8 200 ms and flags the run. A span-level
assertion pinpoints the `draft_answer` LLM call as consuming 60 % of
latency and 90 % of cost — actionable in a way that the trace aggregate
never is.

### mltk's differentiator

Every major observability platform (Phoenix, Langfuse, Datadog, Braintrust)
has converged on span-granular evaluation as of 2026. What they all lack is
**pytest-native span assertions** — tests that run in your CI pipeline,
fail builds, and require no running observability server. mltk fills this
gap with four composable assertions that work directly against the
`SpanTrace` data model.

| Platform | Span eval anchor | pytest-native? | Self-hosted? |
|----------|-----------------|:--------------:|:------------:|
| mltk | `Span` dataclass | YES | YES |
| Arize Phoenix | `span_id` (SpanEvaluations) | NO | YES |
| Langfuse | `observation_id` + `trace_id` | NO | YES |
| Datadog LLM Obs | `span_id` via `export_span()` | NO | NO |
| Braintrust | `span_id` + scope config | NO | NO |

---

## Quick Start

Five lines from import to assertion:

```python
from mltk.domains.llm.span import SpanTrace
from mltk.domains.llm.span_eval import assert_span_budget

trace = SpanTrace.from_dicts([
    {"name": "call_llm", "kind": "llm", "duration_ms": 1200,
     "input_tokens": 300, "output_tokens": 150, "cost_usd": 0.003},
    {"name": "fetch_docs", "kind": "retriever", "duration_ms": 80},
])

result = assert_span_budget(trace, max_total_tokens=1000, max_total_cost_usd=0.01)
assert result.passed, result.message
```

For pytest, use the assertion directly as the test body:

```python
def test_pipeline_stays_within_budget(trace_fixture):
    result = assert_span_budget(
        trace_fixture,
        max_total_tokens=2000,
        max_total_cost_usd=0.05,
        max_spans=10,
    )
    assert result.passed, result.message
```

---

## Span Model

### SpanKind — eight operation types

`SpanKind` follows the **OpenInference** convention used by Arize Phoenix
and 40+ instrumentation packages. Assign a kind to every span so that
kind-specific assertions can filter correctly.

```python
from mltk.domains.llm.span import SpanKind

kind = SpanKind.LLM        # a call to a language model
kind = SpanKind.RETRIEVER  # vector-store document fetch
```

| Kind | What it represents | Typical metrics |
|------|--------------------|-----------------|
| `LLM` | Call to a language model — sends messages, receives completion | `input_tokens`, `output_tokens`, `cost_usd`, `duration_ms` |
| `TOOL` | External function or API call made by the model | `duration_ms`, `status`, `error` |
| `RETRIEVER` | Vector-store or database document fetch | `duration_ms`, `status` |
| `EMBEDDING` | Embedding model call — text to vector | `input_tokens`, `duration_ms` |
| `CHAIN` | Orchestration logic / glue code between steps | `duration_ms` |
| `AGENT` | Top-level autonomous agent invocation | `duration_ms`, `total_tokens` |
| `GUARDRAIL` | Content safety or policy check | `duration_ms`, `status` |
| `RERANKER` | Reranking of candidate document sets | `duration_ms`, `status` |

!!! tip "When to use CHAIN vs AGENT"
    Use `AGENT` for the outermost span that represents an entire autonomous
    run. Use `CHAIN` for orchestration sub-steps within that run — a
    LangChain chain, a LlamaIndex query engine, or a routing step.

### Span — the atomic unit

Each `Span` captures a single operation with its performance metrics,
token usage, cost, and structural links to parent/child spans.

```python
from mltk.domains.llm.span import Span, SpanKind

span = Span(
    name="draft_answer",
    kind=SpanKind.LLM,
    duration_ms=4900.0,
    input_tokens=1440,
    output_tokens=400,
    cost_usd=0.037,
    status="ok",
    error=None,
    parent_id="root-001",
    span_id="llm-draft-001",
    input_text="Summarize the following documents...",
    output_text="Based on the retrieved context...",
    metadata={"model": "gpt-4o", "temperature": 0.2},
)
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | `str` | required | Human-readable operation name |
| `kind` | `SpanKind` | required | Operation type (see table above) |
| `duration_ms` | `float` | `0.0` | Wall-clock duration in milliseconds |
| `input_tokens` | `int` | `0` | Prompt tokens consumed (LLM / EMBEDDING) |
| `output_tokens` | `int` | `0` | Completion tokens generated (LLM) |
| `cost_usd` | `float` | `0.0` | Estimated USD cost for this span |
| `status` | `str` | `"ok"` | `"ok"` or `"error"` |
| `error` | `str \| None` | `None` | Error message if `status == "error"` |
| `parent_id` | `str \| None` | `None` | `span_id` of parent — absent for root spans |
| `span_id` | `str` | `""` (auto-UUID) | Unique identifier for this span |
| `input_text` | `str` | `""` | Raw input text (prompt, query, tool args) |
| `output_text` | `str` | `""` | Raw output text (completion, documents) |
| `metadata` | `dict` | `{}` | Arbitrary key-value span attributes |

### SpanTrace — the tree container

`SpanTrace` is an immutable container that holds all spans from one agent
run. It provides tree-navigation helpers and aggregation properties.

```python
from mltk.domains.llm.span import SpanTrace

trace = SpanTrace(
    spans=[span_a, span_b, span_c],
    trace_id="trace-42",
    total_duration_ms=8200.0,
    metadata={"run": "ci-nightly"},
)
```

**Tree navigation:**

```python
# Spans with no parent — top-level operations
roots = trace.root_spans()

# Direct children of a given span ID
kids = trace.children("agent-root-001")

# All descendants (depth-first) of a given span ID
all_under_agent = trace.descendants("agent-root-001")

# All spans of a specific kind
llm_spans = trace.spans_by_kind(SpanKind.LLM)
retriever_spans = trace.spans_by_kind(SpanKind.RETRIEVER)

# All spans where status == "error"
failures = trace.error_spans()
```

**Aggregation properties:**

```python
trace.total_tokens    # sum of (input_tokens + output_tokens) across all spans
trace.total_cost_usd  # sum of cost_usd across all spans
trace.span_count      # total number of spans
trace.error_count     # number of spans where status == "error"
```

### from_dicts — easy construction

Build a `SpanTrace` from a list of plain dicts. Useful in tests when you
don't want to construct `Span` objects manually.

```python
trace = SpanTrace.from_dicts(
    [
        {
            "name": "agent_run",
            "kind": "agent",
            "duration_ms": 9100,
            "span_id": "root",
        },
        {
            "name": "fetch_context",
            "kind": "retriever",
            "duration_ms": 320,
            "parent_id": "root",
            "span_id": "ret-1",
        },
        {
            "name": "generate_response",
            "kind": "llm",
            "duration_ms": 5400,
            "input_tokens": 900,
            "output_tokens": 280,
            "cost_usd": 0.028,
            "parent_id": "root",
            "span_id": "llm-1",
        },
    ],
    trace_id="demo-trace",
)
```

String kind values (`"LLM"`, `"TOOL"`, `"RETRIEVER"`, etc.) are
automatically coerced to `SpanKind` enum members.

---

## Assertions Reference

All four assertions share the same return type:

```python
from mltk.core.result import TestResult

result.passed       # bool — True = assertion passed
result.message      # human-readable explanation
result.details      # dict with per-span scores, violations
result.duration_ms  # execution time
```

### assert_span_quality

Validates the error rate across spans and optionally runs a judge function
to score individual span outputs for quality.

```python
from mltk.domains.llm.span_eval import assert_span_quality

result = assert_span_quality(
    trace,
    max_error_rate=0.05,         # at most 5 % of spans may be errors
    judge_fn=my_judge,           # optional: Callable[[Span], float]
    min_score=0.7,               # minimum mean judge score across spans
    span_kinds=[SpanKind.LLM],   # restrict judging to these kinds
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `trace` | `SpanTrace` | required | The trace to evaluate |
| `max_error_rate` | `float` | `0.0` | Maximum fraction of spans that may have `status == "error"` |
| `judge_fn` | `Callable[[Span], float] \| None` | `None` | Function that scores a single span's output — returns `[0.0, 1.0]` |
| `min_score` | `float` | `0.7` | Minimum mean judge score across the evaluated spans |
| `span_kinds` | `list[SpanKind] \| None` | `None` | Filter: only these kinds are judged (all kinds judged if `None`) |

!!! note "judge_fn is your evaluation logic"
    `judge_fn` receives the full `Span` object. It can check
    `span.output_text` for hallucinations, validate `span.metadata` for
    expected fields, or call an LLM judge. Return `1.0` for pass, `0.0`
    for fail, or any value in between for partial credit.

```python
def check_answer_quality(span: Span) -> float:
    if not span.output_text:
        return 0.0
    # Simple heuristic — replace with your LLM judge
    if "I don't know" in span.output_text:
        return 0.2
    return 1.0

result = assert_span_quality(
    trace,
    max_error_rate=0.1,
    judge_fn=check_answer_quality,
    min_score=0.8,
    span_kinds=[SpanKind.LLM],
)
```

### assert_span_latency

Enforces per-span and per-kind latency thresholds.

```python
from mltk.domains.llm.span_eval import assert_span_latency

result = assert_span_latency(
    trace,
    max_latency_ms=5000.0,          # no single span may exceed 5 s
    by_kind={
        SpanKind.LLM: 6000.0,       # LLM spans up to 6 s
        SpanKind.RETRIEVER: 500.0,  # retrieval must be under 500 ms
        SpanKind.TOOL: 2000.0,      # tool calls under 2 s
    },
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `trace` | `SpanTrace` | required | The trace to evaluate |
| `max_latency_ms` | `float \| None` | `None` | Global ceiling — applies to every span regardless of kind |
| `by_kind` | `dict[SpanKind, float] \| None` | `None` | Per-kind ceilings — override the global ceiling for specific kinds |

!!! tip "Kind-specific thresholds take precedence"
    When a span matches an entry in `by_kind`, the kind-specific threshold
    applies. The global `max_latency_ms` applies to all spans not covered
    by `by_kind`. Set `max_latency_ms=None` to use only per-kind thresholds.

```python
# Only enforce thresholds for latency-sensitive span kinds
result = assert_span_latency(
    trace,
    by_kind={
        SpanKind.RETRIEVER: 300.0,
        SpanKind.EMBEDDING: 200.0,
    },
)
```

### assert_span_budget

Enforces resource budgets across the whole trace: total tokens, total
cost, number of spans, and number of errors.

```python
from mltk.domains.llm.span_eval import assert_span_budget

result = assert_span_budget(
    trace,
    max_total_tokens=4000,       # all LLM spans combined
    max_total_cost_usd=0.10,     # total USD ceiling
    max_spans=15,                # total operation count
    max_errors=0,                # zero tolerance for errors
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `trace` | `SpanTrace` | required | The trace to evaluate |
| `max_total_tokens` | `int \| None` | `None` | Maximum sum of `(input_tokens + output_tokens)` across all spans |
| `max_total_cost_usd` | `float \| None` | `None` | Maximum sum of `cost_usd` across all spans |
| `max_spans` | `int \| None` | `None` | Maximum total number of spans in the trace |
| `max_errors` | `int` | `0` | Maximum number of spans where `status == "error"` |

!!! warning "Any exceeded budget fails the assertion"
    `assert_span_budget` checks all four limits independently. If any one
    is exceeded, the assertion fails and `result.message` identifies which
    limit was breached and by how much.

```python
# Enforce only cost — useful when token counts are unavailable
result = assert_span_budget(trace, max_total_cost_usd=0.05)

# Enforce only error count — zero-error gate
result = assert_span_budget(trace, max_errors=0)
```

### assert_span_sequence

Validates the structural shape of a trace: which span kinds must appear,
which names must appear, which kinds are forbidden, and minimum span count.

```python
from mltk.domains.llm.span_eval import assert_span_sequence

result = assert_span_sequence(
    trace,
    required_kinds=[SpanKind.RETRIEVER, SpanKind.LLM],
    required_names=["fetch_docs", "generate_response"],
    forbidden_kinds=[SpanKind.AGENT],   # disallow nested agent calls
    min_spans=3,                         # at least 3 spans total
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `trace` | `SpanTrace` | required | The trace to evaluate |
| `required_kinds` | `list[SpanKind] \| None` | `None` | All of these kinds must appear at least once |
| `required_names` | `list[str] \| None` | `None` | All of these span names must appear at least once |
| `forbidden_kinds` | `list[SpanKind] \| None` | `None` | None of these kinds may appear |
| `min_spans` | `int` | `1` | Trace must contain at least this many spans |

!!! note "Structural validation, not ordering"
    `assert_span_sequence` checks *presence and absence* — it does not
    verify ordering. Use it to confirm a RAG pipeline always retrieves
    before generating, or that a safety pipeline always includes a
    `GUARDRAIL` span. For strict ordering, compare parent-child
    relationships in the span tree directly.

```python
# Every RAG response must have at least one retrieval step
result = assert_span_sequence(
    trace,
    required_kinds=[SpanKind.RETRIEVER, SpanKind.LLM],
    min_spans=2,
)

# Safety pipeline: guardrail must run, no nested agents allowed
result = assert_span_sequence(
    trace,
    required_kinds=[SpanKind.GUARDRAIL],
    forbidden_kinds=[SpanKind.AGENT],
)
```

---

## Span Kinds in Depth

Understanding which span kind to assign — and what metrics to expect on
each — makes kind-specific assertions much more informative.

| Kind | Typical parent | Key numeric fields | Common failure modes |
|------|---------------|-------------------|---------------------|
| `AGENT` | None (root) | `duration_ms` | Timeout, runaway loops |
| `CHAIN` | `AGENT`, `CHAIN` | `duration_ms` | Routing error, skipped step |
| `LLM` | `AGENT`, `CHAIN` | `input_tokens`, `output_tokens`, `cost_usd`, `duration_ms` | High cost, hallucination, timeout |
| `TOOL` | `LLM`, `CHAIN` | `duration_ms` | Error status, invalid arguments |
| `RETRIEVER` | `CHAIN`, `AGENT` | `duration_ms` | Slow, no results, wrong documents |
| `EMBEDDING` | `RETRIEVER`, `CHAIN` | `input_tokens`, `duration_ms` | Slow, missing model |
| `RERANKER` | `RETRIEVER`, `CHAIN` | `duration_ms` | Inverted ranking, slow |
| `GUARDRAIL` | `CHAIN`, `AGENT` | `duration_ms` | False positive blocking valid output |

**LLM spans** are the highest-value target for quality and cost
assertions. They carry `input_tokens`, `output_tokens`, and `cost_usd`
and are the only span kind where hallucination or faithfulness judging
makes sense.

**RETRIEVER spans** are the primary target for latency assertions in RAG
pipelines. A retriever exceeding 300–500 ms typically indicates an index
issue or network problem, not a model issue.

**TOOL spans** are the primary target for error-rate assertions. Tool
calls fail for external reasons (API downtime, invalid arguments, rate
limits) that are independent of model quality.

**GUARDRAIL spans** should appear in every safety-sensitive pipeline.
Use `assert_span_sequence(required_kinds=[SpanKind.GUARDRAIL])` as a
structural gate in CI to ensure the safety step is never accidentally
removed.

---

## Tree Structure

### Parent-child relationships

Spans form a tree rooted at one or more root spans (spans with no
`parent_id`). The tree encodes the execution hierarchy — which operations
were initiated by which other operations.

```
AGENT: research_query              (span_id="root", parent_id=None)
  ├─ RETRIEVER: fetch_docs         (span_id="ret-1", parent_id="root")
  ├─ RERANKER: rerank_docs         (span_id="rnk-1", parent_id="root")
  ├─ LLM: draft_answer             (span_id="llm-1", parent_id="root")
  │     └─ TOOL: web_search        (span_id="tool-1", parent_id="llm-1")
  └─ LLM: finalize_answer          (span_id="llm-2", parent_id="root")
```

In this trace, the `draft_answer` LLM span spawned a `web_search` tool
call. The reranker and both LLM calls are siblings — they share the same
parent (`root`).

### Navigating the tree

```python
# Top-level operations
roots = trace.root_spans()
# → [Span(name="research_query", kind=AGENT)]

# Who does the AGENT call directly?
agent_children = trace.children("root")
# → [fetch_docs, rerank_docs, draft_answer, finalize_answer]

# Everything under draft_answer (recursive)
draft_subtree = trace.descendants("llm-1")
# → [Span(name="web_search", kind=TOOL)]

# All LLM spans regardless of depth
llm_spans = trace.spans_by_kind(SpanKind.LLM)
# → [draft_answer, finalize_answer]
```

### Modeling agent hierarchies

For multi-agent systems where one agent delegates to another, represent
each sub-agent as an `AGENT` span parented to the orchestrating agent:

```
AGENT: orchestrator              (span_id="orch")
  ├─ AGENT: research_agent       (span_id="res",  parent_id="orch")
  │     ├─ RETRIEVER: fetch      (span_id="ret",  parent_id="res")
  │     └─ LLM: synthesize       (span_id="llm1", parent_id="res")
  └─ AGENT: writing_agent        (span_id="wrt",  parent_id="orch")
        └─ LLM: draft            (span_id="llm2", parent_id="wrt")
```

`trace.descendants("orch")` returns every span across all sub-agents.
`trace.children("orch")` returns only the two direct sub-agents.

---

## Integration

### Phoenix SpanEvaluations bridge

mltk ships a `PhoenixAdapter` that wraps span assertions as Phoenix
callable evaluators. Results post back to Phoenix as `SpanEvaluations`
objects anchored by `span_id`.

```python
from mltk.integrations.phoenix import PhoenixAdapter
from mltk.domains.llm.span_eval import assert_span_latency

import phoenix as px

# Fetch spans from Phoenix as a DataFrame
client = px.Client()
spans_df = client.get_spans_dataframe(project_name="my-rag-app")

# Wrap the mltk assertion as a Phoenix evaluator
adapter = PhoenixAdapter(
    assert_span_latency,
    max_latency_ms=5000.0,
    by_kind={SpanKind.RETRIEVER: 400.0},
)

# Run and post results back to Phoenix
results_df = adapter.evaluate_dataframe(spans_df)
client.log_evaluations(
    px.trace.SpanEvaluations(
        eval_name="mltk.latency",
        dataframe=results_df,
    )
)
```

The adapter translates each Phoenix span row into a `SpanTrace` with a
single span, runs the assertion, and returns a dataframe with `span_id`,
`label` (`"pass"` / `"fail"`), and `score` columns — the format Phoenix
expects.

### Langfuse observation-level scoring

Langfuse attaches scores to individual observations via `observation_id`.
Capture the ID when the span is created and pass it to `langfuse.score()`
after running assertions.

```python
from langfuse import Langfuse
from mltk.domains.llm.span_eval import assert_span_budget

lf = Langfuse()
trace_lf = lf.trace(name="rag-pipeline")

retrieval_obs = trace_lf.span(name="retrieval", input={"query": q})
# ... run retrieval ...
retrieval_obs.end(output={"docs": docs})

# Build a SpanTrace from the completed operation and assert
span_trace = SpanTrace.from_dicts([
    {"name": "retrieval", "kind": "retriever",
     "duration_ms": retrieval_obs.end_time - retrieval_obs.start_time}
])
result = assert_span_budget(span_trace, max_spans=1, max_errors=0)

# Post result back to Langfuse — attached to the specific observation
lf.score(
    trace_id=trace_lf.id,
    observation_id=retrieval_obs.id,
    name="mltk.budget",
    value=1.0 if result.passed else 0.0,
    comment=result.message,
)
```

!!! note "Both trace_id and observation_id are required"
    Langfuse requires `trace_id` even when scoring a specific observation.
    Omitting it attaches the score to the trace root, not to the span.

### OTEL / MltkTracer export

When using `MltkTracer`, span assertions can be run inline during a
trace session. The tracer emits OTEL-compatible spans with
`openinference.span.kind` set to `"EVALUATOR"` for assertion result spans
— these appear in Phoenix's Evaluations tab rather than the main trace
waterfall.

```python
from mltk.integrations.otel import MltkTracer
from mltk.domains.llm.span_eval import assert_span_quality

with MltkTracer("rag-session") as tracer:
    with tracer.span("fetch_docs", kind="RETRIEVER") as s:
        docs = vector_store.search(query)
        s.set_attribute("doc_count", len(docs))

    with tracer.span("generate", kind="LLM") as s:
        answer = llm.complete(prompt)
        s.set_attribute("output_tokens", count_tokens(answer))

    # Assert on the completed trace
    span_trace = tracer.to_span_trace()
    result = assert_span_quality(
        span_trace,
        max_error_rate=0.0,
        span_kinds=[SpanKind.LLM],
    )
    tracer.record_assertion(result)
```

### Complement: assert_trace_quality (trace-level)

`assert_span_quality` works at the individual span level. For trace-level
quality assertions — overall task completion, final answer correctness,
or agent-level scoring — use the existing `assert_trace_quality` from
`mltk.assertions.trace`:

```python
from mltk.assertions.trace import assert_trace_quality
from mltk.domains.llm.span_eval import assert_span_latency

# Trace-level: did the agent complete the task?
trace_result = assert_trace_quality(
    agent_trace,
    judge_fn=task_completion_judge,
    min_score=0.8,
)

# Span-level: were individual steps fast enough?
latency_result = assert_span_latency(
    span_trace,
    by_kind={SpanKind.RETRIEVER: 400.0, SpanKind.LLM: 6000.0},
)

assert trace_result.passed and latency_result.passed
```

The two assertion families are complementary: trace-level catches
end-to-end failures; span-level pinpoints which component caused them.

---

## Examples

### Agent workflow validation

Verify that a ReAct agent trace contains the expected span types and stays
within error budget:

```python
def test_react_agent_structure(agent_trace: SpanTrace):
    # Must have at least one retrieval AND at least one LLM call
    structure = assert_span_sequence(
        agent_trace,
        required_kinds=[SpanKind.RETRIEVER, SpanKind.LLM],
        min_spans=3,
    )
    assert structure.passed, structure.message

    # No span may error
    quality = assert_span_quality(
        agent_trace,
        max_error_rate=0.0,
    )
    assert quality.passed, quality.message
```

### LLM cost monitoring

Fail CI if any single run exceeds cost or token budget:

```python
def test_llm_cost_within_budget(rag_trace: SpanTrace):
    result = assert_span_budget(
        rag_trace,
        max_total_tokens=3000,
        max_total_cost_usd=0.08,
    )
    assert result.passed, (
        f"Cost gate failed: {result.message}\n"
        f"  total_tokens={rag_trace.total_tokens}, "
        f"  total_cost=${rag_trace.total_cost_usd:.4f}"
    )
```

### Retrieval latency gate

Enforce that the retriever never becomes the bottleneck:

```python
def test_retriever_latency(search_trace: SpanTrace):
    result = assert_span_latency(
        search_trace,
        by_kind={
            SpanKind.RETRIEVER: 300.0,   # 300 ms SLA
            SpanKind.RERANKER: 150.0,    # 150 ms SLA
        },
    )
    if not result.passed:
        slow = [
            s for s in search_trace.spans_by_kind(SpanKind.RETRIEVER)
            if s.duration_ms > 300.0
        ]
        pytest.fail(
            f"{result.message}\n"
            + "\n".join(
                f"  {s.name}: {s.duration_ms:.0f} ms" for s in slow
            )
        )
```

### CI/CD span budget gate

A minimal CI gate that rejects any trace with errors or cost overruns:

```python
# tests/integration/test_pipeline_gates.py

import pytest
from mltk.domains.llm.span import SpanTrace, SpanKind
from mltk.domains.llm.span_eval import (
    assert_span_budget,
    assert_span_latency,
    assert_span_sequence,
)


@pytest.fixture
def pipeline_trace() -> SpanTrace:
    # In real tests, capture from your pipeline's instrumentation
    return SpanTrace.from_dicts([...])


def test_no_errors(pipeline_trace):
    r = assert_span_budget(pipeline_trace, max_errors=0)
    assert r.passed, r.message


def test_cost_ceiling(pipeline_trace):
    r = assert_span_budget(
        pipeline_trace, max_total_cost_usd=0.10
    )
    assert r.passed, r.message


def test_latency_sla(pipeline_trace):
    r = assert_span_latency(
        pipeline_trace,
        max_latency_ms=10_000.0,
        by_kind={
            SpanKind.RETRIEVER: 400.0,
            SpanKind.LLM: 7000.0,
        },
    )
    assert r.passed, r.message


def test_required_steps_present(pipeline_trace):
    r = assert_span_sequence(
        pipeline_trace,
        required_kinds=[
            SpanKind.RETRIEVER,
            SpanKind.LLM,
            SpanKind.GUARDRAIL,
        ],
    )
    assert r.passed, r.message
```

### LLM judge per span

Score the output quality of every LLM span using an in-process judge:

```python
def test_llm_output_quality(rag_trace: SpanTrace):
    def judge(span: Span) -> float:
        if not span.output_text:
            return 0.0
        # Replace with your actual judge — LLM call, NLI model, etc.
        forbidden = ["I cannot", "I don't know", "I'm not sure"]
        if any(phrase in span.output_text for phrase in forbidden):
            return 0.3
        return 1.0

    result = assert_span_quality(
        rag_trace,
        max_error_rate=0.0,
        judge_fn=judge,
        min_score=0.85,
        span_kinds=[SpanKind.LLM],
    )
    assert result.passed, result.message
```

---

## Competitor Comparison

| Feature | mltk | Arize Phoenix | Langfuse | Datadog LLM Obs | Braintrust |
|---------|:----:|:-------------:|:--------:|:---------------:|:----------:|
| pytest-native assertions | YES | NO | NO | NO | NO |
| No server required | YES | Optional | Optional | NO (SaaS) | NO (SaaS) |
| Span kind filtering | YES | YES | YES | YES | YES |
| Per-span judge scoring | YES | YES | YES | YES | YES |
| Latency gates in CI | YES | NO | NO | NO | NO |
| Cost gates in CI | YES | NO | NO | NO | NO |
| Structural (sequence) validation | YES | NO | NO | NO | NO |
| Tree navigation API | YES | NO | NO | NO | NO |
| Open source / self-hosted | YES | YES | YES | NO | NO |
| OpenInference span kinds | YES | YES | partial | NO | partial |
| Phoenix adapter | YES | — | N/A | N/A | N/A |
| Langfuse adapter | YES | N/A | — | N/A | N/A |

**Key differentiation:** mltk is the only library that provides span-level
assertions as first-class pytest constructs. All other platforms require a
running service, an async evaluation pipeline, or a SaaS account. mltk
span assertions run in 0 ms with zero infrastructure, making them suitable
for unit-test-speed CI gates.

---

## Research Citations

The span evaluation model in mltk is grounded in the following research:

**OpenInference Span Kinds (Arize, 2024)**
Used as the canonical span kind taxonomy. The eight-kind model (`LLM`,
`TOOL`, `RETRIEVER`, `EMBEDDING`, `CHAIN`, `AGENT`, `GUARDRAIL`,
`RERANKER`) maps cleanly to agent architectures and is supported by
Phoenix, LlamaIndex, and LangChain instrumentation packages.
[openinference spec](https://arize-ai.github.io/openinference/spec/)

**OTel GenAI Semantic Conventions (OpenTelemetry, 2024–2025)**
The `gen_ai.*` attribute namespace informs the `input_tokens`,
`output_tokens`, and `cost_usd` field design on `Span`. The `SpanKind`
taxonomy bridges OTel conventions with OpenInference kinds.
[opentelemetry.io/docs/specs/semconv/gen-ai/](https://opentelemetry.io/docs/specs/semconv/gen-ai/)

**AgentBoard / Progress Rate (Ma et al., NeurIPS 2024 Oral)**
Established that step-level scoring reveals failure modes (stagnation,
looped strategies) that binary trace-level metrics hide — the core
motivation for span-level assertions.

**Process Reward Models — "Let's Verify Step by Step"**
**(Lightman et al., OpenAI, ICLR 2024)**
Demonstrated that per-step quality labels are more informative than
final-outcome labels for detecting where a reasoning chain fails.
`assert_span_quality` with `judge_fn` implements the evaluation-time
analog of PRM scoring.

**AgentXRay: White-Boxing Agentic Systems**
**(arXiv:2602.05353, 2026)**
Validates execution DAGs against declared workflow specifications.
`assert_span_sequence` implements the presence and absence checks from
this framework without requiring a full DAG schema.

**ToolLLM / ToolEval (OpenBMB, ICLR 2024 Spotlight)**
Established the per-call Thought/Action/Action-Input decomposition.
`assert_span_quality` with `span_kinds=[SpanKind.TOOL]` enables
Action-level (tool choice) quality scoring consistent with the ToolEval
methodology.

**Langfuse observation-level evals (February 2026)**
*Langfuse changelog 2026-02-13 — observation-level-evals*
Platform evidence that span-granular evaluation is now the industry
standard. The `observation_id` integration pattern in the Langfuse adapter
section follows the official SDK API described in this changelog.

**R-83A: Span-Level Evaluation Patterns**
`docs/research/span-evaluation-patterns.md` — internal research brief
covering OpenTelemetry span model, OpenInference taxonomy, and platform
implementations.

**R-83B: Observability-Driven Testing Platforms**
`docs/research/observability-span-evaluation.md` — deep-dive on
Phoenix `SpanEvaluations`, Langfuse observation scoring, Datadog external
evaluations API, and Braintrust span scope model.

**R-83C: Academic Trace Analysis and Agent Evaluation**
`docs/research/academic-trace-analysis.md` — academic landscape across
step-level scoring, PRM/ORM distinction, tool use per-call metrics, and
multi-agent coordination evaluation frameworks.
