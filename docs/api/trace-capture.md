# Agent Trace Capture

Canonical capture record for what a model or agent actually did, at a
declared clearance tier. Existing `AgentTrace`, `SpanTrace`, and the
quality-dict gate remain adapter *outputs* — they do not own capture.

**Module:** `mltk.trace`

Unobservable fields are `None`, never `0`. The clearance tier is a typed
field on the record, not a metadata entry.

---

## Tiers

| Tier | What can be observed |
|------|----------------------|
| `T0` | Closed inference: tool calls, round-trips, retries, tokens, stop reason, per-hop latency — only if recorded |
| `T1` | Instrumented client (fields reserved; always `None` at T0) |
| `T2` | Open inference / logprobs (fields reserved; always `None` at T0) |

---

## Quick Start

```python
from mltk.trace import CaptureSession, ClearanceTier, capture

with CaptureSession(tier=ClearanceTier.T0) as cap:
    cap.record_round_trip(latency_ms=12.0, input_tokens=10, output_tokens=20, stop_reason="stop")
    cap.record_tool_call(name="search", arguments={"q": "weather"})
record = cap.finish()
assert record.input_tokens == 10
assert record.logprobs is None  # T2 field, unobserved at T0


@capture(tier=ClearanceTier.T0)
def model_fn(prompt: str) -> str:
    return "ok"

text, record = model_fn("hi")
```

Project onto existing assertions without changing those schemas:

```python
from mltk.domains.llm.from_capture import to_agent_trace
from mltk.domains.llm.agentic import assert_tool_chain

assert_tool_chain(to_agent_trace(record), expected_tools=["search"])
```

`to_agent_trace` / `to_span_trace` coerce missing numbers to `0` because
those legacy types use `int` / `float`. Honesty stays on `CaptureRecord`.
`to_quality_dict` *omits* `None` keys so `assert_trace_quality` skip-if-missing
still applies.

---

## T1 instrumented client

Wrap any callable. Nested sub-agents share the session. Unobserved T1
fields stay `None`. T2 fields stay `None`.

```python
from mltk.trace import CaptureSession, ClearanceTier, InstrumentedClient

with CaptureSession(tier=ClearanceTier.T1) as cap:
    client = InstrumentedClient(model_fn, session=cap)
    writer = client.subagent("writer")
    writer.call("draft this")
    client.fanout(["q1", "q2"])
    client.cache_hit()
record = cap.finish()
assert record.subagent_calls[0].name == "writer"
assert record.cache_hits == 1
assert record.logprobs is None
```

A T0 session cannot construct `InstrumentedClient` (raises `ValueError`).
