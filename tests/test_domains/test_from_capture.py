"""Adapters: CaptureRecord -> AgentTrace / SpanTrace (projections)."""

from __future__ import annotations

from mltk.domains.llm.agentic import assert_tool_chain
from mltk.domains.llm.from_capture import to_agent_trace, to_span_trace
from mltk.domains.llm.span import SpanKind
from mltk.trace import CapturedCall, CaptureRecord, ClearanceTier


def _record_with_tools() -> CaptureRecord:
    return CaptureRecord(
        tier=ClearanceTier.T0,
        tool_call_count=2,
        tool_calls=(
            CapturedCall(
                name="search", arguments={"q": "weather"},
                result="ok", error=None, duration_ms=10.0,
            ),
            CapturedCall(
                name="calculator", arguments={"expr": "2+2"},
                result="4", error=None, duration_ms=2.0,
            ),
        ),
        input_tokens=11,
        output_tokens=7,
        total_duration_ms=12.0,
    )


def test_to_agent_trace_feeds_assert_tool_chain() -> None:
    rec = _record_with_tools()
    trace = to_agent_trace(rec)
    result = assert_tool_chain(trace, expected_tools=["search", "calculator"])
    assert result.passed
    assert trace.total_tokens == 18
    assert trace.total_duration_ms == 12.0


def test_to_agent_trace_coerces_missing_tokens_to_zero() -> None:
    rec = CaptureRecord(tier=ClearanceTier.T0)
    trace = to_agent_trace(rec)
    assert rec.input_tokens is None
    assert rec.output_tokens is None
    assert rec.total_duration_ms is None
    assert trace.total_tokens == 0
    assert trace.total_duration_ms == 0.0


def test_adapter_does_not_mutate_capture_record() -> None:
    rec = CaptureRecord(tier=ClearanceTier.T0, input_tokens=None)
    to_agent_trace(rec)
    assert rec.input_tokens is None
    assert rec.logprobs is None


def test_to_span_trace_with_tool_calls_has_agent_root() -> None:
    rec = _record_with_tools()
    span_trace = to_span_trace(rec)
    kinds = [s.kind for s in span_trace.spans]
    assert SpanKind.AGENT in kinds
    assert kinds.count(SpanKind.TOOL) == 2


def test_to_span_trace_without_tools_is_single_llm_span() -> None:
    rec = CaptureRecord(
        tier=ClearanceTier.T0,
        total_duration_ms=5.0,
        input_tokens=3,
        output_tokens=4,
    )
    span_trace = to_span_trace(rec)
    assert len(span_trace.spans) == 1
    assert span_trace.spans[0].kind is SpanKind.LLM
