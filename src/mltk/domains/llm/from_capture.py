"""Project a CaptureRecord onto AgentTrace / SpanTrace.

These adapters coerce ``None`` into the legacy zeros those schemas
require. Honesty lives on :class:`~mltk.trace.record.CaptureRecord`;
the projection is lossy by design.
"""

from __future__ import annotations

from mltk.domains.llm.span import Span, SpanKind, SpanTrace
from mltk.domains.llm.trace import AgentTrace, ToolCall
from mltk.trace import CaptureRecord


def to_agent_trace(record: CaptureRecord) -> AgentTrace:
    """Adapt *record* to :class:`AgentTrace`.

    ``None`` token/duration fields become ``0`` / ``0.0``. Tool calls
    missing ``duration_ms`` become ``0.0``.
    """
    calls: list[ToolCall] = []
    if record.tool_calls:
        for captured in record.tool_calls:
            calls.append(
                ToolCall(
                    name=captured.name,
                    arguments=dict(captured.arguments),
                    result=captured.result,
                    error=captured.error,
                    duration_ms=(
                        0.0 if captured.duration_ms is None else captured.duration_ms
                    ),
                )
            )
    tokens = (record.input_tokens or 0) + (record.output_tokens or 0)
    duration = 0.0 if record.total_duration_ms is None else record.total_duration_ms
    return AgentTrace(
        tool_calls=calls,
        total_tokens=tokens,
        total_duration_ms=duration,
    )


def to_span_trace(record: CaptureRecord) -> SpanTrace:
    """Adapt *record* to :class:`SpanTrace`.

    No tool calls → a single ``LLM`` span. Otherwise an ``AGENT`` root
    plus one ``TOOL`` span per captured call.
    """
    duration = 0.0 if record.total_duration_ms is None else record.total_duration_ms
    if not record.tool_calls:
        span = Span(
            name="llm",
            kind=SpanKind.LLM,
            duration_ms=duration,
            input_tokens=record.input_tokens or 0,
            output_tokens=record.output_tokens or 0,
        )
        return SpanTrace(spans=[span], total_duration_ms=duration)

    root = Span(name="agent", kind=SpanKind.AGENT, duration_ms=duration)
    spans: list[Span] = [root]
    for captured in record.tool_calls:
        spans.append(
            Span(
                name=captured.name,
                kind=SpanKind.TOOL,
                duration_ms=(
                    0.0 if captured.duration_ms is None else captured.duration_ms
                ),
                parent_id=root.span_id,
                status="error" if captured.error else "ok",
                error=captured.error,
            )
        )
    return SpanTrace(spans=spans, total_duration_ms=duration)
