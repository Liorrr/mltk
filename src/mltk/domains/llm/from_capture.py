"""Project a CaptureRecord onto AgentTrace / SpanTrace.

:func:`to_agent_trace` keeps every unobserved token or duration as
``None``, so a budget assertion can tell "not measured" from a measured
zero. :func:`to_span_trace` still coerces ``None`` to ``0`` because the
:class:`~mltk.domains.llm.span.Span` fields are not optional; that
projection is lossy, and :class:`~mltk.trace.record.CaptureRecord`
stays the source of truth.
"""

from __future__ import annotations

from mltk.domains.llm.span import Span, SpanKind, SpanTrace
from mltk.domains.llm.trace import AgentTrace, ToolCall
from mltk.trace import CaptureRecord


def to_agent_trace(record: CaptureRecord) -> AgentTrace:
    """Adapt *record* to :class:`AgentTrace`.

    Unobserved fields stay ``None``: ``total_tokens`` is ``None`` unless
    both ``input_tokens`` and ``output_tokens`` were observed (a partial
    count is not a total), ``total_duration_ms`` mirrors the record, and
    each tool call keeps its own ``duration_ms``.
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
                    duration_ms=captured.duration_ms,
                )
            )
    tokens = (
        None
        if record.input_tokens is None or record.output_tokens is None
        else record.input_tokens + record.output_tokens
    )
    return AgentTrace(
        tool_calls=calls,
        total_tokens=tokens,
        total_duration_ms=record.total_duration_ms,
    )


def to_span_trace(record: CaptureRecord) -> SpanTrace:
    """Adapt *record* to :class:`SpanTrace`.

    No tool calls → a single ``LLM`` span. Otherwise an ``AGENT`` root
    plus one ``TOOL`` span per captured call. ``Span`` token and duration
    fields are not optional, so unobserved values become ``0`` here; use
    :func:`to_agent_trace` when a budget check must see "not measured".
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
