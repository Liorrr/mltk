"""Span data model for LLM trace evaluation."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class SpanKind(Enum):
    """Span type classification (OpenInference-aligned)."""

    LLM = "LLM"
    TOOL = "TOOL"
    RETRIEVER = "RETRIEVER"
    EMBEDDING = "EMBEDDING"
    CHAIN = "CHAIN"
    AGENT = "AGENT"
    GUARDRAIL = "GUARDRAIL"
    RERANKER = "RERANKER"


@dataclass
class Span:
    """A single span within an LLM execution trace.

    Attributes:
        name: Human-readable operation name.
        kind: The span kind classification.
        duration_ms: Execution time in milliseconds.
        input_tokens: Number of input/prompt tokens.
        output_tokens: Number of output/completion tokens.
        cost_usd: Estimated cost in USD.
        status: Span status (``"ok"`` or ``"error"``).
        error: Error message if the span failed.
        parent_id: ID of the parent span (None for root).
        span_id: Unique identifier for this span.
        input_text: The input/prompt text.
        output_text: The output/completion text.
        metadata: Arbitrary key-value metadata.
    """

    name: str
    kind: SpanKind
    duration_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    status: str = "ok"
    error: str | None = None
    parent_id: str | None = None
    span_id: str = ""
    input_text: str = ""
    output_text: str = ""
    metadata: dict[str, Any] = field(
        default_factory=dict,
    )

    def __post_init__(self) -> None:
        if not self.span_id:
            self.span_id = str(uuid.uuid4())

    @property
    def total_tokens(self) -> int:
        """Sum of input and output tokens."""
        return self.input_tokens + self.output_tokens

    @property
    def is_error(self) -> bool:
        """Whether this span has error status."""
        return self.status == "error"


@dataclass
class SpanTrace:
    """A collection of spans forming a trace.

    Attributes:
        spans: Ordered list of spans in the trace.
        trace_id: Unique trace identifier.
        total_duration_ms: Wall-clock duration of the trace.
        metadata: Arbitrary trace-level metadata.
    """

    spans: list[Span] = field(default_factory=list)
    trace_id: str = ""
    total_duration_ms: float = 0.0
    metadata: dict[str, Any] = field(
        default_factory=dict,
    )

    def __post_init__(self) -> None:
        if not self.trace_id:
            self.trace_id = str(uuid.uuid4())

    def root_spans(self) -> list[Span]:
        """Return spans with no parent."""
        return [
            s for s in self.spans
            if s.parent_id is None
        ]

    def children(self, span_id: str) -> list[Span]:
        """Return direct children of a span."""
        return [
            s for s in self.spans
            if s.parent_id == span_id
        ]

    def descendants(
        self, span_id: str,
    ) -> list[Span]:
        """Return all descendants of a span."""
        result: list[Span] = []
        for child in self.children(span_id):
            result.append(child)
            result.extend(
                self.descendants(child.span_id)
            )
        return result

    def spans_by_kind(
        self, kind: SpanKind,
    ) -> list[Span]:
        """Return spans matching the given kind."""
        return [
            s for s in self.spans
            if s.kind == kind
        ]

    def error_spans(self) -> list[Span]:
        """Return spans with error status."""
        return [
            s for s in self.spans if s.is_error
        ]

    @property
    def total_tokens(self) -> int:
        """Sum of tokens across all spans."""
        return sum(
            s.total_tokens for s in self.spans
        )

    @property
    def total_cost_usd(self) -> float:
        """Sum of cost across all spans."""
        return sum(
            s.cost_usd for s in self.spans
        )

    @property
    def span_count(self) -> int:
        """Number of spans in the trace."""
        return len(self.spans)

    @property
    def error_count(self) -> int:
        """Number of error spans."""
        return len(self.error_spans())

    @classmethod
    def from_dicts(
        cls,
        spans: list[dict[str, Any]],
    ) -> SpanTrace:
        """Build a SpanTrace from a list of dicts.

        Each dict should have at least ``name`` and
        ``kind`` keys.  ``kind`` can be a string or
        a ``SpanKind`` enum value.

        Args:
            spans: List of span dictionaries.

        Returns:
            A populated SpanTrace instance.
        """
        parsed: list[Span] = []
        for d in spans:
            kind = d.get("kind", "CHAIN")
            if isinstance(kind, str):
                kind = SpanKind(kind)
            parsed.append(
                Span(
                    name=d.get("name", ""),
                    kind=kind,
                    duration_ms=float(
                        d.get("duration_ms", 0.0)
                    ),
                    input_tokens=int(
                        d.get("input_tokens", 0)
                    ),
                    output_tokens=int(
                        d.get("output_tokens", 0)
                    ),
                    cost_usd=float(
                        d.get("cost_usd", 0.0)
                    ),
                    status=d.get("status", "ok"),
                    error=d.get("error"),
                    parent_id=d.get("parent_id"),
                    span_id=d.get(
                        "span_id",
                        str(uuid.uuid4()),
                    ),
                    input_text=d.get(
                        "input_text", ""
                    ),
                    output_text=d.get(
                        "output_text", ""
                    ),
                    metadata=d.get(
                        "metadata", {}
                    ),
                )
            )
        return cls(spans=parsed)


__all__ = ["Span", "SpanKind", "SpanTrace"]
