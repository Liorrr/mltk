"""Frozen capture record. Unobserved fields stay ``None``."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mltk.trace.tier import ClearanceTier


@dataclass(frozen=True)
class CapturedCall:
    """A single tool/function call observed during capture."""

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    result: str | None = None
    error: str | None = None
    duration_ms: float | None = None


@dataclass(frozen=True)
class SubagentCall:
    """A nested sub-agent invocation observed at T1."""

    name: str
    duration_ms: float | None = None


@dataclass(frozen=True)
class FanoutCall:
    """A fan-out of concurrent API calls observed at T1."""

    count: int
    duration_ms: float | None = None


@dataclass(frozen=True)
class RetryEvent:
    """One error/retry hop observed at T1."""

    error: str
    attempt: int


@dataclass(frozen=True)
class CaptureRecord:
    """Canonical capture record for one model/agent run.

    ``tier`` is a typed field, never a metadata entry. Every optional
    field defaults to ``None`` so missing observation is distinct from
    a recorded zero.
    """

    tier: ClearanceTier
    # T0
    tool_call_count: int | None = None
    tool_calls: tuple[CapturedCall, ...] | None = None
    api_call_count: int | None = None
    round_trip_count: int | None = None
    retry_count: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    stop_reason: str | None = None
    per_hop_latency_ms: tuple[float, ...] | None = None
    total_duration_ms: float | None = None
    # T1
    subagent_calls: tuple[SubagentCall, ...] | None = None
    fanout_api_calls: tuple[FanoutCall, ...] | None = None
    error_retry_chain: tuple[RetryEvent, ...] | None = None
    cache_hits: int | None = None
    action_log: tuple[str, ...] | None = None
    # T2
    logprobs: Any | None = None
    top_k: Any | None = None
    sampling_params: dict[str, Any] | None = None
    refusal_internals: Any | None = None
    hidden_state_hooks: Any | None = None
