"""Clearance tiers for trace capture."""

from __future__ import annotations

from enum import Enum


class ClearanceTier(str, Enum):
    """How much of a model run is observable.

    T0
        Closed inference: tool calls, round-trips, retries, tokens,
        stop reason, per-hop latency — only if the caller recorded them.
    T1
        Instrumented client: nested sub-agent calls, fan-out, retry
        chains, cache hits, ordered action log.
    T2
        Open inference: logprobs, top-k, sampling params, refusal
        internals, hidden-state hooks.
    """

    T0 = "t0"
    T1 = "t1"
    T2 = "t2"
