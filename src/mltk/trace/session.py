"""T0 capture session and decorator. No provider SDK."""

from __future__ import annotations

import functools
import time
from collections.abc import Callable
from typing import Any, TypeVar

from mltk.trace.record import (
    CapturedCall,
    CaptureRecord,
    FanoutCall,
    RetryEvent,
    SubagentCall,
)
from mltk.trace.tier import ClearanceTier

F = TypeVar("F", bound=Callable[..., Any])


class CaptureSession:
    """Mutable builder that freezes a :class:`CaptureRecord` on ``finish()``.

    Unobserved fields stay ``None``. Counts become integers only after
    the matching ``record_*`` method runs at least once.
    """

    def __init__(self, *, tier: ClearanceTier) -> None:
        self.tier = tier
        self._hops: list[float] = []
        self._calls: list[CapturedCall] = []
        self._retries: int = 0
        self._retry_seen: bool = False
        self._input_tokens: int | None = None
        self._output_tokens: int | None = None
        self._stop_reason: str | None = None
        self._api_explicit: int | None = None
        self._subagents: list[SubagentCall] = []
        self._fanouts: list[FanoutCall] = []
        self._retry_events: list[RetryEvent] = []
        self._cache_hits: int | None = None
        self._actions: list[str] = []
        self._finished: CaptureRecord | None = None

    def _require_t1(self) -> None:
        if self.tier is not ClearanceTier.T1:
            raise ValueError(
                "T1 observations require CaptureSession(tier=ClearanceTier.T1)"
            )

    def __enter__(self) -> CaptureSession:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def record_round_trip(
        self,
        latency_ms: float,
        *,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        stop_reason: str | None = None,
    ) -> None:
        """Record one closed-inference round trip."""
        self._hops.append(float(latency_ms))
        if input_tokens is not None:
            self._input_tokens = (self._input_tokens or 0) + int(input_tokens)
        if output_tokens is not None:
            self._output_tokens = (self._output_tokens or 0) + int(output_tokens)
        if stop_reason is not None:
            self._stop_reason = stop_reason

    def record_retry(self) -> None:
        """Record one retry attempt (not a successful round trip)."""
        self._retry_seen = True
        self._retries += 1

    def record_tool_call(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        result: str | None = None,
        error: str | None = None,
        duration_ms: float | None = None,
    ) -> None:
        """Record one tool/function call."""
        self._calls.append(
            CapturedCall(
                name=name,
                arguments=dict(arguments or {}),
                result=result,
                error=error,
                duration_ms=duration_ms,
            )
        )

    def record_api_call(self) -> None:
        """Record one underlying API call when it is not 1:1 with round trips."""
        self._api_explicit = (self._api_explicit or 0) + 1

    def record_subagent(self, name: str, *, duration_ms: float | None = None) -> None:
        """Record a nested sub-agent start (T1)."""
        self._require_t1()
        self._subagents.append(SubagentCall(name=name, duration_ms=duration_ms))

    def record_fanout(self, count: int, *, duration_ms: float | None = None) -> None:
        """Record a fan-out of API calls (T1)."""
        self._require_t1()
        self._fanouts.append(FanoutCall(count=int(count), duration_ms=duration_ms))

    def record_error_retry(self, error: str) -> None:
        """Record an error hop in the retry chain (T1) and bump T0 retry count."""
        self._require_t1()
        self.record_retry()
        self._retry_events.append(
            RetryEvent(error=error, attempt=self._retries)
        )

    def record_cache_hit(self) -> None:
        """Record one cache hit (T1)."""
        self._require_t1()
        self._cache_hits = (self._cache_hits or 0) + 1

    def record_action(self, action: str) -> None:
        """Append one ordered action-log entry (T1)."""
        self._require_t1()
        self._actions.append(action)

    def finish(self) -> CaptureRecord:
        """Freeze the session into a :class:`CaptureRecord`."""
        if self._finished is not None:
            return self._finished
        round_trip_count = len(self._hops) if self._hops else None
        if self._api_explicit is not None:
            api_call_count: int | None = self._api_explicit
        else:
            api_call_count = round_trip_count
        self._finished = CaptureRecord(
            tier=self.tier,
            tool_call_count=len(self._calls) if self._calls else None,
            tool_calls=tuple(self._calls) if self._calls else None,
            api_call_count=api_call_count,
            round_trip_count=round_trip_count,
            retry_count=self._retries if self._retry_seen else None,
            input_tokens=self._input_tokens,
            output_tokens=self._output_tokens,
            stop_reason=self._stop_reason,
            per_hop_latency_ms=tuple(self._hops) if self._hops else None,
            total_duration_ms=sum(self._hops) if self._hops else None,
            subagent_calls=tuple(self._subagents) if self._subagents else None,
            fanout_api_calls=tuple(self._fanouts) if self._fanouts else None,
            error_retry_chain=(
                tuple(self._retry_events) if self._retry_events else None
            ),
            cache_hits=self._cache_hits,
            action_log=tuple(self._actions) if self._actions else None,
        )
        return self._finished


def capture(*, tier: ClearanceTier) -> Callable[[F], Callable[..., tuple[Any, CaptureRecord]]]:
    """Wrap a callable: time one round trip and return ``(result, record)``.

    Does not scrape usage from the return value — missing tokens stay
    ``None``.
    """

    def decorator(fn: F) -> Callable[..., tuple[Any, CaptureRecord]]:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> tuple[Any, CaptureRecord]:
            session = CaptureSession(tier=tier)
            start = time.perf_counter()
            result = fn(*args, **kwargs)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            session.record_round_trip(latency_ms=elapsed_ms)
            return result, session.finish()

        return wrapper

    return decorator
