"""T0 capture session and decorator. No provider SDK."""

from __future__ import annotations

import functools
import time
from collections.abc import Callable
from typing import Any, TypeVar

from mltk.trace.record import CapturedCall, CaptureRecord
from mltk.trace.tier import ClearanceTier

F = TypeVar("F", bound=Callable[..., Any])


class CaptureSession:
    """Mutable builder that freezes a :class:`CaptureRecord` on ``finish()``.

    Unobserved fields stay ``None``. Counts become integers only after
    the matching ``record_*`` method runs at least once.

    ``total_duration_ms`` sums every observed segment -- round-trip
    latencies plus any tool-call ``duration_ms``. Time the caller never
    recorded is not in it, so treat it as "observed duration", not
    wall clock.
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
        self._finished: CaptureRecord | None = None

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

    def finish(self) -> CaptureRecord:
        """Freeze the session into a :class:`CaptureRecord`."""
        if self._finished is not None:
            return self._finished
        round_trip_count = len(self._hops) if self._hops else None
        if self._api_explicit is not None:
            api_call_count: int | None = self._api_explicit
        else:
            api_call_count = round_trip_count
        # Total spans every observed segment, not just round trips.
        # record_tool_call and record_round_trip are distinct APIs -- a
        # tool call is never also a hop -- so summing both cannot double
        # count. Tool calls left without a duration_ms contribute
        # nothing, the same way an unobserved field stays None.
        tool_ms = sum(
            c.duration_ms for c in self._calls if c.duration_ms is not None
        )
        observed_ms = sum(self._hops) + tool_ms
        total_duration_ms = observed_ms if (self._hops or tool_ms) else None
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
            total_duration_ms=total_duration_ms,
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
