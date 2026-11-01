"""T1 instrumented client proxy. No vendor SDK."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from typing import Any

from mltk.trace.session import CaptureSession
from mltk.trace.tier import ClearanceTier


class InstrumentedClient:
    """Wrap a callable and record T1 observations onto a capture session.

    ``inner`` is provider-agnostic: any ``Callable[..., Any]``. Nested
    ``subagent()`` shares the same session so fan-out and child agents
    land on one :class:`~mltk.trace.record.CaptureRecord`.
    """

    def __init__(self, inner: Callable[..., Any], *, session: CaptureSession) -> None:
        if session.tier is not ClearanceTier.T1:
            raise ValueError(
                "InstrumentedClient requires CaptureSession(tier=ClearanceTier.T1)"
            )
        self._inner = inner
        self.session = session

    def call(self, *args: Any, **kwargs: Any) -> Any:
        """Invoke ``inner`` and record one round trip (or a retry hop on error)."""
        self.session.record_action("call")
        start = time.perf_counter()
        try:
            result = self._inner(*args, **kwargs)
        except Exception as exc:
            self.session.record_error_retry(str(exc))
            raise
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        self.session.record_round_trip(latency_ms=elapsed_ms)
        return result

    def fanout(self, requests: Sequence[Any]) -> list[Any]:
        """Call ``inner`` once per request and record the fan-out."""
        self.session.record_action("fanout")
        start = time.perf_counter()
        results = [self._inner(req) for req in requests]
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        self.session.record_fanout(len(requests), duration_ms=elapsed_ms)
        return results

    def subagent(self, name: str) -> InstrumentedClient:
        """Return a child client bound to the same session."""
        self.session.record_action(f"subagent:{name}")
        self.session.record_subagent(name)
        return InstrumentedClient(self._inner, session=self.session)

    def cache_hit(self) -> None:
        """Record a cache hit without invoking ``inner``."""
        self.session.record_cache_hit()
        self.session.record_action("cache_hit")
