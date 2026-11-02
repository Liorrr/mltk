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
        if not session.tier.permits(ClearanceTier.T1):
            raise ValueError(
                "InstrumentedClient requires CaptureSession(tier=ClearanceTier.T1)"
            )
        self._inner = inner
        self.session = session
        self._agent_idx: int | None = None

    def call(self, *args: Any, **kwargs: Any) -> Any:
        """Invoke ``inner`` and record one round trip (or a retry hop on error)."""
        self.session.record_action("call")
        start = time.perf_counter()
        try:
            result = self._inner(*args, **kwargs)
        except Exception as exc:
            failed_ms = (time.perf_counter() - start) * 1000.0
            self.session.record_error_retry(str(exc), duration_ms=failed_ms)
            raise
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        self.session.record_round_trip(latency_ms=elapsed_ms)
        return result

    def fanout(self, requests: Sequence[Any]) -> list[Any]:
        """Call ``inner`` once per request and record the fan-out."""
        self.session.record_action("fanout")
        start = time.perf_counter()
        issued = 0
        results: list[Any] = []
        try:
            for req in requests:
                issued += 1
                try:
                    results.append(self._inner(req))
                except Exception as exc:
                    self.session.record_error_retry(str(exc))
                    raise
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            self.session.record_fanout(issued, duration_ms=elapsed_ms)
        return results

    def subagent(self, name: str) -> InstrumentedClient:
        """Return a child client bound to the same session."""
        self.session.record_action(f"subagent:{name}")
        idx = self.session.record_subagent(name, parent=self._agent_idx)
        child = InstrumentedClient(self._inner, session=self.session)
        child._agent_idx = idx
        return child

    def cache_hit(self) -> None:
        """Record a cache hit without invoking ``inner``."""
        self.session.record_cache_hit()
        self.session.record_action("cache_hit")
