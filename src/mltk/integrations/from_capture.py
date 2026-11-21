"""Project a CaptureRecord onto the quality-dict consumed by assert_trace_quality.

Keys whose source field is ``None`` are omitted, so the quality gate sees
them as not measured. Only ``latency_ms`` is emitted today: a captured
trace has no cost or quality score, so a gate on ``max_cost_usd``, or on
``min_score`` without a ``judge_fn``, fails as unmeasured unless it passes
``on_unmeasured="skip"`` (or ``"pass"``).
"""

from __future__ import annotations

from typing import Any

from mltk.trace import CaptureRecord


def to_quality_dict(record: CaptureRecord) -> dict[str, Any]:
    """Flat dict for :func:`~mltk.integrations.trace_quality.assert_trace_quality`.

    Emits ``latency_ms`` (from ``total_duration_ms``) when it was observed,
    and nothing else. A missing key is "not measured" to the gate, whose
    default ``on_unmeasured="fail"`` fails a budget on it.
    """
    payload: dict[str, Any] = {}
    if record.total_duration_ms is not None:
        payload["latency_ms"] = record.total_duration_ms
    return payload
