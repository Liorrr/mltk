"""Project a CaptureRecord onto the quality-dict consumed by assert_trace_quality.

Keys whose source field is ``None`` are omitted so the quality gate's
skip-if-missing behaviour applies.
"""

from __future__ import annotations

from typing import Any

from mltk.trace import CaptureRecord


def to_quality_dict(record: CaptureRecord) -> dict[str, Any]:
    """Flat dict for :func:`~mltk.integrations.trace_quality.assert_trace_quality`."""
    payload: dict[str, Any] = {}
    if record.total_duration_ms is not None:
        payload["latency_ms"] = record.total_duration_ms
    return payload
