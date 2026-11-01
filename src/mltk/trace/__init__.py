"""Canonical agent-trace capture — clearance-tiered records.

``mltk.trace`` owns the capture record. Existing schemas
(``AgentTrace``, ``SpanTrace``, the quality dict) are adapter *outputs*
in their own packages and must not be imported here.
"""

from __future__ import annotations

from mltk.trace.record import CapturedCall, CaptureRecord
from mltk.trace.session import CaptureSession, capture
from mltk.trace.tier import ClearanceTier

__all__ = [
    "CaptureRecord",
    "CaptureSession",
    "CapturedCall",
    "ClearanceTier",
    "capture",
]
