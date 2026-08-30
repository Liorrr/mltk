"""Adapter: CaptureRecord -> quality-dict for assert_trace_quality."""

from __future__ import annotations

import pytest

from mltk.core.assertion import MltkAssertionError
from mltk.integrations.from_capture import to_quality_dict
from mltk.integrations.trace_quality import assert_trace_quality
from mltk.trace import CaptureRecord, ClearanceTier


def test_quality_dict_omits_none_keys() -> None:
    rec = CaptureRecord(tier=ClearanceTier.T0, total_duration_ms=None)
    payload = to_quality_dict(rec)
    assert "latency_ms" not in payload
    result = assert_trace_quality(payload, max_latency_ms=1000)
    assert result.passed


def test_quality_dict_includes_observed_latency() -> None:
    rec = CaptureRecord(tier=ClearanceTier.T0, total_duration_ms=450.0)
    payload = to_quality_dict(rec)
    assert payload["latency_ms"] == 450.0
    result = assert_trace_quality(payload, max_latency_ms=2000)
    assert result.passed


def test_quality_dict_fails_when_latency_exceeds_budget() -> None:
    rec = CaptureRecord(tier=ClearanceTier.T0, total_duration_ms=3000.0)
    payload = to_quality_dict(rec)
    with pytest.raises(MltkAssertionError):
        assert_trace_quality(payload, max_latency_ms=2000)
