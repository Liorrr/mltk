"""CaptureRecord honesty: unobserved fields are None; tier is typed."""

from __future__ import annotations

from mltk.trace import CapturedCall, CaptureRecord, ClearanceTier


def test_t0_unobserved_fields_are_none() -> None:
    rec = CaptureRecord(tier=ClearanceTier.T0)
    assert rec.input_tokens is None
    assert rec.output_tokens is None
    assert rec.stop_reason is None
    assert rec.logprobs is None
    assert rec.sampling_params is None
    assert rec.cache_hits is None
    assert rec.hidden_state_hooks is None


def test_tier_is_enum_not_string_metadata() -> None:
    rec = CaptureRecord(tier=ClearanceTier.T0)
    assert rec.tier is ClearanceTier.T0
    assert not hasattr(rec, "metadata")


def test_t0_does_not_default_counts_to_zero() -> None:
    rec = CaptureRecord(tier=ClearanceTier.T0)
    assert rec.tool_call_count is None
    assert rec.retry_count is None
    assert rec.round_trip_count is None


def test_observed_tool_calls_set_count() -> None:
    call = CapturedCall(
        name="search",
        arguments={"q": "x"},
        result=None,
        error=None,
        duration_ms=1.5,
    )
    rec = CaptureRecord(
        tier=ClearanceTier.T0,
        tool_calls=(call,),
        tool_call_count=1,
    )
    assert rec.tool_call_count == 1
    assert rec.logprobs is None


def test_trace_package_does_not_import_llm_or_integrations() -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parents[2] / "src" / "mltk" / "trace"
    for path in root.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "mltk.domains" not in text, path.name
        assert "mltk.integrations" not in text, path.name
