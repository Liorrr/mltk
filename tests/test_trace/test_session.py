"""CaptureSession and @capture decorator — T0 observation only."""

from __future__ import annotations

from mltk.trace import CaptureSession, ClearanceTier, capture


def test_session_records_round_trip_and_leaves_tokens_none() -> None:
    with CaptureSession(tier=ClearanceTier.T0) as cap:
        cap.record_round_trip(latency_ms=12.0)
    rec = cap.finish()
    assert rec.tier is ClearanceTier.T0
    assert rec.round_trip_count == 1
    assert rec.per_hop_latency_ms == (12.0,)
    assert rec.total_duration_ms == 12.0
    assert rec.input_tokens is None
    assert rec.logprobs is None


def test_session_records_usage_only_when_given() -> None:
    with CaptureSession(tier=ClearanceTier.T0) as cap:
        cap.record_round_trip(
            latency_ms=5.0,
            input_tokens=10,
            output_tokens=20,
            stop_reason="stop",
        )
    rec = cap.finish()
    assert rec.input_tokens == 10
    assert rec.output_tokens == 20
    assert rec.stop_reason == "stop"


def test_decorator_wraps_callable_timing() -> None:
    @capture(tier=ClearanceTier.T0)
    def model_fn(prompt: str) -> str:
        return "ok"

    text, rec = model_fn("hi")
    assert text == "ok"
    assert rec.round_trip_count == 1
    assert rec.total_duration_ms is not None
    assert rec.total_duration_ms >= 0.0
    assert rec.input_tokens is None


def test_retry_count_increments_on_record_retry() -> None:
    with CaptureSession(tier=ClearanceTier.T0) as cap:
        cap.record_retry()
        cap.record_retry()
        cap.record_round_trip(latency_ms=1.0)
    rec = cap.finish()
    assert rec.retry_count == 2
    assert rec.round_trip_count == 1


def test_api_call_count_defaults_to_round_trip_count() -> None:
    with CaptureSession(tier=ClearanceTier.T0) as cap:
        cap.record_round_trip(latency_ms=3.0)
        cap.record_round_trip(latency_ms=4.0)
    rec = cap.finish()
    assert rec.round_trip_count == 2
    assert rec.api_call_count == 2
    assert rec.total_duration_ms == 7.0


def test_record_tool_call_sets_count() -> None:
    with CaptureSession(tier=ClearanceTier.T0) as cap:
        cap.record_tool_call(name="search", arguments={"q": "weather"})
        cap.record_tool_call(name="calculator", arguments={"expr": "2+2"})
    rec = cap.finish()
    assert rec.tool_call_count == 2
    assert rec.tool_calls is not None
    assert rec.tool_calls[0].name == "search"
    assert rec.tool_calls[1].name == "calculator"
