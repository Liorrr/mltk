"""T1 instrumented client — nested calls, fan-out, retries, cache, action log."""

from __future__ import annotations

import pytest

from mltk.trace import CaptureSession, ClearanceTier, InstrumentedClient


def test_t0_session_rejects_t1_client() -> None:
    with CaptureSession(tier=ClearanceTier.T0) as cap:
        with pytest.raises(ValueError, match="T1"):
            InstrumentedClient(lambda x: x, session=cap)


def test_t2_session_accepts_t1_client_and_observations() -> None:
    with CaptureSession(tier=ClearanceTier.T2) as cap:
        client = InstrumentedClient(lambda x: x, session=cap)
        client.cache_hit()
    rec = cap.finish()
    assert rec.cache_hits == 1


def test_call_records_round_trip_and_action_log() -> None:
    with CaptureSession(tier=ClearanceTier.T1) as cap:
        client = InstrumentedClient(lambda prompt: f"out:{prompt}", session=cap)
        result = client.call("hi")
    rec = cap.finish()
    assert result == "out:hi"
    assert rec.tier is ClearanceTier.T1
    assert rec.round_trip_count == 1
    assert rec.action_log == ("call",)
    assert rec.logprobs is None
    assert rec.subagent_calls is None
    assert rec.cache_hits is None


def test_subagent_nesting_is_recorded() -> None:
    with CaptureSession(tier=ClearanceTier.T1) as cap:
        client = InstrumentedClient(lambda x: x, session=cap)
        child = client.subagent("researcher")
        grandchild = child.subagent("writer")
        grandchild.call("draft")
    rec = cap.finish()
    assert rec.subagent_calls is not None
    names = tuple(c.name for c in rec.subagent_calls)
    assert names == ("researcher", "writer")
    assert rec.subagent_calls[0].parent is None
    assert rec.subagent_calls[1].parent == 0
    assert rec.action_log == ("subagent:researcher", "subagent:writer", "call")


def test_sibling_subagents_have_root_parent() -> None:
    with CaptureSession(tier=ClearanceTier.T1) as cap:
        client = InstrumentedClient(lambda x: x, session=cap)
        client.subagent("researcher")
        client.subagent("writer")
    rec = cap.finish()
    assert rec.subagent_calls is not None
    assert rec.subagent_calls[1].parent is None


def test_fanout_records_count() -> None:
    with CaptureSession(tier=ClearanceTier.T1) as cap:
        client = InstrumentedClient(lambda x: x * 2, session=cap)
        out = client.fanout([1, 2, 3])
    rec = cap.finish()
    assert out == [2, 4, 6]
    assert rec.fanout_api_calls is not None
    assert rec.fanout_api_calls[0].count == 3
    assert rec.action_log == ("fanout",)


def test_error_retry_chain_on_failure() -> None:
    def boom(_prompt: str) -> str:
        raise RuntimeError("upstream 503")

    with CaptureSession(tier=ClearanceTier.T1) as cap:
        client = InstrumentedClient(boom, session=cap)
        with pytest.raises(RuntimeError, match="503"):
            client.call("x")
    rec = cap.finish()
    assert rec.error_retry_chain is not None
    assert rec.error_retry_chain[0].error == "upstream 503"
    assert rec.error_retry_chain[0].attempt == 1
    assert rec.error_retry_chain[0].duration_ms is not None
    assert rec.total_duration_ms is not None
    assert rec.retry_count == 1


def test_fanout_failure_records_issued_calls_and_error() -> None:
    def fail_middle(request: str) -> str:
        if request == "b":
            raise RuntimeError("upstream 503")
        return request

    with CaptureSession(tier=ClearanceTier.T1) as cap:
        client = InstrumentedClient(fail_middle, session=cap)
        with pytest.raises(RuntimeError, match="503"):
            client.fanout(["a", "b", "c"])
    rec = cap.finish()
    assert rec.fanout_api_calls is not None
    assert rec.fanout_api_calls[0].count == 2
    assert rec.error_retry_chain is not None
    assert rec.error_retry_chain[0].error == "upstream 503"
    assert rec.error_retry_chain[0].duration_ms is None


def test_cache_hits_are_none_until_recorded() -> None:
    with CaptureSession(tier=ClearanceTier.T1) as cap:
        client = InstrumentedClient(lambda x: x, session=cap)
        client.cache_hit()
        client.cache_hit()
    rec = cap.finish()
    assert rec.cache_hits == 2
    assert rec.action_log == ("cache_hit", "cache_hit")


def test_t1_unobserved_fields_stay_none() -> None:
    with CaptureSession(tier=ClearanceTier.T1) as cap:
        cap.record_round_trip(latency_ms=1.0)
    rec = cap.finish()
    assert rec.cache_hits is None
    assert rec.subagent_calls is None
    assert rec.fanout_api_calls is None
    assert rec.error_retry_chain is None
    assert rec.action_log is None
    assert rec.hidden_state_hooks is None
