"""Tests for mltk.domains.llm.span + span_eval."""

from __future__ import annotations

import pytest

from mltk.core.assertion import MltkAssertionError
from mltk.domains.llm.span import (
    Span,
    SpanKind,
    SpanTrace,
)
from mltk.domains.llm.span_eval import (
    assert_span_budget,
    assert_span_latency,
    assert_span_quality,
    assert_span_sequence,
)

# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

_ALL_KINDS = list(SpanKind)


def _make_span(
    name: str = "op",
    kind: SpanKind = SpanKind.LLM,
    duration_ms: float = 10.0,
    input_tokens: int = 100,
    output_tokens: int = 50,
    cost_usd: float = 0.001,
    status: str = "ok",
    error: str | None = None,
    parent_id: str | None = None,
    span_id: str = "",
    metadata: dict | None = None,
) -> Span:
    return Span(
        name=name,
        kind=kind,
        duration_ms=duration_ms,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost_usd,
        status=status,
        error=error,
        parent_id=parent_id,
        span_id=span_id or "",
        metadata=metadata or {},
    )


def _trace(*spans: Span) -> SpanTrace:
    return SpanTrace(spans=list(spans))


# =========================================================
# 1. SpanTrace.from_dicts with all 8 SpanKind values
# =========================================================


class TestFromDictsAllKinds:
    """SpanTrace.from_dicts handles every SpanKind."""

    @pytest.mark.parametrize("kind", _ALL_KINDS)
    def test_from_dicts_each_kind(
        self, kind: SpanKind,
    ) -> None:
        # SCENARIO: Build trace from dict with kind str.
        # WHY: All 8 SpanKind values must round-trip.
        # EXPECTED: Span.kind equals the enum member.
        trace = SpanTrace.from_dicts(
            [{"name": "s", "kind": kind.value}]
        )
        assert trace.spans[0].kind == kind

    def test_from_dicts_all_eight_at_once(
        self,
    ) -> None:
        # SCENARIO: 8 spans, one per kind.
        # WHY: Full-set coverage in a single trace.
        # EXPECTED: 8 spans, all kinds represented.
        dicts = [
            {"name": k.value, "kind": k.value}
            for k in _ALL_KINDS
        ]
        trace = SpanTrace.from_dicts(dicts)
        assert trace.span_count == 8
        found = {s.kind for s in trace.spans}
        assert found == set(_ALL_KINDS)


# =========================================================
# 2. Span with zero duration and zero tokens
# =========================================================


class TestSpanZeroValues:
    """Span edge-case: zero duration and tokens."""

    def test_zero_duration_zero_tokens(self) -> None:
        # SCENARIO: Span with all numeric fields at 0.
        # WHY: Instantaneous/no-op spans are valid.
        # EXPECTED: total_tokens=0, is_error=False.
        s = _make_span(
            duration_ms=0.0,
            input_tokens=0,
            output_tokens=0,
            cost_usd=0.0,
        )
        assert s.total_tokens == 0
        assert s.duration_ms == 0.0
        assert s.is_error is False


# =========================================================
# 3. SpanTrace with 100 spans (scale test)
# =========================================================


class TestSpanTraceScale:
    """SpanTrace: 100 spans scale test."""

    def test_100_spans(self) -> None:
        # SCENARIO: Trace containing 100 spans.
        # WHY: Must handle realistic agent traces.
        # EXPECTED: span_count=100, token sum correct.
        spans = [
            _make_span(
                name=f"op_{i}",
                input_tokens=10,
                output_tokens=5,
            )
            for i in range(100)
        ]
        trace = SpanTrace(spans=spans)
        assert trace.span_count == 100
        assert trace.total_tokens == 100 * 15


# =========================================================
# 4. assert_span_quality with judge_fn that raises
# =========================================================


class TestSpanQualityJudgeRaises:
    """assert_span_quality: judge_fn raises."""

    def test_judge_fn_exception_propagates(
        self,
    ) -> None:
        # SCENARIO: judge_fn raises RuntimeError.
        # WHY: Callers must see the error, not silence.
        # EXPECTED: RuntimeError propagates.
        def bad_judge(span: Span) -> float:
            raise RuntimeError("judge broke")

        trace = _trace(
            _make_span(name="a"),
        )
        with pytest.raises(RuntimeError):
            assert_span_quality(
                trace, judge_fn=bad_judge,
            )


# =========================================================
# 5. assert_span_latency: by_kind overrides global
# =========================================================


class TestSpanLatencyByKindOverride:
    """assert_span_latency: by_kind overrides global."""

    def test_by_kind_overrides_global(
        self,
    ) -> None:
        # SCENARIO: Global 50ms but LLM kind allows 200ms.
        # WHY: Kind-specific thresholds must take priority.
        # EXPECTED: LLM span at 150ms passes; TOOL at 60ms
        #           fails against global 50ms.
        trace = _trace(
            _make_span(
                name="llm_call",
                kind=SpanKind.LLM,
                duration_ms=150.0,
            ),
            _make_span(
                name="tool_call",
                kind=SpanKind.TOOL,
                duration_ms=60.0,
            ),
        )
        with pytest.raises(MltkAssertionError):
            assert_span_latency(
                trace,
                max_latency_ms=50.0,
                by_kind={SpanKind.LLM: 200.0},
            )


# =========================================================
# 6. assert_span_budget: exact boundary values
# =========================================================


class TestSpanBudgetBoundary:
    """assert_span_budget: exact boundary values."""

    def test_exact_token_limit_passes(self) -> None:
        # SCENARIO: Tokens exactly equal to limit.
        # WHY: Boundary: equal should pass (not >).
        # EXPECTED: passes.
        trace = _trace(
            _make_span(
                input_tokens=500,
                output_tokens=500,
            ),
        )
        result = assert_span_budget(
            trace, max_total_tokens=1000,
        )
        assert result.passed is True

    def test_one_over_token_limit_fails(
        self,
    ) -> None:
        # SCENARIO: Tokens one over the limit.
        # WHY: Must be strictly > to fail.
        # EXPECTED: MltkAssertionError raised.
        trace = _trace(
            _make_span(
                input_tokens=501,
                output_tokens=500,
            ),
        )
        with pytest.raises(MltkAssertionError):
            assert_span_budget(
                trace, max_total_tokens=1000,
            )

    def test_exact_cost_limit_passes(self) -> None:
        # SCENARIO: Cost exactly at limit.
        # WHY: Boundary must pass.
        # EXPECTED: passes.
        trace = _trace(
            _make_span(cost_usd=0.10),
        )
        result = assert_span_budget(
            trace, max_total_cost_usd=0.10,
        )
        assert result.passed is True

    def test_exact_span_count_passes(self) -> None:
        # SCENARIO: Span count exactly at limit.
        # WHY: Boundary must pass.
        # EXPECTED: passes.
        trace = _trace(
            _make_span(name="a"),
            _make_span(name="b"),
            _make_span(name="c"),
        )
        result = assert_span_budget(
            trace, max_spans=3,
        )
        assert result.passed is True


# =========================================================
# 7. assert_span_sequence: all kinds required
# =========================================================


class TestSpanSequenceAllKinds:
    """assert_span_sequence: require all 8 kinds."""

    def test_all_kinds_present(self) -> None:
        # SCENARIO: Trace has one span per kind.
        # WHY: Full coverage of required_kinds.
        # EXPECTED: passes, missing_kinds empty.
        spans = [
            _make_span(name=k.value, kind=k)
            for k in _ALL_KINDS
        ]
        trace = SpanTrace(spans=spans)
        result = assert_span_sequence(
            trace, required_kinds=_ALL_KINDS,
        )
        assert result.passed is True
        assert result.details["missing_kinds"] == []

    def test_missing_one_kind_fails(self) -> None:
        # SCENARIO: 7 of 8 kinds present.
        # WHY: One missing must fail.
        # EXPECTED: MltkAssertionError, AGENT missing.
        spans = [
            _make_span(name=k.value, kind=k)
            for k in _ALL_KINDS
            if k != SpanKind.AGENT
        ]
        trace = SpanTrace(spans=spans)
        with pytest.raises(MltkAssertionError) as exc:
            assert_span_sequence(
                trace,
                required_kinds=_ALL_KINDS,
            )
        r = exc.value.result
        assert "AGENT" in r.details["missing_kinds"]


# =========================================================
# 8. SpanTrace.descendants: deep nesting (5 levels)
# =========================================================


class TestDescendantsDeepNesting:
    """SpanTrace.descendants with 5-level depth."""

    def test_five_levels(self) -> None:
        # SCENARIO: Chain of 5 parent-child spans.
        # WHY: descendants must recurse correctly.
        # EXPECTED: root has 4 descendants.
        spans = []
        parent = None
        for i in range(5):
            sid = f"span_{i}"
            spans.append(
                _make_span(
                    name=f"level_{i}",
                    span_id=sid,
                    parent_id=parent,
                )
            )
            parent = sid
        trace = SpanTrace(spans=spans)
        desc = trace.descendants("span_0")
        assert len(desc) == 4
        names = [d.name for d in desc]
        assert "level_1" in names
        assert "level_4" in names


# =========================================================
# 9. Span metadata with nested dict values
# =========================================================


class TestSpanMetadataNested:
    """Span metadata with nested dict values."""

    def test_nested_metadata(self) -> None:
        # SCENARIO: Metadata contains nested dicts.
        # WHY: Arbitrary metadata depth must be stored.
        # EXPECTED: Nested values accessible.
        meta = {
            "model": {
                "name": "gpt-4o",
                "params": {"temp": 0.7},
            },
        }
        s = _make_span(metadata=meta)
        assert s.metadata["model"]["name"] == "gpt-4o"
        nested = s.metadata["model"]["params"]
        assert nested["temp"] == 0.7


# =========================================================
# 10. assert_span_quality filtering to empty span set
# =========================================================


class TestSpanQualityFilterEmpty:
    """assert_span_quality: filter produces empty set."""

    def test_filter_to_empty_passes(self) -> None:
        # SCENARIO: Filter by RERANKER but none exist.
        # WHY: No spans to evaluate means no failures.
        # EXPECTED: passes, span_count=0.
        trace = _trace(
            _make_span(kind=SpanKind.LLM),
            _make_span(kind=SpanKind.TOOL),
        )
        result = assert_span_quality(
            trace,
            span_kinds=[SpanKind.RERANKER],
        )
        assert result.passed is True
        assert result.details["span_count"] == 0
