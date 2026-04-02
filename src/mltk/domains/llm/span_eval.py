"""Span-level evaluation assertions."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from mltk.core.assertion import assert_true, timed_assertion
from mltk.core.result import Severity, TestResult
from mltk.domains.llm.span import Span, SpanKind, SpanTrace

__all__ = [
    "assert_span_quality",
    "assert_span_latency",
    "assert_span_budget",
    "assert_span_sequence",
]


def _filter_spans(
    trace: SpanTrace,
    span_kinds: list[SpanKind] | None,
) -> list[Span]:
    """Return spans filtered by kind, or all spans if None."""
    if span_kinds is None:
        return list(trace.spans)
    kinds = set(span_kinds)
    return [
        s for s in trace.spans if s.kind in kinds
    ]


@timed_assertion
def assert_span_quality(
    trace: SpanTrace,
    max_error_rate: float = 0.0,
    judge_fn: Callable[[Span], float] | None = None,
    min_score: float = 0.5,
    span_kinds: list[SpanKind] | None = None,
) -> TestResult:
    """Assert quality across spans in a trace.

    Checks error rate and optional judge scores.

    Args:
        trace: The span trace to evaluate.
        max_error_rate: Maximum allowed error fraction.
        judge_fn: Optional callable returning 0.0-1.0.
        min_score: Minimum acceptable judge score.
        span_kinds: Filter to these kinds only.

    Returns:
        TestResult named ``llm.span_eval.quality``.
    """
    spans = _filter_spans(trace, span_kinds)
    span_count = len(spans)
    error_count = sum(1 for s in spans if s.is_error)
    error_rate = (
        error_count / span_count if span_count else 0.0
    )

    violations: list[str] = []

    if error_rate > max_error_rate:
        violations.append(
            f"error rate {error_rate:.2%}"
            f" > {max_error_rate:.2%} threshold"
        )

    per_span_scores: list[dict[str, Any]] = []
    failing_spans: list[str] = []

    if judge_fn is not None:
        for s in spans:
            score = judge_fn(s)
            per_span_scores.append(
                {
                    "span_id": s.span_id,
                    "name": s.name,
                    "score": score,
                }
            )
            if score < min_score:
                failing_spans.append(
                    f"({s.name}): "
                    f"{score:.3f} < {min_score}"
                )

    if failing_spans:
        violations.append(
            f"{len(failing_spans)} span(s)"
            f" below min_score {min_score}"
        )

    passed = len(violations) == 0
    message = (
        "All span quality checks passed"
        if passed
        else "; ".join(violations)
    )

    return assert_true(
        passed,
        name="llm.span_eval.quality",
        message=message,
        severity=Severity.CRITICAL,
        span_count=span_count,
        error_rate=round(error_rate, 4),
        error_count=error_count,
        per_span_scores=per_span_scores,
        min_score=min_score,
        failing_spans=failing_spans,
    )


@timed_assertion
def assert_span_latency(
    trace: SpanTrace,
    max_latency_ms: float | None = None,
    by_kind: dict[SpanKind, float] | None = None,
) -> TestResult:
    """Assert spans meet latency thresholds.

    Args:
        trace: The span trace to evaluate.
        max_latency_ms: Global max latency in ms.
        by_kind: Per-kind max latency overrides.

    Returns:
        TestResult named ``llm.span_eval.latency``.

    Raises:
        ValueError: If neither threshold is set.
    """
    if max_latency_ms is None and by_kind is None:
        raise ValueError(
            "At least one of max_latency_ms"
            " or by_kind must be provided"
        )

    violations: list[dict[str, Any]] = []
    max_observed = max(
        (s.duration_ms for s in trace.spans),
        default=0.0,
    )
    per_kind_max: dict[str, float] = {}

    for s in trace.spans:
        threshold = None
        if by_kind and s.kind in by_kind:
            threshold = by_kind[s.kind]
        elif max_latency_ms is not None:
            threshold = max_latency_ms

        if threshold is not None:
            kind_key = s.kind.value
            cur = per_kind_max.get(kind_key, 0.0)
            if s.duration_ms > cur:
                per_kind_max[kind_key] = s.duration_ms
            if s.duration_ms > threshold:
                violations.append(
                    {
                        "span_id": s.span_id,
                        "name": s.name,
                        "kind": kind_key,
                        "duration_ms": round(
                            s.duration_ms, 1
                        ),
                        "threshold_ms": threshold,
                    }
                )

    passed = len(violations) == 0
    span_count = len(trace.spans)

    if not passed:
        worst = max(
            violations,
            key=lambda v: v["duration_ms"],
        )
        message = (
            f"{len(violations)} span(s)"
            f" exceeded latency; worst: "
            f"{worst['name']}"
            f" at {worst['duration_ms']:.1f}ms"
            f" (limit {worst['threshold_ms']}ms)"
        )
    else:
        message = (
            f"All {span_count} spans within"
            f" latency thresholds"
            f" (max observed:"
            f" {max_observed:.1f}ms)"
        )

    return assert_true(
        passed,
        name="llm.span_eval.latency",
        message=message,
        severity=Severity.CRITICAL,
        max_observed_ms=round(max_observed, 1),
        per_kind_max=per_kind_max,
        span_count=span_count,
        violations=violations,
    )


@timed_assertion
def assert_span_budget(
    trace: SpanTrace,
    max_total_tokens: int | None = None,
    max_total_cost_usd: float | None = None,
    max_spans: int | None = None,
    max_errors: int = 0,
) -> TestResult:
    """Assert trace stays within resource budget.

    Args:
        trace: The span trace to evaluate.
        max_total_tokens: Max sum of tokens.
        max_total_cost_usd: Max total cost in USD.
        max_spans: Max number of spans.
        max_errors: Max number of error spans.

    Returns:
        TestResult named ``llm.span_eval.budget``.

    Raises:
        ValueError: If no budget constraint is set.
    """
    if (
        max_total_tokens is None
        and max_total_cost_usd is None
        and max_spans is None
    ):
        raise ValueError(
            "At least one of max_total_tokens,"
            " max_total_cost_usd, or max_spans"
            " must be set"
        )

    violations: list[str] = []
    total_tokens = trace.total_tokens
    total_cost = trace.total_cost_usd
    span_count = trace.span_count
    error_count = trace.error_count

    if (
        max_total_tokens is not None
        and total_tokens > max_total_tokens
    ):
        violations.append(
            f"tokens {total_tokens:,}"
            f" > {max_total_tokens:,} limit"
        )
    if (
        max_total_cost_usd is not None
        and total_cost > max_total_cost_usd
    ):
        violations.append(
            f"cost ${total_cost:.4f}"
            f" > ${max_total_cost_usd:.4f}"
        )
    if (
        max_spans is not None
        and span_count > max_spans
    ):
        violations.append(
            f"spans {span_count}"
            f" > {max_spans} limit"
        )
    if error_count > max_errors:
        violations.append(
            f"errors {error_count}"
            f" > {max_errors} limit"
        )

    passed = len(violations) == 0
    message = (
        "Trace within all budget constraints"
        if passed
        else "; ".join(violations)
    )

    return assert_true(
        passed,
        name="llm.span_eval.budget",
        message=message,
        severity=Severity.CRITICAL,
        total_tokens=total_tokens,
        max_total_tokens=max_total_tokens,
        total_cost_usd=round(total_cost, 6),
        max_total_cost_usd=max_total_cost_usd,
        span_count=span_count,
        max_spans=max_spans,
        error_count=error_count,
        max_errors=max_errors,
        violations=violations,
    )


@timed_assertion
def assert_span_sequence(
    trace: SpanTrace,
    required_kinds: list[SpanKind] | None = None,
    required_names: list[str] | None = None,
    forbidden_kinds: list[SpanKind] | None = None,
    min_spans: int = 1,
) -> TestResult:
    """Assert trace contains expected span structure.

    Args:
        trace: The span trace to evaluate.
        required_kinds: SpanKind values that must appear.
        required_names: Span names that must appear.
        forbidden_kinds: SpanKind values that must not appear.
        min_spans: Minimum number of spans required.

    Returns:
        TestResult named ``llm.span_eval.sequence``.
    """
    present_kinds = sorted(
        {s.kind.value for s in trace.spans}
    )
    present_names = {s.name for s in trace.spans}
    span_count = len(trace.spans)

    violations: list[str] = []

    if span_count < min_spans:
        violations.append(
            f"span count {span_count}"
            f" < min_spans {min_spans}"
        )

    missing_kinds: list[str] = []
    if required_kinds:
        for k in required_kinds:
            if k.value not in present_kinds:
                missing_kinds.append(k.value)
        if missing_kinds:
            violations.append(
                "missing required kinds: "
                + ", ".join(missing_kinds)
            )

    missing_names: list[str] = []
    if required_names:
        for n in required_names:
            if n not in present_names:
                missing_names.append(n)
        if missing_names:
            violations.append(
                "missing required names: "
                + ", ".join(missing_names)
            )

    forbidden_found: list[str] = []
    if forbidden_kinds:
        for k in forbidden_kinds:
            if k.value in present_kinds:
                forbidden_found.append(k.value)
        if forbidden_found:
            violations.append(
                "forbidden kinds found: "
                + ", ".join(forbidden_found)
            )

    passed = len(violations) == 0
    message = (
        f"Trace structure valid: {span_count}"
        f" spans, kinds="
        + ", ".join(present_kinds)
        if passed
        else "; ".join(violations)
    )

    return assert_true(
        passed,
        name="llm.span_eval.sequence",
        message=message,
        severity=Severity.CRITICAL,
        span_count=span_count,
        present_kinds=present_kinds,
        missing_kinds=missing_kinds,
        missing_names=missing_names,
        forbidden_found=forbidden_found,
    )
