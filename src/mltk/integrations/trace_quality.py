"""Trace quality gate -- single pass/fail signal for production LLM traces.

CI/CD pipelines need a single PASS/FAIL decision for each production
trace before promoting a model or deployment.  Without a unified gate,
teams scatter quality checks across latency monitors, cost dashboards,
and manual review -- any one of which can silently break.

``assert_trace_quality`` bundles the most common production checks into
one assertion:

- **Latency**: Is the end-to-end response time within SLA?
- **Cost**: Does the total token cost stay under budget?
- **Quality score**: Does a judge function rate the output above a
  minimum threshold?

**Why a single assertion instead of three separate ones:**

Pipeline YAML becomes one line instead of three.  The failure message
tells you *which* checks failed and *why*, so you do not need to dig
through multiple assertion outputs.  Teams that want individual checks
can still use the separate latency/cost/quality assertions -- this
gate is a convenience wrapper for the common "check everything" case.

**Trace format:**

The ``trace`` dict should contain keys that match what observability
platforms export.  Common keys:

- ``latency_ms`` (float): End-to-end latency in milliseconds.
- ``cost_usd`` (float): Total cost in US dollars.
- ``score`` (float): Quality score from 0.0 to 1.0.
- ``output`` (str): The LLM output text (used by ``judge_fn``).
- ``input`` (str): The original prompt (used by ``judge_fn``).

Any key can be absent -- the gate only checks thresholds that you
explicitly set via parameters.

Typical usage::

    from mltk.integrations.trace_quality import assert_trace_quality

    # Gate a deployment: trace must meet all thresholds
    result = assert_trace_quality(
        trace={"latency_ms": 450, "cost_usd": 0.003, "score": 0.92},
        max_latency_ms=2000,
        max_cost_usd=0.01,
        min_score=0.8,
    )
    assert result.passed

    # With a custom judge function for quality scoring
    def my_judge(trace):
        return 0.95 if "correct" in trace.get("output", "") else 0.1

    result = assert_trace_quality(
        trace={"output": "The correct answer is 42", "latency_ms": 100},
        max_latency_ms=500,
        judge_fn=my_judge,
        min_score=0.5,
    )
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from mltk.core.assertion import assert_true, timed_assertion
from mltk.core.result import Severity, TestResult


@timed_assertion
def assert_trace_quality(
    trace: dict[str, Any],
    *,
    max_latency_ms: float | None = None,
    max_cost_usd: float | None = None,
    min_score: float | None = None,
    judge_fn: Callable[[dict[str, Any]], float] | None = None,
) -> TestResult:
    """Assert that an LLM trace meets production quality thresholds.

    Checks up to three dimensions of trace quality.  Each dimension
    is only checked when its corresponding threshold parameter is set.
    If all specified thresholds pass, the assertion passes.  If any
    threshold fails, the assertion fails with a message listing every
    violation.

    **Latency check** (``max_latency_ms``):

    Reads ``trace["latency_ms"]`` and compares it to the threshold.
    If the key is missing, the check is skipped (not failed).  This
    lets you use the same gate for traces that may or may not include
    timing data.

    **Cost check** (``max_cost_usd``):

    Reads ``trace["cost_usd"]`` and compares it to the threshold.
    Same skip-if-missing behavior as latency.

    **Quality score check** (``min_score``):

    If ``judge_fn`` is provided, it is called with the full trace
    dict and must return a float (0.0--1.0).  The returned value is
    compared against ``min_score``.

    If ``judge_fn`` is not provided, the check reads
    ``trace["score"]`` instead.  This supports platforms that
    pre-compute quality scores (e.g., Phoenix evaluations, Langfuse
    scores).

    Args:
        trace: Dict containing trace data.  Expected keys:
            ``latency_ms`` (float), ``cost_usd`` (float),
            ``score`` (float), ``output`` (str), ``input`` (str).
        max_latency_ms: Maximum acceptable latency in milliseconds.
            None means do not check latency.
        max_cost_usd: Maximum acceptable cost in US dollars.  None
            means do not check cost.
        min_score: Minimum acceptable quality score (0.0--1.0).
            None means do not check quality.
        judge_fn: Optional callable that takes the trace dict and
            returns a quality score (0.0--1.0).  When provided,
            this overrides ``trace["score"]`` for the quality check.

    Returns:
        A ``TestResult`` with ``name="integrations.trace_quality"``.
        On failure, the ``message`` lists every violated threshold.
        The ``details`` dict contains the actual values for each
        checked dimension.

    Example::

        # All thresholds pass
        result = assert_trace_quality(
            {"latency_ms": 100, "cost_usd": 0.001, "score": 0.95},
            max_latency_ms=500,
            max_cost_usd=0.01,
            min_score=0.8,
        )
        assert result.passed

        # Latency violation
        result = assert_trace_quality(
            {"latency_ms": 3000},
            max_latency_ms=2000,
        )
        assert not result.passed
    """
    failures: list[str] = []
    details: dict[str, Any] = {}

    # -- Latency check --
    if max_latency_ms is not None:
        actual_latency = trace.get("latency_ms")
        if actual_latency is not None:
            details["latency_ms"] = actual_latency
            details["max_latency_ms"] = max_latency_ms
            if actual_latency > max_latency_ms:
                failures.append(
                    f"latency {actual_latency:.1f}ms > "
                    f"{max_latency_ms:.1f}ms threshold"
                )

    # -- Cost check --
    if max_cost_usd is not None:
        actual_cost = trace.get("cost_usd")
        if actual_cost is not None:
            details["cost_usd"] = actual_cost
            details["max_cost_usd"] = max_cost_usd
            if actual_cost > max_cost_usd:
                failures.append(
                    f"cost ${actual_cost:.4f} > "
                    f"${max_cost_usd:.4f} threshold"
                )

    # -- Quality score check --
    if min_score is not None:
        if judge_fn is not None:
            actual_score = judge_fn(trace)
        else:
            actual_score = trace.get("score")

        if actual_score is not None:
            details["score"] = actual_score
            details["min_score"] = min_score
            if actual_score < min_score:
                failures.append(
                    f"score {actual_score:.3f} < "
                    f"{min_score:.3f} threshold"
                )

    passed = len(failures) == 0

    if passed:
        message = "All trace quality checks passed"
    else:
        message = "; ".join(failures)

    return assert_true(
        passed,
        name="integrations.trace_quality",
        message=message,
        severity=Severity.CRITICAL,
        **details,
    )
