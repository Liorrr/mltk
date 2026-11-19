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

The gate only checks thresholds that you explicitly set via parameters.
A budgeted key that is absent was not measured: by default that fails
(``on_unmeasured="fail"``); pass ``on_unmeasured="skip"`` or ``"pass"``
for traces that do not record it.

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
from mltk.core.error_policy import (
    EVAL_ON_ERROR,
    EvalOnError,
    check_zero_threshold,
    unknown_on_error_result,
)
from mltk.core.judge_parse import call_judge, callback_score
from mltk.core.result import Severity, TestResult
from mltk.domains.llm.trace import (
    ON_UNMEASURED_OPTIONS,
    OnUnmeasured,
    is_valid_measurement,
)

_NAME = "integrations.trace_quality"


@timed_assertion
def assert_trace_quality(
    trace: dict[str, Any],
    *,
    max_latency_ms: float | None = None,
    max_cost_usd: float | None = None,
    min_score: float | None = None,
    judge_fn: Callable[[dict[str, Any]], float] | None = None,
    on_error: EvalOnError = "fail_if_all",
    on_unmeasured: OnUnmeasured = "fail",
) -> TestResult:
    """Assert that an LLM trace meets production quality thresholds.

    Checks up to three dimensions of trace quality.  Each dimension
    is only checked when its corresponding threshold parameter is set.
    If all specified thresholds pass, the assertion passes.  If any
    threshold fails, the assertion fails with a message listing every
    violation.

    **Latency check** (``max_latency_ms``):

    Reads ``trace["latency_ms"]`` and compares it to the threshold.

    **Cost check** (``max_cost_usd``):

    Reads ``trace["cost_usd"]`` and compares it to the threshold.

    **Quality score check** (``min_score``):

    If ``judge_fn`` is provided, it is called with the full trace
    dict and must return a score in ``[0, 1]``.  The returned value
    is compared against ``min_score``.  A judge that raises, or
    returns ``None``, a boolean, a non-finite number or a value
    outside ``[0, 1]``, is a judge **error** under ``on_error``.

    If ``judge_fn`` is not provided, the check reads
    ``trace["score"]`` instead.  This supports platforms that
    pre-compute quality scores (e.g., Phoenix evaluations, Langfuse
    scores).  A precomputed score must be a finite number (not a
    boolean); its scale is not checked.

    **Unmeasured values** (``on_unmeasured``):

    A budgeted key that is missing (or ``None``) was not measured,
    so its budget cannot be verified:

    - ``"fail"`` (default): the result fails, naming the unmeasured
      keys. Measured budgets are still checked and reported.
    - ``"pass"``: unmeasured budgets are treated as met.
    - ``"skip"``: unmeasured budgets are dropped; if no budget was
      measured, the result is skipped.

    A measured ``0`` is checked normally. A latency or cost that is
    not a finite, non-negative number, or a precomputed score that is
    not a finite number, is invalid and fails under every policy.
    A judge returning ``None`` is a judge error, not an unmeasured
    score.

    Args:
        trace: Dict containing trace data.  Expected keys:
            ``latency_ms`` (float), ``cost_usd`` (float),
            ``score`` (float), ``output`` (str), ``input`` (str).
        max_latency_ms: Maximum acceptable latency in milliseconds.
            None means do not check latency.
        max_cost_usd: Maximum acceptable cost in US dollars.  None
            means do not check cost.
        min_score: Minimum acceptable quality score.  None means do
            not check quality.
        judge_fn: Optional callable that takes the trace dict and
            returns a quality score in ``[0, 1]``.  When provided,
            this overrides ``trace["score"]`` for the quality check.
        on_error: Policy for a judge error: ``"fail_if_all"``
            (default) and ``"strict"`` fail the result (there is one
            judge call); ``"zero"`` scores the call ``0.0``.
        on_unmeasured: Policy for a budgeted key that is missing:
            ``"fail"`` (default), ``"pass"`` or ``"skip"``.

    Returns:
        A ``TestResult`` with ``name="integrations.trace_quality"``.
        On failure, the ``message`` lists every violated threshold.
        The ``details`` dict always contains ``checked`` (the
        dimensions that were measured and compared), ``unmeasured``,
        ``invalid``, ``on_unmeasured`` and ``on_error``, plus the
        actual value and threshold for each checked dimension. With a
        ``judge_fn`` it also has ``errored_count``, ``n_valid`` and
        ``errors``.

    Raises:
        ValueError: If ``on_error="zero"`` and ``min_score <= 0``.

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
    if on_error not in EVAL_ON_ERROR:
        return unknown_on_error_result(_NAME, on_error, EVAL_ON_ERROR)
    if on_unmeasured not in ON_UNMEASURED_OPTIONS:
        return assert_true(
            False,
            name=_NAME,
            message=(
                f"Unknown on_unmeasured: '{on_unmeasured}'. "
                f"Supported: {', '.join(ON_UNMEASURED_OPTIONS)}"
            ),
            severity=Severity.CRITICAL,
            on_unmeasured=on_unmeasured,
        )
    check_zero_threshold(on_error, min_score=min_score)

    failures: list[str] = []
    checked: list[str] = []
    unmeasured: list[str] = []
    invalid: list[str] = []
    details: dict[str, Any] = {}

    for key, budget in (
        ("latency_ms", max_latency_ms),
        ("cost_usd", max_cost_usd),
    ):
        if budget is None:
            continue
        actual = trace.get(key)
        if actual is None:
            unmeasured.append(key)
            continue
        if not is_valid_measurement(actual):
            invalid.append(f"{key}={actual!r}")
            continue
        checked.append(key)
        details[key] = actual
        details[f"max_{key}"] = budget
        if actual > budget:
            failures.append(
                f"latency {actual:.1f}ms > {budget:.1f}ms threshold"
                if key == "latency_ms"
                else f"cost ${actual:.4f} > ${budget:.4f} threshold"
            )

    if min_score is not None:
        score: float | None = None
        if judge_fn is not None:
            score, error = call_judge(judge_fn, trace)
            details.update(
                errored_count=0 if error is None else 1,
                n_valid=1 if error is None else 0,
                errors=[] if error is None else [{"error": error}],
            )
            if error is not None:
                if on_error == "zero":
                    score = 0.0
                else:
                    failures.append(
                        f"judge error: {error} (on_error='{on_error}')"
                    )
        else:
            raw = trace.get("score")
            if raw is None:
                unmeasured.append("score")
            else:
                score, error = callback_score(raw, check_range=False)
                if error is not None:
                    invalid.append(f"score={raw!r}")
        if score is not None:
            checked.append("score")
            details["score"] = score
            details["min_score"] = min_score
            if score < min_score:
                failures.append(
                    f"score {score:.3f} < {min_score:.3f} threshold"
                    + (
                        " (judge error scored 0.0, on_error='zero')"
                        if details.get("errored_count") else ""
                    )
                )

    details.update(
        checked=checked,
        unmeasured=unmeasured,
        invalid=invalid,
        on_unmeasured=on_unmeasured,
        on_error=on_error,
    )
    missing = ", ".join(unmeasured)
    if invalid:
        # NaN, infinite or negative values are not measurements; comparing
        # them with a threshold would pass silently (NaN > x is False).
        failures.append(
            f"invalid measurement {', '.join(invalid)} "
            "(latency and cost must be finite and non-negative; "
            "a score must be a finite number)"
        )

    if failures:
        passed = False
        message = "; ".join(failures)
        if unmeasured:
            message += f"; not measured: {missing}"
    elif unmeasured and on_unmeasured == "fail":
        passed = False
        message = (
            f"Trace quality unverifiable: {missing} not measured "
            "(on_unmeasured='fail')"
        )
    elif unmeasured and on_unmeasured == "skip" and not checked:
        return assert_true(
            True,
            name=_NAME,
            message=f"Skipped: {missing} not measured (on_unmeasured='skip')",
            severity=Severity.INFO,
            skipped=True,
            reason=f"{missing} not measured",
            **details,
        )
    else:
        passed = True
        message = "All trace quality checks passed"
        if unmeasured:
            message += (
                f" ({missing} not measured, on_unmeasured='{on_unmeasured}')"
            )

    return assert_true(
        passed,
        name=_NAME,
        message=message,
        severity=Severity.CRITICAL,
        **details,
    )
