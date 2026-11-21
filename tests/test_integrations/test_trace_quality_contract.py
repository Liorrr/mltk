"""Contract tests for ``assert_trace_quality``: unmeasured values and score checks.

A budgeted value that is missing from the trace was not measured, and
``on_unmeasured`` decides the verdict (``"fail"`` by default). A value that
is not a finite, non-negative number fails under every policy. A judge that
returns ``None`` is a judge error, not an unmeasured score. A precomputed
``trace["score"]`` is checked for finiteness only.
"""

from __future__ import annotations

from collections.abc import Callable
from decimal import Decimal
from typing import Any

import numpy as np
import pytest

from mltk.core.assertion import MltkAssertionError
from mltk.core.result import TestResult as Result
from mltk.integrations.from_capture import to_quality_dict
from mltk.integrations.trace_quality import assert_trace_quality
from mltk.trace import CaptureRecord, ClearanceTier


def _run(trace: dict[str, Any], **kw: Any) -> Result:
    try:
        return assert_trace_quality(trace, **kw)
    except MltkAssertionError as exc:
        return exc.result


BUDGETS = [
    pytest.param("latency_ms", {"max_latency_ms": 500.0}, id="latency"),
    pytest.param("cost_usd", {"max_cost_usd": 0.01}, id="cost"),
    pytest.param("score", {"min_score": 0.5}, id="score"),
]


# ---------------------------------------------------------------------------
# D6: on_unmeasured
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("key", "budget"), BUDGETS)
def test_missing_budgeted_value_fails_by_default(key: str, budget: dict) -> None:
    result = _run({"output": "x"}, **budget)
    assert result.passed is False
    assert result.details["unmeasured"] == [key]
    assert result.details["on_unmeasured"] == "fail"
    assert "not measured" in result.message


@pytest.mark.parametrize(("key", "budget"), BUDGETS)
def test_missing_budgeted_value_passes_under_pass(key: str, budget: dict) -> None:
    result = _run({"output": "x"}, on_unmeasured="pass", **budget)
    assert result.passed is True
    assert result.details["unmeasured"] == [key]
    assert key in result.message


@pytest.mark.parametrize(("key", "budget"), BUDGETS)
def test_missing_budgeted_value_is_skipped_under_skip(key: str, budget: dict) -> None:
    result = _run({"output": "x"}, on_unmeasured="skip", **budget)
    assert result.passed is True
    assert result.details["skipped"] is True
    assert result.details["unmeasured"] == [key]


def test_skip_still_checks_the_measured_budgets() -> None:
    # Mirror: skip drops only the unmeasured budget; a measured overrun fails.
    result = _run(
        {"latency_ms": 900.0}, max_latency_ms=500.0, max_cost_usd=0.01,
        on_unmeasured="skip",
    )
    assert result.passed is False
    assert result.details["unmeasured"] == ["cost_usd"]
    assert "latency" in result.message


def test_skip_with_a_measured_budget_within_limits_passes() -> None:
    result = _run(
        {"latency_ms": 100.0}, max_latency_ms=500.0, max_cost_usd=0.01,
        on_unmeasured="skip",
    )
    assert result.passed is True
    assert result.details.get("skipped") is not True
    assert result.details["checked"] == ["latency_ms"]


@pytest.mark.parametrize(("key", "budget"), BUDGETS)
def test_measured_zero_is_a_measurement(key: str, budget: dict) -> None:
    # Mirror of "missing": 0 is measured, and meets latency/cost budgets.
    result = _run({key: 0.0}, **budget)
    if key == "score":
        assert result.passed is False  # 0.0 < min_score, measured
        assert result.details["unmeasured"] == []
    else:
        assert result.passed is True
        assert result.details["checked"] == [key]


@pytest.mark.parametrize(("key", "budget"), BUDGETS)
@pytest.mark.parametrize("on_unmeasured", ["fail", "pass", "skip"])
@pytest.mark.parametrize(
    "value",
    [float("nan"), float("inf"), float("-inf")],
    ids=["nan", "inf", "-inf"],
)
def test_non_finite_value_fails_under_every_policy(
    key: str, budget: dict, on_unmeasured: str, value: float,
) -> None:
    result = _run({key: value}, on_unmeasured=on_unmeasured, **budget)
    assert result.passed is False
    assert [entry.split("=")[0] for entry in result.details["invalid"]] == [key]


@pytest.mark.parametrize("key", ["latency_ms", "cost_usd"])
@pytest.mark.parametrize(
    "value",
    [Decimal("0.004"), np.array(0.004), np.float32(0.004), np.int64(0), 0],
    ids=["Decimal", "0d-array", "np.float32", "np.int64", "int"],
)
def test_numeric_latency_or_cost_types_are_measurements(key: str, value: object) -> None:
    # Values are read with float(), as the score is: these are measured and
    # within budget (before this check, the gate compared them directly).
    result = _run({key: value}, max_latency_ms=500.0, max_cost_usd=0.01,
                  on_unmeasured="skip")
    assert result.passed is True
    assert result.details["invalid"] == []
    assert result.details["checked"] == [key]
    assert result.details[key] == pytest.approx(float(value))


@pytest.mark.parametrize("key", ["latency_ms", "cost_usd"])
def test_numeric_type_over_budget_still_fails(key: str) -> None:
    # Mirror: a Decimal over the budget is compared, not waved through.
    result = _run({key: Decimal("999")}, max_latency_ms=500.0, max_cost_usd=0.01,
                  on_unmeasured="skip")
    assert result.passed is False
    assert result.details["invalid"] == []
    assert result.details["checked"] == [key]


@pytest.mark.parametrize("key", ["latency_ms", "cost_usd"])
@pytest.mark.parametrize(
    "value",
    ["450", b"450", True, np.bool_(True), Decimal("NaN"), Decimal("-1"), [1.0]],
    ids=["str", "bytes", "True", "np.bool_", "Decimal-NaN", "Decimal-neg", "list"],
)
def test_non_measurements_stay_invalid(key: str, value: object) -> None:
    result = _run({key: value}, max_latency_ms=500.0, max_cost_usd=0.01,
                  on_unmeasured="pass")
    assert result.passed is False
    assert [entry.split("=")[0] for entry in result.details["invalid"]] == [key]


@pytest.mark.parametrize("key", ["latency_ms", "cost_usd"])
@pytest.mark.parametrize("on_unmeasured", ["fail", "pass", "skip"])
def test_negative_latency_or_cost_fails_under_every_policy(
    key: str, on_unmeasured: str,
) -> None:
    result = _run(
        {key: -5.0}, max_latency_ms=500.0, max_cost_usd=0.01,
        on_unmeasured=on_unmeasured,
    )
    assert result.passed is False
    assert any(entry.startswith(f"{key}=") for entry in result.details["invalid"])


@pytest.mark.parametrize("on_unmeasured", ["fail", "pass", "skip"])
def test_judge_returning_none_is_a_judge_error_not_unmeasured(on_unmeasured: str) -> None:
    result = _run(
        {"output": "x"}, min_score=0.5, judge_fn=lambda trace: None,
        on_unmeasured=on_unmeasured,
    )
    assert result.passed is False
    assert result.details["errored_count"] == 1
    assert result.details["unmeasured"] == []


def test_unknown_on_unmeasured_fails() -> None:
    result = _run({"latency_ms": 1.0}, max_latency_ms=5.0, on_unmeasured="ignore")
    assert result.passed is False
    assert "Unknown on_unmeasured" in result.message


def test_details_are_never_empty() -> None:
    result = _run({})
    assert result.passed is True
    assert result.details["checked"] == []
    assert result.details["unmeasured"] == []
    assert result.details["invalid"] == []
    assert result.details["on_unmeasured"] == "fail"


def test_unmeasured_checks_are_listed_with_their_names() -> None:
    result = _run({}, max_latency_ms=1.0, max_cost_usd=1.0, min_score=0.5,
                  on_unmeasured="pass")
    assert result.details["unmeasured"] == ["latency_ms", "cost_usd", "score"]
    assert result.details["checked"] == []


# ---------------------------------------------------------------------------
# D2: a precomputed trace["score"] is checked for finiteness only
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [0.8, np.float32(0.8), "0.8", Decimal("0.8"), -0.1, 1.5, 7.0],
    ids=["0.8", "np.float32", "str", "Decimal", "-0.1", "1.5", "7.0"],
)
def test_precomputed_score_needs_only_to_be_finite(value: object) -> None:
    result = _run({"score": value}, min_score=0.5)
    assert result.details["invalid"] == []
    assert result.details["score"] == pytest.approx(float(value))
    assert result.passed is (float(value) >= 0.5)


@pytest.mark.parametrize(
    "value",
    [True, np.bool_(True), float("nan"), float("inf")],
    ids=["True", "np.bool_", "NaN", "inf"],
)
def test_precomputed_non_score_is_invalid(value: object) -> None:
    result = _run({"score": value}, min_score=0.5, on_unmeasured="pass")
    assert result.passed is False
    assert len(result.details["invalid"]) == 1
    assert result.details["invalid"][0].startswith("score=")


def test_precomputed_none_score_is_unmeasured() -> None:
    result = _run({"score": None}, min_score=0.5)
    assert result.passed is False
    assert result.details["unmeasured"] == ["score"]


# ---------------------------------------------------------------------------
# Captured traces: to_quality_dict emits latency only
# ---------------------------------------------------------------------------


def _record(**fields: Any) -> CaptureRecord:
    return CaptureRecord(tier=ClearanceTier.T0, **fields)


def test_captured_latency_is_checked() -> None:
    payload = to_quality_dict(_record(total_duration_ms=120.0))
    assert _run(payload, max_latency_ms=500.0).passed is True


def test_captured_trace_without_cost_needs_skip_for_a_cost_budget() -> None:
    payload = to_quality_dict(_record(total_duration_ms=120.0))
    failed = _run(payload, max_latency_ms=500.0, max_cost_usd=0.01)
    assert failed.passed is False
    assert failed.details["unmeasured"] == ["cost_usd"]
    opted_out = _run(payload, max_latency_ms=500.0, max_cost_usd=0.01, on_unmeasured="skip")
    assert opted_out.passed is True


def test_judge_score_is_used_when_trace_has_no_score() -> None:
    payload = to_quality_dict(_record(total_duration_ms=120.0))
    result = _run(payload, min_score=0.5, judge_fn=_judge(0.9))
    assert result.passed is True
    assert result.details["unmeasured"] == []


def _judge(value: object) -> Callable[[dict[str, Any]], object]:
    return lambda trace: value
