"""Tests for the shared judge-callback score validator.

A judge callback returns a number. ``callback_score`` coerces it with
``float()`` (so numeric strings, NumPy scalars and ``Decimal`` keep working)
and rejects values that are not a score: ``None``, booleans, non-numbers,
non-finite values and, by default, anything outside ``[0, 1]``.
"""

from __future__ import annotations

from decimal import Decimal

import numpy as np
import pytest

from mltk.core.error_policy import check_zero_threshold
from mltk.core.judge_parse import call_judge, callback_score

# The D2 value matrix: every value is either a valid score or a judge error.
ACCEPTED = [
    pytest.param(0.8, id="float"),
    pytest.param(np.float32(0.8), id="np.float32"),
    pytest.param("0.8", id="numeric-str"),
    pytest.param(Decimal("0.8"), id="Decimal"),
]
REJECTED_ANY_SCALE = [
    pytest.param(True, id="True"),
    pytest.param(np.bool_(True), id="np.bool_"),
    pytest.param(None, id="None"),
    pytest.param(float("nan"), id="NaN"),
    pytest.param(float("inf"), id="inf"),
]
OUT_OF_UNIT_RANGE = [
    pytest.param(-0.1, id="-0.1"),
    pytest.param(1.5, id="1.5"),
    pytest.param(7.0, id="7.0"),
]


@pytest.mark.parametrize("value", ACCEPTED)
def test_valid_scores_are_accepted(value: object) -> None:
    score, error = callback_score(value)
    assert error is None
    assert score == pytest.approx(0.8, abs=1e-6)
    assert type(score) is float


@pytest.mark.parametrize("value", REJECTED_ANY_SCALE + OUT_OF_UNIT_RANGE)
def test_non_scores_are_errors(value: object) -> None:
    score, error = callback_score(value)
    assert score is None
    assert error


@pytest.mark.parametrize("value", OUT_OF_UNIT_RANGE)
def test_range_check_can_be_disabled(value: object) -> None:
    # Mirror: without the range check the same values are valid scores.
    score, error = callback_score(value, check_range=False)
    assert error is None
    assert score == float(value)


@pytest.mark.parametrize("value", REJECTED_ANY_SCALE)
def test_non_scores_stay_errors_without_range_check(value: object) -> None:
    score, error = callback_score(value, check_range=False)
    assert score is None
    assert error


@pytest.mark.parametrize("value", [0.0, 1.0, 0, 1, np.int64(1), np.float64(0.0)])
def test_unit_range_boundaries_are_valid(value: object) -> None:
    # Integers (including NumPy integers) are numbers, not booleans.
    score, error = callback_score(value)
    assert error is None
    assert score == float(value)


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(np.bool_(False), id="np.bool_-False"),
        pytest.param(False, id="False"),
        pytest.param(np.array(True), id="0d-bool-array"),
    ],
)
def test_every_boolean_form_is_an_error(value: object) -> None:
    score, error = callback_score(value)
    assert score is None
    assert "boolean" in error


@pytest.mark.parametrize(
    "value",
    [
        pytest.param("1/10", id="fraction-str"),
        pytest.param("Score: 0.8", id="prose"),
        pytest.param("nan", id="nan-str"),
        pytest.param(Decimal("NaN"), id="Decimal-NaN"),
        pytest.param(Decimal("sNaN"), id="Decimal-sNaN"),
        pytest.param(complex(0.5, 0), id="complex"),
        pytest.param([0.8], id="list"),
        pytest.param(object(), id="object"),
    ],
)
def test_values_float_cannot_read_as_a_finite_score_are_errors(value: object) -> None:
    # A callback returns a number; text is not parsed for a score here.
    score, error = callback_score(value)
    assert score is None
    assert error


def test_zero_d_float_array_is_a_score() -> None:
    # Mirror of the 0-d boolean array: a 0-d numeric array is a score.
    score, error = callback_score(np.array(0.25))
    assert error is None
    assert score == 0.25


def test_call_judge_reports_exceptions_as_errors() -> None:
    def judge(a: str, b: str) -> float:
        raise RuntimeError("judge unavailable")

    score, error = call_judge(judge, "x", "y")
    assert score is None
    assert error == "RuntimeError: judge unavailable"


def test_call_judge_passes_arguments_and_validates() -> None:
    seen: list[tuple[str, str]] = []

    def judge(a: str, b: str) -> float:
        seen.append((a, b))
        return 0.4

    assert call_judge(judge, "x", "y") == (0.4, None)
    assert seen == [("x", "y")]
    assert call_judge(lambda a: 7.0, "x")[0] is None
    assert call_judge(lambda a: 7.0, "x", check_range=False) == (7.0, None)


def test_call_judge_does_not_swallow_keyboard_interrupt() -> None:
    def judge(a: str) -> float:
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        call_judge(judge, "x")


@pytest.mark.parametrize("threshold", [0, 0.0, -0.1])
def test_zero_policy_rejects_thresholds_an_errored_zero_would_meet(
    threshold: float,
) -> None:
    with pytest.raises(ValueError, match="min_score"):
        check_zero_threshold("zero", min_score=threshold)


@pytest.mark.parametrize("threshold", [0.5, 1e-9, None])
def test_zero_policy_accepts_positive_or_unset_thresholds(
    threshold: float | None,
) -> None:
    check_zero_threshold("zero", min_score=threshold)


@pytest.mark.parametrize("on_error", ["fail_if_all", "strict"])
def test_other_policies_accept_a_zero_threshold(on_error: str) -> None:
    # Mirror: only on_error="zero" turns an error into a 0.0 score.
    check_zero_threshold(on_error, min_score=0.0)
