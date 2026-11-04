"""Tests for shared error-policy helpers."""

from __future__ import annotations

from typing import get_args

import pytest

from mltk.core.assertion import MltkAssertionError
from mltk.core.error_policy import (
    EVAL_ON_ERROR,
    RED_TEAM_ON_ERROR,
    EvalOnError,
    RedTeamOnError,
    apply_error_policy,
    error_policy_verdict,
    unknown_on_error_result,
)
from mltk.core.result import Severity


def test_policy_literals_and_tuples() -> None:
    assert get_args(RedTeamOnError) == ("fail_if_all", "strict", "resist")
    assert get_args(EvalOnError) == ("fail_if_all", "strict", "zero")
    assert RED_TEAM_ON_ERROR == get_args(RedTeamOnError)
    assert EVAL_ON_ERROR == get_args(EvalOnError)


def test_unknown_policy_critical_raises_with_details() -> None:
    with pytest.raises(MltkAssertionError) as exc_info:
        unknown_on_error_result("test.policy", "bad", ("good",), answer=1)
    result = exc_info.value.result
    assert result.message == "Unknown on_error 'bad'. Valid values: good"
    assert result.passed is False
    assert result.details["answer"] == 1


def test_unknown_policy_warning_returns_result() -> None:
    result = unknown_on_error_result(
        "test.policy", "bad", ("good",), severity=Severity.WARNING
    )
    assert result.passed is False
    assert result.severity is Severity.WARNING


@pytest.mark.parametrize(
    ("on_error", "errored", "total", "expected"),
    [
        ("strict", 1, 2, "strict"),
        ("strict", 0, 2, None),
        ("fail_if_all", 2, 2, "all_errored"),
        ("fail_if_all", 0, 0, None),
        ("fail_if_all", 1, 2, None),
        ("resist", 2, 2, None),
        ("zero", 2, 2, None),
    ],
)
def test_error_policy_verdict(
    on_error: str, errored: int, total: int, expected: str | None
) -> None:
    assert error_policy_verdict(on_error, errored, total) == expected


@pytest.mark.parametrize(
    ("passed", "message", "on_error", "errored", "total", "expected"),
    [
        (
            True,
            "score ok",
            "strict",
            1,
            2,
            (False, "1/2 model calls raised (on_error='strict'); score ok"),
        ),
        (
            True,
            "score ok",
            "fail_if_all",
            2,
            2,
            (False, "all 2 model calls raised — endpoint unreachable?; score ok"),
        ),
        (True, "score ok", "resist", 1, 2, (True, "score ok (1/2 calls errored)")),
        (True, "score ok", "zero", 0, 2, (True, "score ok")),
    ],
)
def test_apply_error_policy(
    passed: bool,
    message: str,
    on_error: str,
    errored: int,
    total: int,
    expected: tuple[bool, str],
) -> None:
    assert apply_error_policy(passed, message, on_error, errored, total) == expected


def test_reexports_and_shared_aliases() -> None:
    from mltk.core.error_policy import apply_error_policy
    from mltk.domains.llm.red_team import assertions, session
    from mltk.domains.nlp import security

    assert assertions._apply_error_policy is apply_error_policy
    assert assertions.OnError is RedTeamOnError
    assert session.OnError is RedTeamOnError
    assert security.OnError is RedTeamOnError
