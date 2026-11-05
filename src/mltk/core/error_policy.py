"""Shared error policies for red-team and evaluation assertions.

Red-team ``resist`` keeps the legacy behavior: errors are not failures and
therefore lean toward a pass. Eval ``zero`` likewise keeps the legacy
behavior: errored samples count as ``0.0`` and therefore lean toward a fail.
"""

from __future__ import annotations

from typing import Literal, get_args

from mltk.core.assertion import assert_true
from mltk.core.result import Severity, TestResult

RedTeamOnError = Literal["fail_if_all", "strict", "resist"]
EvalOnError = Literal["fail_if_all", "strict", "zero"]
RED_TEAM_ON_ERROR: tuple[str, ...] = get_args(RedTeamOnError)
EVAL_ON_ERROR: tuple[str, ...] = get_args(EvalOnError)


def unknown_on_error_result(
    name: str,
    on_error: str,
    allowed: tuple[str, ...],
    severity: Severity = Severity.CRITICAL,
    **details: object,
) -> TestResult:
    """Build a failed result for an unknown error policy.

    Args:
        name: Assertion name for the result (e.g. ``"eval.task.qa"``).
        on_error: The rejected policy value.
        allowed: Valid policy values, listed in the message in this order.
        severity: Result severity. CRITICAL (the default) raises.
        **details: Extra fields stored in the result details.

    Returns:
        A failed TestResult (only when ``severity`` is not CRITICAL).

    Raises:
        MltkAssertionError: With the default CRITICAL severity.
    """
    return assert_true(
        False,
        name=name,
        message=f"Unknown on_error '{on_error}'. Valid values: {', '.join(allowed)}",
        severity=severity,
        **details,
    )


def error_policy_verdict(
    on_error: str, errored_count: int, total: int
) -> Literal["strict", "all_errored"] | None:
    """Return the policy verdict that overrides a score, if any.

    Args:
        on_error: A validated policy value.
        errored_count: Calls or samples that errored.
        total: Calls or samples attempted.

    Returns:
        ``"strict"`` when ``on_error="strict"`` and anything errored,
        ``"all_errored"`` when ``on_error="fail_if_all"`` and every one of a
        non-zero total errored, else None. ``resist`` and ``zero`` keep the
        legacy behaviour and never produce a verdict.
    """
    if on_error == "strict" and errored_count:
        return "strict"
    if on_error == "fail_if_all" and total > 0 and errored_count == total:
        return "all_errored"
    return None


def apply_error_policy(
    passed: bool,
    message: str,
    on_error: str,
    errored_count: int,
    total: int,
) -> tuple[bool, str]:
    """Apply *on_error* to a score-based verdict and its message.

    Args:
        passed: Verdict from the score alone (errors not considered).
        message: Message matching that score-based verdict.
        on_error: ``fail_if_all``, ``strict`` or ``resist``.
        errored_count: Model calls that raised.
        total: Model calls attempted.

    Returns:
        ``(passed, message)`` after the policy.  When errors cause the
        failure, the message leads with the error reason.
    """
    if on_error == "strict" and errored_count:
        reason = (
            f"{errored_count}/{total} model calls raised "
            "(on_error='strict')"
        )
    elif on_error == "fail_if_all" and total > 0 and errored_count == total:
        reason = f"all {total} model calls raised — endpoint unreachable?"
    else:
        if errored_count:
            message += f" ({errored_count}/{total} calls errored)"
        return passed, message
    return False, f"{reason}; {message}"
