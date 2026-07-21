"""Shared empty-input policy helpers.

Every assertion family that accepts an ``on_empty`` policy funnels through
these helpers so the policy behaves identically across domains. Extracted
from six drifted per-module copies (``data/synthetic``, ``domains/codegen``,
``domains/llm/conversation``, ``domains/llm/rag``, ``domains/llm/retrieval``,
``domains/recommendation``) in the S102 adversarial-review batch.

The copies had drifted: two parameterised ``severity`` while four hardcoded
``Severity.CRITICAL``. The unified signatures below default ``severity`` to
``Severity.CRITICAL``, which reproduces both variants exactly — hardcoded
callers omit the argument, parameterised callers pass theirs.
"""
from __future__ import annotations

from mltk.core.assertion import assert_true
from mltk.core.result import Severity, TestResult

ON_EMPTY_OPTIONS = ("fail", "skip", "pass")


def unknown_on_empty_result(
    name: str,
    on_empty: str,
    severity: Severity = Severity.CRITICAL,
) -> TestResult:
    """Return a failed result for an unsupported ``on_empty`` policy."""
    return assert_true(
        False,
        name=name,
        message=(
            f"Unknown on_empty: '{on_empty}'. "
            f"Supported: {', '.join(ON_EMPTY_OPTIONS)}"
        ),
        severity=severity,
        on_empty=on_empty,
    )


def empty_input_result(
    *,
    name: str,
    reason: str,
    on_empty: str,
    legacy_message: str,
    severity: Severity = Severity.CRITICAL,
    **legacy_details: object,
) -> TestResult:
    """Apply the configured empty-input policy."""
    if on_empty == "fail":
        return assert_true(
            False,
            name=name,
            message=f"{reason} -- empty input is not allowed",
            severity=severity,
        )
    if on_empty == "skip":
        return assert_true(
            True,
            name=name,
            message=f"Skipped: {reason}",
            severity=Severity.INFO,
            skipped=True,
            reason=reason,
        )
    return assert_true(
        True,
        name=name,
        message=legacy_message,
        severity=severity,
        **legacy_details,
    )
