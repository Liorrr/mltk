"""Base assertion framework for mltk."""

from __future__ import annotations

import functools
import time
from typing import Any

from mltk.core.result import Severity, TestResult


class MltkAssertionError(AssertionError):
    """Raised when an mltk assertion fails. Carries the TestResult."""

    def __init__(self, result: TestResult) -> None:
        self.result = result
        super().__init__(result.message)


def assert_true(
    condition: bool,
    name: str,
    message: str,
    severity: Severity = Severity.CRITICAL,
    **details: Any,
) -> TestResult:
    """Base assertion. Raises MltkAssertionError if condition is False.

    Args:
        condition: Boolean condition to assert.
        name: Test name identifier (e.g., "data.schema").
        message: Human-readable result message.
        severity: Severity level; CRITICAL raises on failure.
        **details: Additional key-value details stored in TestResult.

    Returns:
        TestResult capturing the assertion outcome, timing, and details.

    Example:
        >>> result = assert_true(len(df) > 0, name="data.non_empty", message="Has rows")
    """
    result = TestResult(
        name=name,
        passed=condition,
        severity=severity,
        message=message,
        details=details,
    )
    if not condition and severity == Severity.CRITICAL:
        raise MltkAssertionError(result)
    return result


def timed_assertion(func):  # type: ignore[no-untyped-def]
    """Decorator that adds wall-clock timing to assertion functions.

    Wraps any function returning a TestResult, measuring elapsed time
    in milliseconds and storing it in ``result.duration_ms``.

    Args:
        func: Assertion function that returns a TestResult.

    Returns:
        Wrapped function with the same signature that populates duration_ms.

    Example:
        >>> @timed_assertion
        ... def assert_fast(data):
        ...     return assert_true(len(data) > 0, "fast", "ok")
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> TestResult:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed_ms = (time.perf_counter() - start) * 1000
        result.duration_ms = elapsed_ms
        return result

    return wrapper
