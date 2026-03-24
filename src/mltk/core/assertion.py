"""Base assertion framework for mltk."""

from __future__ import annotations

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
    """Base assertion. Raises MltkAssertionError if condition is False."""
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
    """Decorator that adds timing to assertion functions."""

    def wrapper(*args: Any, **kwargs: Any) -> TestResult:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed_ms = (time.perf_counter() - start) * 1000
        result.duration_ms = elapsed_ms
        return result

    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__
    return wrapper
