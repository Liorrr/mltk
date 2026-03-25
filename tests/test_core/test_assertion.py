"""Tests for mltk.core.assertion -- the foundational assert_true helper.

assert_true is the building block for every mltk assertion. It converts a
boolean condition into a TestResult with severity, message, and metadata.
These tests verify the four critical behaviors:
1. True condition returns a passing result
2. False condition with CRITICAL severity raises MltkAssertionError
3. False condition with WARNING severity does NOT raise (allows CI to continue)
4. Extra kwargs are carried through as result details for report generation
"""

import pytest

from mltk.core.assertion import MltkAssertionError, assert_true
from mltk.core.result import Severity


def test_assert_true_passes() -> None:
    """PASS: True condition produces a passing TestResult.

    WHY: The simplest case -- when the check passes, the result should
    indicate success so CI pipelines continue. Every higher-level
    assertion (assert_no_drift, assert_metric, etc.) delegates to this.
    Expected: result.passed is True.
    """
    result = assert_true(True, name="check", message="ok")
    assert result.passed is True


def test_assert_true_fails_critical() -> None:
    """FAIL: False condition with CRITICAL severity raises MltkAssertionError.

    WHY: Critical failures must halt the pipeline immediately. If a model
    metric is below threshold, we cannot silently continue to deployment.
    The exception carries the full TestResult for structured error handling.
    Expected: MltkAssertionError raised, result.passed is False, severity is CRITICAL.
    """
    with pytest.raises(MltkAssertionError) as exc_info:
        assert_true(False, name="check", message="failed")
    assert exc_info.value.result.passed is False
    assert exc_info.value.result.severity == Severity.CRITICAL


def test_assert_true_warning_does_not_raise() -> None:
    """WARN: False condition with WARNING severity returns result without raising.

    WHY: Some checks are informational -- e.g., "data is slightly stale"
    should be logged but not block deployment. WARNING-severity failures
    return a failed result but do NOT raise an exception.
    Expected: result.passed is False, severity is WARNING, no exception.
    """
    result = assert_true(False, name="check", message="warn", severity=Severity.WARNING)
    assert result.passed is False
    assert result.severity == Severity.WARNING


def test_assert_true_carries_details() -> None:
    """PASS: Extra kwargs are stored in result.details for report generation.

    WHY: Assertions like assert_metric pass score=0.95, threshold=0.9 as
    kwargs. These must appear in the HTML report and JSON output so users
    can see exact values, not just pass/fail. Without this, debugging
    failures requires re-running tests.
    Expected: result.details contains score and threshold values.
    """
    result = assert_true(True, name="check", message="ok", score=0.95, threshold=0.9)
    assert result.details["score"] == 0.95
    assert result.details["threshold"] == 0.9
