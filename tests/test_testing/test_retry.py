"""Tests for mltk.testing.retry — confidence-interval retry."""
from __future__ import annotations

import pytest

from mltk.testing.retry import RetryResult, retry_until_confident

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _always_pass() -> None:
    pass


def _always_fail() -> None:
    raise AssertionError("always fails")


class _FailAfterN:
    """Passes for the first N calls, then always fails."""

    def __init__(self, pass_for: int) -> None:
        self._pass_for = pass_for
        self._calls = 0

    def __call__(self) -> None:
        self._calls += 1
        if self._calls > self._pass_for:
            raise AssertionError("failing now")


class _PassRateController:
    """Produces exactly `pass_count` passes out of every `period` calls."""

    def __init__(self, pass_count: int, period: int) -> None:
        self._pass_count = pass_count
        self._period = period
        self._calls = 0

    def __call__(self) -> None:
        pos = self._calls % self._period
        self._calls += 1
        if pos >= self._pass_count:
            raise AssertionError("controlled failure")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_retry_always_pass():
    # SCENARIO: function never raises
    # WHY: should reach confident pass verdict
    # EXPECTED: is_passing=True, pass_rate=1.0
    result = retry_until_confident(_always_pass, min_runs=3, max_runs=10)

    assert isinstance(result, RetryResult)
    assert result.pass_rate == pytest.approx(1.0)
    assert result.is_passing is True
    assert result.fail_count == 0


def test_retry_always_fail():
    # SCENARIO: function always raises
    # WHY: should reach confident fail verdict
    # EXPECTED: is_passing=False, pass_rate=0.0
    result = retry_until_confident(_always_fail, min_runs=3, max_runs=10)

    assert result.pass_rate == pytest.approx(0.0)
    assert result.is_passing is False
    assert result.pass_count == 0


def test_retry_flaky():
    # SCENARIO: 50% pass rate (alternating pass/fail)
    # WHY: result should be not confidently passing with default failure_threshold=0.5
    # EXPECTED: is_passing=False (CI lower bound won't exceed 0.5 at 50% rate)
    controller = _PassRateController(pass_count=1, period=2)
    result = retry_until_confident(
        controller,
        min_runs=3,
        max_runs=10,
        confidence=0.95,
        failure_threshold=0.5,
    )

    assert isinstance(result, RetryResult)
    assert 0.0 < result.pass_rate < 1.0
    # With 50% pass rate the lower CI bound won't exceed 0.5
    assert result.is_passing is False


def test_confidence_interval():
    # SCENARIO: always-passing function with 95% confidence
    # WHY: CI bounds must be valid probabilities and logically consistent
    # EXPECTED: 0 <= lower <= upper <= 1, lower > 0.5 for a stable pass
    result = retry_until_confident(
        _always_pass,
        min_runs=5,
        max_runs=10,
        confidence=0.95,
        failure_threshold=0.5,
    )

    assert 0.0 <= result.confidence_lower <= result.confidence_upper <= 1.0
    assert result.confidence_lower > 0.5


def test_result_fields_populated():
    # SCENARIO: inspect every field of RetryResult
    # WHY: callers depend on all fields being present and typed correctly
    # EXPECTED: all fields have correct types and consistent sums
    result = retry_until_confident(_always_pass, min_runs=3, max_runs=5)

    assert isinstance(result.pass_count, int)
    assert isinstance(result.fail_count, int)
    assert isinstance(result.pass_rate, float)
    assert isinstance(result.confidence_lower, float)
    assert isinstance(result.confidence_upper, float)
    assert isinstance(result.is_passing, bool)
    assert result.pass_count + result.fail_count >= 3


def test_min_runs_respected():
    # SCENARIO: min_runs=5 — at least 5 executions must happen
    # WHY: premature exit before min_runs would give unreliable CI
    # EXPECTED: pass_count + fail_count >= 5
    result = retry_until_confident(
        _always_pass,
        min_runs=5,
        max_runs=10,
    )

    assert result.pass_count + result.fail_count >= 5
