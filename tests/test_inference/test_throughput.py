"""Tests for mltk.inference.throughput -- requests per second validation.

Throughput tests ensure a model serving endpoint can handle the expected
request volume. A model that processes only 5 RPS when the traffic pattern
requires 100 RPS will cause request queuing, timeouts, and SLA breaches.
These tests verify that assert_throughput correctly measures RPS, tracks
errors separately, and supports concurrent worker mode.
"""

import time

import pytest

from mltk.core.assertion import MltkAssertionError
from mltk.inference.throughput import assert_throughput


def _fast_func(x: int = 1) -> int:
    """Simulates a high-throughput model (sub-millisecond inference)."""
    return x * 2


def _slow_func(x: int = 1) -> int:
    """Simulates a low-throughput model (~100ms per call, max 10 RPS)."""
    time.sleep(0.1)
    return x * 2


def _error_func(x: int = 1) -> int:
    """Simulates a model that crashes on every request."""
    raise ValueError("broken")


class TestAssertThroughput:
    """Throughput (RPS) tests.

    Validates that assert_throughput correctly benchmarks request volume,
    gates on minimum RPS thresholds, tracks error rates, and supports
    concurrent worker scaling.
    """

    def test_fast_function_passes(self) -> None:
        """PASS: Fast function exceeds minimum 10 RPS threshold.

        WHY: A lightweight model should easily handle thousands of RPS.
        This is the happy path for production serving capacity checks.
        Expected: result.passed is True, actual_rps >= 10.
        """
        result = assert_throughput(_fast_func, 1, min_rps=10, duration=0.5)
        assert result.passed is True
        assert result.details["actual_rps"] >= 10

    def test_slow_function_fails(self) -> None:
        """FAIL: Slow function cannot reach 100 RPS.

        WHY: A model taking 100ms per call can do at most ~10 RPS single-threaded.
        Requesting 100 RPS is impossible. This catches capacity mismatches
        before deployment causes request queuing and timeouts.
        Expected: MltkAssertionError raised.
        """
        with pytest.raises(MltkAssertionError):
            assert_throughput(_slow_func, 1, min_rps=100, duration=0.5)

    def test_error_tracking(self) -> None:
        """PASS (min_rps=0): Errors tracked separately from successful completions.

        WHY: In production, a model that crashes 50% of the time still
        "processes" requests. Error rate must be tracked separately from
        throughput so teams can distinguish "slow" from "broken."
        Expected: errors > 0, error_rate > 0 in result details.
        """
        result = assert_throughput(_error_func, 1, min_rps=0, duration=0.2)
        assert result.details["errors"] > 0
        assert result.details["error_rate"] > 0

    def test_concurrent_mode(self) -> None:
        """PASS: Concurrent workers increase effective throughput.

        WHY: Many model servers run multiple workers in parallel (e.g.,
        Gunicorn workers, Ray Serve replicas). Throughput tests must
        simulate this to give realistic RPS numbers.
        Expected: result.passed is True, concurrency=2 recorded in details.
        """
        result = assert_throughput(
            _fast_func, 1, min_rps=10, duration=0.5, concurrency=2
        )
        assert result.passed is True
        assert result.details["concurrency"] == 2

    def test_details_contain_stats(self) -> None:
        """PASS: Result details contain all throughput statistics for reporting.

        WHY: CI dashboards and HTML reports need actual_rps, completed count,
        error count, and duration to track serving capacity trends over
        model versions.
        Expected: All four stat keys present in result.details.
        """
        result = assert_throughput(_fast_func, 1, min_rps=1, duration=0.2)
        assert "actual_rps" in result.details
        assert "completed" in result.details
        assert "errors" in result.details
        assert "duration" in result.details
