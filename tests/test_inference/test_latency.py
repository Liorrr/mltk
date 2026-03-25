"""Tests for mltk.inference.latency -- response time percentile validation.

Latency tests gate model deployments on response time SLAs. A model that
takes 500ms per prediction when the SLA is 100ms will cause timeouts,
dropped requests, and degraded user experience. These tests verify that
assert_latency correctly measures percentiles (P50, P95, P99), excludes
warmup iterations (JIT compilation, cache priming), and fails on threshold
breaches.
"""

import time

import pytest

from mltk.core.assertion import MltkAssertionError
from mltk.inference.latency import assert_cold_start, assert_latency


def _fast_func(x: int = 1) -> int:
    """Simulates a fast model inference (sub-millisecond)."""
    return x * 2


def _slow_func(x: int = 1) -> int:
    """Simulates a slow model inference (~50ms per call)."""
    time.sleep(0.05)
    return x * 2


class TestAssertLatency:
    """Latency percentile tests.

    Validates that assert_latency benchmarks a function, computes accurate
    percentile statistics, and gates on user-specified thresholds.
    """

    def test_fast_function_passes(self) -> None:
        """PASS: Fast function meets P50 and P95 thresholds.

        WHY: A sub-millisecond function should easily pass 50ms thresholds.
        This is the happy path for lightweight models (e.g., logistic regression,
        small decision trees).
        Expected: result.passed is True, P50 < 50ms.
        """
        result = assert_latency(_fast_func, 1, p50=50.0, p95=100.0, iterations=20, warmup=2)
        assert result.passed is True
        assert result.details["p50"] < 50.0

    def test_slow_function_fails(self) -> None:
        """FAIL: Slow function exceeds P95 threshold of 10ms.

        WHY: A model that takes 50ms per call violates a 10ms P95 SLA.
        This must block deployment to prevent latency-related outages.
        Expected: MltkAssertionError raised.
        """
        with pytest.raises(MltkAssertionError):
            assert_latency(_slow_func, 1, p95=10.0, iterations=5, warmup=1)

    def test_warmup_excluded(self) -> None:
        """PASS: Warmup iterations are excluded from percentile calculation.

        WHY: The first few calls to a model often include JIT compilation,
        cache priming, or lazy initialization. Including them would inflate
        latency numbers and create false failures. Warmup runs are discarded.
        Expected: result.details["iterations"] reflects only measured iterations.
        """
        result = assert_latency(_fast_func, 1, p50=50.0, iterations=10, warmup=3)
        assert result.details["iterations"] == 10

    def test_no_threshold_error(self) -> None:
        """FAIL: Calling assert_latency without any threshold is a user error.

        WHY: If no percentile threshold is specified, the test has no pass/fail
        criteria and would always pass vacuously. This catches misconfiguration.
        Expected: MltkAssertionError with "At least one" message.
        """
        with pytest.raises(MltkAssertionError) as exc:
            assert_latency(_fast_func, 1)
        assert "At least one" in str(exc.value)

    def test_full_distribution_in_details(self) -> None:
        """PASS: Result details contain complete latency distribution statistics.

        WHY: The HTML report and CI logs need full statistics (P50, P95, P99,
        mean, std, min, max) for performance monitoring dashboards. Teams use
        these to track latency trends over releases.
        Expected: All seven stat keys present in result.details.
        """
        result = assert_latency(_fast_func, 1, p50=100.0, iterations=20, warmup=2)
        assert "p50" in result.details
        assert "p95" in result.details
        assert "p99" in result.details
        assert "mean" in result.details
        assert "std" in result.details
        assert "min" in result.details
        assert "max" in result.details


class TestAssertColdStart:
    """Cold start (first-call) latency tests.

    Cold start latency matters for serverless deployments (AWS Lambda,
    Cloud Run) where the first request after a scale-up event must still
    meet SLA. These tests verify the first-call measurement works correctly.
    """

    def test_fast_cold_start(self) -> None:
        """PASS: Fast function cold start is within the 1000ms threshold.

        WHY: Even the first call to a lightweight model should be fast.
        This verifies there is no unexpected initialization overhead.
        Expected: result.passed is True.
        """
        result = assert_cold_start(_fast_func, 1, max_ms=1000.0)
        assert result.passed is True

    def test_slow_cold_start(self) -> None:
        """FAIL: Slow function exceeds 10ms cold start threshold.

        WHY: A 50ms cold start violates a strict 10ms requirement, which
        is realistic for latency-sensitive applications (e.g., ad serving,
        real-time bidding).
        Expected: MltkAssertionError raised.
        """
        with pytest.raises(MltkAssertionError):
            assert_cold_start(_slow_func, 1, max_ms=10.0)
