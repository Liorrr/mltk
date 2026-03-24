"""Tests for mltk.inference.latency -- response time percentile validation."""

import time

import pytest

from mltk.core.assertion import MltkAssertionError
from mltk.inference.latency import assert_cold_start, assert_latency


def _fast_func(x: int = 1) -> int:
    return x * 2


def _slow_func(x: int = 1) -> int:
    time.sleep(0.05)
    return x * 2


class TestAssertLatency:
    """Latency percentile tests."""

    def test_fast_function_passes(self) -> None:
        """PASS: Fast function meets all percentile thresholds."""
        result = assert_latency(_fast_func, 1, p50=50.0, p95=100.0, iterations=20, warmup=2)
        assert result.passed is True
        assert result.details["p50"] < 50.0

    def test_slow_function_fails(self) -> None:
        """FAIL: Slow function exceeds P95 threshold."""
        with pytest.raises(MltkAssertionError):
            assert_latency(_slow_func, 1, p95=10.0, iterations=5, warmup=1)

    def test_warmup_excluded(self) -> None:
        """Warmup iterations are not counted in percentiles."""
        result = assert_latency(_fast_func, 1, p50=50.0, iterations=10, warmup=3)
        assert result.details["iterations"] == 10

    def test_no_threshold_error(self) -> None:
        """FAIL: Must specify at least one percentile threshold."""
        with pytest.raises(MltkAssertionError) as exc:
            assert_latency(_fast_func, 1)
        assert "At least one" in str(exc.value)

    def test_full_distribution_in_details(self) -> None:
        """Result details contain full latency statistics."""
        result = assert_latency(_fast_func, 1, p50=100.0, iterations=20, warmup=2)
        assert "p50" in result.details
        assert "p95" in result.details
        assert "p99" in result.details
        assert "mean" in result.details
        assert "std" in result.details
        assert "min" in result.details
        assert "max" in result.details


class TestAssertColdStart:
    """Cold start (first-call) latency tests."""

    def test_fast_cold_start(self) -> None:
        """PASS: Fast function cold start within threshold."""
        result = assert_cold_start(_fast_func, 1, max_ms=1000.0)
        assert result.passed is True

    def test_slow_cold_start(self) -> None:
        """FAIL: Slow function exceeds cold start threshold."""
        with pytest.raises(MltkAssertionError):
            assert_cold_start(_slow_func, 1, max_ms=10.0)
