"""Tests for mltk.inference.throughput -- requests per second validation."""

import time

import pytest

from mltk.core.assertion import MltkAssertionError
from mltk.inference.throughput import assert_throughput


def _fast_func(x: int = 1) -> int:
    return x * 2


def _slow_func(x: int = 1) -> int:
    time.sleep(0.1)
    return x * 2


def _error_func(x: int = 1) -> int:
    raise ValueError("broken")


class TestAssertThroughput:
    """Throughput (RPS) tests."""

    def test_fast_function_passes(self) -> None:
        """PASS: Fast function exceeds minimum RPS."""
        result = assert_throughput(_fast_func, 1, min_rps=10, duration=0.5)
        assert result.passed is True
        assert result.details["actual_rps"] >= 10

    def test_slow_function_fails(self) -> None:
        """FAIL: Slow function below minimum RPS."""
        with pytest.raises(MltkAssertionError):
            assert_throughput(_slow_func, 1, min_rps=100, duration=0.5)

    def test_error_tracking(self) -> None:
        """Errors are tracked separately from completions."""
        result = assert_throughput(_error_func, 1, min_rps=0, duration=0.2)
        assert result.details["errors"] > 0
        assert result.details["error_rate"] > 0

    def test_concurrent_mode(self) -> None:
        """Concurrent workers increase throughput."""
        result = assert_throughput(
            _fast_func, 1, min_rps=10, duration=0.5, concurrency=2
        )
        assert result.passed is True
        assert result.details["concurrency"] == 2

    def test_details_contain_stats(self) -> None:
        """Result details contain throughput statistics."""
        result = assert_throughput(_fast_func, 1, min_rps=1, duration=0.2)
        assert "actual_rps" in result.details
        assert "completed" in result.details
        assert "errors" in result.details
        assert "duration" in result.details
