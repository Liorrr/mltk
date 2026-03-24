"""Inference throughput testing -- validate requests per second capacity.

Duration-based measurement gives realistic RPS numbers. Concurrent mode
uses ThreadPoolExecutor to simulate multi-client load.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from mltk.core.assertion import assert_true, timed_assertion
from mltk.core.result import Severity, TestResult


@timed_assertion
def assert_throughput(
    func: Callable[..., Any],
    *args: Any,
    min_rps: float = 100.0,
    duration: float = 5.0,
    concurrency: int = 1,
) -> TestResult:
    """Assert model serves at least min_rps requests per second.

    Args:
        func: Function to benchmark.
        *args: Arguments to pass to func.
        min_rps: Minimum required requests per second.
        duration: Test duration in seconds.
        concurrency: Number of concurrent workers.

    Returns:
        TestResult with throughput statistics.
    """
    completed = 0
    errors = 0
    end_time = time.perf_counter() + duration

    if concurrency <= 1:
        # Sequential mode
        while time.perf_counter() < end_time:
            try:
                func(*args)
                completed += 1
            except Exception:
                errors += 1
    else:
        # Concurrent mode
        def _worker() -> tuple[int, int]:
            local_completed = 0
            local_errors = 0
            while time.perf_counter() < end_time:
                try:
                    func(*args)
                    local_completed += 1
                except Exception:
                    local_errors += 1
            return local_completed, local_errors

        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [executor.submit(_worker) for _ in range(concurrency)]
            for f in futures:
                c, e = f.result()
                completed += c
                errors += e

    actual_rps = completed / duration if duration > 0 else 0.0
    error_rate = errors / max(completed + errors, 1)
    passed = actual_rps >= min_rps

    message = (
        f"Throughput: {actual_rps:.1f} RPS >= {min_rps} (completed={completed})"
        if passed
        else f"Throughput too low: {actual_rps:.1f} RPS < {min_rps} (completed={completed})"
    )

    return assert_true(
        passed,
        name="inference.throughput",
        message=message,
        severity=Severity.CRITICAL,
        actual_rps=actual_rps,
        min_rps=min_rps,
        completed=completed,
        errors=errors,
        error_rate=error_rate,
        duration=duration,
        concurrency=concurrency,
    )
