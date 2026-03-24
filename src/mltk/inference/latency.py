"""Inference latency testing -- validate response time percentiles.

Industry SLAs: P95 <50ms for classification, <200ms for NLP, <1s for LLM TTFT.
Warmup phase is mandatory — first calls are 10-100x slower due to JIT/cache/GPU compilation.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import numpy as np

from mltk.core.assertion import assert_true, timed_assertion
from mltk.core.result import Severity, TestResult


@timed_assertion
def assert_latency(
    func: Callable[..., Any],
    *args: Any,
    p50: float | None = None,
    p95: float | None = None,
    p99: float | None = None,
    iterations: int = 100,
    warmup: int = 5,
) -> TestResult:
    """Assert inference latency percentiles are within bounds.

    Args:
        func: Function to benchmark.
        *args: Arguments to pass to func.
        p50: Max P50 latency in milliseconds.
        p95: Max P95 latency in milliseconds.
        p99: Max P99 latency in milliseconds.
        iterations: Number of measurement iterations.
        warmup: Warmup iterations (excluded from measurement).

    Returns:
        TestResult with full latency distribution.
    """
    if p50 is None and p95 is None and p99 is None:
        return assert_true(
            False,
            name="inference.latency",
            message="At least one percentile threshold required (p50, p95, or p99)",
            severity=Severity.CRITICAL,
        )

    # Warmup phase
    for _ in range(warmup):
        func(*args)

    # Measurement phase
    latencies: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        func(*args)
        elapsed_ms = (time.perf_counter() - start) * 1000
        latencies.append(elapsed_ms)

    arr = np.array(latencies)
    actual_p50 = float(np.percentile(arr, 50))
    actual_p95 = float(np.percentile(arr, 95))
    actual_p99 = float(np.percentile(arr, 99))

    # Check thresholds
    failures = []
    if p50 is not None and actual_p50 > p50:
        failures.append(f"P50={actual_p50:.2f}ms > {p50}ms")
    if p95 is not None and actual_p95 > p95:
        failures.append(f"P95={actual_p95:.2f}ms > {p95}ms")
    if p99 is not None and actual_p99 > p99:
        failures.append(f"P99={actual_p99:.2f}ms > {p99}ms")

    passed = len(failures) == 0
    message = (
        f"Latency OK: P50={actual_p50:.2f}ms, P95={actual_p95:.2f}ms, P99={actual_p99:.2f}ms"
        if passed
        else f"Latency exceeded: {'; '.join(failures)}"
    )

    return assert_true(
        passed,
        name="inference.latency",
        message=message,
        severity=Severity.CRITICAL,
        p50=actual_p50,
        p95=actual_p95,
        p99=actual_p99,
        min=float(arr.min()),
        max=float(arr.max()),
        mean=float(arr.mean()),
        std=float(arr.std()),
        iterations=iterations,
        warmup=warmup,
        thresholds={"p50": p50, "p95": p95, "p99": p99},
    )


@timed_assertion
def assert_cold_start(
    func: Callable[..., Any],
    *args: Any,
    max_ms: float = 2000.0,
) -> TestResult:
    """Assert first-call latency (cold start) is within bounds.

    Args:
        func: Function to benchmark (should include model loading).
        *args: Arguments to pass to func.
        max_ms: Maximum allowed cold start time in milliseconds.

    Returns:
        TestResult with cold start timing.
    """
    start = time.perf_counter()
    func(*args)
    cold_ms = (time.perf_counter() - start) * 1000

    passed = cold_ms <= max_ms
    message = (
        f"Cold start: {cold_ms:.2f}ms <= {max_ms}ms"
        if passed
        else f"Cold start too slow: {cold_ms:.2f}ms > {max_ms}ms"
    )

    return assert_true(
        passed,
        name="inference.cold_start",
        message=message,
        severity=Severity.CRITICAL,
        cold_start_ms=cold_ms,
        max_ms=max_ms,
    )
