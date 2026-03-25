"""LLM-specific latency — Time to First Token and Inter-Token Latency."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from mltk.core.assertion import assert_true, timed_assertion
from mltk.core.result import Severity, TestResult


@timed_assertion
def assert_ttft(
    func: Callable[..., Any],
    *args: Any,
    max_ms: float = 1000.0,
    iterations: int = 5,
) -> TestResult:
    """Assert Time to First Token is within bounds.

    Measures how long until the LLM starts producing output.
    Industry benchmark: chatbot TTFT P50 < 500ms.

    Args:
        func: Function that returns first token/chunk (or full response).
        *args: Arguments to pass to func.
        max_ms: Maximum allowed TTFT in milliseconds.
        iterations: Number of measurements (takes median).

    Returns:
        TestResult with TTFT statistics.

    Example:
        >>> def get_first_token(prompt): return llm.generate(prompt, max_tokens=1)
        >>> assert_ttft(get_first_token, "Hello", max_ms=1000.0)
    """
    latencies = []
    for _ in range(iterations):
        start = time.perf_counter()
        func(*args)
        elapsed_ms = (time.perf_counter() - start) * 1000
        latencies.append(elapsed_ms)

    import numpy as np

    arr = np.array(latencies)
    median_ms = float(np.median(arr))

    passed = median_ms <= max_ms
    message = (
        f"TTFT: {median_ms:.2f}ms <= {max_ms}ms (median of {iterations})"
        if passed
        else f"TTFT too slow: {median_ms:.2f}ms > {max_ms}ms"
    )

    return assert_true(
        passed, name="llm.ttft", message=message,
        severity=Severity.CRITICAL,
        median_ms=median_ms, max_ms=max_ms,
        min_ms=float(arr.min()), max_observed_ms=float(arr.max()),
        iterations=iterations,
    )


@timed_assertion
def assert_itl(
    func: Callable[..., Any],
    *args: Any,
    max_ms: float = 50.0,
    num_tokens: int = 10,
) -> TestResult:
    """Assert Inter-Token Latency is within bounds.

    Measures average time between consecutive tokens.
    Industry benchmark: ITL P50 < 50ms.

    Args:
        func: Function that generates tokens (called num_tokens times).
        *args: Arguments to pass to func.
        max_ms: Maximum allowed average ITL in milliseconds.
        num_tokens: Number of token generations to measure.

    Returns:
        TestResult with ITL statistics.

    Example:
        >>> def gen_token(state): return next_token(state)
        >>> assert_itl(gen_token, initial_state, max_ms=50.0, num_tokens=20)
    """
    intervals = []
    for _ in range(num_tokens):
        start = time.perf_counter()
        func(*args)
        elapsed_ms = (time.perf_counter() - start) * 1000
        intervals.append(elapsed_ms)

    import numpy as np

    arr = np.array(intervals)
    avg_ms = float(arr.mean())

    passed = avg_ms <= max_ms
    message = (
        f"ITL: {avg_ms:.2f}ms <= {max_ms}ms (avg of {num_tokens} tokens)"
        if passed
        else f"ITL too slow: {avg_ms:.2f}ms > {max_ms}ms"
    )

    return assert_true(
        passed, name="llm.itl", message=message,
        severity=Severity.CRITICAL,
        avg_ms=avg_ms, max_ms=max_ms,
        min_ms=float(arr.min()), max_observed_ms=float(arr.max()),
        num_tokens=num_tokens,
    )
