"""Flaky test detection — identify non-deterministic ML tests."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class FlakySummary:
    """Summary of a flakiness detection run."""

    test_name: str
    pass_count: int
    fail_count: int
    pass_rate: float
    is_flaky: bool


def detect_flaky(
    func: Callable[[], None],
    runs: int = 5,
    threshold: float = 0.8,
    test_name: str | None = None,
) -> FlakySummary:
    """Run a test function N times and detect if it's flaky.

    A test is flaky if 0 < pass_rate < threshold.  If it always passes
    or always fails it is either stable or broken — not flaky.

    Args:
        func: Zero-argument callable to exercise.  Raises on failure.
        runs: Number of executions.  Must be >= 1.
        threshold: Pass-rate below which a partial-pass test is considered
            flaky.  Range (0, 1].  Default 0.8.
        test_name: Human-readable label; defaults to ``func.__name__``.

    Returns:
        :class:`FlakySummary` with per-run counts, pass rate, and flakiness
        verdict.

    Example:
        >>> import random
        >>> def sometimes_fails():
        ...     if random.random() < 0.4:
        ...         raise AssertionError("oops")
        >>> summary = detect_flaky(sometimes_fails, runs=20)
        >>> summary.is_flaky
        True
    """
    name = test_name or getattr(func, "__name__", repr(func))

    pass_count = 0
    fail_count = 0

    for _ in range(runs):
        try:
            func()
            pass_count += 1
        except Exception:  # noqa: BLE001
            fail_count += 1

    pass_rate = pass_count / runs if runs > 0 else 0.0
    # Always-pass (rate == 1.0) and always-fail (rate == 0.0) are NOT flaky.
    is_flaky = 0.0 < pass_rate < threshold

    return FlakySummary(
        test_name=name,
        pass_count=pass_count,
        fail_count=fail_count,
        pass_rate=pass_rate,
        is_flaky=is_flaky,
    )
