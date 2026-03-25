"""Memory leak detection — verify training doesn't leak memory over time.

Memory leaks in training are often caused by accumulating tensors in Python
lists (e.g., storing loss.item() but accidentally storing the tensor), keeping
references to the computation graph alive, or unreleased CUDA buffers. These
assertions work on plain lists of memory readings so they are framework and
device agnostic.
"""

from __future__ import annotations

import numpy as np

from mltk.core.assertion import assert_true, timed_assertion
from mltk.core.result import Severity, TestResult


@timed_assertion
def assert_no_memory_leak(
    memory_readings_mb: list[float],
    max_growth_mb: float = 100.0,
    window: int = 10,
) -> TestResult:
    """Assert memory doesn't grow unbounded during training.

    Compares the mean of the last ``window`` readings to the mean of the first
    ``window`` readings. Growth exceeding ``max_growth_mb`` indicates a memory
    leak — common causes include tensor accumulation in a Python list,
    unreleased CUDA buffers, or optimizer state growing unexpectedly.

    When fewer than ``2 * window`` readings are available, the window is
    clamped to ``len(readings) // 2`` (minimum 1) so the check degrades
    gracefully rather than raising an error.

    Args:
        memory_readings_mb: Sequence of RSS/GPU memory measurements in MB,
            taken once per training step or epoch.
        max_growth_mb: Maximum allowed memory growth from start to end window.
        window: Number of readings to average at the start and end.

    Returns:
        TestResult with ``growth_mb``, ``start_mean_mb``, ``end_mean_mb``,
        ``window``, and ``max_growth_mb`` details.

    Example:
        >>> readings = [500.0 + i * 0.1 for i in range(100)]  # tiny drift
        >>> assert_no_memory_leak(readings, max_growth_mb=100.0, window=10)
    """
    n = len(memory_readings_mb)
    arr = np.asarray(memory_readings_mb, dtype=float)

    # Gracefully handle short sequences
    effective_window = min(window, max(1, n // 2))

    start_mean = float(np.mean(arr[:effective_window]))
    end_mean = float(np.mean(arr[-effective_window:]))
    growth_mb = end_mean - start_mean

    passed = growth_mb <= max_growth_mb
    message = (
        f"No memory leak: growth={growth_mb:.1f} MB <= max={max_growth_mb} MB "
        f"(start={start_mean:.1f} MB, end={end_mean:.1f} MB, window={effective_window})"
        if passed
        else (
            f"Memory leak detected: growth={growth_mb:.1f} MB > max={max_growth_mb} MB "
            f"(start={start_mean:.1f} MB -> end={end_mean:.1f} MB, "
            f"window={effective_window})"
        )
    )

    return assert_true(
        passed,
        name="training.no_memory_leak",
        message=message,
        severity=Severity.CRITICAL,
        growth_mb=round(growth_mb, 3),
        start_mean_mb=round(start_mean, 3),
        end_mean_mb=round(end_mean, 3),
        window=effective_window,
        max_growth_mb=max_growth_mb,
        num_readings=n,
    )


@timed_assertion
def assert_loss_is_detached(
    memory_per_step_mb: list[float],
    max_growth_per_step_mb: float = 1.0,
) -> TestResult:
    """Assert loss tensor isn't accumulating the computation graph.

    If ``loss`` is stored without calling ``.item()`` or ``.detach()``, the
    entire computation graph is kept alive for every step. This manifests as
    memory increasing monotonically and linearly — roughly proportional to
    model size per step. This assertion fits a linear trend to the memory
    readings and fails if the slope exceeds ``max_growth_per_step_mb``.

    When fewer than 2 readings are provided the check passes (not enough data
    to estimate a trend).

    Args:
        memory_per_step_mb: Memory usage in MB recorded after each training
            step (len >= 2 for a meaningful check).
        max_growth_per_step_mb: Maximum allowed memory growth per step in MB.

    Returns:
        TestResult with ``slope_mb_per_step``, ``max_growth_per_step_mb``,
        and ``num_steps`` details.

    Example:
        >>> steps = [500.0 + i * 0.05 for i in range(50)]  # negligible drift
        >>> assert_loss_is_detached(steps, max_growth_per_step_mb=1.0)
    """
    n = len(memory_per_step_mb)

    if n < 2:
        return assert_true(
            True,
            name="training.loss_is_detached",
            message=f"Too few readings ({n}) to estimate trend — skipping check",
            severity=Severity.CRITICAL,
            slope_mb_per_step=0.0,
            max_growth_per_step_mb=max_growth_per_step_mb,
            num_steps=n,
        )

    arr = np.asarray(memory_per_step_mb, dtype=float)
    steps = np.arange(n, dtype=float)

    # Linear regression slope via least squares
    slope = float(np.polyfit(steps, arr, 1)[0])

    passed = slope <= max_growth_per_step_mb
    message = (
        f"Loss detach OK: memory slope={slope:.4f} MB/step <= "
        f"max={max_growth_per_step_mb} MB/step"
        if passed
        else (
            f"Computation graph leak: memory slope={slope:.4f} MB/step > "
            f"max={max_growth_per_step_mb} MB/step. "
            f"Call loss.item() or loss.detach() before storing."
        )
    )

    return assert_true(
        passed,
        name="training.loss_is_detached",
        message=message,
        severity=Severity.CRITICAL,
        slope_mb_per_step=round(slope, 6),
        max_growth_per_step_mb=max_growth_per_step_mb,
        num_steps=n,
    )
