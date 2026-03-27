"""Streaming drift detection -- detect distribution shifts in real-time data streams.

Batch drift tests compare two fixed windows. Streaming detectors process one
observation at a time, maintaining internal state, and fire the instant a
change-point is detected. This is critical for latency-sensitive production
systems that cannot wait for a full batch to accumulate.

Implements two complementary algorithms:
- CUSUM: Cumulative Sum control chart. Simple, fast, well-understood.
  Best for detecting sustained mean shifts of known magnitude.
- ADWIN: ADaptive WINdowing. Automatically adjusts window size using the
  Hoeffding bound. No tuning of drift magnitude required -- it adapts.
  Best for detecting arbitrary distributional shifts with unknown timing.

Both detectors conform to BaseDriftDetector and can be swapped freely.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from mltk.core.assertion import assert_true, timed_assertion
from mltk.core.result import Severity, TestResult

# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class BaseDriftDetector(ABC):
    """Abstract base for per-element streaming drift detectors."""

    @abstractmethod
    def update(self, value: float) -> bool:
        """Feed one observation. Returns True if drift detected."""
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset detector state."""
        ...

    @property
    @abstractmethod
    def drift_detected(self) -> bool:
        """Whether drift has been detected since last reset."""
        ...


# ---------------------------------------------------------------------------
# CUSUM — Cumulative Sum change-point detector
# ---------------------------------------------------------------------------


class CUSUMDetector(BaseDriftDetector):
    """Cumulative Sum (CUSUM) change-point detector.

    Tracks cumulative sums of deviations from a target mean in both the
    positive and negative direction. When either cumulative sum exceeds
    the threshold, drift is declared.

    Args:
        threshold: Decision boundary for the cumulative sum (default 5.0).
        drift_level: Minimum deviation magnitude (allowance parameter).
            Deviations smaller than this are absorbed. Default 0.5.
        target_mean: Expected mean of the stream. If None, it is estimated
            from the first ``warmup`` observations.
        warmup: Number of initial observations used to estimate target_mean
            when target_mean is None. Default 30.

    Example:
        >>> det = CUSUMDetector(threshold=4.0, drift_level=0.5)
        >>> for x in stable_stream:
        ...     det.update(x)
        >>> assert not det.drift_detected
    """

    def __init__(
        self,
        threshold: float = 5.0,
        drift_level: float = 0.5,
        target_mean: float | None = None,
        warmup: int = 30,
    ) -> None:
        self.threshold = threshold
        self.drift_level = drift_level
        self._target_mean = target_mean
        self.warmup = warmup

        self._s_pos: float = 0.0
        self._s_neg: float = 0.0
        self._drift: bool = False
        self._warmup_buffer: list[float] = []
        self._warmed_up: bool = target_mean is not None

    def update(self, value: float) -> bool:
        """Feed one observation. Returns True if drift detected at this step."""
        # Warmup phase: collect samples to estimate target mean
        if not self._warmed_up:
            self._warmup_buffer.append(value)
            if len(self._warmup_buffer) >= self.warmup:
                self._target_mean = float(np.mean(self._warmup_buffer))
                self._warmed_up = True
                self._warmup_buffer.clear()
            return False

        assert self._target_mean is not None  # guaranteed by warmup logic

        deviation = value - self._target_mean
        self._s_pos = max(0.0, self._s_pos + deviation - self.drift_level)
        self._s_neg = max(0.0, self._s_neg - deviation - self.drift_level)

        if max(self._s_pos, self._s_neg) > self.threshold:
            self._drift = True
            return True
        return False

    def reset(self) -> None:
        """Reset detector state (keeps target_mean if already estimated)."""
        self._s_pos = 0.0
        self._s_neg = 0.0
        self._drift = False

    @property
    def drift_detected(self) -> bool:
        """Whether drift has been detected since last reset."""
        return self._drift


# ---------------------------------------------------------------------------
# ADWIN — ADaptive WINdowing detector
# ---------------------------------------------------------------------------


class _Bucket:
    """A bucket in the ADWIN exponential histogram.

    Each bucket stores a compressed summary of consecutive observations:
    the count (how many observations) and their total sum.
    """

    __slots__ = ("total", "variance", "count")

    def __init__(self, total: float = 0.0, variance: float = 0.0, count: int = 0) -> None:
        self.total = total
        self.variance = variance
        self.count = count


class ADWINDetector(BaseDriftDetector):
    """ADaptive WINdowing (ADWIN) change detector.

    Maintains a variable-length window of recent observations using an
    exponential histogram (buckets of exponentially growing size). On each
    new observation, checks whether the window can be split into two
    sub-windows whose means differ by more than a Hoeffding-bound
    threshold. If so, the older sub-window is dropped.

    This implementation uses the ADWIN2 bucket compression scheme from
    Bifet & Gavalda (2007) for O(log W) memory and amortised O(log W)
    per-element time.

    Args:
        delta: Confidence parameter for the Hoeffding bound. Smaller values
            require stronger evidence before declaring drift. Default 0.002.
        min_window: Minimum number of observations before drift checking
            begins. Default 30.
        max_buckets: Maximum number of buckets per level in the exponential
            histogram. When exceeded, two buckets are merged into the next
            level. Default 5.

    Example:
        >>> det = ADWINDetector(delta=0.01)
        >>> for x in data_stream:
        ...     if det.update(x):
        ...         print("Drift detected!")
    """

    def __init__(
        self,
        delta: float = 0.002,
        min_window: int = 30,
        max_buckets: int = 5,
    ) -> None:
        self.delta = delta
        self.min_window = min_window
        self.max_buckets = max_buckets

        # Exponential histogram: list of lists of _Bucket.
        # Level 0 holds individual observations, level k holds buckets
        # each covering 2^k observations.
        self._levels: list[list[_Bucket]] = []
        self._total: float = 0.0
        self._variance: float = 0.0
        self._count: int = 0
        self._drift: bool = False

    # -- public interface ---------------------------------------------------

    def update(self, value: float) -> bool:
        """Feed one observation. Returns True if drift detected at this step."""
        self._insert(value)
        if self._count < self.min_window:
            return False
        detected = self._check_and_shrink()
        if detected:
            self._drift = True
        return detected

    def reset(self) -> None:
        """Reset detector state entirely."""
        self._levels.clear()
        self._total = 0.0
        self._variance = 0.0
        self._count = 0
        self._drift = False

    @property
    def drift_detected(self) -> bool:
        """Whether drift has been detected since last reset."""
        return self._drift

    @property
    def window_size(self) -> int:
        """Current number of observations in the adaptive window."""
        return self._count

    # -- internal -----------------------------------------------------------

    def _insert(self, value: float) -> None:
        """Insert a single observation into the exponential histogram."""
        # Create a level-0 bucket for this observation.
        bucket = _Bucket(total=value, variance=0.0, count=1)

        if len(self._levels) == 0:
            self._levels.append([])
        self._levels[0].insert(0, bucket)

        # Update global stats (Welford online variance).
        self._count += 1
        old_mean = (self._total / (self._count - 1)) if self._count > 1 else 0.0
        self._total += value
        new_mean = self._total / self._count
        self._variance += (value - old_mean) * (value - new_mean)

        # Compress: if level k has more than max_buckets entries, merge the
        # two oldest (rightmost) into level k+1.
        self._compress()

    def _compress(self) -> None:
        """Merge buckets when a level overflows max_buckets."""
        for level_idx in range(len(self._levels)):
            level = self._levels[level_idx]
            if len(level) <= self.max_buckets:
                break
            # Merge the two oldest (rightmost) buckets
            b2 = level.pop()  # oldest
            b1 = level.pop()  # second oldest
            merged = _Bucket(
                total=b1.total + b2.total,
                variance=(
                    b1.variance + b2.variance
                    + (b1.total / b1.count - b2.total / b2.count) ** 2
                    * b1.count * b2.count / (b1.count + b2.count)
                ),
                count=b1.count + b2.count,
            )
            # Push merged bucket to next level
            if level_idx + 1 >= len(self._levels):
                self._levels.append([])
            self._levels[level_idx + 1].insert(0, merged)

    def _check_and_shrink(self) -> bool:
        """Check all possible split points and shrink window if drift found.

        Iterates from the oldest bucket to the newest. For each candidate
        split, computes the Hoeffding bound and tests whether the means of
        the two sub-windows differ significantly.

        Returns True if drift was detected (and window was shrunk).
        """
        found_drift = False

        # Walk from oldest level down to newest, checking splits at each
        # bucket boundary. n1 = right (newer) sub-window, n0 = left (older).
        while True:
            made_cut = False
            # Accumulate right (newer) sub-window stats
            n1: int = 0
            sum1: float = 0.0

            for level_idx in range(len(self._levels)):
                level = self._levels[level_idx]
                for bucket_idx in range(len(level)):
                    # This bucket is part of the right sub-window
                    b = level[bucket_idx]
                    n1 += b.count
                    sum1 += b.total

                    # Left sub-window = total - right
                    n0 = self._count - n1
                    if n0 < self.min_window // 2 or n1 < self.min_window // 2:
                        continue

                    mean0 = (self._total - sum1) / n0
                    mean1 = sum1 / n1

                    # Hoeffding bound with harmonic-mean weighting
                    m = 1.0 / (1.0 / n0 + 1.0 / n1)
                    delta_prime = self.delta / math.log(self._count)
                    if delta_prime <= 0:
                        continue
                    epsilon = math.sqrt((1.0 / (2.0 * m)) * math.log(4.0 / delta_prime))

                    if abs(mean0 - mean1) >= epsilon:
                        # Drift detected: remove oldest sub-window
                        self._remove_oldest(n0)
                        found_drift = True
                        made_cut = True
                        break
                if made_cut:
                    break
            if not made_cut:
                break

        return found_drift

    def _remove_oldest(self, n_to_remove: int) -> None:
        """Remove the n_to_remove oldest observations from the window.

        Drops entire buckets from the oldest end (rightmost entries at
        the highest levels) until enough observations are removed. Then
        recomputes global statistics from remaining buckets.
        """
        removed = 0
        # Remove from the highest (oldest) levels first
        for level_idx in range(len(self._levels) - 1, -1, -1):
            level = self._levels[level_idx]
            while level and removed < n_to_remove:
                b = level[-1]  # oldest bucket at this level
                if removed + b.count <= n_to_remove:
                    level.pop()
                    removed += b.count
                else:
                    break
            if removed >= n_to_remove:
                break

        # Clean up empty levels from the top
        while self._levels and len(self._levels[-1]) == 0:
            self._levels.pop()

        # Recompute global stats from remaining buckets
        self._total = 0.0
        self._count = 0
        self._variance = 0.0
        for level in self._levels:
            for b in level:
                self._total += b.total
                self._count += b.count
                self._variance += b.variance


# ---------------------------------------------------------------------------
# Public assertion wrapper
# ---------------------------------------------------------------------------


_METHODS: dict[str, type[BaseDriftDetector]] = {
    "cusum": CUSUMDetector,
    "adwin": ADWINDetector,
}


@timed_assertion
def assert_no_streaming_drift(
    observations: list[float] | np.ndarray,
    method: str = "adwin",
    **kwargs: Any,
) -> TestResult:
    """Assert that a stream of observations shows no distributional drift.

    Creates a streaming detector, feeds all observations one at a time, and
    reports whether drift was detected. Unlike batch drift tests that compare
    two fixed windows, this processes elements sequentially and fires at the
    exact change-point.

    Args:
        observations: Time-ordered numeric observations to scan.
        method: Detection algorithm -- ``"adwin"`` (default) or ``"cusum"``.
        **kwargs: Extra keyword arguments forwarded to the detector constructor
            (e.g., ``delta``, ``threshold``, ``drift_level``).

    Returns:
        TestResult with details including ``drift_detected`` (bool),
        ``drift_point`` (int index or -1), ``method`` (str), and
        ``window_size`` (int, ADWIN only).

    Example:
        >>> stream = [0.5] * 100 + [5.0] * 100
        >>> assert_no_streaming_drift(stream, method="cusum", threshold=4.0)
    """
    name = "monitor.streaming_drift"

    if method not in _METHODS:
        return assert_true(
            False,
            name=name,
            message=f"Unknown method: '{method}'. Supported: {sorted(_METHODS.keys())}",
            severity=Severity.CRITICAL,
            method=method,
        )

    detector = _METHODS[method](**kwargs)
    arr = np.asarray(observations, dtype=np.float64).ravel()

    drift_point = -1
    for i, val in enumerate(arr):
        if detector.update(float(val)):
            drift_point = i
            break

    passed = not detector.drift_detected
    window_size = detector.window_size if isinstance(detector, ADWINDetector) else len(arr)

    message = (
        f"No streaming drift detected ({method}, {len(arr)} observations)"
        if passed
        else (
            f"Streaming drift detected at index {drift_point} "
            f"({method}, {len(arr)} observations)"
        )
    )

    return assert_true(
        passed,
        name=name,
        message=message,
        severity=Severity.CRITICAL,
        drift_detected=not passed,
        drift_point=drift_point,
        method=method,
        window_size=window_size,
        n_observations=len(arr),
    )
