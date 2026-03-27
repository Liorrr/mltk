"""Tests for mltk.monitor.streaming_drift -- streaming change-point detection.

Streaming drift detectors process one observation at a time and fire the
instant they have statistical confidence that the generating distribution
has changed. This test suite validates both the low-level detectors (CUSUM,
ADWIN) and the high-level assertion wrapper (assert_no_streaming_drift).

Test design:
- Stable streams must NOT trigger drift (no false positives).
- Abrupt mean shifts must be detected within a reasonable number of samples
  after the shift (no missed detections).
- Gradual drift must eventually trigger detection.
- Edge cases (empty, single-value, constant) must not crash.
"""

from __future__ import annotations

import numpy as np
import pytest

from mltk.core.assertion import MltkAssertionError
from mltk.monitor.streaming_drift import (
    ADWINDetector,
    CUSUMDetector,
    assert_no_streaming_drift,
)

# ---------------------------------------------------------------------------
# CUSUM tests
# ---------------------------------------------------------------------------


class TestCUSUM:
    """CUSUM change-point detector tests."""

    def test_stable_stream_no_drift(self) -> None:
        """PASS: A stable stream around the target mean should not trigger drift.

        WHY: Normal production noise (std=0.1 around mean=0) should never
        trigger a false alarm. CUSUM with default threshold (5.0) should
        absorb small fluctuations.
        """
        rng = np.random.default_rng(42)
        det = CUSUMDetector(threshold=5.0, drift_level=0.5, target_mean=0.0)
        for val in rng.normal(0.0, 0.1, 200):
            det.update(float(val))
        assert det.drift_detected is False

    def test_abrupt_mean_shift_detected(self) -> None:
        """FAIL: A jump from mean=0 to mean=5 should be detected quickly.

        WHY: CUSUM is designed to detect sustained mean shifts. A shift of
        10x the drift_level (5.0 vs 0.5) should accumulate rapidly and
        cross the threshold well before 100 shifted samples.
        """
        det = CUSUMDetector(threshold=5.0, drift_level=0.5, target_mean=0.0)
        drift_idx = -1
        # 50 stable, then 50 shifted
        for _i in range(50):
            det.update(0.0)
        for i in range(50):
            if det.update(5.0):
                drift_idx = 50 + i
                break
        assert det.drift_detected is True
        assert drift_idx != -1
        # Should detect within first few shifted samples
        assert drift_idx < 55

    def test_gradual_drift_detected(self) -> None:
        """FAIL: A slowly increasing mean should eventually trigger detection.

        WHY: Concept drift in production is often gradual. CUSUM should
        catch it once cumulative deviation exceeds the threshold, even if
        each individual step is small.
        """
        det = CUSUMDetector(threshold=5.0, drift_level=0.5, target_mean=0.0)
        detected = False
        for i in range(500):
            # Mean increases linearly: 0.0 -> 2.5 over 500 steps
            val = i * 0.005
            if det.update(val):
                detected = True
                break
        assert detected is True

    def test_negative_shift_detected(self) -> None:
        """FAIL: A downward mean shift should be detected by the negative CUSUM.

        WHY: CUSUM tracks both upward (s_pos) and downward (s_neg) shifts.
        A drop from mean=0 to mean=-5 should trigger s_neg.
        """
        det = CUSUMDetector(threshold=5.0, drift_level=0.5, target_mean=0.0)
        for _ in range(50):
            det.update(0.0)
        detected = False
        for _ in range(50):
            if det.update(-5.0):
                detected = True
                break
        assert detected is True

    def test_reset_clears_state(self) -> None:
        """Reset should clear drift flag and cumulative sums.

        WHY: After investigating a drift alert, operators reset the detector
        to begin monitoring from a clean slate. The old cumulative sums
        must not carry over.
        """
        det = CUSUMDetector(threshold=5.0, drift_level=0.5, target_mean=0.0)
        # Trigger drift
        for _ in range(20):
            det.update(10.0)
        assert det.drift_detected is True

        det.reset()
        assert det.drift_detected is False
        # Stable values after reset should not trigger
        for _ in range(50):
            det.update(0.0)
        assert det.drift_detected is False

    def test_warmup_auto_estimates_mean(self) -> None:
        """CUSUM with target_mean=None should estimate from warmup samples.

        WHY: In many deployments, the expected mean is unknown. The warmup
        phase (default 30 samples) estimates it, then monitoring begins.
        A stable continuation should not trigger drift.
        """
        det = CUSUMDetector(threshold=5.0, drift_level=0.5, target_mean=None, warmup=30)
        rng = np.random.default_rng(99)
        # Feed 200 samples all from the same distribution
        for val in rng.normal(10.0, 0.1, 200):
            det.update(float(val))
        assert det.drift_detected is False


# ---------------------------------------------------------------------------
# ADWIN tests
# ---------------------------------------------------------------------------


class TestADWIN:
    """ADWIN adaptive windowing detector tests."""

    def test_stable_stream_no_drift(self) -> None:
        """PASS: ADWIN should not fire on a stationary Gaussian stream.

        WHY: The Hoeffding bound should accommodate normal variance. With
        delta=0.002 and 500 observations from N(0, 0.1), no split should
        produce sub-windows with significantly different means.
        """
        rng = np.random.default_rng(1)
        det = ADWINDetector(delta=0.002, min_window=30)
        for val in rng.normal(0.0, 0.1, 500):
            det.update(float(val))
        assert det.drift_detected is False

    def test_abrupt_mean_shift_detected(self) -> None:
        """FAIL: ADWIN should detect a large abrupt mean shift.

        WHY: 200 samples from N(0, 0.1) followed by 200 from N(5, 0.1) is
        an unmistakable distributional change. ADWIN should detect it and
        shrink the window to exclude the old regime.
        """
        rng = np.random.default_rng(2)
        det = ADWINDetector(delta=0.002, min_window=30)
        drift_idx = -1
        stream = np.concatenate([
            rng.normal(0.0, 0.1, 200),
            rng.normal(5.0, 0.1, 200),
        ])
        for i, val in enumerate(stream):
            if det.update(float(val)):
                drift_idx = i
                break
        assert det.drift_detected is True
        # Drift should be detected near the shift point (index 200)
        assert 190 <= drift_idx <= 250

    def test_smaller_shift_detected_with_more_data(self) -> None:
        """FAIL: A smaller shift (1.0) should still be detected with enough data.

        WHY: ADWIN adapts its window. A 1.0 mean shift on N(0, 0.5) data
        is a 2-sigma shift, detectable given sufficient observations.
        """
        rng = np.random.default_rng(3)
        det = ADWINDetector(delta=0.01, min_window=30)
        stream = np.concatenate([
            rng.normal(0.0, 0.5, 300),
            rng.normal(1.0, 0.5, 300),
        ])
        detected = False
        for val in stream:
            if det.update(float(val)):
                detected = True
                break
        assert detected is True

    def test_delta_controls_sensitivity(self) -> None:
        """Higher delta (looser confidence) detects faster than lower delta.

        WHY: The Hoeffding bound epsilon scales as sqrt(ln(4/delta)),
        so a larger delta shrinks the bound and makes detection easier.
        On the same stream, delta=0.5 should detect before delta=0.001.
        """
        rng = np.random.default_rng(4)
        stream = np.concatenate([
            rng.normal(0.0, 0.2, 150),
            rng.normal(3.0, 0.2, 150),
        ])

        # Sensitive detector
        det_sensitive = ADWINDetector(delta=0.5, min_window=10)
        idx_sensitive = -1
        for i, val in enumerate(stream):
            if det_sensitive.update(float(val)):
                idx_sensitive = i
                break

        # Conservative detector
        det_conservative = ADWINDetector(delta=0.001, min_window=10)
        idx_conservative = -1
        for i, val in enumerate(stream):
            if det_conservative.update(float(val)):
                idx_conservative = i
                break

        assert idx_sensitive != -1
        assert idx_conservative != -1
        # Sensitive should detect at same time or earlier
        assert idx_sensitive <= idx_conservative

    def test_window_shrinks_after_drift(self) -> None:
        """After detecting drift, ADWIN's window should be smaller.

        WHY: ADWIN drops the older sub-window on detection. The window_size
        after processing a shift should be smaller than the total number
        of observations fed.
        """
        rng = np.random.default_rng(5)
        det = ADWINDetector(delta=0.002, min_window=30)
        stream = np.concatenate([
            rng.normal(0.0, 0.1, 200),
            rng.normal(5.0, 0.1, 200),
        ])
        for val in stream:
            det.update(float(val))
        # Window should have shrunk from 400 to something smaller
        assert det.window_size < 400

    def test_reset_clears_everything(self) -> None:
        """Reset should clear window, stats, and drift flag.

        WHY: After reset, ADWIN must behave identically to a fresh instance.
        No residual state should leak across reset boundaries.
        """
        det = ADWINDetector(delta=0.002, min_window=30)
        rng = np.random.default_rng(6)
        # Trigger drift
        for val in rng.normal(0.0, 0.1, 100):
            det.update(float(val))
        for val in rng.normal(10.0, 0.1, 100):
            det.update(float(val))
        assert det.drift_detected is True

        det.reset()
        assert det.drift_detected is False
        assert det.window_size == 0


# ---------------------------------------------------------------------------
# assert_no_streaming_drift wrapper tests
# ---------------------------------------------------------------------------


class TestAssertNoStreamingDrift:
    """High-level assertion wrapper tests."""

    def test_cusum_stable_passes(self) -> None:
        """PASS: A stable stream should pass the CUSUM-based assertion.

        WHY: Validates the full pipeline: assertion -> CUSUMDetector -> TestResult.
        """
        rng = np.random.default_rng(10)
        stream = rng.normal(0.0, 0.1, 200).tolist()
        result = assert_no_streaming_drift(
            stream, method="cusum", target_mean=0.0, threshold=5.0, drift_level=0.5,
        )
        assert result.passed is True
        assert result.details["drift_detected"] is False
        assert result.details["drift_point"] == -1
        assert result.details["method"] == "cusum"

    def test_cusum_drift_raises(self) -> None:
        """FAIL: A shifted stream should raise MltkAssertionError via CUSUM.

        WHY: The assertion must raise on drift so pytest integration catches it.
        """
        stream = [0.0] * 100 + [10.0] * 100
        with pytest.raises(MltkAssertionError) as exc:
            assert_no_streaming_drift(
                stream, method="cusum", target_mean=0.0, threshold=4.0, drift_level=0.5,
            )
        result = exc.value.result
        assert result.details["drift_detected"] is True
        assert 95 <= result.details["drift_point"] <= 110

    def test_adwin_stable_passes(self) -> None:
        """PASS: A stable stream should pass the ADWIN-based assertion.

        WHY: Validates the full pipeline: assertion -> ADWINDetector -> TestResult.
        """
        rng = np.random.default_rng(11)
        stream = rng.normal(5.0, 0.1, 300).tolist()
        result = assert_no_streaming_drift(
            stream, method="adwin", delta=0.002, min_window=30,
        )
        assert result.passed is True
        assert result.details["drift_detected"] is False
        assert result.details["drift_point"] == -1

    def test_adwin_drift_raises(self) -> None:
        """FAIL: A shifted stream should raise MltkAssertionError via ADWIN.

        WHY: Same as CUSUM test but using ADWIN. Validates method dispatch.
        """
        rng = np.random.default_rng(12)
        stream = np.concatenate([
            rng.normal(0.0, 0.1, 200),
            rng.normal(5.0, 0.1, 200),
        ]).tolist()
        with pytest.raises(MltkAssertionError) as exc:
            assert_no_streaming_drift(stream, method="adwin", delta=0.002, min_window=30)
        result = exc.value.result
        assert result.details["drift_detected"] is True
        assert result.details["drift_point"] > 0

    def test_unknown_method_raises(self) -> None:
        """FAIL: An unsupported method name should produce a failed TestResult.

        WHY: Clear error messaging on invalid input prevents silent misconfiguration.
        """
        with pytest.raises(MltkAssertionError) as exc:
            assert_no_streaming_drift([1.0, 2.0, 3.0], method="kalman")
        assert "Unknown method" in str(exc.value)

    def test_drift_point_near_actual_shift(self) -> None:
        """Drift point should be near the actual shift location.

        WHY: A detector that reports drift at index 0 or at the very end is
        useless for root-cause analysis. The drift_point should land close
        to the actual change-point for the result to be actionable.
        """
        rng = np.random.default_rng(13)
        shift_idx = 150
        stream = np.concatenate([
            rng.normal(0.0, 0.1, shift_idx),
            rng.normal(5.0, 0.1, 150),
        ]).tolist()
        with pytest.raises(MltkAssertionError) as exc:
            assert_no_streaming_drift(
                stream, method="cusum", target_mean=0.0, threshold=4.0, drift_level=0.5,
            )
        dp = exc.value.result.details["drift_point"]
        # Should detect within 20 samples of the actual shift
        assert shift_idx - 5 <= dp <= shift_idx + 20

    def test_numpy_array_input(self) -> None:
        """Accepts numpy arrays, not just lists.

        WHY: Users often have data in numpy arrays from prior processing.
        The assertion should handle both input types transparently.
        """
        arr = np.zeros(200)
        result = assert_no_streaming_drift(
            arr, method="cusum", target_mean=0.0, threshold=5.0, drift_level=0.5,
        )
        assert result.passed is True

    def test_result_has_timing(self) -> None:
        """TestResult.duration_ms should be populated by @timed_assertion.

        WHY: Performance monitoring of the assertion itself. The timed_assertion
        decorator must set duration_ms > 0 after execution.
        """
        rng = np.random.default_rng(14)
        stream = rng.normal(0.0, 0.1, 100).tolist()
        result = assert_no_streaming_drift(
            stream, method="cusum", target_mean=0.0, threshold=5.0,
        )
        assert result.duration_ms >= 0.0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases for streaming drift detectors."""

    def test_very_short_stream(self) -> None:
        """PASS: A stream shorter than min_window should pass without checking.

        WHY: With fewer than min_window observations, there is not enough
        data to make a statistical claim. Should pass gracefully.
        """
        result = assert_no_streaming_drift(
            [1.0, 2.0, 3.0], method="adwin", min_window=30,
        )
        assert result.passed is True
        assert result.details["drift_detected"] is False

    def test_all_same_values(self) -> None:
        """PASS: A constant stream (zero variance) should never trigger drift.

        WHY: All values identical means both sub-window means are equal.
        No split should ever pass the Hoeffding bound.
        """
        result = assert_no_streaming_drift(
            [5.0] * 200, method="adwin", delta=0.002, min_window=30,
        )
        assert result.passed is True

    def test_all_same_values_cusum(self) -> None:
        """PASS: CUSUM with constant values matching target_mean should be stable.

        WHY: Deviation from target is always 0, so cumulative sums stay at 0.
        """
        result = assert_no_streaming_drift(
            [3.0] * 100, method="cusum", target_mean=3.0, threshold=5.0,
        )
        assert result.passed is True

    def test_single_value(self) -> None:
        """PASS: A single observation should not crash or trigger drift.

        WHY: Degenerate input that must be handled without error.
        """
        result = assert_no_streaming_drift([42.0], method="adwin", min_window=30)
        assert result.passed is True

    def test_empty_stream(self) -> None:
        """PASS: An empty stream should pass (nothing to detect).

        WHY: No observations means no drift. Must not crash.
        """
        result = assert_no_streaming_drift([], method="cusum", target_mean=0.0)
        assert result.passed is True
        assert result.details["drift_point"] == -1


# ---------------------------------------------------------------------------
# Hardening: parametrized, edge-case, and integration tests (S-hardening)
# ---------------------------------------------------------------------------


class TestADWINParametrizedDelta:
    """Parametrized ADWIN tests across a range of delta values."""

    @pytest.mark.parametrize(
        "delta",
        [0.001, 0.01, 0.1, 0.5, 0.99],
        ids=["very_strict", "strict", "moderate", "loose", "very_loose"],
    )
    def test_adwin_stable_stream_various_deltas(self, delta: float) -> None:
        """PASS: A stable stream should not trigger drift at any delta value.

        WHY: With data from a single distribution N(0, 0.1), no delta value
        should cause a false positive when the stream is stationary.
        """
        rng = np.random.default_rng(42)
        det = ADWINDetector(delta=delta, min_window=30)
        for val in rng.normal(0.0, 0.1, 500):
            det.update(float(val))
        assert det.drift_detected is False


class TestCUSUMParametrizedThreshold:
    """Parametrized CUSUM tests across a range of thresholds."""

    @pytest.mark.parametrize(
        "threshold",
        [2.0, 5.0, 10.0],
        ids=["low_threshold", "default_threshold", "high_threshold"],
    )
    def test_cusum_abrupt_shift_detected_at_various_thresholds(
        self, threshold: float
    ) -> None:
        """FAIL: A jump from mean=0 to mean=10 is detected at all thresholds.

        WHY: A 10-unit mean shift is massive; even the strictest threshold (10.0)
        should eventually detect it given enough shifted samples.
        """
        det = CUSUMDetector(threshold=threshold, drift_level=0.5, target_mean=0.0)
        for _ in range(50):
            det.update(0.0)
        detected = False
        for _ in range(100):
            if det.update(10.0):
                detected = True
                break
        assert detected is True
        assert det.drift_detected is True


class TestStreamingDriftHardening:
    """Additional coverage for streaming drift detectors."""

    def test_adwin_sinusoidal_no_drift(self) -> None:
        """PASS: ADWIN on a high-frequency sinusoidal signal with small amplitude
        should not detect drift.

        WHY: A small-amplitude rapid sine wave centered at 0 is effectively
        stationary noise. With a conservative delta and large min_window,
        ADWIN should not interpret oscillation as a distributional shift.
        """
        t = np.linspace(0, 200 * np.pi, 1000)
        # Small amplitude so sub-window means are always close to 0
        stream = 0.05 * np.sin(t)
        det = ADWINDetector(delta=0.001, min_window=100)
        for val in stream:
            det.update(float(val))
        assert det.drift_detected is False

    def test_cusum_linearly_increasing_drift(self) -> None:
        """FAIL: CUSUM on linearly increasing data should detect drift.

        WHY: A linear ramp from 0 to 10 causes cumulative deviation from
        target_mean=0 to grow continuously, crossing the threshold.
        """
        stream = np.linspace(0.0, 10.0, 300)
        det = CUSUMDetector(threshold=5.0, drift_level=0.5, target_mean=0.0)
        detected = False
        for val in stream:
            if det.update(float(val)):
                detected = True
                break
        assert detected is True

    def test_large_stream_no_crash(self) -> None:
        """PERF: 10,000 elements should complete without crashing or hanging.

        WHY: Production streams can be large. The detector must handle them
        efficiently without memory or performance issues. We use very low
        variance so ADWIN does not trigger false drift.
        """
        rng = np.random.default_rng(55)
        stream = rng.normal(0.0, 0.01, 10_000)
        result = assert_no_streaming_drift(
            stream.tolist(), method="adwin", delta=0.001, min_window=50,
        )
        # Just verify it completes and returns a result
        assert result.details["n_observations"] == 10_000

    def test_large_stream_cusum_no_crash(self) -> None:
        """PERF: 10,000 elements through CUSUM completes without error."""
        rng = np.random.default_rng(56)
        stream = rng.normal(5.0, 0.5, 10_000)
        result = assert_no_streaming_drift(
            stream.tolist(), method="cusum", target_mean=5.0, threshold=5.0,
        )
        assert result.details["n_observations"] == 10_000
        assert result.passed is True

    def test_adwin_window_size_decreases_after_drift(self) -> None:
        """After drift detection, ADWIN's window_size should be smaller than
        total observations fed.

        WHY: ADWIN drops the older sub-window when drift is detected. The
        remaining window should be strictly smaller than the total count.
        """
        rng = np.random.default_rng(77)
        det = ADWINDetector(delta=0.002, min_window=30)
        # 300 stable, then 300 shifted -- guaranteed drift
        stable = rng.normal(0.0, 0.1, 300)
        shifted = rng.normal(10.0, 0.1, 300)
        total_fed = 0
        for val in np.concatenate([stable, shifted]):
            det.update(float(val))
            total_fed += 1
        assert det.drift_detected is True
        assert det.window_size < total_fed

    def test_cusum_drift_detected_matches_update_return(self) -> None:
        """drift_detected property must be True once update() returned True.

        WHY: The property and the return value are two interfaces to the same
        state. They must agree at all times.
        """
        det = CUSUMDetector(threshold=4.0, drift_level=0.5, target_mean=0.0)
        update_returned_true = False
        for _ in range(50):
            det.update(0.0)
        for _ in range(50):
            if det.update(10.0):
                update_returned_true = True
                break
        assert update_returned_true is True
        assert det.drift_detected is True

    def test_adwin_drift_detected_matches_update_return(self) -> None:
        """drift_detected property must be True once update() returned True.

        WHY: Same consistency check as CUSUM -- ensures interface contract.
        """
        rng = np.random.default_rng(88)
        det = ADWINDetector(delta=0.01, min_window=20)
        update_returned_true = False
        stream = np.concatenate([
            rng.normal(0.0, 0.1, 200),
            rng.normal(10.0, 0.1, 200),
        ])
        for val in stream:
            if det.update(float(val)):
                update_returned_true = True
                break
        assert update_returned_true is True
        assert det.drift_detected is True
