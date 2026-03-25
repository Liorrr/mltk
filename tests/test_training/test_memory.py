"""Tests for mltk.training.memory — memory leak and computation graph leak detection."""

import numpy as np
import pytest

from mltk.core.assertion import MltkAssertionError
from mltk.training.memory import (
    assert_loss_is_detached,
    assert_no_memory_leak,
)


class TestNoMemoryLeak:
    """assert_no_memory_leak — RSS/GPU memory growth over training."""

    def test_no_memory_leak_stable(self) -> None:
        # SCENARIO: Memory oscillates around 500 MB with tiny noise throughout training
        # WHY: Healthy training loop — no tensor accumulation, allocator reuses buffers
        # EXPECTED: passed=True, growth well within max_growth_mb=100
        rng = np.random.default_rng(42)
        readings = [500.0 + rng.normal(0, 2.0) for _ in range(50)]
        result = assert_no_memory_leak(readings, max_growth_mb=100.0, window=10)
        assert result.passed is True
        assert result.details["growth_mb"] < 100.0

    def test_memory_leak_detected(self) -> None:
        # SCENARIO: Memory grows from 500 MB to 800 MB over 100 steps (3 MB/step)
        # WHY: Loss stored as tensor each step; computation graph keeps accumulating
        # EXPECTED: MltkAssertionError, growth_mb > 100
        readings = [500.0 + i * 3.0 for i in range(100)]
        with pytest.raises(MltkAssertionError) as exc:
            assert_no_memory_leak(readings, max_growth_mb=100.0, window=10)
        result = exc.value.result
        assert result.passed is False
        assert result.details["growth_mb"] > 100.0
        assert "leak" in result.message.lower()

    def test_memory_leak_custom_threshold(self) -> None:
        # SCENARIO: Memory grows 50 MB total; strict threshold of 20 MB triggers failure
        # WHY: Production models may require tighter budgets (e.g., 16 GB GPU, large batch)
        # EXPECTED: MltkAssertionError raised
        readings = [200.0 + i * 0.5 for i in range(100)]  # total growth = 49.5 MB
        with pytest.raises(MltkAssertionError):
            assert_no_memory_leak(readings, max_growth_mb=20.0, window=10)

    def test_memory_leak_stores_diagnostic_details(self) -> None:
        # SCENARIO: Stable readings; verify all detail keys are populated
        # WHY: start_mean_mb, end_mean_mb, growth_mb are needed for dashboards/reports
        # EXPECTED: all detail keys present with correct types
        readings = [400.0 + float(i) * 0.01 for i in range(40)]
        result = assert_no_memory_leak(readings, max_growth_mb=100.0, window=10)
        assert "growth_mb" in result.details
        assert "start_mean_mb" in result.details
        assert "end_mean_mb" in result.details
        assert "window" in result.details
        assert "num_readings" in result.details
        assert result.details["num_readings"] == 40

    def test_few_readings_handled_gracefully(self) -> None:
        # SCENARIO: Only 3 memory readings available (very early in training)
        # WHY: window=10 > len=3 — must not crash; clamp to len//2=1
        # EXPECTED: assertion completes without exception, effective_window=1
        readings = [512.0, 514.0, 513.0]
        result = assert_no_memory_leak(readings, max_growth_mb=100.0, window=10)
        assert result.details["window"] == 1

    def test_memory_leak_returns_duration(self) -> None:
        # SCENARIO: @timed_assertion decorator is active
        # WHY: Timing metadata must be populated for every assertion
        # EXPECTED: duration_ms >= 0
        readings = [300.0] * 20
        result = assert_no_memory_leak(readings, max_growth_mb=100.0, window=5)
        assert result.duration_ms >= 0.0


class TestLossIsDetached:
    """assert_loss_is_detached — computation graph accumulation detection."""

    def test_loss_detached_ok(self) -> None:
        # SCENARIO: Memory per step is flat (±noise) around 600 MB
        # WHY: loss.item() is called each step — no graph references held
        # EXPECTED: passed=True, slope near zero
        rng = np.random.default_rng(7)
        steps = [600.0 + rng.normal(0, 0.5) for _ in range(60)]
        result = assert_loss_is_detached(steps, max_growth_per_step_mb=1.0)
        assert result.passed is True
        assert abs(result.details["slope_mb_per_step"]) < 1.0

    def test_loss_not_detached(self) -> None:
        # SCENARIO: Memory grows ~5 MB per step (graph retained for each loss)
        # WHY: Classic PyTorch bug: appending `loss` (not `loss.item()`) to a list
        # EXPECTED: MltkAssertionError, slope > max_growth_per_step_mb
        steps = [400.0 + i * 5.0 for i in range(50)]
        with pytest.raises(MltkAssertionError) as exc:
            assert_loss_is_detached(steps, max_growth_per_step_mb=1.0)
        result = exc.value.result
        assert result.passed is False
        assert result.details["slope_mb_per_step"] > 1.0
        assert "graph" in result.message.lower() or "slope" in result.message.lower()

    def test_loss_detached_stores_slope_and_steps(self) -> None:
        # SCENARIO: Healthy run; verify detail keys and step count
        # WHY: slope_mb_per_step and num_steps needed for diagnostics
        # EXPECTED: slope_mb_per_step present, num_steps == len(input)
        steps = [350.0 + float(i) * 0.02 for i in range(30)]
        result = assert_loss_is_detached(steps, max_growth_per_step_mb=1.0)
        assert "slope_mb_per_step" in result.details
        assert result.details["num_steps"] == 30

    def test_few_readings_handled_gracefully(self) -> None:
        # SCENARIO: Only 1 reading — cannot estimate a trend
        # WHY: assert must not crash on minimal data; skip is the correct behavior
        # EXPECTED: passed=True with a descriptive skip message
        result = assert_loss_is_detached([512.0], max_growth_per_step_mb=1.0)
        assert result.passed is True
        assert result.details["slope_mb_per_step"] == 0.0
        assert "too few" in result.message.lower()

    def test_loss_not_detached_tight_threshold(self) -> None:
        # SCENARIO: Slope is 0.8 MB/step, threshold is 0.5 MB/step (strict budget)
        # WHY: Large models on small GPUs need tight per-step budgets
        # EXPECTED: MltkAssertionError raised
        steps = [256.0 + i * 0.8 for i in range(50)]
        with pytest.raises(MltkAssertionError):
            assert_loss_is_detached(steps, max_growth_per_step_mb=0.5)

    def test_loss_detached_returns_duration(self) -> None:
        # SCENARIO: @timed_assertion decorator is active
        # WHY: Timing metadata must be populated for every assertion
        # EXPECTED: duration_ms >= 0
        steps = [500.0] * 25
        result = assert_loss_is_detached(steps, max_growth_per_step_mb=1.0)
        assert result.duration_ms >= 0.0
