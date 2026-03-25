"""Tests for mltk.training.distributed — effective batch size and gradient sync."""

import numpy as np
import pytest

from mltk.core.assertion import MltkAssertionError
from mltk.training.distributed import (
    assert_effective_batch_size,
    assert_gradient_sync,
)


class TestEffectiveBatchSize:
    """assert_effective_batch_size — world_size x local_batch correctness."""

    def test_effective_batch_size_correct(self) -> None:
        # SCENARIO: 4 GPUs, 32 samples each -> 128 total
        # WHY: Standard DDP setup; effective batch == expected
        # EXPECTED: passed=True, effective_batch_size=128
        result = assert_effective_batch_size(
            local_batch_size=32, world_size=4, expected_batch_size=128
        )
        assert result.passed is True
        assert result.details["effective_batch_size"] == 128

    def test_effective_batch_size_wrong(self) -> None:
        # SCENARIO: 4 GPUs x 32 = 128 but recipe was tuned for 64
        # WHY: Scaling world_size without checking expected_batch_size is a
        #      common mistake — gradients aggregate differently than expected
        # EXPECTED: MltkAssertionError raised, effective_batch_size=128 != 64
        with pytest.raises(MltkAssertionError) as exc:
            assert_effective_batch_size(
                local_batch_size=32, world_size=4, expected_batch_size=64
            )
        result = exc.value.result
        assert result.passed is False
        assert result.details["effective_batch_size"] == 128
        assert result.details["expected_batch_size"] == 64
        assert "mismatch" in result.message.lower()

    def test_effective_batch_size_single_gpu(self) -> None:
        # SCENARIO: world_size=1 (single GPU or CPU run)
        # WHY: Edge case — no distribution, effective == local
        # EXPECTED: passed=True
        result = assert_effective_batch_size(
            local_batch_size=64, world_size=1, expected_batch_size=64
        )
        assert result.passed is True
        assert result.details["world_size"] == 1

    def test_effective_batch_size_stores_all_details(self) -> None:
        # SCENARIO: Valid config; verify all detail keys are present
        # WHY: Downstream diagnostics and reports rely on structured details
        # EXPECTED: details contains local_batch_size, world_size, expected_batch_size
        result = assert_effective_batch_size(
            local_batch_size=16, world_size=8, expected_batch_size=128
        )
        assert "local_batch_size" in result.details
        assert "world_size" in result.details
        assert "expected_batch_size" in result.details
        assert "effective_batch_size" in result.details

    def test_effective_batch_size_returns_duration(self) -> None:
        # SCENARIO: @timed_assertion decorator is active
        # WHY: Every assertion must populate timing metadata
        # EXPECTED: duration_ms >= 0
        result = assert_effective_batch_size(
            local_batch_size=8, world_size=2, expected_batch_size=16
        )
        assert result.duration_ms >= 0.0


class TestGradientSync:
    """assert_gradient_sync — all-reduce gradient equality across ranks."""

    def test_gradient_sync_ok(self) -> None:
        # SCENARIO: Rank 0 and rank 1 have identical gradient arrays
        # WHY: After a correct all-reduce, both ranks should hold the same averaged grads
        # EXPECTED: passed=True, diverged_layers=[], max_diff=0.0
        grads = [np.array([0.01, 0.02, -0.03]), np.array([0.1, -0.05, 0.08])]
        result = assert_gradient_sync(grads, grads, tolerance=1e-5)
        assert result.passed is True
        assert result.details["diverged_layers"] == []
        assert result.details["max_diff"] == 0.0

    def test_gradient_sync_diverged(self) -> None:
        # SCENARIO: Rank 1 gradients are completely different from rank 0
        # WHY: A broken all-reduce or missing barrier leaves ranks with their local grads
        # EXPECTED: MltkAssertionError, diverged_layers contains affected indices
        grads_rank0 = [np.array([0.01, 0.02]), np.array([0.1, -0.05])]
        grads_rank1 = [np.array([0.99, 0.88]), np.array([-0.7, 0.3])]
        with pytest.raises(MltkAssertionError) as exc:
            assert_gradient_sync(grads_rank0, grads_rank1, tolerance=1e-5)
        result = exc.value.result
        assert result.passed is False
        assert len(result.details["diverged_layers"]) > 0
        assert "desync" in result.message.lower()

    def test_gradient_sync_within_tolerance(self) -> None:
        # SCENARIO: Gradients differ by 1e-7 — well within tolerance of 1e-5
        # WHY: FP32 all-reduce introduces tiny numerical noise; tolerance must absorb it
        # EXPECTED: passed=True (difference < tolerance)
        rng = np.random.default_rng(0)
        base = [rng.normal(0, 0.1, 10) for _ in range(3)]
        perturbed = [g + rng.normal(0, 1e-7, 10) for g in base]
        result = assert_gradient_sync(base, perturbed, tolerance=1e-5)
        assert result.passed is True

    def test_gradient_sync_exactly_at_tolerance_boundary(self) -> None:
        # SCENARIO: One layer diff is exactly tolerance — should fail (strictly >)
        # WHY: Boundary condition must be explicit; > not >=
        # EXPECTED: passed=False, diverged_layers=[0]
        grads_rank0 = [np.array([0.0])]
        grads_rank1 = [np.array([1e-5 + 1e-10])]  # just above 1e-5
        with pytest.raises(MltkAssertionError) as exc:
            assert_gradient_sync(grads_rank0, grads_rank1, tolerance=1e-5)
        assert 0 in exc.value.result.details["diverged_layers"]

    def test_gradient_sync_length_mismatch(self) -> None:
        # SCENARIO: Rank 0 has 3 layers, rank 1 has 2 layers
        # WHY: Model architecture mismatch or serialization bug — must fail clearly
        # EXPECTED: MltkAssertionError with informative message
        grads0 = [np.array([0.1]), np.array([0.2]), np.array([0.3])]
        grads1 = [np.array([0.1]), np.array([0.2])]
        with pytest.raises(MltkAssertionError) as exc:
            assert_gradient_sync(grads0, grads1)
        assert "mismatch" in exc.value.result.message.lower()

    def test_gradient_sync_returns_duration(self) -> None:
        # SCENARIO: @timed_assertion decorator is active
        # WHY: Timing metadata must be populated for every assertion
        # EXPECTED: duration_ms >= 0
        g = [np.array([0.05, -0.05])]
        result = assert_gradient_sync(g, g)
        assert result.duration_ms >= 0.0
