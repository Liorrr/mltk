"""Tests for extended distributed assertions — N-rank sync, alignment, divergence, clipping."""

import numpy as np
import pytest

from mltk.core.assertion import MltkAssertionError
from mltk.training.distributed import (
    assert_gradient_alignment,
    assert_gradient_clipped,
    assert_n_rank_gradient_sync,
    assert_weight_divergence,
)

# ---------------------------------------------------------------------------
# assert_n_rank_gradient_sync
# ---------------------------------------------------------------------------


class TestNRankGradientSync:
    """assert_n_rank_gradient_sync — generalized N-rank all-reduce verification."""

    def test_three_ranks_synced(self) -> None:
        # SCENARIO: 3 ranks with identical gradients after all-reduce
        # WHY: Standard 3-GPU DDP — must detect that all ranks agree
        # EXPECTED: passed=True, diverged_pairs=[], num_ranks=3
        g = [np.array([0.01, 0.02, -0.03]), np.array([0.1, -0.05])]
        result = assert_n_rank_gradient_sync([g, g, g], tolerance=1e-5)
        assert result.passed is True
        assert result.details["diverged_pairs"] == []
        assert result.details["num_ranks"] == 3
        assert result.details["num_layers"] == 2
        assert result.details["max_diff"] == 0.0

    def test_three_ranks_desynced(self) -> None:
        # SCENARIO: Rank 2 holds stale gradients, diverging from ranks 0/1
        # WHY: Broken barrier or dropped collective on one rank
        # EXPECTED: MltkAssertionError, diverged_pairs contains rank 2 entries
        g0 = [np.array([0.01, 0.02])]
        g1 = [np.array([0.01, 0.02])]
        g2 = [np.array([0.99, 0.88])]
        with pytest.raises(MltkAssertionError) as exc:
            assert_n_rank_gradient_sync([g0, g1, g2], tolerance=1e-5)
        result = exc.value.result
        assert result.passed is False
        assert len(result.details["diverged_pairs"]) > 0
        # All diverged pairs must involve rank 2
        for pair in result.details["diverged_pairs"]:
            assert 2 in (pair[0], pair[1])

    def test_single_rank(self) -> None:
        # SCENARIO: world_size=1, only one rank
        # WHY: Edge case — trivially synced, should pass
        # EXPECTED: passed=True, num_ranks=1
        g = [np.array([0.5, -0.3])]
        result = assert_n_rank_gradient_sync([g], tolerance=1e-5)
        assert result.passed is True
        assert result.details["num_ranks"] == 1

    def test_layer_count_mismatch(self) -> None:
        # SCENARIO: Rank 0 has 2 layers, rank 1 has 1 layer
        # WHY: Model architecture mismatch across ranks — immediate failure
        # EXPECTED: MltkAssertionError with mismatch message
        g0 = [np.array([0.1]), np.array([0.2])]
        g1 = [np.array([0.1])]
        with pytest.raises(MltkAssertionError) as exc:
            assert_n_rank_gradient_sync([g0, g1])
        assert "mismatch" in exc.value.result.message.lower()

    def test_single_layer(self) -> None:
        # SCENARIO: 4 ranks, each with a single-layer gradient — all identical
        # WHY: Single-layer networks (logistic regression) are common in testing
        # EXPECTED: passed=True, num_layers=1
        g = [np.array([0.42, -0.13, 0.07])]
        result = assert_n_rank_gradient_sync([g, g, g, g], tolerance=1e-5)
        assert result.passed is True
        assert result.details["num_layers"] == 1
        assert result.details["num_ranks"] == 4

    def test_returns_duration(self) -> None:
        # SCENARIO: @timed_assertion decorator is active
        # WHY: Timing metadata must be populated
        # EXPECTED: duration_ms >= 0
        g = [np.array([0.1])]
        result = assert_n_rank_gradient_sync([g, g])
        assert result.duration_ms >= 0.0


# ---------------------------------------------------------------------------
# assert_gradient_alignment
# ---------------------------------------------------------------------------


class TestGradientAlignment:
    """assert_gradient_alignment — cosine similarity between gradient vectors."""

    def test_identical_grads_aligned(self) -> None:
        # SCENARIO: Identical gradient vectors from two ranks
        # WHY: After correct all-reduce, cosine similarity should be 1.0
        # EXPECTED: passed=True, cosine_similarity=1.0
        g = [np.array([1.0, 2.0, 3.0]), np.array([-0.5, 0.7])]
        result = assert_gradient_alignment(g, g, min_cosine=0.9)
        assert result.passed is True
        assert result.details["cosine_similarity"] == pytest.approx(1.0, abs=1e-10)

    def test_orthogonal_grads_fail(self) -> None:
        # SCENARIO: Orthogonal gradient vectors (cosine ~ 0)
        # WHY: Ranks optimizing in perpendicular directions = broken sync
        # EXPECTED: MltkAssertionError, cosine_similarity near 0
        g_a = [np.array([1.0, 0.0])]
        g_b = [np.array([0.0, 1.0])]
        with pytest.raises(MltkAssertionError) as exc:
            assert_gradient_alignment(g_a, g_b, min_cosine=0.9)
        result = exc.value.result
        assert result.passed is False
        assert result.details["cosine_similarity"] == pytest.approx(0.0, abs=1e-10)

    def test_opposite_grads_fail(self) -> None:
        # SCENARIO: Anti-parallel gradients (cosine = -1)
        # WHY: One rank ascending while the other descends — catastrophic
        # EXPECTED: MltkAssertionError, cosine_similarity ~ -1.0
        g_a = [np.array([1.0, 2.0, 3.0])]
        g_b = [np.array([-1.0, -2.0, -3.0])]
        with pytest.raises(MltkAssertionError) as exc:
            assert_gradient_alignment(g_a, g_b, min_cosine=0.9)
        result = exc.value.result
        assert result.details["cosine_similarity"] == pytest.approx(-1.0, abs=1e-10)

    def test_mostly_aligned_passes(self) -> None:
        # SCENARIO: Gradients differ slightly but cosine > min_cosine
        # WHY: Small perturbations from different micro-batches are expected
        # EXPECTED: passed=True
        rng = np.random.default_rng(42)
        base = [rng.normal(0, 1, 100)]
        perturbed = [base[0] + rng.normal(0, 0.1, 100)]
        result = assert_gradient_alignment(base, perturbed, min_cosine=0.9)
        assert result.passed is True
        assert result.details["cosine_similarity"] > 0.9

    def test_zero_gradient_fails(self) -> None:
        # SCENARIO: One rank has a zero gradient vector
        # WHY: Zero gradient norm makes cosine undefined; treated as 0
        # EXPECTED: MltkAssertionError, cosine_similarity=0.0
        g_a = [np.array([1.0, 2.0])]
        g_b = [np.array([0.0, 0.0])]
        with pytest.raises(MltkAssertionError) as exc:
            assert_gradient_alignment(g_a, g_b, min_cosine=0.9)
        assert exc.value.result.details["cosine_similarity"] == 0.0


# ---------------------------------------------------------------------------
# assert_weight_divergence
# ---------------------------------------------------------------------------


class TestWeightDivergence:
    """assert_weight_divergence — L2 distance between weight checkpoints."""

    def test_identical_weights_pass(self) -> None:
        # SCENARIO: Same weight snapshot compared to itself
        # WHY: L2 distance = 0.0, trivially within any threshold
        # EXPECTED: passed=True, l2_distance=0.0
        w = [np.array([0.5, -0.3, 0.1]), np.array([0.8, -0.2])]
        result = assert_weight_divergence(w, w, max_l2_distance=0.01)
        assert result.passed is True
        assert result.details["l2_distance"] == 0.0
        assert result.details["num_params"] == 5

    def test_diverged_weights_fail(self) -> None:
        # SCENARIO: Weights have drifted far apart (L2 >> threshold)
        # WHY: Missed all-reduce or different learning rates per rank
        # EXPECTED: MltkAssertionError, l2_distance > max_l2_distance
        w_a = [np.array([0.0, 0.0, 0.0])]
        w_b = [np.array([1.0, 1.0, 1.0])]
        with pytest.raises(MltkAssertionError) as exc:
            assert_weight_divergence(w_a, w_b, max_l2_distance=0.01)
        result = exc.value.result
        assert result.passed is False
        assert result.details["l2_distance"] == pytest.approx(np.sqrt(3.0), abs=1e-10)
        assert "divergence exceeded" in result.message.lower()

    def test_within_threshold_passes(self) -> None:
        # SCENARIO: Tiny perturbation — L2 distance just under threshold
        # WHY: Normal numerical jitter after a few steps
        # EXPECTED: passed=True
        w_a = [np.array([1.0, 2.0, 3.0])]
        w_b = [np.array([1.001, 2.001, 3.001])]
        result = assert_weight_divergence(w_a, w_b, max_l2_distance=0.01)
        assert result.passed is True
        assert result.details["l2_distance"] < 0.01

    def test_single_layer(self) -> None:
        # SCENARIO: Single-layer model weights compared
        # WHY: Edge case — only one layer, should still work
        # EXPECTED: passed=True, num_params = array length
        w = [np.array([0.42])]
        result = assert_weight_divergence(w, w, max_l2_distance=0.01)
        assert result.passed is True
        assert result.details["num_params"] == 1


# ---------------------------------------------------------------------------
# assert_gradient_clipped
# ---------------------------------------------------------------------------


class TestGradientClipped:
    """assert_gradient_clipped — global gradient norm vs max_norm."""

    def test_norm_under_max_passes(self) -> None:
        # SCENARIO: Small gradients, well within clipping threshold
        # WHY: After proper clip_grad_norm_, global norm <= max_norm
        # EXPECTED: passed=True, global_norm < max_norm
        g = [np.array([0.1, 0.2]), np.array([-0.05, 0.15])]
        result = assert_gradient_clipped(g, max_norm=1.0)
        assert result.passed is True
        assert result.details["global_norm"] < 1.0
        assert result.details["max_norm"] == 1.0
        assert result.details["num_layers"] == 2

    def test_norm_over_max_fails(self) -> None:
        # SCENARIO: Large gradients exceed the clipping threshold
        # WHY: Clipping was not applied or called before backward
        # EXPECTED: MltkAssertionError, global_norm > max_norm
        g = [np.array([10.0, 20.0, 30.0])]
        with pytest.raises(MltkAssertionError) as exc:
            assert_gradient_clipped(g, max_norm=1.0)
        result = exc.value.result
        assert result.passed is False
        assert result.details["global_norm"] > 1.0
        assert "not clipped" in result.message.lower()

    def test_exactly_at_max_passes(self) -> None:
        # SCENARIO: Global norm equals max_norm exactly
        # WHY: Boundary condition — <= means equality passes
        # EXPECTED: passed=True
        # Construct a vector with known norm: [max_norm, 0]
        g = [np.array([5.0, 0.0])]
        result = assert_gradient_clipped(g, max_norm=5.0)
        assert result.passed is True
        assert result.details["global_norm"] == pytest.approx(5.0, abs=1e-10)

    def test_single_layer_single_element(self) -> None:
        # SCENARIO: Single scalar gradient
        # WHY: Simplest possible gradient — edge case for concatenation
        # EXPECTED: global_norm == abs(value)
        g = [np.array([0.3])]
        result = assert_gradient_clipped(g, max_norm=1.0)
        assert result.passed is True
        assert result.details["global_norm"] == pytest.approx(0.3, abs=1e-10)
        assert result.details["num_layers"] == 1

    def test_returns_duration(self) -> None:
        # SCENARIO: @timed_assertion decorator is active
        # WHY: Timing metadata must be populated
        # EXPECTED: duration_ms >= 0
        g = [np.array([0.1, 0.2])]
        result = assert_gradient_clipped(g, max_norm=1.0)
        assert result.duration_ms >= 0.0
