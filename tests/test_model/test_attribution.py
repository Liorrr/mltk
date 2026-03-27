"""Tests for mltk.model.attribution -- feature attribution stability.

Feature attribution methods (SHAP, LIME, integrated gradients) use
randomized sampling, so two runs on the same data may produce different
importance vectors.  These tests verify the two stability assertions:

- **assert_top_k_stable**: checks whether the same features appear in the
  top-K across two runs (set overlap).
- **assert_attribution_cosine_stability**: checks whether the full
  attribution vectors point in the same direction (cosine similarity).
"""

import numpy as np
import pytest

from mltk.core.assertion import MltkAssertionError
from mltk.model.attribution import (
    assert_attribution_cosine_stability,
    assert_top_k_stable,
)

# =========================================================================
# assert_top_k_stable
# =========================================================================

class TestAssertTopKStable:
    """Tests for top-K feature overlap between attribution runs."""

    def test_identical_attributions(self) -> None:
        """PASS: Same vector twice -- overlap is 1.0.

        When SHAP is run with a fixed seed, the output is deterministic.
        Two identical vectors must always have perfect top-K agreement.
        """
        a = np.array([0.5, 0.1, 0.9, 0.3, 0.8, 0.05, 0.7])
        result = assert_top_k_stable(a, a, k=5, min_overlap=1.0)
        assert result.passed is True
        assert result.details["overlap"] == 1.0
        assert result.details["common_features"] == 5

    def test_same_top_k_different_magnitudes(self) -> None:
        """PASS: Top features are the same, magnitudes differ.

        Two SHAP runs might agree on which features matter but assign
        slightly different importance values.  Top-K only cares about
        the set of indices, not the exact values.
        """
        a = np.array([0.5, 0.1, 0.9, 0.3, 0.8])
        b = np.array([0.6, 0.2, 0.85, 0.25, 0.7])
        # Both runs: top-3 indices are {2, 4, 0}
        result = assert_top_k_stable(a, b, k=3, min_overlap=0.8)
        assert result.passed is True
        assert result.details["overlap"] >= 0.8

    def test_completely_different_top_features(self) -> None:
        """FAIL: Top features are entirely different between runs.

        This simulates a pathological case where SHAP explanations are
        essentially random -- no trust can be placed in the explanation.
        """
        # Run A: features 5,6,7 are large
        a = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.9, 0.8, 0.7])
        # Run B: features 0,1,2 are large
        b = np.array([0.9, 0.8, 0.7, 0.04, 0.05, 0.01, 0.02, 0.03])
        with pytest.raises(MltkAssertionError) as exc:
            assert_top_k_stable(a, b, k=3, min_overlap=0.5)
        assert "Unstable" in str(exc.value)
        assert exc.value.result.details["overlap"] == 0.0

    def test_partial_overlap_at_threshold(self) -> None:
        """PASS: Overlap exactly at the min_overlap boundary.

        With k=5 and min_overlap=0.6, we need at least 3 out of 5
        features in common.  This test has exactly 3 overlapping.
        """
        # Top-5 of A: indices {4,5,6,7,8}  (values 0.6..1.0)
        a = np.array([0.0, 0.1, 0.2, 0.3, 0.6, 0.7, 0.8, 0.9, 1.0])
        # Top-5 of B: indices {0,1,6,7,8}  (values 0.8..0.95)
        b = np.array([0.8, 0.85, 0.0, 0.0, 0.0, 0.0, 0.9, 0.92, 0.95])
        # Common: {6, 7, 8} -> overlap = 3/5 = 0.6
        result = assert_top_k_stable(a, b, k=5, min_overlap=0.6)
        assert result.passed is True
        assert result.details["overlap"] == pytest.approx(0.6)
        assert result.details["common_features"] == 3

    def test_k_larger_than_features(self) -> None:
        """PASS: k > n_features is clamped gracefully.

        If a model has only 3 features but the user asks for top-10,
        we clamp to k=3 rather than raising an error.  With identical
        vectors, overlap is 1.0.
        """
        a = np.array([0.5, 0.2, 0.8])
        result = assert_top_k_stable(a, a, k=10, min_overlap=1.0)
        assert result.passed is True
        assert result.details["k"] == 3  # Clamped to n_features
        assert result.details["overlap"] == 1.0

    def test_single_feature(self) -> None:
        """PASS: Trivial case with one feature.

        Even a single-feature model can be checked.  The top-1 index
        is always 0, so overlap is 1.0.
        """
        a = np.array([0.42])
        b = np.array([0.99])
        result = assert_top_k_stable(a, b, k=1, min_overlap=1.0)
        assert result.passed is True
        assert result.details["k"] == 1
        assert result.details["overlap"] == 1.0

    def test_negative_attributions(self) -> None:
        """PASS: Absolute value is used, so sign doesn't matter.

        SHAP values can be negative (feature pushes prediction down).
        A feature with attribution -0.9 is more important than one with
        +0.1.  We use absolute values when ranking.
        """
        # Feature 0 has the largest absolute value in both (-0.9)
        a = np.array([-0.9, 0.1, 0.3, -0.8, 0.05])
        b = np.array([-0.85, 0.15, 0.25, -0.75, 0.1])
        # Top-3 by abs: A={0,3,2}, B={0,3,2}
        result = assert_top_k_stable(a, b, k=3, min_overlap=1.0)
        assert result.passed is True
        assert result.details["overlap"] == 1.0


# =========================================================================
# assert_attribution_cosine_stability
# =========================================================================

class TestAssertAttributionCosineStability:
    """Tests for cosine similarity between attribution vectors."""

    def test_identical_vectors(self) -> None:
        """PASS: Same vector twice -- cosine is ~1.0.

        A deterministic attribution method produces identical output,
        so cosine must be effectively perfect.  Due to floating-point
        arithmetic, dot(a,a)/(||a||*||a||) can be very slightly below
        1.0, so we threshold at 0.999 and verify the value rounds to 1.0.
        """
        a = np.array([0.5, 0.1, 0.9, 0.3])
        result = assert_attribution_cosine_stability(a, a, min_cosine=0.999)
        assert result.passed is True
        assert result.details["cosine_similarity"] == pytest.approx(1.0)

    def test_proportional_vectors(self) -> None:
        """PASS: Proportional vectors (a = 2*b) have cosine 1.0.

        If one run assigns exactly double the importance to every feature,
        the *direction* is the same.  Cosine is scale-invariant, so this
        is a perfect score.  Top-K would also pass, but this tests that
        cosine correctly ignores uniform scaling.
        """
        a = np.array([1.0, 2.0, 3.0, 4.0])
        b = 2.0 * a
        result = assert_attribution_cosine_stability(a, b, min_cosine=0.99)
        assert result.passed is True
        assert result.details["cosine_similarity"] == pytest.approx(1.0)

    def test_orthogonal_vectors(self) -> None:
        """FAIL: Orthogonal vectors have cosine ~0.0.

        Two attribution vectors that share no directional information
        are essentially unrelated explanations.  This should fail any
        reasonable min_cosine threshold.
        """
        a = np.array([1.0, 0.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0, 0.0])
        with pytest.raises(MltkAssertionError) as exc:
            assert_attribution_cosine_stability(a, b, min_cosine=0.5)
        assert exc.value.result.details["cosine_similarity"] == pytest.approx(0.0)

    def test_near_identical_with_noise(self) -> None:
        """PASS: Small noise produces cosine very close to 1.0.

        Typical scenario: two SHAP runs with different seeds but enough
        samples.  The attribution vectors are nearly identical, with
        tiny floating-point differences.
        """
        rng = np.random.default_rng(42)
        a = np.array([0.5, 0.1, 0.9, 0.3, 0.8])
        noise = rng.normal(0, 0.01, a.shape)
        b = a + noise
        result = assert_attribution_cosine_stability(a, b, min_cosine=0.99)
        assert result.passed is True
        assert result.details["cosine_similarity"] > 0.99

    def test_2d_arrays_per_sample_cosine(self) -> None:
        """PASS: 2-D input computes per-sample cosine, reports mean.

        Multi-sample SHAP values are shape (n_samples, n_features).
        Cosine is computed for each row independently, then averaged.
        """
        rng = np.random.default_rng(42)
        a = rng.normal(0, 1, (10, 5))
        noise = rng.normal(0, 0.01, a.shape)
        b = a + noise
        result = assert_attribution_cosine_stability(a, b, min_cosine=0.99)
        assert result.passed is True
        assert result.details["n_samples"] == 10
        assert result.details["n_features"] == 5
        assert result.details["cosine_similarity"] > 0.99

    def test_zero_vector(self) -> None:
        """FAIL: Zero vector has cosine 0.0.

        A zero attribution vector means "no feature matters" -- this is
        a degenerate explanation.  The cosine is defined as 0.0 (not NaN),
        and it should fail any positive min_cosine threshold.
        """
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([0.5, 0.1, 0.3])
        with pytest.raises(MltkAssertionError) as exc:
            assert_attribution_cosine_stability(a, b, min_cosine=0.5)
        assert exc.value.result.details["cosine_similarity"] == pytest.approx(0.0)

    def test_opposite_vectors(self) -> None:
        """FAIL: Opposite vectors (a = -b) have cosine -1.0.

        If one run says "age increases prediction" and another says
        "age decreases prediction" for every feature, the explanations
        are contradictory.  Cosine = -1.0, which fails any positive
        min_cosine threshold.
        """
        a = np.array([1.0, 2.0, 3.0])
        b = -a
        with pytest.raises(MltkAssertionError) as exc:
            assert_attribution_cosine_stability(a, b, min_cosine=0.0)
        assert exc.value.result.details["cosine_similarity"] == pytest.approx(-1.0)
