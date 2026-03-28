"""Tests for mltk.domains.multimodal -- multimodal alignment and consistency.

Multimodal models map different modalities (image, text, audio) into a
shared embedding space. These tests validate two properties:

1. Image-text alignment: paired image and text embeddings should have high
   cosine similarity. Misalignment breaks image retrieval, captioning, and
   visual question answering.

2. Cross-modal consistency: predictions from different modalities on the
   SAME content should agree. Disagreement reveals dangerous modality-specific
   failures (e.g., text says "safe" but image says "unsafe").
"""

import numpy as np
import pytest

from mltk.core.assertion import MltkAssertionError
from mltk.domains.multimodal import (
    assert_cross_modal_consistency,
    assert_image_text_alignment,
)


class TestImageTextAlignment:
    """Image-text alignment tests using cosine similarity.

    Cosine similarity between matched image-text pairs should be high.
    In a well-trained model like CLIP, matched pairs score ~0.3 (random)
    vs ~0.7+ (aligned). We use synthetic embeddings to test the assertion
    logic without requiring an actual model.
    """

    def test_aligned_embeddings_pass(self) -> None:
        """PASS: Nearly identical embeddings yield cosine ~1.0.

        WHY: When image and text embeddings point in nearly the same
        direction, the model has learned strong cross-modal alignment.
        This is the ideal state after successful training.
        Expected: avg_cosine close to 1.0, passes min_cosine=0.5.
        """
        img = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        txt = np.array([[0.95, 0.05, 0.0], [0.05, 0.95, 0.0]])
        result = assert_image_text_alignment(img, txt, min_cosine=0.5)
        assert result.passed is True
        assert result.details["avg_cosine"] > 0.9
        assert result.details["n_pairs"] == 2

    def test_misaligned_embeddings_fail(self) -> None:
        """FAIL: Orthogonal embeddings yield cosine ~0.0.

        WHY: After a bad fine-tuning run, image and text embeddings can
        become orthogonal (pointing in unrelated directions). The model
        produces irrelevant captions and returns wrong images for queries.
        Expected: MltkAssertionError raised.
        """
        img = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        txt = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
        with pytest.raises(MltkAssertionError):
            assert_image_text_alignment(img, txt, min_cosine=0.5)

    def test_single_pair(self) -> None:
        """PASS: Alignment works with a single image-text pair.

        WHY: Some evaluations use a single sample (e.g., production
        monitoring on individual requests). The assertion should handle
        n_pairs=1 without division or shape errors.
        Expected: n_pairs=1 in details, cosine computed correctly.
        """
        img = np.array([[0.6, 0.8, 0.0]])
        txt = np.array([[0.6, 0.8, 0.0]])
        result = assert_image_text_alignment(img, txt, min_cosine=0.9)
        assert result.passed is True
        assert result.details["n_pairs"] == 1
        assert abs(result.details["avg_cosine"] - 1.0) < 1e-6

    def test_many_pairs(self) -> None:
        """PASS: Alignment scales to many pairs.

        WHY: Batch evaluation on hundreds of image-text pairs is the
        standard workflow. avg_cosine should be the mean across all pairs,
        and min/max pair cosines should capture the range.
        Expected: 100 pairs, min_pair_cosine and max_pair_cosine in details.
        """
        rng = np.random.RandomState(42)
        base = rng.randn(100, 64)
        noise = rng.randn(100, 64) * 0.1
        img = base
        txt = base + noise  # Small perturbation = high similarity
        result = assert_image_text_alignment(img, txt, min_cosine=0.5)
        assert result.passed is True
        assert result.details["n_pairs"] == 100
        assert result.details["min_pair_cosine"] <= result.details["max_pair_cosine"]

    def test_empty_embeddings_fail(self) -> None:
        """FAIL: Empty arrays cannot be evaluated.

        WHY: An empty embedding array means no data was provided, likely
        a pipeline bug. The assertion should fail with a clear message
        rather than silently passing or crashing.
        Expected: MltkAssertionError raised.
        """
        img = np.array([]).reshape(0, 3)
        txt = np.array([]).reshape(0, 3)
        with pytest.raises(MltkAssertionError):
            assert_image_text_alignment(img, txt, min_cosine=0.5)

    def test_1d_input_treated_as_single_pair(self) -> None:
        """PASS: 1D vectors are treated as a single embedding pair.

        WHY: Users may pass flat 1D arrays instead of 2D. The assertion
        should reshape gracefully rather than error on ndim mismatch.
        Expected: Treated as 1 pair, passes with identical vectors.
        """
        img = np.array([1.0, 0.0, 0.0])
        txt = np.array([1.0, 0.0, 0.0])
        result = assert_image_text_alignment(img, txt, min_cosine=0.9)
        assert result.passed is True
        assert result.details["n_pairs"] == 1


class TestCrossModalConsistency:
    """Cross-modal consistency tests.

    Two modalities processing the same content should produce the same
    predictions. Disagreement rate reveals modality-specific failures.
    """

    def test_perfect_agreement(self) -> None:
        """PASS: All predictions match across modalities (100% agreement).

        WHY: The ideal case -- text classification and image classification
        agree on every sample. The model's multimodal fusion is working.
        Expected: agreement_rate=1.0, no disagreements.
        """
        a = ["cat", "dog", "bird"]
        b = ["cat", "dog", "bird"]
        result = assert_cross_modal_consistency(a, b, min_agreement=0.9)
        assert result.passed is True
        assert result.details["agreement_rate"] == 1.0
        assert result.details["disagreements"] == []

    def test_zero_agreement_fails(self) -> None:
        """FAIL: No predictions match (0% agreement).

        WHY: Complete disagreement means one modality is fundamentally
        broken. For example, an OCR model reading "cat" from every image
        while the text model correctly classifies. This is a critical
        integration failure.
        Expected: MltkAssertionError raised.
        """
        a = [0, 1, 2, 3]
        b = [3, 2, 1, 0]
        with pytest.raises(MltkAssertionError):
            assert_cross_modal_consistency(a, b, min_agreement=0.5)

    def test_partial_agreement_with_details(self) -> None:
        """PASS (at low threshold): 75% agreement, indices of disagreements.

        WHY: Partial disagreement is common in production. The assertion
        returns which indices disagree so engineers can inspect the specific
        failing cases (e.g., "sample 2 is always wrong from camera modality").
        Expected: agreement_rate=0.75, disagreements=[2].
        """
        a = np.array([1, 1, 0, 1])
        b = np.array([1, 1, 1, 1])
        result = assert_cross_modal_consistency(a, b, min_agreement=0.7)
        assert result.passed is True
        assert result.details["agreement_rate"] == 0.75
        assert result.details["n_agreed"] == 3
        assert result.details["n_total"] == 4
        assert result.details["disagreements"] == [2]

    def test_empty_predictions_fail(self) -> None:
        """FAIL: Empty prediction arrays cannot be compared.

        WHY: Empty predictions mean the pipeline produced no output,
        likely a data loading or model inference failure. Must fail
        rather than vacuously pass.
        Expected: MltkAssertionError raised.
        """
        with pytest.raises(MltkAssertionError):
            assert_cross_modal_consistency([], [], min_agreement=0.5)


# -------------------------------------------------------------------
# Parametrized & edge-case tests (hardening)
# -------------------------------------------------------------------


class TestMinCosineParametrized:
    """Parametrize min_cosine thresholds."""

    @pytest.mark.parametrize(
        "min_cos", [0.0, 0.3, 0.5, 0.8, 0.99]
    )
    def test_identical_embeddings_vs_threshold(
        self, min_cos: float
    ) -> None:
        """Identical vectors (cos=1) pass any threshold."""
        v = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        r = assert_image_text_alignment(
            v, v, min_cosine=min_cos
        )
        assert r.passed is True
        assert abs(r.details["avg_cosine"] - 1.0) < 1e-6


class TestHighDimensionalEmbeddings:
    """High-dimensional embeddings (512-d)."""

    def test_512d_aligned(self) -> None:
        """512-d embeddings with small noise pass."""
        rng = np.random.RandomState(42)
        base = rng.randn(10, 512)
        noise = rng.randn(10, 512) * 0.05
        r = assert_image_text_alignment(
            base, base + noise, min_cosine=0.5
        )
        assert r.passed is True
        assert r.details["n_pairs"] == 10
        assert r.details["avg_cosine"] > 0.9


class TestCrossModalStringPredictions:
    """Cross-modal with string predictions."""

    def test_string_predictions_agreement(self) -> None:
        """String labels are compared correctly."""
        a = ["cat", "dog", "cat", "bird", "fish"]
        b = ["cat", "dog", "dog", "bird", "fish"]
        r = assert_cross_modal_consistency(
            a, b, min_agreement=0.7
        )
        assert r.passed is True
        assert r.details["agreement_rate"] == 0.8
        assert r.details["disagreements"] == [2]


class TestNormalizedVsUnnormalized:
    """Normalized vs unnormalized embeddings."""

    def test_unnormalized_same_direction(self) -> None:
        """Unnormalized but same-direction vectors pass."""
        img = np.array([[10.0, 0.0, 0.0]])
        txt = np.array([[0.001, 0.0, 0.0]])
        r = assert_image_text_alignment(
            img, txt, min_cosine=0.9
        )
        assert r.passed is True
        assert abs(r.details["avg_cosine"] - 1.0) < 1e-6


class TestSinglePairAlignment:
    """Single pair alignment edge case."""

    def test_single_pair_misaligned(self) -> None:
        """Single pair below threshold fails cleanly."""
        img = np.array([[1.0, 0.0]])
        txt = np.array([[0.0, 1.0]])
        with pytest.raises(MltkAssertionError):
            assert_image_text_alignment(
                img, txt, min_cosine=0.5
            )
