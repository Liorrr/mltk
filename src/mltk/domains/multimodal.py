"""Multimodal evaluation -- alignment and consistency across modalities.

Multimodal AI models (CLIP, BLIP, LLaVA, GPT-4V) learn to map different
modalities -- images, text, audio -- into a shared embedding space. When
working correctly, semantically related inputs from different modalities
land close together: a photo of a dog and the sentence "a photo of a dog"
should have high cosine similarity.

These assertions validate two critical multimodal properties:

1. **Image-text alignment**: Do image and text embeddings agree?
   Misalignment after fine-tuning or quantization leads to wrong captions,
   broken image search, and hallucinated visual descriptions.

2. **Cross-modal consistency**: When the SAME content is processed through
   different modalities, do predictions agree? A medical report saying
   "no tumor" while the image classifier says "tumor detected" is a
   life-threatening inconsistency.

mltk does NOT compute embeddings -- that is the model's job. These
assertions take pre-computed embeddings or predictions and validate
their quality.
"""

from __future__ import annotations

import numpy as np

from mltk.core.assertion import assert_true, timed_assertion
from mltk.core.result import Severity, TestResult

__all__ = [
    "assert_image_text_alignment",
    "assert_cross_modal_consistency",
]


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors.

    Cosine similarity measures the angle between two vectors in embedding
    space, ignoring magnitude. It ranges from -1 (opposite) through 0
    (orthogonal) to 1 (identical direction).

    Returns 0.0 if either vector has zero norm (degenerate embedding).
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))

@timed_assertion
def assert_image_text_alignment(
    image_embeddings: np.ndarray,
    text_embeddings: np.ndarray,
    min_cosine: float = 0.5,
) -> TestResult:
    """Assert that image and text embeddings are aligned in shared space.

    Multimodal models like CLIP encode images and text into the same
    vector space. Aligned pairs (e.g., a photo and its caption) should
    have high cosine similarity. This assertion computes per-pair cosine
    similarity and checks that the average meets a minimum threshold.

    Use this after fine-tuning, quantization, or data updates to catch
    alignment degradation before it reaches production.

    Args:
        image_embeddings: Image embeddings, shape ``(n_pairs, dim)`` or
            ``(dim,)`` for a single pair. Each row is one image embedding.
        text_embeddings: Text embeddings, same shape as image_embeddings.
            Each row corresponds to the matching image embedding.
        min_cosine: Minimum average cosine similarity required (default 0.5).
            CLIP-quality models typically score 0.25-0.35 on random pairs
            and 0.7+ on matched pairs.

    Returns:
        TestResult with details: ``avg_cosine``, ``min_cosine``,
        ``min_pair_cosine``, ``max_pair_cosine``, ``n_pairs``.

    Raises:
        MltkAssertionError: If avg_cosine < min_cosine (CRITICAL severity).

    Example:
        >>> import numpy as np
        >>> # Simulate aligned embeddings (high similarity)
        >>> img = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        >>> txt = np.array([[0.9, 0.1, 0.0], [0.1, 0.9, 0.0]])
        >>> result = assert_image_text_alignment(img, txt, min_cosine=0.5)
        >>> result.passed
        True
    """
    img = np.asarray(image_embeddings, dtype=float)
    txt = np.asarray(text_embeddings, dtype=float)

    # Handle 1D input (single pair) by reshaping to 2D
    if img.ndim == 1:
        img = img.reshape(1, -1)
    if txt.ndim == 1:
        txt = txt.reshape(1, -1)

    n_pairs = img.shape[0]

    if n_pairs == 0:
        return assert_true(
            False,
            name="multimodal.image_text_alignment",
            message="Cannot compute alignment on empty embeddings",
            severity=Severity.CRITICAL,
        )

    # Compute per-pair cosine similarity
    cosines = [_cosine_similarity(img[i], txt[i]) for i in range(n_pairs)]

    avg_cosine = float(np.mean(cosines))
    min_pair_cosine = float(np.min(cosines))
    max_pair_cosine = float(np.max(cosines))

    passed = avg_cosine >= min_cosine
    message = (
        f"Image-text alignment: avg_cosine={avg_cosine:.4f} >= {min_cosine}"
        if passed
        else f"Image-text alignment too low: avg_cosine={avg_cosine:.4f} < {min_cosine}"
    )

    return assert_true(
        passed,
        name="multimodal.image_text_alignment",
        message=message,
        severity=Severity.CRITICAL,
        avg_cosine=avg_cosine,
        min_cosine=min_cosine,
        min_pair_cosine=min_pair_cosine,
        max_pair_cosine=max_pair_cosine,
        n_pairs=n_pairs,
    )

@timed_assertion
def assert_cross_modal_consistency(
    predictions_a: np.ndarray | list,
    predictions_b: np.ndarray | list,
    min_agreement: float = 0.8,
) -> TestResult:
    """Assert that predictions from two modalities agree on the same content.

    When a model processes the same input through different modalities
    (e.g., text description vs. image of the same scene), the predictions
    should be consistent. Disagreement reveals modality-specific biases
    or failures in multimodal fusion.

    Real-world examples where this catches bugs:
    - Medical: text report says "benign" but image classifier says "malignant"
    - Autonomous driving: LIDAR detects obstacle but camera classifier says "clear"
    - Content moderation: text is safe but image contains violations

    Args:
        predictions_a: Predictions from modality A (array or list).
        predictions_b: Predictions from modality B (array or list).
            Must have the same length as predictions_a.
        min_agreement: Minimum fraction of samples that must agree
            (default 0.8 = 80% agreement).

    Returns:
        TestResult with details: ``agreement_rate``, ``min_agreement``,
        ``n_agreed``, ``n_total``, ``disagreements`` (list of disagreeing indices).

    Raises:
        MltkAssertionError: If agreement_rate < min_agreement (CRITICAL severity).

    Example:
        >>> preds_text = ["cat", "dog", "cat", "bird"]
        >>> preds_image = ["cat", "dog", "dog", "bird"]
        >>> result = assert_cross_modal_consistency(preds_text, preds_image, min_agreement=0.7)
        >>> result.details["agreement_rate"]
        0.75
    """
    # Convert to comparable form
    if isinstance(predictions_a, np.ndarray):
        a = predictions_a.ravel()
    else:
        a = list(predictions_a)

    if isinstance(predictions_b, np.ndarray):
        b = predictions_b.ravel()
    else:
        b = list(predictions_b)

    n_total = len(a)

    if n_total == 0:
        return assert_true(
            False,
            name="multimodal.cross_modal_consistency",
            message="Cannot compute consistency on empty predictions",
            severity=Severity.CRITICAL,
        )

    # Element-wise comparison
    disagreements: list[int] = []
    n_agreed = 0
    for i in range(n_total):
        if isinstance(a, np.ndarray):
            eq = a[i] == b[i]
        else:
            eq = a[i] == b[i]
        if eq:
            n_agreed += 1
        else:
            disagreements.append(i)

    agreement_rate = n_agreed / n_total

    passed = agreement_rate >= min_agreement
    message = (
        f"Cross-modal consistency: {agreement_rate:.4f} >= {min_agreement}"
        if passed
        else f"Cross-modal consistency too low: {agreement_rate:.4f} < {min_agreement}"
    )

    return assert_true(
        passed,
        name="multimodal.cross_modal_consistency",
        message=message,
        severity=Severity.CRITICAL,
        agreement_rate=agreement_rate,
        min_agreement=min_agreement,
        n_agreed=n_agreed,
        n_total=n_total,
        disagreements=disagreements,
    )
