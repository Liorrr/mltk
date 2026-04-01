"""Multimodal alignment and consistency assertions.

Multimodal AI models (CLIP, BLIP, LLaVA, GPT-4V) learn to map different
modalities -- images, text, audio -- into a shared embedding space. When
working correctly, semantically related inputs from different modalities
land close together: a photo of a dog and the sentence "a photo of a dog"
should have high cosine similarity.

This module validates four multimodal properties:

1. **Image-text alignment** (embedding-based): Do image and text
   embeddings agree?  Misalignment after fine-tuning or quantization
   leads to wrong captions and hallucinated visual descriptions.

2. **Cross-modal consistency** (embedding-based): When the SAME content
   is processed through different modalities, do predictions agree?

3. **Prompt faithfulness** (LLM-judge): Does a generated image
   faithfully represent the text prompt that produced it?

4. **Image coherence** (LLM-judge): Is an image coherent with its
   surrounding text context in a document or article?

Embedding-based assertions take pre-computed embeddings. LLM-judge
assertions use the ``judge_fn`` pattern from ``mltk.domains.llm.judge``.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable

import numpy as np

from mltk.core.assertion import assert_true, timed_assertion
from mltk.core.result import Severity, TestResult
from mltk.domains.multimodal._image import (
    ImageInput,
    _build_image_prompt,
)

__all__ = [
    "assert_image_text_alignment",
    "assert_cross_modal_consistency",
    "assert_prompt_faithfulness",
    "assert_image_coherence",
]

# ---------------------------------------------------------------
# Score parsing (reused from judge.py pattern)
# ---------------------------------------------------------------

_FLOAT_PATTERN = re.compile(r"(\d+(?:\.\d+)?)")


def _parse_score(raw: str) -> float | None:
    """Extract the first numeric value from a judge response."""
    # Try JSON first
    try:
        data = json.loads(raw)
        if isinstance(data, dict) and "score" in data:
            return float(data["score"])
    except (json.JSONDecodeError, ValueError, TypeError):
        pass
    # Fallback to regex
    match = _FLOAT_PATTERN.search(raw.strip())
    if match:
        return float(match.group(1))
    return None


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


# ---------------------------------------------------------------
# LLM-as-Judge alignment assertions (Sprint A)
# ---------------------------------------------------------------


@timed_assertion
def assert_prompt_faithfulness(
    prompt: str,
    image: ImageInput | None,
    judge_fn: Callable[[str], str],
    min_score: float = 0.7,
    image_description: str | None = None,
) -> TestResult:
    """Assert that an image faithfully represents a text prompt.

    Text-to-image models (DALL-E 3, Stable Diffusion, Flux) generate
    images from text prompts.  "Faithfulness" means the image contains
    what the prompt asked for -- correct objects, attributes, spatial
    relationships, style, and mood.

    The assertion builds an evaluation prompt and passes it to
    ``judge_fn``, which calls any LLM (local or cloud) and returns
    a score.  mltk does NOT own the LLM call.

    Two paths (design decision D3):
    - **VLM path**: Pass the raw image + a VLM-capable judge.
    - **Text path**: Pre-describe the image, pass the description.

    Args:
        prompt: The text prompt that was used to generate the image.
        image: Image source (file path, bytes) or None when using
            image_description.
        judge_fn: Callable that takes a prompt string and returns
            the judge's response containing a score (0.0 to 1.0).
        min_score: Minimum faithfulness score to pass (default 0.7).
        image_description: Optional pre-computed text description
            of the image.  When provided, the ``image`` parameter
            is ignored and Pillow is not required.

    Returns:
        TestResult with details: ``score``, ``min_score``,
        ``prompt_text``.

    Raises:
        MltkAssertionError: If score < min_score (CRITICAL severity).

    Example:
        >>> def mock_judge(p: str) -> str:
        ...     return '{"score": 0.85, "reasoning": "Matches"}'
        >>> result = assert_prompt_faithfulness(
        ...     prompt="A red car on a highway",
        ...     image=None,
        ...     judge_fn=mock_judge,
        ...     image_description="Photo of a red sedan on a road.",
        ... )
        >>> result.passed
        True
    """
    instruction = (
        "You are an impartial evaluation judge.\n\n"
        "## Task\n"
        "Rate how faithfully the image represents the "
        "following text prompt. Consider: correct objects, "
        "attributes (color, size, shape), spatial layout, "
        "style, and mood.\n\n"
        f"## Text Prompt\n{prompt}\n\n"
        "## Scoring\n"
        "Return a JSON object with 'score' (0.0 to 1.0) and "
        "'reasoning' (brief explanation).\n"
        "1.0 = perfect match to the prompt.\n"
        "0.0 = completely unrelated image.\n"
    )

    eval_prompt = _build_image_prompt(
        instruction,
        image=image,
        image_description=image_description,
    )

    score: float | None = None
    error: str | None = None

    try:
        raw = judge_fn(eval_prompt)
        score = _parse_score(str(raw))
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    if score is None:
        score = 0.0
        if error is None:
            error = "Could not parse score from judge response"

    passed = score >= min_score

    message = (
        f"Prompt faithfulness: {score:.4f} >= "
        f"{min_score}"
        if passed
        else f"Prompt faithfulness too low: {score:.4f} < "
        f"{min_score}"
    )

    return assert_true(
        passed,
        name="multimodal.alignment.prompt_faithfulness",
        message=message,
        severity=Severity.CRITICAL,
        score=round(score, 4),
        min_score=min_score,
        prompt_text=prompt,
        error=error,
    )


@timed_assertion
def assert_image_coherence(
    text: str,
    image: ImageInput | None,
    judge_fn: Callable[[str], str],
    min_score: float = 0.7,
    image_description: str | None = None,
) -> TestResult:
    """Assert that an image is coherent with surrounding text context.

    In documents, articles, and presentations, images should be
    contextually relevant to the text they accompany.  An image of
    a sunset in a medical report about bone fractures is incoherent
    -- even if both the image and text are individually valid.

    This differs from ``assert_prompt_faithfulness``: faithfulness
    checks "did the generator follow instructions," while coherence
    checks "does this image belong with this text."

    Args:
        text: The text context that the image accompanies.
        image: Image source (file path, bytes) or None when using
            image_description.
        judge_fn: Callable that takes a prompt string and returns
            the judge's response containing a score (0.0 to 1.0).
        min_score: Minimum coherence score to pass (default 0.7).
        image_description: Optional pre-computed text description
            of the image.

    Returns:
        TestResult with details: ``score``, ``min_score``,
        ``text_context``.

    Raises:
        MltkAssertionError: If score < min_score (CRITICAL severity).

    Example:
        >>> def mock_judge(p: str) -> str:
        ...     return '{"score": 0.9, "reasoning": "Relevant"}'
        >>> result = assert_image_coherence(
        ...     text="The patient's X-ray shows a fracture.",
        ...     image=None,
        ...     judge_fn=mock_judge,
        ...     image_description="An X-ray of a broken femur.",
        ... )
        >>> result.passed
        True
    """
    instruction = (
        "You are an impartial evaluation judge.\n\n"
        "## Task\n"
        "Rate how coherent the image is with the following "
        "text context. A coherent image is contextually "
        "relevant, supports the text's message, and would "
        "make sense to a reader seeing both together.\n\n"
        f"## Text Context\n{text}\n\n"
        "## Scoring\n"
        "Return a JSON object with 'score' (0.0 to 1.0) and "
        "'reasoning' (brief explanation).\n"
        "1.0 = perfectly coherent with the text.\n"
        "0.0 = completely unrelated or contradictory.\n"
    )

    eval_prompt = _build_image_prompt(
        instruction,
        image=image,
        image_description=image_description,
    )

    score: float | None = None
    error: str | None = None

    try:
        raw = judge_fn(eval_prompt)
        score = _parse_score(str(raw))
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    if score is None:
        score = 0.0
        if error is None:
            error = "Could not parse score from judge response"

    passed = score >= min_score

    message = (
        f"Image coherence: {score:.4f} >= "
        f"{min_score}"
        if passed
        else f"Image coherence too low: {score:.4f} < "
        f"{min_score}"
    )

    return assert_true(
        passed,
        name="multimodal.alignment.image_coherence",
        message=message,
        severity=Severity.CRITICAL,
        score=round(score, 4),
        min_score=min_score,
        text_context=text,
        error=error,
    )
