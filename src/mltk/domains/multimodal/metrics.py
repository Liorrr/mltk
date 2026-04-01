"""Numerical multimodal metrics -- CLIPScore and edit preservation.

These assertions compute quantitative scores from embeddings or pixel
data.  Unlike the LLM-judge assertions in ``alignment.py`` and
``vlm.py``, no ``judge_fn`` is needed -- metrics are deterministic
functions of numerical inputs.

Two assertions:

1. **CLIPScore** (``assert_clip_score``): Measures image-text alignment
   via cosine similarity.  Two paths:
   - *Zero-dep*: pass pre-computed numpy embeddings.
   - *Model*: pass raw image + text; lazy-imports ``open_clip`` to
     encode both.  Requires ``pip install mltk[clip]``.

2. **Edit preservation** (``assert_edit_preservation``): Measures how
   much an edited image preserves the original.  Two methods:
   - *SSIM*: Structural Similarity Index (requires scikit-image).
   - *pixel_diff*: Mean absolute pixel difference (pure numpy).

Both methods resize to ``max_image_size`` before comparison to cap
memory usage (risk R2 from S78 review).
"""

from __future__ import annotations

import io
from functools import lru_cache
from typing import Any

import numpy as np

from mltk.core.assertion import assert_true, timed_assertion
from mltk.core.result import Severity, TestResult
from mltk.domains.multimodal._image import ImageInput, load_image

__all__ = [
    "assert_clip_score",
    "assert_edit_preservation",
]


# ---------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors.

    Cosine similarity measures the angle between two vectors,
    returning a value in [-1, 1].  Identical directions yield 1.0,
    orthogonal vectors yield 0.0, and opposite directions yield -1.0.

    Returns 0.0 if either vector has zero norm (degenerate case).
    """
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a.ravel(), b.ravel()) / (norm_a * norm_b))


def _load_and_resize(
    source: ImageInput,
    max_size: int,
) -> np.ndarray:
    """Load an image and resize to fit within max_size x max_size.

    Uses Pillow to decode and resize.  Returns a numpy array in
    uint8, shape (H, W) for grayscale or (H, W, 3) for RGB.

    Raises:
        ImportError: If Pillow is not installed.
    """
    try:
        from PIL import Image
    except ImportError:
        raise ImportError(
            "Pillow is required for image comparison. "
            "Install with: pip install mltk[multimodal]"
        ) from None

    raw = load_image(source)
    img = Image.open(io.BytesIO(raw)).convert("RGB")

    # Resize preserving aspect ratio
    w, h = img.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)

    return np.asarray(img, dtype=np.uint8)


# ---------------------------------------------------------------
# CLIPScore -- lazy model loading
# ---------------------------------------------------------------


@lru_cache(maxsize=1)
def _get_clip_model(
    model_name: str = "ViT-B-32",
) -> tuple[Any, Any, Any]:
    """Lazy-load an open-clip model, transforms, and tokenizer.

    Cached so the ~340 MB model is loaded at most once per process.
    Returns ``(model, preprocess, tokenizer)``.

    Raises:
        ImportError: If open-clip-torch is not installed.
    """
    try:
        import open_clip
    except ImportError:
        raise ImportError(
            "open-clip-torch is required for CLIPScore "
            "with raw image/text inputs. Install with: "
            "pip install mltk[clip]"
        ) from None

    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name,
        pretrained="laion2b_s34b_b79k",
    )
    tokenizer = open_clip.get_tokenizer(model_name)
    model.eval()
    return model, preprocess, tokenizer


def _encode_clip(
    image: ImageInput,
    text: str,
    model_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode image and text through a CLIP model.

    Returns L2-normalized embedding vectors as numpy arrays.

    Raises:
        ImportError: If open-clip-torch or torch not installed.
    """
    try:
        import torch
    except ImportError:
        raise ImportError(
            "PyTorch is required for CLIPScore model path. "
            "Install with: pip install torch"
        ) from None

    try:
        from PIL import Image
    except ImportError:
        raise ImportError(
            "Pillow is required for CLIPScore model path. "
            "Install with: pip install mltk[multimodal]"
        ) from None

    model, preprocess, tokenizer = _get_clip_model(model_name)

    # Encode image
    raw = load_image(image)
    pil_img = Image.open(io.BytesIO(raw)).convert("RGB")
    img_tensor = preprocess(pil_img).unsqueeze(0)
    text_tokens = tokenizer([text])

    with torch.no_grad():
        img_features = model.encode_image(img_tensor)
        txt_features = model.encode_text(text_tokens)
        img_features /= img_features.norm(dim=-1, keepdim=True)
        txt_features /= txt_features.norm(dim=-1, keepdim=True)

    return (
        img_features.squeeze(0).cpu().numpy(),
        txt_features.squeeze(0).cpu().numpy(),
    )


# ---------------------------------------------------------------
# Assertions
# ---------------------------------------------------------------


@timed_assertion
def assert_clip_score(
    image: ImageInput | None = None,
    text: str | None = None,
    image_embedding: np.ndarray | None = None,
    text_embedding: np.ndarray | None = None,
    min_score: float = 0.25,
    model_name: str = "ViT-B-32",
) -> TestResult:
    """Assert that image-text CLIPScore meets a minimum threshold.

    CLIPScore measures how well an image and text match in CLIP's
    shared embedding space.  Higher scores mean the image and text
    are more semantically aligned.

    **Two paths** (design decision D2):

    1. **Zero-dep embedding path**: Pass ``image_embedding`` and
       ``text_embedding`` as pre-computed numpy vectors.  mltk
       computes cosine similarity -- no model loading, no GPU.

    2. **Model path**: Pass ``image`` (file/bytes) and ``text``
       (string).  mltk lazy-imports ``open_clip`` to encode both.
       Requires ``pip install mltk[clip]``.

    Why 0.25 default?  CLIP cosine similarities are lower than
    typical embedding spaces.  ViT-B/32 matched pairs score
    0.25-0.35; strong matches reach 0.35+.

    Args:
        image: Image source for model-path encoding.
        text: Text string for model-path encoding.
        image_embedding: Pre-computed image embedding (1-D numpy).
        text_embedding: Pre-computed text embedding (1-D numpy).
        min_score: Minimum cosine similarity to pass (default 0.25).
        model_name: CLIP model variant (default "ViT-B-32").

    Returns:
        TestResult with details: ``score``, ``min_score``,
        ``method`` ("embedding" or "model").

    Raises:
        MltkAssertionError: If score < min_score (CRITICAL).
        ValueError: If neither embedding pair nor image+text given.

    Example:
        >>> emb = np.array([1.0, 0.0, 0.0])
        >>> result = assert_clip_score(
        ...     image_embedding=emb,
        ...     text_embedding=emb,
        ...     min_score=0.9,
        ... )
        >>> result.passed
        True
    """
    score: float
    method: str
    error: str | None = None

    if (
        image_embedding is not None
        and text_embedding is not None
    ):
        # Zero-dep embedding path
        method = "embedding"
        img_emb = np.asarray(image_embedding, dtype=float)
        txt_emb = np.asarray(text_embedding, dtype=float)
        score = _cosine_similarity(img_emb, txt_emb)

    elif image is not None and text is not None:
        # Model path -- lazy-load open-clip
        method = "model"
        try:
            img_emb, txt_emb = _encode_clip(
                image, text, model_name
            )
            score = _cosine_similarity(img_emb, txt_emb)
        except ImportError as exc:
            score = 0.0
            error = str(exc)
        except Exception as exc:
            score = 0.0
            error = f"{type(exc).__name__}: {exc}"

    else:
        raise ValueError(
            "Provide either (image_embedding + text_embedding) "
            "for zero-dep cosine similarity, or (image + text) "
            "for model-based encoding.  Got neither complete pair."
        )

    passed = score >= min_score

    message = (
        f"CLIPScore ({method}): {score:.4f} >= {min_score}"
        if passed
        else f"CLIPScore too low ({method}): "
        f"{score:.4f} < {min_score}"
    )

    return assert_true(
        passed,
        name="multimodal.metrics.clip_score",
        message=message,
        severity=Severity.CRITICAL,
        score=round(score, 4),
        min_score=min_score,
        method=method,
        error=error,
    )


@timed_assertion
def assert_edit_preservation(
    original: ImageInput,
    edited: ImageInput,
    method: str = "ssim",
    threshold: float = 0.8,
    max_image_size: int = 512,
) -> TestResult:
    """Assert that an edited image preserves enough of the original.

    Image editing (inpainting, style transfer, super-resolution)
    should change the target region while preserving the rest.
    This assertion quantifies preservation using either SSIM or
    pixel-level difference.

    **Two methods** (design decision D3):

    1. **SSIM** (``method="ssim"``): Structural Similarity Index
       (Wang et al., 2004).  Considers luminance, contrast, and
       structure.  Requires ``scikit-image``.  Falls back to error
       if not installed.

    2. **pixel_diff** (``method="pixel_diff"``): Pure numpy.
       ``score = 1.0 - mean(|original - edited|) / 255``.
       Simpler but less perceptually meaningful than SSIM.

    Both methods resize images to ``max_image_size`` before
    comparison to cap memory usage.

    Args:
        original: The original image before editing.
        edited: The edited image to compare against original.
        method: Comparison method -- "ssim" or "pixel_diff".
        threshold: Minimum similarity score to pass (default 0.8).
        max_image_size: Maximum dimension for resize (default 512).
            Larger images are scaled down before comparison.

    Returns:
        TestResult with details: ``score``, ``threshold``,
        ``method``, ``max_image_size``.

    Raises:
        MltkAssertionError: If score < threshold (CRITICAL).
        ValueError: If method is not "ssim" or "pixel_diff".

    Example:
        >>> # Identical images should score 1.0
        >>> result = assert_edit_preservation(
        ...     original=b"...",  # same PNG bytes
        ...     edited=b"...",    # same PNG bytes
        ...     method="pixel_diff",
        ... )
    """
    if method not in ("ssim", "pixel_diff"):
        raise ValueError(
            f"method must be 'ssim' or 'pixel_diff', "
            f"got '{method}'"
        )

    error: str | None = None

    try:
        img_orig = _load_and_resize(original, max_image_size)
        img_edit = _load_and_resize(edited, max_image_size)
    except ImportError as exc:
        return assert_true(
            False,
            name="multimodal.metrics.edit_preservation",
            message=str(exc),
            severity=Severity.CRITICAL,
            score=0.0,
            threshold=threshold,
            method=method,
            error=str(exc),
        )

    # Ensure same shape for comparison
    min_h = min(img_orig.shape[0], img_edit.shape[0])
    min_w = min(img_orig.shape[1], img_edit.shape[1])
    img_orig = img_orig[:min_h, :min_w]
    img_edit = img_edit[:min_h, :min_w]

    if method == "ssim":
        try:
            from skimage.metrics import structural_similarity
        except ImportError:
            raise ImportError(
                "scikit-image is required for SSIM. "
                "Install with: pip install mltk[multimodal]"
            ) from None

        # Convert to grayscale for SSIM
        if img_orig.ndim == 3:
            gray_orig = np.mean(
                img_orig.astype(float), axis=2
            )
            gray_edit = np.mean(
                img_edit.astype(float), axis=2
            )
        else:
            gray_orig = img_orig.astype(float)
            gray_edit = img_edit.astype(float)

        score = float(structural_similarity(
            gray_orig,
            gray_edit,
            data_range=255.0,
        ))

    else:
        # pixel_diff: 1.0 - mean(|a - b|) / 255
        diff = np.abs(
            img_orig.astype(float) - img_edit.astype(float)
        )
        score = 1.0 - float(np.mean(diff)) / 255.0

    passed = score >= threshold

    message = (
        f"Edit preservation ({method}): "
        f"{score:.4f} >= {threshold}"
        if passed
        else f"Edit preservation too low ({method}): "
        f"{score:.4f} < {threshold}"
    )

    return assert_true(
        passed,
        name="multimodal.metrics.edit_preservation",
        message=message,
        severity=Severity.CRITICAL,
        score=round(score, 4),
        threshold=threshold,
        method=method,
        max_image_size=max_image_size,
        error=error,
    )
