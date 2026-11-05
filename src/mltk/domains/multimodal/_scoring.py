"""Shared judge-response parsing and vector math for multimodal assertions.

Single home for the helpers that alignment, metrics, and VLM
assertions all need, so a fix in score parsing or similarity math
lands everywhere at once instead of drifting across per-module copies.
"""

from __future__ import annotations

import numpy as np

from mltk.core.judge_parse import parse_judge_score

_parse_score = parse_judge_score


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors.

    Cosine similarity measures the angle between two vectors in
    embedding space, ignoring magnitude. Identical directions yield
    1.0, orthogonal vectors 0.0, and opposite directions -1.0.
    Inputs are flattened, so row vectors and 1-D arrays both work.

    Returns 0.0 if either vector has zero norm (degenerate embedding).
    """
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a.ravel(), b.ravel()) / (norm_a * norm_b))
