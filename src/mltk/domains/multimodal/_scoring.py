"""Shared judge-response parsing and vector math for multimodal assertions.

Single home for the helpers that alignment, metrics, and VLM
assertions all need, so a fix in score parsing or similarity math
lands everywhere at once instead of drifting across per-module copies.
"""

from __future__ import annotations

import json
import re

import numpy as np

_FLOAT_PATTERN = re.compile(r"(\d+(?:\.\d+)?)")


def _parse_score(raw: str) -> float | None:
    """Extract the numeric score from a judge response.

    A JSON object with a ``"score"`` key is honored first, so extra
    numeric fields cannot shadow the actual score. An explicit
    ``score`` field is authoritative — if it cannot be converted
    (null, "N/A"), grading failed and None is returned rather than
    scraping an unrelated number from the raw JSON. Non-JSON
    responses fall back to the first number in the text. Returns
    None if no number is found.
    """
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        data = None
    if isinstance(data, dict) and "score" in data:
        try:
            return float(data["score"])
        except (ValueError, TypeError):
            return None
    match = _FLOAT_PATTERN.search(raw.strip())
    if match:
        return float(match.group(1))
    return None


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
