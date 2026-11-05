"""Shared parsing for numeric scores returned by judge models."""

from __future__ import annotations

import json
import re

_FLOAT_PATTERN = re.compile(r"(\d+(?:\.\d+)?)")


def parse_judge_score(raw: str) -> float | None:
    """Extract the numeric score from a judge response.

    Structured responses are honored first: a JSON object with a
    ``"score"`` key returns that value, so extra numeric fields
    (reasoning steps, indices) cannot shadow the actual score. An
    explicit ``score`` field is authoritative — if it cannot be
    converted (null, "N/A"), grading failed and None is returned
    rather than scraping an unrelated number from the raw JSON.
    Non-JSON responses fall back to the first number in the text.

    Returns None if no number is found, letting the caller handle the
    failure gracefully.
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
