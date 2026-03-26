"""Shared helpers for report and compliance generators."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _load_results(results_path: str | Path) -> list[dict[str, Any]]:
    """Load test results from a JSON file.

    The file may contain either a list of result dicts or a dict with a
    ``"results"`` key (as produced by some mltk collectors).

    Args:
        results_path: Path to a JSON file containing mltk test results.

    Returns:
        A list of result dicts.

    Raises:
        ValueError: If the JSON structure is neither a list nor a dict with
            a ``"results"`` key.
        FileNotFoundError: If *results_path* does not exist.
    """
    path = Path(results_path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict) and "results" in raw:
        return raw["results"]
    raise ValueError(
        f"Cannot parse results from {path}. "
        "Expected a JSON list or a dict with a 'results' key."
    )
