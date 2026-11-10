"""Shared identity-column validation for evaluation file loaders."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def validate_id_column(id_column: str | None) -> None:
    if id_column is not None and not isinstance(id_column, str):
        raise ValueError("id_column must be a string or None")


def extract_sample_id(
    record: Mapping[str, Any], id_column: str | None, *, context: str
) -> str | None:
    """Extract a string ID without coercing values or changing metadata."""
    if id_column is None:
        return None
    value = record.get(id_column)
    if value is not None and not isinstance(value, str):
        raise ValueError(
            f"{context}: identity column {id_column!r} must contain a string "
            f"or null; got {type(value).__name__}"
        )
    return value or None
