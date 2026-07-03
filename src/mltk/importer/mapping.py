"""Deterministic column-role auto-mapping for the smart dataset importer.

Given raw column names, coarse dtypes, and a peek at the row data,
infers a :class:`~mltk.importer.schema.ColumnRole` for every column
using name + dtype heuristics -- no ML, no guessing, fully reproducible.
"""

from __future__ import annotations

from typing import Any

from mltk.importer.schema import ColumnMapping, ColumnRole

_INPUT_KEYWORDS = ("input", "prompt", "question", "query")
_CONTEXT_KEYWORDS = ("context", "passage", "document", "chunk", "retrieved")
_GOLDEN_KEYWORDS = (
    "answer",
    "target",
    "expected",
    "golden",
    "reference",
    "output",
    "completion",
    "response",
)
_LABEL_KEYWORDS = ("label", "class", "category")
_METADATA_KEYWORDS = (
    "id",
    "index",
    "source",
    "metadata",
    "split",
    "timestamp",
    "date",
)


def auto_map_columns(
    columns: list[str],
    dtypes: dict[str, str],
    rows: list[dict[str, Any]],
) -> ColumnMapping:
    """Infer a :class:`ColumnRole` for every column via name + dtype heuristics.

    Pure and deterministic: the same ``(columns, dtypes, rows)`` always
    produce the same result. Never guesses wildly -- a column is only
    assigned a specific role when a heuristic confidently matches;
    everything else is left as :attr:`ColumnRole.UNKNOWN` for the user
    to resolve via :meth:`~mltk.importer.schema.ColumnMapping.override`.

    Heuristics are matched case-insensitively against each column name
    (substring or exact match) in this priority order, first match wins
    per column:

    1. ``input``, ``prompt``, ``question``, ``query`` -> ``INPUT``.
    2. ``context``, ``passage``, ``document``, ``chunk``, ``retrieved``
       -> ``CONTEXT``.
    3. ``answer``, ``target``, ``expected``, ``golden``, ``reference``,
       ``output``, ``completion``, ``response`` (only when the column's
       dtype is ``"string"``) -> ``GOLDEN``.
    4. ``label``, ``class``, ``category`` -> ``LABEL`` (regardless of
       dtype).
    5. ``id``, ``index``, ``source``, ``metadata``, ``split``,
       ``timestamp``, ``date`` -> ``METADATA``.
    6. A column named ``text`` becomes ``INPUT`` only if it is the sole
       remaining free-text candidate and no other column already
       matched ``INPUT`` via rule 1. Everything still unmatched becomes
       ``UNKNOWN``.

    Args:
        columns: Column names in original order.
        dtypes: Column name -> coarse dtype string (``"string"``,
            ``"numeric"``, ``"boolean"``, or ``"other"``).
        rows: Row dicts, used only to populate the returned mapping's
            ``samples`` (from the first row, for preview purposes).

    Returns:
        A :class:`ColumnMapping` with one role per column, in original
        column order.

    Example:
        >>> mapping = auto_map_columns(
        ...     columns=["question", "answer", "id"],
        ...     dtypes={
        ...         "question": "string",
        ...         "answer": "string",
        ...         "id": "numeric",
        ...     },
        ...     rows=[{"question": "2+2?", "answer": "4", "id": 1}],
        ... )
        >>> mapping.roles["question"].value
        'input'
        >>> mapping.roles["answer"].value
        'golden'
        >>> mapping.roles["id"].value
        'metadata'
    """
    decisions: dict[str, ColumnRole] = {}

    # Rule 1: direct INPUT keyword match -- highest priority, checked
    # over all columns before anything else so it can never be
    # shadowed by a later, lower-priority rule.
    for column in columns:
        if any(kw in column.lower() for kw in _INPUT_KEYWORDS):
            decisions[column] = ColumnRole.INPUT

    # Rules 2-5, plus text-candidate collection for rule 6.
    text_candidates: list[str] = []
    for column in columns:
        if column in decisions:
            continue
        lowered = column.lower()

        if any(kw in lowered for kw in _CONTEXT_KEYWORDS):
            decisions[column] = ColumnRole.CONTEXT
        elif (
            any(kw in lowered for kw in _GOLDEN_KEYWORDS)
            and dtypes.get(column) == "string"
        ):
            decisions[column] = ColumnRole.GOLDEN
        elif any(kw in lowered for kw in _LABEL_KEYWORDS):
            decisions[column] = ColumnRole.LABEL
        elif any(kw in lowered for kw in _METADATA_KEYWORDS):
            decisions[column] = ColumnRole.METADATA
        elif "text" in lowered:
            text_candidates.append(column)
        else:
            decisions[column] = ColumnRole.UNKNOWN

    # Rule 6: a lone free-text column becomes INPUT only if no column
    # already claimed INPUT via rule 1.
    has_input = ColumnRole.INPUT in decisions.values()
    if not has_input and len(text_candidates) == 1:
        decisions[text_candidates[0]] = ColumnRole.INPUT
    else:
        for column in text_candidates:
            decisions[column] = ColumnRole.UNKNOWN

    roles = {column: decisions[column] for column in columns}
    samples = dict(rows[0]) if rows else {}
    return ColumnMapping(roles=roles, samples=samples)
