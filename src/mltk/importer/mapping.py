"""Deterministic column-role auto-mapping for the smart dataset importer.

Given raw column names, coarse dtypes, and a peek at the row data,
infers a :class:`~mltk.importer.schema.ColumnRole` for every column
using name + dtype heuristics -- no ML, no guessing, fully reproducible.
"""

from __future__ import annotations

import re
from typing import Any

from mltk.importer.schema import ColumnMapping, ColumnRole

_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_TOKEN_SPLIT = re.compile(r"[^a-z0-9]+")

_INPUT_KEYWORDS = frozenset({"input", "prompt", "question", "query"})
_CONTEXT_KEYWORDS = frozenset(
    {"context", "passage", "document", "chunk", "retrieved"}
)
_GOLDEN_KEYWORDS = frozenset(
    {
        "answer",
        "target",
        "expected",
        "golden",
        "reference",
        "output",
        "completion",
        "response",
    }
)
_LABEL_KEYWORDS = frozenset({"label", "class", "category"})
# Rule (a): checked against only the LAST token of a column name, before
# any other rule runs -- a suffix like "_id" or "_date" marks a column as
# bookkeeping metadata even if an earlier token also looks like an INPUT
# or CONTEXT keyword (e.g. "question_id", "document_id").
_METADATA_SUFFIX_TOKENS = frozenset(
    {"id", "idx", "index", "uid", "uuid", "timestamp", "date", "split"}
)
# Rule (f): checked against ANY token, as the last-resort keyword match.
_METADATA_KEYWORDS = frozenset(
    {"id", "idx", "index", "source", "metadata", "split", "timestamp", "date"}
)


def _tokenize(column: str) -> list[str]:
    """Split a column name into lowercase word tokens.

    Splits on runs of non-alphanumeric characters (``_``, ``-``, spaces,
    etc.) and on lowercase-to-uppercase camelCase boundaries, then
    lowercases every token. Used so keyword matching is whole-token
    (set membership), never a raw substring check.

    Args:
        column: Raw column name.

    Returns:
        List of lowercase tokens, in order. Empty list for a column name
        with no alphanumeric characters (e.g. ``""``).

    Example:
        >>> _tokenize("question_id")
        ['question', 'id']
        >>> _tokenize("InputTokens")
        ['input', 'tokens']
    """
    with_boundaries = _CAMEL_BOUNDARY.sub("_", column)
    lowered = with_boundaries.lower()
    return [token for token in _TOKEN_SPLIT.split(lowered) if token]


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

    Each column name is tokenized (see :func:`_tokenize`) and matched
    against keyword sets by WHOLE TOKEN, never by substring. Per column,
    the first matching rule wins, in this priority order:

    1. Metadata-suffix rule: if the column's LAST token is one of
       ``id``, ``idx``, ``index``, ``uid``, ``uuid``, ``timestamp``,
       ``date``, ``split`` -> ``METADATA``. This runs before every
       other rule, so ``question_id`` and ``document_id`` are
       ``METADATA`` even though ``question``/``document`` would
       otherwise match a lower rule.
    2. Any token in ``input``, ``prompt``, ``question``, ``query`` AND
       dtype is ``"string"`` -> INPUT candidate.
    3. Any token in ``context``, ``passage``, ``document``, ``chunk``,
       ``retrieved`` -> ``CONTEXT``.
    4. Any token in ``answer``, ``target``, ``expected``, ``golden``,
       ``reference``, ``output``, ``completion``, ``response`` AND
       (dtype is ``"string"`` OR the whole lowered column name is
       itself exactly one of those keywords AND dtype is ``"numeric"``)
       -> ``GOLDEN``. This lets a numeric ``answer`` column (e.g. a
       math-QA dataset where pandas infers a numeric dtype) still map
       to GOLDEN, while a compound numeric column like
       ``output_tokens`` does not.
    5. Any token in ``label``, ``class``, ``category`` -> ``LABEL``
       (regardless of dtype).
    6. Any token in ``id``, ``idx``, ``index``, ``source``,
       ``metadata``, ``split``, ``timestamp``, ``date`` -> ``METADATA``.
    7. Otherwise -> pending ``UNKNOWN``, subject to the text fallback
       below.

    Two global steps run after every column has a preliminary decision:

    - Single-INPUT invariant: if more than one column matched rule 2,
      only the FIRST (in original column order) keeps ``INPUT`` --
      every other INPUT candidate is demoted to ``UNKNOWN`` so it is
      surfaced by :meth:`~mltk.importer.schema.ColumnMapping.preview`/
      :meth:`~mltk.importer.schema.ColumnMapping.validate` and passed
      through to sample metadata by
      :meth:`~mltk.importer.schema.ImportResult.to_eval_dataset`,
      instead of being silently dropped.
    - Text fallback: among columns still ``UNKNOWN`` after the above,
      collect those with token ``text`` AND dtype ``"string"``. If no
      column has role ``INPUT`` and there is exactly one such
      candidate, it becomes ``INPUT``. Otherwise all such candidates
      stay ``UNKNOWN``.

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
    input_candidates: list[str] = []
    text_pending: list[str] = []

    for column in columns:
        tokens = _tokenize(column)
        token_set = set(tokens)
        dtype = dtypes.get(column)
        last_token = tokens[-1] if tokens else None

        if last_token in _METADATA_SUFFIX_TOKENS:
            decisions[column] = ColumnRole.METADATA
            continue

        if token_set & _INPUT_KEYWORDS and dtype == "string":
            decisions[column] = ColumnRole.INPUT
            input_candidates.append(column)
            continue

        if token_set & _CONTEXT_KEYWORDS:
            decisions[column] = ColumnRole.CONTEXT
            continue

        if token_set & _GOLDEN_KEYWORDS and (
            dtype == "string"
            or (column.lower() in _GOLDEN_KEYWORDS and dtype == "numeric")
        ):
            decisions[column] = ColumnRole.GOLDEN
            continue

        if token_set & _LABEL_KEYWORDS:
            decisions[column] = ColumnRole.LABEL
            continue

        if token_set & _METADATA_KEYWORDS:
            decisions[column] = ColumnRole.METADATA
            continue

        decisions[column] = ColumnRole.UNKNOWN
        if "text" in token_set and dtype == "string":
            text_pending.append(column)

    # Single-INPUT invariant: only the first INPUT candidate survives.
    if len(input_candidates) > 1:
        for column in input_candidates[1:]:
            decisions[column] = ColumnRole.UNKNOWN

    # Text fallback: only fires when no column has claimed INPUT at all.
    has_input = bool(input_candidates)
    if not has_input and len(text_pending) == 1:
        decisions[text_pending[0]] = ColumnRole.INPUT

    roles = {column: decisions[column] for column in columns}
    samples = dict(rows[0]) if rows else {}
    return ColumnMapping(roles=roles, samples=samples)
