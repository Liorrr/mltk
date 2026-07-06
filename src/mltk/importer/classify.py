"""Deterministic task-type classification for imported datasets.

Given a finalized :class:`~mltk.importer.schema.ColumnMapping`, classifies
the dataset into the task type that downstream suite generation and pytest
emission use to select appropriate mltk assertions. The classifier is
role-presence-based only -- no ML, no data peeking, and no guessing.

Architecture::

    raw rows + columns + dtypes
        |
        v
    mltk.importer.mapping.auto_map_columns() -> ColumnMapping
        |
        v
    mltk.importer.classify.classify_task() -> TaskType
        |
        v
    mltk.importer.suite_gen.build_suite() / codegen.generate_pytest()
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

from mltk.importer.mapping import _tokenize
from mltk.importer.schema import ColumnMapping, ColumnRole

if TYPE_CHECKING:
    from mltk.eval.dataset import EvalDataset

_SUMMARY_GOLDEN_TOKENS = frozenset(
    {"summary", "summaries", "highlights", "tldr", "abstract"}
)


class TaskType(Enum):
    """High-level ML task type inferred from importer column roles."""

    CLASSIFICATION = "classification"
    QA_RAG = "qa_rag"
    SUMMARIZATION = "summarization"
    GENERATION = "generation"
    RETRIEVAL = "retrieval"


def classify_task(
    mapping: ColumnMapping,
    dataset: EvalDataset | None = None,
) -> TaskType:
    """Classify an imported dataset from its semantic column roles.

    Pure and deterministic: the same :class:`ColumnMapping` always
    produces the same :class:`TaskType`. The optional *dataset* is
    intentionally unused in this sprint and is reserved for future
    data-peek heuristics, such as comparing golden-length to
    input-length ratios after an :class:`~mltk.eval.dataset.EvalDataset`
    has been materialized.

    Decision table, in priority order:

    1. ``CONTEXT`` present and ``GOLDEN`` present -> ``QA_RAG``.
    2. ``CONTEXT`` present and no ``GOLDEN`` -> ``RETRIEVAL``.
    3. ``LABEL`` present and no ``GOLDEN`` -> ``CLASSIFICATION``.
    4. ``GOLDEN`` present and a golden column name contains a summary
       keyword as a whole token -> ``SUMMARIZATION``.
    5. ``GOLDEN`` present otherwise -> ``GENERATION``.
    6. No ``CONTEXT``, ``GOLDEN``, or ``LABEL`` -> ``GENERATION``.

    Args:
        mapping: Column-to-role mapping produced by auto-mapping and any
            caller overrides.
        dataset: Reserved for future data-aware classification
            heuristics; accepted for forward compatibility and ignored
            by the current deterministic rules.

    Returns:
        The inferred :class:`TaskType`.

    Example:
        >>> mapping = ColumnMapping(
        ...     roles={
        ...         "question": ColumnRole.INPUT,
        ...         "answer": ColumnRole.GOLDEN,
        ...         "passage": ColumnRole.CONTEXT,
        ...     }
        ... )
        >>> classify_task(mapping)
        <TaskType.QA_RAG: 'qa_rag'>
    """
    context_cols = mapping.columns_with_role(ColumnRole.CONTEXT)
    golden_cols = mapping.columns_with_role(ColumnRole.GOLDEN)
    label_cols = mapping.columns_with_role(ColumnRole.LABEL)

    if context_cols and golden_cols:
        return TaskType.QA_RAG

    if context_cols and not golden_cols:
        return TaskType.RETRIEVAL

    if label_cols and not golden_cols:
        return TaskType.CLASSIFICATION

    if golden_cols and any(_is_summary_golden_column(c) for c in golden_cols):
        return TaskType.SUMMARIZATION

    return TaskType.GENERATION


def _is_summary_golden_column(column: str) -> bool:
    """True if *column* has a whole-token summarization keyword."""
    return bool(set(_tokenize(column)) & _SUMMARY_GOLDEN_TOKENS)
