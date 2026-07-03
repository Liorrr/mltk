"""Normalized schema types for the smart dataset importer.

Defines the semantic column roles a source dataset column can play, the
(auto-inferred, user-overridable) column-to-role mapping, and the
normalized ``ImportResult`` produced by loading a dataset from any
source (CSV, JSON, Parquet, HuggingFace Hub).

Architecture::

    raw rows + columns + dtypes
        |
        v
    mltk.importer.mapping.auto_map_columns() -> ColumnMapping
        |
        v
    ImportResult (source, columns, dtypes, rows, mapping)
        |
        v
    ImportResult.to_eval_dataset() -> EvalDataset
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mltk.eval.dataset import EvalDataset


class ColumnRole(Enum):
    """Semantic role a source column plays in the normalized dataset."""

    INPUT = "input"  # the prompt/question fed to the model
    GOLDEN = "golden"  # expected answer / reference -> EvalSample.target
    CONTEXT = "context"  # retrieval context / passage (RAG) -> metadata["context"]
    LABEL = "label"  # classification label -> metadata["label"]
    METADATA = "metadata"  # passthrough, arbitrary metadata
    IGNORE = "ignore"  # explicitly excluded from the normalized dataset
    UNKNOWN = "unknown"  # could not be inferred -- always surfaced, never guessed


def _is_missing(value: Any) -> bool:
    """True if *value* should be treated as an absent cell.

    Covers ``None``, NaN floats (e.g. from pandas), and blank/whitespace
    strings.

    Args:
        value: A single cell value from an :class:`ImportResult` row.

    Returns:
        True if the cell should be treated as missing data.
    """
    if value is None:
        return True
    if isinstance(value, float) and value != value:  # NaN != NaN
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    return False


def _stringify(value: Any) -> str | None:
    """Convert a cell value to ``str``, or ``None`` if it's missing.

    Prevents missing cells (``None``/NaN/blank) from being coerced into
    the literal string ``"None"`` by a bare ``str(value)`` call.

    Args:
        value: A single cell value from an :class:`ImportResult` row.

    Returns:
        ``str(value)``, or ``None`` if the cell is missing.
    """
    return None if _is_missing(value) else str(value)


@dataclass
class ColumnMapping:
    """User-overridable mapping from source columns to semantic roles.

    Produced by :func:`mltk.importer.mapping.auto_map_columns` and then
    optionally adjusted by the caller via :meth:`override` before
    materializing an :class:`~mltk.eval.dataset.EvalDataset` with
    :meth:`ImportResult.to_eval_dataset`.

    Args:
        roles: Mapping from source column name to inferred/assigned role,
            in original column order.
        samples: Mapping from source column name to one example value,
            used to render :meth:`preview`.

    Example:
        >>> mapping = ColumnMapping(
        ...     roles={
        ...         "question": ColumnRole.INPUT,
        ...         "answer": ColumnRole.GOLDEN,
        ...     },
        ...     samples={"question": "2+2?", "answer": "4"},
        ... )
        >>> mapping.columns_with_role(ColumnRole.INPUT)
        ['question']
    """

    roles: dict[str, ColumnRole]
    samples: dict[str, Any] = field(default_factory=dict)

    def preview(self) -> str:
        """Render a human-readable column | role | sample preview table.

        Columns whose role is :attr:`ColumnRole.UNKNOWN` are flagged
        with a trailing ``(needs review)`` marker so they are never
        silently hidden -- the caller can spot exactly which columns
        still need a manual :meth:`override` before the mapping is used.

        Returns:
            A newline-joined table with one header row and one row per
            column, formatted as ``"column | role | sample"``.

        Example:
            >>> m = ColumnMapping(
            ...     roles={"q": ColumnRole.INPUT, "x": ColumnRole.UNKNOWN},
            ...     samples={"q": "2+2?", "x": "??"},
            ... )
            >>> print(m.preview())
            column | role | sample
            q | input | 2+2?
            x | unknown | ?? (needs review)
        """
        lines = ["column | role | sample"]
        for column, role in self.roles.items():
            sample = self.samples.get(column, "")
            marker = " (needs review)" if role is ColumnRole.UNKNOWN else ""
            lines.append(f"{column} | {role.value} | {sample}{marker}")
        return "\n".join(lines)

    def override(self, column: str, role: ColumnRole) -> ColumnMapping:
        """Return a new mapping with *column* reassigned to *role*.

        Does not mutate ``self`` -- the ``roles`` dict is copied so the
        original mapping remains valid for comparison/undo.

        Args:
            column: Source column name to reassign.
            role: New role to assign to *column*.

        Returns:
            A new :class:`ColumnMapping` with the override applied.

        Raises:
            ValueError: If *column* is not present in ``self.roles``.

        Example:
            >>> m = ColumnMapping(roles={"x": ColumnRole.UNKNOWN})
            >>> m2 = m.override("x", ColumnRole.INPUT)
            >>> m2.roles["x"] is ColumnRole.INPUT
            True
            >>> m.roles["x"] is ColumnRole.UNKNOWN
            True
        """
        if column not in self.roles:
            raise ValueError(
                f"Column '{column}' not found in mapping. "
                f"Known columns: {list(self.roles.keys())}"
            )
        new_roles = dict(self.roles)
        new_roles[column] = role
        return ColumnMapping(roles=new_roles, samples=dict(self.samples))

    def columns_with_role(self, role: ColumnRole) -> list[str]:
        """Return all column names currently assigned *role*.

        Args:
            role: The role to filter by.

        Returns:
            Column names with role *role*, in original column order.

        Example:
            >>> m = ColumnMapping(
            ...     roles={"a": ColumnRole.LABEL, "b": ColumnRole.LABEL},
            ... )
            >>> m.columns_with_role(ColumnRole.LABEL)
            ['a', 'b']
        """
        return [column for column, r in self.roles.items() if r is role]

    def validate(self) -> list[str]:
        """Check the mapping for problems that need user attention.

        Does not raise -- callers decide what to do with the returned
        problem list (e.g. block, warn, or prompt for overrides).

        Returns:
            Human-readable problem strings. Empty list means the
            mapping is valid. Checks performed:

            - zero columns have role ``INPUT`` (every dataset needs
              at least one).
            - any ``UNKNOWN`` roles are present (lists which columns
              -- this is reported, never silently resolved).

        Example:
            >>> m = ColumnMapping(roles={"x": ColumnRole.UNKNOWN})
            >>> problems = m.validate()
            >>> len(problems)
            2
        """
        problems: list[str] = []
        if not self.columns_with_role(ColumnRole.INPUT):
            problems.append(
                "no column has role INPUT -- every dataset needs at "
                "least one input column"
            )
        unknown_columns = self.columns_with_role(ColumnRole.UNKNOWN)
        if unknown_columns:
            problems.append(
                "columns with UNKNOWN role need review: "
                + ", ".join(unknown_columns)
            )
        return problems


@dataclass
class ImportResult:
    """Normalized output of loading a dataset from any source.

    Args:
        source: Human-readable origin (file path, URL, or HF dataset id).
        columns: Column names in original order.
        dtypes: Column name -> coarse dtype string: ``"string"``,
            ``"numeric"``, ``"boolean"``, or ``"other"``.
        rows: The actual data, one dict per row, keyed by column name.
        mapping: The (auto-inferred, possibly user-overridden) column
            mapping.

    Example:
        >>> result = ImportResult(
        ...     source="qa.csv",
        ...     columns=["question", "answer"],
        ...     dtypes={"question": "string", "answer": "string"},
        ...     rows=[{"question": "2+2?", "answer": "4"}],
        ...     mapping=ColumnMapping(
        ...         roles={
        ...             "question": ColumnRole.INPUT,
        ...             "answer": ColumnRole.GOLDEN,
        ...         },
        ...     ),
        ... )
        >>> ds = result.to_eval_dataset(name="qa", version="1.0.0")
        >>> ds.sample_count
        1
    """

    source: str
    columns: list[str]
    dtypes: dict[str, str]
    rows: list[dict[str, Any]]
    mapping: ColumnMapping

    def to_eval_dataset(
        self,
        name: str,
        version: str = "0.1.0",
        mapping: ColumnMapping | None = None,
    ) -> EvalDataset:
        """Materialize an :class:`~mltk.eval.dataset.EvalDataset` from this result.

        Uses *mapping* if given, else ``self.mapping``.

        Construction rules:

        - ``INPUT``: the first column with role ``INPUT`` becomes
          ``EvalSample.input`` (``str(value)``, or ``""`` if the cell is
          missing/NaN/empty -- ``EvalSample.input`` is non-optional).
          Raises if there is no ``INPUT`` column at all.
        - ``GOLDEN``: the first column with role ``GOLDEN`` becomes
          ``EvalSample.target`` (``str(value)``, or ``None`` if the cell
          is missing/NaN/empty). Any additional ``GOLDEN`` columns go
          into ``metadata["references"]`` as a list of ``str | None``
          (missing cells become ``None``, never the literal string
          ``"None"``).
        - ``CONTEXT``: all ``CONTEXT`` columns go into
          ``metadata["context"]`` -- a single ``str | None`` if there is
          exactly one such column, else a list of ``str | None``
          preserving column order. Missing cells become ``None``.
        - ``LABEL``: all ``LABEL`` columns go into ``metadata["label"]``
          -- a single value if there is exactly one such column, else a
          dict of ``{column: value}``.
        - ``METADATA`` and ``UNKNOWN`` columns pass through into sample
          metadata keyed by their own column name. ``IGNORE`` columns
          are dropped entirely.

        Args:
            name: Name for the resulting dataset.
            version: Semver version string (default ``"0.1.0"``).
            mapping: Column mapping to use instead of ``self.mapping``.

        Returns:
            A constructed :class:`~mltk.eval.dataset.EvalDataset` with
            one sample per row in ``self.rows``, in original row order,
            and a card stamped with ``source=self.source``.

        Raises:
            ValueError: If the effective mapping has zero columns with
                role ``INPUT``.

        Example:
            >>> result = ImportResult(
            ...     source="qa.csv",
            ...     columns=["question", "answer"],
            ...     dtypes={"question": "string", "answer": "string"},
            ...     rows=[{"question": "2+2?", "answer": "4"}],
            ...     mapping=ColumnMapping(
            ...         roles={
            ...             "question": ColumnRole.INPUT,
            ...             "answer": ColumnRole.GOLDEN,
            ...         },
            ...     ),
            ... )
            >>> ds = result.to_eval_dataset("qa")
            >>> ds.samples[0].input
            '2+2?'
            >>> ds.samples[0].target
            '4'
        """
        from mltk.eval._types import EvalSample
        from mltk.eval.dataset import DatasetCard, EvalDataset

        effective = mapping if mapping is not None else self.mapping

        input_cols = effective.columns_with_role(ColumnRole.INPUT)
        if not input_cols:
            raise ValueError(
                "Cannot build an EvalDataset: no column has role "
                "INPUT. Assign at least one column to ColumnRole.INPUT "
                "via ColumnMapping.override() before calling "
                "to_eval_dataset()."
            )
        input_col = input_cols[0]

        golden_cols = effective.columns_with_role(ColumnRole.GOLDEN)
        golden_col = golden_cols[0] if golden_cols else None
        extra_golden_cols = golden_cols[1:]

        context_cols = effective.columns_with_role(ColumnRole.CONTEXT)
        label_cols = effective.columns_with_role(ColumnRole.LABEL)
        passthrough_roles = (ColumnRole.METADATA, ColumnRole.UNKNOWN)
        passthrough_cols = [
            column
            for column, role in effective.roles.items()
            if role in passthrough_roles
        ]

        samples: list[EvalSample] = []
        for row in self.rows:
            input_value = _stringify(row.get(input_col)) or ""

            target_value: str | None = None
            if golden_col is not None:
                target_value = _stringify(row.get(golden_col))

            metadata: dict[str, Any] = {}

            if extra_golden_cols:
                metadata["references"] = [
                    _stringify(row.get(c)) for c in extra_golden_cols
                ]

            if context_cols:
                if len(context_cols) == 1:
                    metadata["context"] = _stringify(
                        row.get(context_cols[0])
                    )
                else:
                    metadata["context"] = [
                        _stringify(row.get(c)) for c in context_cols
                    ]

            if label_cols:
                if len(label_cols) == 1:
                    metadata["label"] = row.get(label_cols[0])
                else:
                    metadata["label"] = {
                        c: row.get(c) for c in label_cols
                    }

            for c in passthrough_cols:
                metadata[c] = row.get(c)

            samples.append(
                EvalSample(
                    input=input_value,
                    target=target_value,
                    metadata=metadata,
                )
            )

        card = DatasetCard(source=self.source)
        return EvalDataset(
            name=name, version=version, samples=samples, card=card
        )
