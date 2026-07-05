"""Multi-source dataset loading for the Smart Dataset Importer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from datasets import load_dataset as _hf_load_dataset

    _DATASETS_AVAILABLE = True
except ImportError:  # pragma: no cover
    _DATASETS_AVAILABLE = False

from mltk.importer.mapping import auto_map_columns
from mltk.importer.schema import ColumnMapping, ColumnRole, ImportResult

_LOCAL_EXTENSIONS = {".csv", ".json", ".parquet", ".pq"}

_LoadedTable = tuple[list[str], dict[str, str], list[dict[str, Any]]]


class DatasetImporter:
    """Loads a dataset from a local file, URL, or HuggingFace Hub id and normalizes it."""

    @staticmethod
    def load(
        source: str,
        *,
        split: str | None = None,
        input_column: str | None = None,
        target_column: str | None = None,
    ) -> ImportResult:
        """Load a dataset from `source` and return a normalized ImportResult.

        Args:
            source: A local file path (``.csv``/``.json``/``.parquet``/``.pq``),
                a URL to such a file, or a HuggingFace Hub dataset id.
            split: Dataset split to load when `source` is a HuggingFace Hub id.
                Defaults to ``"train"``. Ignored for local files and URLs.
            input_column: Optional column name to force-assign the INPUT role,
                applied after auto-mapping via an exclusive override -- any
                other column auto-mapped to INPUT is demoted to UNKNOWN, so
                `input_column` is guaranteed to become the effective
                ``EvalSample.input`` once ``to_eval_dataset()`` is called.
            target_column: Optional column name to force-assign the GOLDEN
                role, applied after auto-mapping via the same exclusive
                override, guaranteeing `target_column` becomes the effective
                ``EvalSample.target``.

        Returns:
            An ImportResult with normalized columns/dtypes/rows and a
            ColumnMapping (auto-inferred, then overridden if requested).

        Raises:
            FileNotFoundError: If `source` looks like a local path but the
                file does not exist.
            ValueError: If a local file has an unrecognized extension.
            NotImplementedError: If `source` is a URL (not yet supported).
            ImportError: If `source` is treated as a HuggingFace Hub id but
                the optional ``datasets`` package is not installed.

        Example:
            >>> result = DatasetImporter.load("qa.csv")  # doctest: +SKIP
            >>> result.mapping.columns_with_role(ColumnRole.INPUT)
            ['question']
        """
        path = Path(source)
        is_url = source.startswith("http://") or source.startswith("https://")
        # A local candidate is either an existing regular file, or a path
        # that doesn't exist yet but has a recognized local extension (so a
        # typo'd .csv still raises FileNotFoundError instead of silently
        # falling through to the HuggingFace branch). An EXISTING directory
        # (e.g. a local dir that happens to share a name with a HF Hub id)
        # is deliberately excluded here so it falls through to the HF branch
        # below rather than raising a bogus "unrecognized extension" error.
        is_local = not is_url and (
            path.is_file()
            or (not path.exists() and path.suffix.lower() in _LOCAL_EXTENSIONS)
        )

        if is_local:
            columns, dtypes, rows = _load_local_file(path)
        elif is_url:
            raise NotImplementedError(
                "URL sources are not yet supported; download the file locally first"
            )
        else:
            if not _DATASETS_AVAILABLE:
                raise ImportError(
                    f"HuggingFace `datasets` package is required to load '{source}'. "
                    "Install with: pip install mltk[importer]"
                )
            columns, dtypes, rows = _load_hf_dataset(source, split=split)

        mapping: ColumnMapping = auto_map_columns(columns, dtypes, rows)
        result = ImportResult(
            source=source, columns=columns, dtypes=dtypes, rows=rows, mapping=mapping
        )

        if input_column is not None:
            result.mapping = result.mapping.override(
                input_column, ColumnRole.INPUT, exclusive=True
            )
        if target_column is not None:
            result.mapping = result.mapping.override(
                target_column, ColumnRole.GOLDEN, exclusive=True
            )

        return result


def _load_local_file(path: Path) -> _LoadedTable:
    """Load a local CSV, JSON, or Parquet file into columns/dtypes/rows.

    Args:
        path: Local filesystem path. Extension determines the loader:
            ``.csv`` -> :func:`pandas.read_csv`;
            ``.json`` -> manual parsing (supports bare array or
            ``{"samples": [...]}`` shapes);
            ``.parquet``/``.pq`` -> :func:`pandas.read_parquet`.

    Returns:
        A ``(columns, dtypes, rows)`` tuple.

    Raises:
        FileNotFoundError: If the path does not exist.
        ValueError: If the file extension is not recognized.
    """
    if not path.exists():
        raise FileNotFoundError(f"Data source not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        return _dataframe_to_table(pd.read_csv(path))
    if suffix in (".parquet", ".pq"):
        return _dataframe_to_table(pd.read_parquet(path))
    if suffix == ".json":
        return _load_json_file(path)
    raise ValueError(f"Unrecognized file extension {suffix!r} for {path}")


def _load_json_file(path: Path) -> _LoadedTable:
    """Load a JSON file, supporting bare-array or ``{"samples": [...]}`` shapes.

    Args:
        path: Local path to a ``.json`` file.

    Returns:
        A ``(columns, dtypes, rows)`` tuple.

    Raises:
        ValueError: If the JSON content is neither a list nor an object with
            a ``samples`` key whose value is itself a list.
    """
    with path.open(encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        records = data
    elif isinstance(data, dict) and isinstance(data.get("samples"), list):
        records = data["samples"]
    else:
        raise ValueError(
            f"Unsupported JSON shape in {path}: expected a list or an object "
            "with a 'samples' key whose value is a list"
        )

    return _dataframe_to_table(pd.DataFrame(records))


def _load_hf_dataset(source: str, *, split: str | None) -> _LoadedTable:
    """Load a dataset from the HuggingFace Hub into columns/dtypes/rows.

    Args:
        source: HuggingFace Hub dataset id.
        split: Split to load; defaults to ``"train"`` when None.

    Returns:
        A ``(columns, dtypes, rows)`` tuple.
    """
    ds = _hf_load_dataset(source, split=split or "train")
    rows: list[dict[str, Any]] = ds.to_list()
    columns: list[str] = list(ds.column_names)
    dtypes = {column: _infer_value_dtype(rows, column) for column in columns}
    return columns, dtypes, rows


def _dataframe_to_table(df: pd.DataFrame) -> _LoadedTable:
    """Convert a DataFrame into the internal ``(columns, dtypes, rows)`` shape."""
    columns = list(df.columns)
    dtypes = {column: _infer_series_dtype(df[column]) for column in columns}
    rows = df.to_dict(orient="records")
    return columns, dtypes, rows


def _infer_series_dtype(series: pd.Series) -> str:
    """Coarsely classify a pandas Series as string/numeric/boolean/other."""
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    if pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series):
        return "string"
    return "other"


def _infer_value_dtype(rows: list[dict[str, Any]], column: str) -> str:
    """Coarsely classify a column's dtype from the first non-null value seen."""
    for row in rows:
        value = row.get(column)
        if value is None:
            continue
        if isinstance(value, bool):
            return "boolean"
        if isinstance(value, str):
            return "string"
        if isinstance(value, (int, float)):
            return "numeric"
        return "other"
    return "other"
