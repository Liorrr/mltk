"""Smart Dataset Importer -- load, normalize, and map any dataset for mltk.

Point mltk at a dataset (HuggingFace Hub id, or a local CSV/JSON/Parquet
file) and get back a normalized :class:`ImportResult` with an
auto-inferred, user-overridable column-role mapping. From there,
:meth:`ImportResult.to_eval_dataset` materializes an
:class:`~mltk.eval.dataset.EvalDataset` ready for
:class:`~mltk.eval.task.EvalTask` or :class:`~mltk.eval.dataset.DatasetRegistry`.

This package is intentionally standalone -- it is not imported by
``mltk``'s top-level ``__init__.py``, so the optional ``datasets``
(HuggingFace) dependency is never loaded on a plain ``import mltk``.
Install it with ``pip install mltk[importer]``.

Quick start::

    from mltk.importer import DatasetImporter

    result = DatasetImporter.load("qa.csv")
    print(result.mapping.preview())
    dataset = result.to_eval_dataset(name="my-qa")
"""

from __future__ import annotations

from mltk.importer.loader import DatasetImporter
from mltk.importer.mapping import auto_map_columns
from mltk.importer.schema import ColumnMapping, ColumnRole, ImportResult

__all__ = [
    "DatasetImporter",
    "auto_map_columns",
    "ColumnMapping",
    "ColumnRole",
    "ImportResult",
]
