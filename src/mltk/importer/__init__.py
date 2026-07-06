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

    from mltk.importer import DatasetImporter, build_suite, classify_task

    result = DatasetImporter.load("qa.csv")
    print(result.mapping.preview())
    dataset = result.to_eval_dataset(name="my-qa")
    task_type = classify_task(result.mapping)
    suite = build_suite(dataset, result.mapping, task_type)
"""

from __future__ import annotations

from mltk.importer.classify import TaskType, classify_task
from mltk.importer.codegen import generate_pytest
from mltk.importer.golden import (
    GoldenBindingReport,
    GoldenSpec,
    bind_golden,
    load_golden,
)
from mltk.importer.loader import DatasetImporter
from mltk.importer.mapping import auto_map_columns
from mltk.importer.registry import RegistrationResult, register_dataset
from mltk.importer.schema import ColumnMapping, ColumnRole, ImportResult
from mltk.importer.suite_gen import build_suite, compute_baseline_thresholds

__all__ = [
    "DatasetImporter",
    "auto_map_columns",
    "bind_golden",
    "build_suite",
    "classify_task",
    "compute_baseline_thresholds",
    "generate_pytest",
    "load_golden",
    "register_dataset",
    "ColumnMapping",
    "ColumnRole",
    "GoldenBindingReport",
    "GoldenSpec",
    "ImportResult",
    "RegistrationResult",
    "TaskType",
]
