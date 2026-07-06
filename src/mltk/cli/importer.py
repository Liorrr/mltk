"""CLI entrypoint for importing datasets into mltk pytest suites."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Annotated

import typer


def import_dataset(
    source: Annotated[
        str,
        typer.Argument(help="Local file path or HuggingFace dataset id."),
    ],
    split: Annotated[
        str | None,
        typer.Option(
            "--split",
            help="Dataset split for HuggingFace sources.",
        ),
    ] = None,
    input_column: Annotated[
        str | None,
        typer.Option(
            "--input-column",
            help="Column to force-map as the eval input.",
        ),
    ] = None,
    target_column: Annotated[
        str | None,
        typer.Option(
            "--target-column",
            help="Column to force-map as the eval target.",
        ),
    ] = None,
    name: Annotated[
        str | None,
        typer.Option(
            "--name",
            help="Dataset name for the generated eval dataset.",
        ),
    ] = None,
    output: Annotated[
        str | None,
        typer.Option(
            "--output",
            help="Path for the generated pytest scaffold.",
        ),
    ] = None,
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            help="Overwrite an existing generated pytest file.",
        ),
    ] = False,
    no_emit: Annotated[
        bool,
        typer.Option(
            "--no-emit",
            help="Preview and build the suite without writing pytest.",
        ),
    ] = False,
) -> None:
    """Import a dataset, preview mapping, build a suite, and emit pytest."""
    try:
        from mltk.importer.classify import classify_task
        from mltk.importer.codegen import generate_pytest
        from mltk.importer.loader import DatasetImporter
        from mltk.importer.schema import ColumnRole
        from mltk.importer.suite_gen import build_suite
    except ImportError as err:
        print(  # noqa: T201
            "The mltk importer requires optional dependencies. "
            "Install them with: pip install mltk[importer]"
        )
        raise typer.Exit(1) from err

    try:
        result = DatasetImporter.load(
            source,
            split=split,
            input_column=input_column,
            target_column=target_column,
        )
    except (
        FileNotFoundError,
        ImportError,
        NotImplementedError,
        ValueError,
    ) as err:
        print(f"Import failed: {err}")  # noqa: T201
        raise typer.Exit(1) from err

    print(result.mapping.preview())  # noqa: T201

    problems = result.mapping.validate()
    for problem in problems:
        print(f"warning: {problem}")  # noqa: T201

    if not result.mapping.columns_with_role(ColumnRole.INPUT):
        raise typer.Exit(1)

    task_type = classify_task(result.mapping)
    task_type_value = _task_type_value(task_type)
    print(f"task type: {task_type_value}")  # noqa: T201

    dataset_name = name or _source_stem(source)
    eval_dataset = result.to_eval_dataset(name=dataset_name)
    suite = build_suite(eval_dataset, result.mapping, task_type)
    print(  # noqa: T201
        f"suite: {_suite_name(suite, dataset_name)} "
        f"({_suite_assertion_count(suite)} registered assertions)"
    )

    if no_emit:
        return

    output_path = (
        Path(output)
        if output is not None
        else _default_output_path(source, task_type_value)
    )
    if output_path.exists() and not force:
        print(  # noqa: T201
            f"Output file already exists: {output_path}. "
            "Use --force to overwrite."
        )
        raise typer.Exit(1)

    generate_pytest(
        result,
        task_type,
        dataset_name=dataset_name,
        output_path=str(output_path),
    )
    print(f"pytest file written: {output_path}")  # noqa: T201
    print(  # noqa: T201
        "Tier-2 tests are skipped until predict_fn is wired. "
        "Set MLTK_PREDICT_FN=module:callable or edit the fixture."
    )


def _source_stem(source: str) -> str:
    """Return a human-readable stem for a file path or dataset id."""
    stripped = source.rstrip("/\\")
    if not stripped:
        return "dataset"
    stem = Path(stripped).stem
    return stem or "dataset"


def _task_type_value(task_type: object) -> str:
    """Return the stable string value for a TaskType-like object."""
    value = getattr(task_type, "value", task_type)
    return str(value)


def _default_output_path(source: str, task_type_value: str) -> Path:
    """Return the default committable pytest scaffold path."""
    source_part = _sanitize_filename_part(_source_stem(source))
    task_part = _sanitize_filename_part(task_type_value)
    return Path(f"test_{source_part}_{task_part}.py")


def _sanitize_filename_part(value: str) -> str:
    """Sanitize a value for use in generated test filenames."""
    sanitized = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()
    return sanitized or "dataset"


def _suite_name(suite: object, fallback: str) -> str:
    """Return a suite name without requiring a concrete suite type."""
    name = getattr(suite, "name", None)
    return str(name) if name else fallback


def _suite_assertion_count(suite: object) -> int:
    """Return the number of registered assertions in a suite-like object."""
    try:
        return len(suite)  # type: ignore[arg-type]
    except TypeError:
        pending = getattr(suite, "_pending", None)
        if pending is not None:
            return len(pending)
        return 0
