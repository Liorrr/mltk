"""Pytest code generation for imported datasets.

Turns an :class:`~mltk.importer.schema.ImportResult` into a
self-contained pytest scaffold that can be committed and tightened over
time.  Tier 1 tests run immediately against imported rows; Tier 2 tests
are wired to a single ``predict_fn`` fixture so users can un-skip model
quality checks with one edit or one environment variable.
"""

from __future__ import annotations

import ast
import os
import re
import textwrap
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from mltk.importer.schema import ColumnRole

if TYPE_CHECKING:
    from mltk.importer.classify import TaskType
    from mltk.importer.schema import ImportResult

__all__ = ["generate_pytest"]

_QUALITY_KEYS = (
    "min_samples",
    "min_target_coverage",
    "max_duplicate_rate",
    "min_categories",
)


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------


def _sanitize_name(raw: str) -> str:
    """Turn an arbitrary source stem into a valid Python identifier."""
    cleaned = re.sub(r"[^a-zA-Z0-9_]", "_", raw)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    if not cleaned:
        cleaned = "dataset"
    if not re.match(r"[a-zA-Z_]", cleaned):
        cleaned = f"dataset_{cleaned}"
    return cleaned.lower()


def _source_stem(source: str) -> str:
    """Return the final path/id component without a file extension."""
    trimmed = str(source).rstrip("/\\")
    if not trimmed:
        return "dataset"
    filename = re.split(r"[\\/]", trimmed)[-1]
    stem = os.path.splitext(filename)[0]
    return stem or filename or "dataset"


def _default_dataset_name(source: str) -> str:
    """Return the default emitted dataset name for *source*."""
    return _sanitize_name(_source_stem(source))


def _escape_docstring_text(value: str) -> str:
    """Escape arbitrary text for inclusion inside a triple-quoted string."""
    escaped = value.encode("unicode_escape").decode("ascii")
    return escaped.replace('"""', '\\"\\"\\"')


def _task_value(task_type: TaskType) -> str:
    """Normalize a TaskType enum member or raw task string."""
    return str(getattr(task_type, "value", task_type)).lower()


def _is_missing_value(value: Any) -> bool:
    """Return True for None, NaN, or blank string values."""
    if value is None:
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _columns_have_missing(df: pd.DataFrame, columns: list[str]) -> bool:
    """Return True if any requested column is absent or has missing data."""
    for column in columns:
        if column not in df.columns:
            return True
        for value in df[column].tolist():
            if _is_missing_value(value):
                return True
    return False


def _dedupe(values: list[str]) -> list[str]:
    """Preserve-order de-duplication."""
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _required_no_null_columns(import_result: ImportResult) -> list[str]:
    """Return effective INPUT plus all GOLDEN columns."""
    input_cols = import_result.mapping.columns_with_role(ColumnRole.INPUT)
    golden_cols = import_result.mapping.columns_with_role(ColumnRole.GOLDEN)
    return _dedupe(input_cols[:1] + golden_cols)


def _format_dict_literal(
    values: dict[str, Any],
    *,
    item_indent: int,
    closing_indent: int,
) -> str:
    """Render a deterministic multi-line Python dict literal."""
    item_prefix = " " * item_indent
    closing_prefix = " " * closing_indent
    lines = ["{"]
    for key, value in values.items():
        lines.append(f"{item_prefix}{key!r}: {value!r},")
    lines.append(f"{closing_prefix}}}")
    return "\n".join(lines)


def _format_list_literal(
    values: list[str],
    *,
    item_indent: int,
    closing_indent: int,
) -> str:
    """Render a deterministic multi-line Python list literal."""
    item_prefix = " " * item_indent
    closing_prefix = " " * closing_indent
    lines = ["["]
    for value in values:
        lines.append(f"{item_prefix}{value!r},")
    lines.append(f"{closing_prefix}]")
    return "\n".join(lines)


def _load_baseline_thresholds(dataset: Any) -> dict[str, Any]:
    """Compute baseline thresholds via the suite_gen contract."""
    from mltk.importer.suite_gen import compute_baseline_thresholds

    thresholds = compute_baseline_thresholds(dataset)
    return {key: thresholds[key] for key in _QUALITY_KEYS}


def _format_threshold(value: Any) -> str:
    """Return a stable Python literal for a threshold value."""
    if isinstance(value, float):
        return repr(float(value))
    if isinstance(value, int):
        return repr(int(value))
    return repr(value)


# ------------------------------------------------------------------
# Code block builders
# ------------------------------------------------------------------


def _build_header(source: str) -> str:
    """Return the file-level docstring and imports prelude."""
    source_text = _escape_docstring_text(source)
    return textwrap.dedent(f'''\
        """Auto-generated by mltk import {source_text}.

        This is a pytest scaffold you can commit and edit freely.
        Tier 1 tests run now against imported rows.
        Tier 2 tests are skipped until the predict_fn fixture is wired.
        Assertions that need fields the import cannot infer are left as
        comments with a short "requires ..." note.
        """

        from __future__ import annotations

        import importlib
        import os

        import pandas as pd
        import pytest
    ''')


def _build_imports(task: str, *, include_no_nulls: bool) -> str:
    """Return assertion imports for the emitted file."""
    data_import = (
        "from mltk.data import assert_no_nulls, assert_schema"
        if include_no_nulls
        else "from mltk.data import assert_schema"
    )
    imports = [
        data_import,
        "from mltk.eval.dataset import assert_dataset_quality",
        "from mltk.importer import DatasetImporter",
    ]

    if task in {"qa_rag", "retrieval"}:
        imports.append(
            textwrap.dedent("""\
                from mltk.domains.llm.rag import (
                    assert_answer_relevancy,
                    assert_context_relevancy,
                    assert_faithfulness,
                )""").rstrip()
            if task == "qa_rag"
            else "from mltk.domains.llm.rag import assert_context_relevancy"
        )
    elif task == "classification":
        imports.append("from mltk.model.metrics import assert_metric")
    elif task == "summarization":
        imports.append(
            textwrap.dedent("""\
                from mltk.domains.llm.summarization import (
                    assert_summary_compression,
                    assert_summary_coverage,
                    assert_summary_faithfulness,
                )""").rstrip()
        )
    elif task == "generation":
        imports.append(
            "from mltk.domains.llm.text_quality import assert_output_format"
        )
    else:
        raise ValueError(f"Unsupported task_type: {task!r}")

    return "\n\n" + "\n".join(imports)


def _build_fixtures(
    source: str,
    dataset_name: str,
    *,
    load_kwargs: dict[str, str] | None = None,
) -> str:
    """Return importer/dataframe/dataset/predict_fn fixtures."""
    load_kwargs_lines = ""
    if load_kwargs:
        load_kwargs_lines = "\n" + "\n".join(
            f"                {key}={value!r},"
            for key, value in sorted(load_kwargs.items())
        )

    return textwrap.dedent(f'''\n
        @pytest.fixture(scope="module")
        def import_result():
            """Load the imported dataset."""
            return DatasetImporter.load(
                {source!r},{load_kwargs_lines}
            )


        @pytest.fixture(scope="module")
        def df(import_result):
            """Rebuild the imported rows as a pandas DataFrame."""
            return pd.DataFrame(import_result.rows)


        @pytest.fixture(scope="module")
        def dataset(import_result):
            """Materialize the imported rows as an EvalDataset."""
            return import_result.to_eval_dataset(
                name={dataset_name!r},
                version="0.1.0",
            )


        @pytest.fixture(scope="module")
        def predict_fn():
            """Return a callable that maps prompt -> prediction."""
            spec = os.environ.get("MLTK_PREDICT_FN")
            if spec:
                if ":" not in spec:
                    raise ValueError(
                        "MLTK_PREDICT_FN must be 'module:callable', "
                        f"got {{spec!r}}"
                    )
                module_name, callable_name = spec.split(":", 1)
                module = importlib.import_module(module_name)
                fn = getattr(module, callable_name)
                if not callable(fn):
                    raise TypeError(
                        "MLTK_PREDICT_FN must point to a callable"
                    )
                return fn
            pytest.skip(
                "Set MLTK_PREDICT_FN=module:callable or edit this fixture "
                "to return your model's predict function "
                "(prompt -> prediction)."
            )
    ''')


def _build_data_sanity_class(
    schema: dict[str, str],
    no_null_columns: list[str],
    include_no_nulls: bool,
    thresholds: dict[str, Any],
) -> str:
    """Return the Tier 1 data sanity test class."""
    schema_literal = _format_dict_literal(
        schema, item_indent=16, closing_indent=12
    )
    lines = [
        "",
        "",
        "class TestDataSanity:",
        '    """Tier 1 tests that run against the imported rows."""',
        "",
        "    def test_schema(self, df):",
        "        result = assert_schema(",
        "            df,",
        f"            expected={schema_literal},",
        "        )",
        "        assert result.passed",
    ]

    if include_no_nulls:
        columns_literal = _format_list_literal(
            no_null_columns, item_indent=16, closing_indent=12
        )
        lines.extend(
            [
                "",
                "    def test_no_nulls(self, df):",
                "        result = assert_no_nulls(",
                "            df,",
                f"            columns={columns_literal},",
                "        )",
                "        assert result.passed",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "    # test_no_nulls omitted: baseline has missing values in",
                "    # the INPUT or GOLDEN columns imported from the source.",
            ]
        )

    quality_kwargs = _build_quality_kwargs(thresholds)
    lines.extend(
        [
            "",
            "    def test_dataset_quality(self, dataset):",
            "        result = assert_dataset_quality(",
            "            dataset,",
            quality_kwargs,
            "        )",
            "        assert result.passed",
        ]
    )

    return "\n".join(lines) + "\n"


def _build_quality_kwargs(thresholds: dict[str, Any]) -> str:
    """Return explicit dataset-quality kwargs with baseline comments."""
    lines: list[str] = []
    for key in _QUALITY_KEYS:
        value = thresholds[key]
        if key == "min_categories" and value is None:
            continue
        lines.append(
            " " * 12
            + f"{key}={_format_threshold(value)}, "
            + "# baseline from import \u2014 tighten as needed"
        )
    return "\n".join(lines)


def _build_model_quality_class(task: str) -> str:
    """Return the Tier 2 task-specific test class."""
    body_builders = {
        "qa_rag": _build_qa_rag_tests,
        "retrieval": _build_retrieval_tests,
        "classification": _build_classification_tests,
        "summarization": _build_summarization_tests,
        "generation": _build_generation_tests,
    }
    try:
        body = body_builders[task]()
    except KeyError as exc:
        raise ValueError(f"Unsupported task_type: {task!r}") from exc

    indented_body = textwrap.indent(body.rstrip(), "    ")
    return (
        "\n\nclass TestModelQuality:\n"
        '    """Tier 2 tests skipped until predict_fn is wired."""\n\n'
        f"{indented_body}\n"
    )


def _build_qa_rag_tests() -> str:
    """Return QA/RAG model-quality tests."""
    return textwrap.dedent("""\
            def test_faithfulness(self, predict_fn, dataset):
                for sample in dataset.samples:
                    context = sample.metadata.get("context")
                    if not context:
                        continue
                    pred = predict_fn(sample.input)
                    result = assert_faithfulness(pred, context)
                    assert result.passed

            def test_answer_relevancy(self, predict_fn, dataset):
                for sample in dataset.samples:
                    if not sample.input:
                        continue
                    pred = predict_fn(sample.input)
                    result = assert_answer_relevancy(sample.input, pred)
                    assert result.passed

            def test_context_relevancy(self, predict_fn, dataset):
                for sample in dataset.samples:
                    context = sample.metadata.get("context")
                    if not sample.input or not context:
                        continue
                    result = assert_context_relevancy(sample.input, context)
                    assert result.passed
    """)


def _build_retrieval_tests() -> str:
    """Return retrieval model-quality tests."""
    return textwrap.dedent("""\
            def test_context_relevancy(self, predict_fn, dataset):
                for sample in dataset.samples:
                    context = sample.metadata.get("context")
                    if not sample.input or not context:
                        continue
                    result = assert_context_relevancy(sample.input, context)
                    assert result.passed
    """)


def _build_classification_tests() -> str:
    """Return classification model-quality tests."""
    return textwrap.dedent("""\
            def test_metric(self, predict_fn, dataset):
                y_true = []
                y_pred = []
                for sample in dataset.samples:
                    if sample.target is None:
                        continue
                    y_true.append(sample.target)
                    y_pred.append(predict_fn(sample.input))
                if not y_true:
                    return
                result = assert_metric(
                    y_true,
                    y_pred,
                    metric="accuracy",
                    threshold=0.8,
                )
                assert result.passed

            # assert_no_bias requires protected-attribute columns.
            # assert_slice_performance requires slice key columns.
    """)


def _build_summarization_tests() -> str:
    """Return summarization model-quality tests."""
    return textwrap.dedent("""\
            def test_summary_coverage(self, predict_fn, dataset):
                for sample in dataset.samples:
                    source = sample.metadata.get("context") or sample.input
                    if isinstance(source, list):
                        source = " ".join(part for part in source if part)
                    if not source:
                        continue
                    pred = predict_fn(sample.input)
                    result = assert_summary_coverage(source, pred)
                    assert result.passed

            def test_summary_compression(self, predict_fn, dataset):
                for sample in dataset.samples:
                    source = sample.metadata.get("context") or sample.input
                    if isinstance(source, list):
                        source = " ".join(part for part in source if part)
                    if not source:
                        continue
                    pred = predict_fn(sample.input)
                    result = assert_summary_compression(source, pred)
                    assert result.passed

            def test_summary_faithfulness(self, predict_fn, dataset):
                for sample in dataset.samples:
                    source = sample.metadata.get("context") or sample.input
                    if isinstance(source, list):
                        source = " ".join(part for part in source if part)
                    if not source:
                        continue
                    pred = predict_fn(sample.input)
                    result = assert_summary_faithfulness(source, pred)
                    assert result.passed
    """)


def _build_generation_tests() -> str:
    """Return generation model-quality tests."""
    return textwrap.dedent("""\
            def test_output_format(self, predict_fn, dataset):
                for sample in dataset.samples:
                    if not sample.input:
                        continue
                    pred = predict_fn(sample.input)
                    result = assert_output_format(
                        pred,
                        pattern=r"\\S",
                        description="non-empty string",
                    )
                    assert result.passed

            # assert_json_schema requires a task-specific schema that
            # cannot be inferred from imported rows.
            # assert_json_schema(pred, schema={...})
    """)


def _validate_syntax(code: str) -> None:
    """Parse *code* with ast.parse and raise ValueError with context."""
    try:
        ast.parse(code, filename="<mltk-import-codegen>")
    except SyntaxError as exc:
        context = exc.text.strip() if exc.text else ""
        raise ValueError(
            "Generated pytest file is invalid: "
            f"{exc.msg} at line {exc.lineno}: {context}"
        ) from exc


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------


def generate_pytest(
    import_result: ImportResult,
    task_type: TaskType,
    *,
    dataset_name: str | None = None,
    output_path: str | os.PathLike[str] | None = None,
    load_kwargs: dict[str, str] | None = None,
) -> str:
    """Generate a self-contained pytest file for an imported dataset.

    Args:
        import_result: Normalized imported dataset metadata and rows.
        task_type: Import task type controlling Tier 2 assertion shape.
        dataset_name: Optional EvalDataset name.  Defaults to a
            sanitized stem of ``import_result.source``.
        output_path: Optional path to write the generated file to.
            Parent directories are created automatically.
        load_kwargs: Optional keyword arguments to preserve when the
            generated fixture reloads ``import_result.source``.

    Returns:
        Generated Python source ending with a newline.

    Raises:
        ValueError: If task_type is unsupported or generated syntax is
            invalid.
    """
    resolved_dataset_name = (
        dataset_name
        if dataset_name is not None
        else _default_dataset_name(import_result.source)
    )
    task = _task_value(task_type)

    df = pd.DataFrame(import_result.rows)
    schema = {column: str(dtype) for column, dtype in df.dtypes.items()}
    no_null_columns = _required_no_null_columns(import_result)
    include_no_nulls = bool(no_null_columns) and not _columns_have_missing(
        df, no_null_columns
    )
    dataset = import_result.to_eval_dataset(
        name=resolved_dataset_name,
        version="0.1.0",
    )
    thresholds = _load_baseline_thresholds(dataset)

    parts = [
        _build_header(import_result.source),
        _build_imports(task, include_no_nulls=include_no_nulls),
        _build_fixtures(
            import_result.source,
            resolved_dataset_name,
            load_kwargs=load_kwargs,
        ),
        _build_data_sanity_class(
            schema,
            no_null_columns,
            include_no_nulls,
            thresholds,
        ),
        _build_model_quality_class(task),
    ]
    code = "\n".join(part.rstrip() for part in parts).rstrip() + "\n"

    _validate_syntax(code)

    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(code, encoding="utf-8")

    return code
