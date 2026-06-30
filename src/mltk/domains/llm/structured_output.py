"""Structured-output validation assertions for LLM responses."""

from __future__ import annotations

import json
from typing import Any

from mltk.core.assertion import assert_true, timed_assertion
from mltk.core.result import Severity, TestResult

# ---------------------------------------------------------------------------
# Optional dependency guards
# ---------------------------------------------------------------------------

try:
    import jsonschema
    from jsonschema.exceptions import SchemaError as _JsonSchemaSchemaError
    from jsonschema.exceptions import ValidationError as _JsonSchemaError

    _JSONSCHEMA_AVAILABLE = True
except ImportError:  # pragma: no cover
    _JSONSCHEMA_AVAILABLE = False
    _JsonSchemaError = Exception  # type: ignore[assignment,misc]  # dummy, never reached
    _JsonSchemaSchemaError = Exception  # type: ignore[assignment,misc]  # dummy

try:
    import pydantic
    from pydantic import ValidationError as _PydanticValidationError

    _PYDANTIC_AVAILABLE = True
    _PYDANTIC_V2 = int(pydantic.__version__.split(".")[0]) >= 2
except ImportError:  # pragma: no cover
    _PYDANTIC_AVAILABLE = False
    _PYDANTIC_V2 = False
    _PydanticValidationError = Exception  # type: ignore[assignment,misc]  # dummy


# ---------------------------------------------------------------------------
# Assertions
# ---------------------------------------------------------------------------


@timed_assertion
def assert_valid_json(
    text: str,
    *,
    severity: Severity = Severity.CRITICAL,
) -> TestResult:
    """Assert that a string is valid JSON.

    Args:
        text: The string to validate as JSON.
        severity: Severity level; CRITICAL raises MltkAssertionError on failure.

    Returns:
        TestResult with parsing details including ``parsed_type`` and ``text_length``.

    Example:
        >>> assert_valid_json('{"key": "value"}')
    """
    try:
        parsed = json.loads(text)
        parsed_type = type(parsed).__name__
        return assert_true(
            True,
            name="llm.valid_json",
            message=f"Valid JSON ({parsed_type})",
            severity=severity,
            parsed_type=parsed_type,
            text_length=len(text),
        )
    except (json.JSONDecodeError, TypeError) as exc:
        text_str = str(text)
        return assert_true(
            False,
            name="llm.valid_json",
            message=f"Invalid JSON: {exc}",
            severity=severity,
            json_error=str(exc),
            text_preview=text_str[:100] + ("..." if len(text_str) > 100 else ""),
        )


@timed_assertion
def assert_json_schema(
    output: Any,
    schema: dict,
    *,
    severity: Severity = Severity.CRITICAL,
) -> TestResult:
    """Assert that output conforms to a JSON Schema.

    ``output`` may be a Python dict/list or a JSON string; if a string it is
    parsed first -- a parse failure is itself a validation failure.

    Args:
        output: The value to validate. A ``str`` is parsed as JSON first.
        schema: JSON Schema dict (Draft 7 / Draft 2020-12 supported by jsonschema).
        severity: Severity level; CRITICAL raises MltkAssertionError on failure.

    Returns:
        TestResult with ``output_type``, ``validation_error``, and ``schema_path``.

    Raises:
        ImportError: If ``jsonschema`` is not installed.

    Example:
        >>> schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        >>> assert_json_schema({"name": "Alice"}, schema)
    """
    if not _JSONSCHEMA_AVAILABLE:
        raise ImportError(
            "assert_json_schema requires jsonschema: pip install jsonschema"
        )

    if isinstance(output, str):
        try:
            data: Any = json.loads(output)
        except json.JSONDecodeError as exc:
            return assert_true(
                False,
                name="llm.json_schema",
                message=f"Cannot parse output as JSON: {exc}",
                severity=severity,
                json_error=str(exc),
            )
    else:
        data = output

    try:
        jsonschema.validate(data, schema)  # type: ignore[possibly-undefined]
    except _JsonSchemaSchemaError as exc:  # type: ignore[possibly-undefined]
        # The schema itself is malformed (test-author error), not the output.
        return assert_true(
            False,
            name="llm.json_schema",
            message=f"Invalid JSON Schema provided: {getattr(exc, 'message', exc)}",
            severity=severity,
            schema_error=getattr(exc, "message", str(exc)),
            output_type=type(data).__name__,
        )
    except _JsonSchemaError as exc:  # type: ignore[possibly-undefined]
        # json_path was added in jsonschema 4.18; fall back to list(e.path)
        schema_path = getattr(exc, "json_path", None) or list(exc.path)  # type: ignore[union-attr]
        return assert_true(
            False,
            name="llm.json_schema",
            message=f"JSON Schema validation failed: {exc.message}",  # type: ignore[union-attr]
            severity=severity,
            validation_error=exc.message,  # type: ignore[union-attr]
            schema_path=schema_path,
            output_type=type(data).__name__,
        )

    return assert_true(
        True,
        name="llm.json_schema",
        message="Output conforms to JSON Schema",
        severity=severity,
        output_type=type(data).__name__,
    )


@timed_assertion
def assert_pydantic_schema(
    output: Any,
    model: type,
    *,
    severity: Severity = Severity.CRITICAL,
) -> TestResult:
    """Assert that output validates against a Pydantic model.

    ``output`` may be a Python dict or a JSON string; strings are validated
    via ``model_validate_json`` (pydantic v2) or ``parse_raw`` (pydantic v1).

    Args:
        output: The value to validate. A ``str`` is passed as raw JSON.
        model: A Pydantic ``BaseModel`` subclass to validate against.
        severity: Severity level; CRITICAL raises MltkAssertionError on failure.

    Returns:
        TestResult with ``model``, ``error_count``, and ``errors`` details.

    Raises:
        ImportError: If ``pydantic`` is not installed.

    Example:
        >>> from pydantic import BaseModel
        >>> class Person(BaseModel):
        ...     name: str
        ...     age: int
        >>> assert_pydantic_schema({"name": "Alice", "age": 30}, Person)
    """
    if not _PYDANTIC_AVAILABLE:
        raise ImportError(
            "assert_pydantic_schema requires pydantic: pip install pydantic"
        )

    if not (isinstance(model, type) and issubclass(model, pydantic.BaseModel)):
        raise TypeError(
            "assert_pydantic_schema 'model' must be a pydantic BaseModel subclass, "
            f"got {model!r}"
        )

    model_name = getattr(model, "__name__", str(model))
    try:
        if isinstance(output, str):
            if _PYDANTIC_V2:
                model.model_validate_json(output)  # type: ignore[attr-defined]
            else:
                model.parse_raw(output)  # type: ignore[attr-defined]
        else:
            if _PYDANTIC_V2:
                model.model_validate(output)  # type: ignore[attr-defined]
            else:
                model.parse_obj(output)  # type: ignore[attr-defined]
        return assert_true(
            True,
            name="llm.pydantic_schema",
            message=f"Output validates against {model_name}",
            severity=severity,
            model=model_name,
        )
    except _PydanticValidationError as exc:  # type: ignore[possibly-undefined]
        error_count = len(exc.errors())  # type: ignore[union-attr]
        summary = str(exc)
        if len(summary) > 300:
            summary = summary[:300] + "..."
        return assert_true(
            False,
            name="llm.pydantic_schema",
            message=f"Pydantic validation failed ({error_count} error(s)): {summary}",
            severity=severity,
            model=model_name,
            error_count=error_count,
            errors=exc.errors(),  # type: ignore[union-attr]
        )
