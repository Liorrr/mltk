"""API contract testing -- validate inference function input/output schemas.

Catches training-serving skew where preprocessing differs between
training and serving code. Validates request/response formats.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from mltk.core.assertion import assert_true, timed_assertion
from mltk.core.result import Severity, TestResult


def _validate_schema(data: Any, schema: dict[str, Any], label: str) -> list[str]:
    """Validate data against JSON Schema. Returns list of errors."""
    try:
        import jsonschema

        validator = jsonschema.Draft7Validator(schema)
        return [f"{label}: {e.message}" for e in validator.iter_errors(data)]
    except ImportError:
        # Fallback: basic type checking
        return _basic_type_check(data, schema, label)


def _basic_type_check(
    data: Any, schema: dict[str, Any], label: str
) -> list[str]:
    """Basic type checking fallback when jsonschema is not installed."""
    errors = []
    expected_type = schema.get("type")

    type_map = {
        "object": dict,
        "array": list,
        "string": str,
        "number": (int, float),
        "integer": int,
        "boolean": bool,
    }

    if expected_type and expected_type in type_map:
        expected = type_map[expected_type]
        if not isinstance(data, expected):
            errors.append(
                f"{label}: expected type '{expected_type}', "
                f"got '{type(data).__name__}'"
            )

    if expected_type == "object" and isinstance(data, dict):
        required = schema.get("required", [])
        for field in required:
            if field not in data:
                errors.append(f"{label}: missing required field '{field}'")

    return errors


@timed_assertion
def assert_api_contract(
    func: Callable[..., Any],
    input_data: Any,
    input_schema: dict[str, Any] | None = None,
    output_schema: dict[str, Any] | None = None,
) -> TestResult:
    """Assert inference function input/output matches schema.

    Args:
        func: Inference function to test.
        input_data: Sample input to pass to func.
        input_schema: JSON Schema dict for input validation.
        output_schema: JSON Schema dict for output validation.

    Returns:
        TestResult with validation details.
    """
    errors: list[str] = []

    # Validate input
    if input_schema is not None:
        input_errors = _validate_schema(input_data, input_schema, "input")
        errors.extend(input_errors)

    # Call function
    try:
        output = func(input_data)
    except Exception as e:
        return assert_true(
            False,
            name="inference.contract",
            message=f"Function raised {type(e).__name__}: {e}",
            severity=Severity.CRITICAL,
        )

    # Validate output
    if output_schema is not None:
        output_errors = _validate_schema(output, output_schema, "output")
        errors.extend(output_errors)

    passed = len(errors) == 0
    message = (
        "API contract valid"
        if passed
        else f"Contract violations: {'; '.join(errors)}"
    )

    return assert_true(
        passed,
        name="inference.contract",
        message=message,
        severity=Severity.CRITICAL,
        errors=errors,
        has_input_schema=input_schema is not None,
        has_output_schema=output_schema is not None,
    )
