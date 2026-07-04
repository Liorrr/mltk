"""Tests for structured-output validation assertions in mltk.domains.llm."""

from __future__ import annotations

import pytest

from mltk.core.assertion import MltkAssertionError
from mltk.core.result import Severity
from mltk.domains.llm.structured_output import (
    assert_json_schema,
    assert_pydantic_schema,
    assert_valid_json,
)

# ---------------------------------------------------------------------------
# assert_valid_json
# ---------------------------------------------------------------------------


class TestAssertValidJson:
    """Tests for assert_valid_json."""

    def test_valid_object_passes(self) -> None:
        """PASS: Well-formed JSON object parses and passes."""
        result = assert_valid_json('{"key": "value", "count": 42}')
        assert result.passed is True
        assert result.details["parsed_type"] == "dict"

    def test_valid_array_passes(self) -> None:
        """PASS: Well-formed JSON array parses and passes."""
        result = assert_valid_json("[1, 2, 3]")
        assert result.passed is True
        assert result.details["parsed_type"] == "list"

    def test_valid_scalar_passes(self) -> None:
        """PASS: JSON number is valid JSON."""
        result = assert_valid_json("42")
        assert result.passed is True

    def test_malformed_json_raises(self) -> None:
        """FAIL CRITICAL: Malformed JSON raises MltkAssertionError (subclass of AssertionError)."""
        with pytest.raises(AssertionError):
            assert_valid_json("{bad json}")

    def test_malformed_json_message_contains_error_text(self) -> None:
        """FAIL: Failure message includes the JSONDecodeError description."""
        with pytest.raises(MltkAssertionError) as exc_info:
            assert_valid_json("not json at all")
        assert "Invalid JSON" in str(exc_info.value)

    def test_warning_severity_returns_failed_result(self) -> None:
        """FAIL WARNING: Non-critical severity returns a failed TestResult without raising."""
        result = assert_valid_json("{bad}", severity=Severity.WARNING)
        assert result.passed is False
        assert result.severity == Severity.WARNING

    def test_empty_string_fails(self) -> None:
        """FAIL: Empty string is not valid JSON."""
        with pytest.raises(AssertionError):
            assert_valid_json("")

    def test_text_length_in_details(self) -> None:
        """PASS: text_length is surfaced in details on success."""
        text = '{"hello": "world"}'
        result = assert_valid_json(text)
        assert result.details["text_length"] == len(text)

    def test_json_error_in_details_on_failure(self) -> None:
        """FAIL WARNING: json_error detail is populated on parse failure."""
        result = assert_valid_json("[unclosed", severity=Severity.WARNING)
        assert "json_error" in result.details
        assert result.details["json_error"]  # non-empty


# ---------------------------------------------------------------------------
# assert_json_schema
# ---------------------------------------------------------------------------

_SIMPLE_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
    },
    "required": ["name"],
}


class TestAssertJsonSchema:
    """Tests for assert_json_schema -- requires jsonschema installed."""

    @classmethod
    def setup_class(cls) -> None:
        """Skip entire class when jsonschema is not installed."""
        pytest.importorskip("jsonschema")

    def test_matching_dict_passes(self) -> None:
        """PASS: Dict satisfying the schema validates successfully."""
        result = assert_json_schema({"name": "Alice", "age": 30}, _SIMPLE_SCHEMA)
        assert result.passed is True
        assert result.details["output_type"] == "dict"

    def test_violating_dict_raises(self) -> None:
        """FAIL CRITICAL: Dict with wrong field type raises MltkAssertionError."""
        with pytest.raises(AssertionError):
            assert_json_schema({"name": 123}, _SIMPLE_SCHEMA)

    def test_json_string_input_passes(self) -> None:
        """PASS: A JSON string is parsed and validated successfully."""
        result = assert_json_schema('{"name": "Bob"}', _SIMPLE_SCHEMA)
        assert result.passed is True

    def test_json_string_violating_raises(self) -> None:
        """FAIL CRITICAL: JSON string with wrong-type field raises."""
        with pytest.raises(AssertionError):
            assert_json_schema('{"name": 999}', _SIMPLE_SCHEMA)

    def test_wrong_type_field_reported_in_details(self) -> None:
        """FAIL: Validation details include validation_error and schema_path."""
        with pytest.raises(MltkAssertionError) as exc_info:
            assert_json_schema({"name": 123}, _SIMPLE_SCHEMA)
        details = exc_info.value.result.details
        assert "validation_error" in details
        assert "schema_path" in details

    def test_missing_required_field_raises(self) -> None:
        """FAIL CRITICAL: Missing required field raises."""
        with pytest.raises(AssertionError):
            assert_json_schema({"age": 5}, _SIMPLE_SCHEMA)

    def test_warning_severity_non_raising(self) -> None:
        """FAIL WARNING: WARNING severity returns failed result without raising."""
        result = assert_json_schema({"name": 999}, _SIMPLE_SCHEMA, severity=Severity.WARNING)
        assert result.passed is False
        assert result.severity == Severity.WARNING

    def test_invalid_json_string_fails_gracefully(self) -> None:
        """FAIL CRITICAL: Malformed JSON string fails with parse error, not crash."""
        with pytest.raises(AssertionError):
            assert_json_schema("not valid json", _SIMPLE_SCHEMA)

    def test_importerror_when_unavailable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """FAIL: ImportError with pip-install hint when _JSONSCHEMA_AVAILABLE is False."""
        import mltk.domains.llm.structured_output as mod

        monkeypatch.setattr(mod, "_JSONSCHEMA_AVAILABLE", False)
        with pytest.raises(ImportError, match="pip install jsonschema"):
            assert_json_schema({"name": "x"}, _SIMPLE_SCHEMA)


# ---------------------------------------------------------------------------
# assert_pydantic_schema
# ---------------------------------------------------------------------------


class TestAssertPydanticSchema:
    """Tests for assert_pydantic_schema -- requires pydantic installed."""

    @classmethod
    def setup_class(cls) -> None:
        """Skip entire class when pydantic is not installed; build the test model."""
        pydantic = pytest.importorskip("pydantic")

        class _PersonModel(pydantic.BaseModel):
            """Minimal model used across all tests in this class."""

            name: str
            age: int

        cls._Person = _PersonModel  # type: ignore[attr-defined]

    def test_valid_dict_passes(self) -> None:
        """PASS: Dict matching model validates and passes."""
        result = assert_pydantic_schema({"name": "Alice", "age": 30}, self._Person)
        assert result.passed is True
        assert result.details["model"] == "_PersonModel"

    def test_missing_required_field_raises(self) -> None:
        """FAIL CRITICAL: Missing required field raises MltkAssertionError."""
        with pytest.raises(AssertionError):
            assert_pydantic_schema({"name": "Alice"}, self._Person)

    def test_json_string_input_passes(self) -> None:
        """PASS: Valid JSON string is parsed and validated."""
        result = assert_pydantic_schema('{"name": "Bob", "age": 25}', self._Person)
        assert result.passed is True

    def test_invalid_json_string_raises(self) -> None:
        """FAIL CRITICAL: JSON string with wrong types raises."""
        with pytest.raises(AssertionError):
            assert_pydantic_schema('{"name": 123, "age": "not-an-int"}', self._Person)

    def test_warning_severity_non_raising(self) -> None:
        """FAIL WARNING: WARNING severity returns failed result without raising."""
        result = assert_pydantic_schema({"age": 30}, self._Person, severity=Severity.WARNING)
        assert result.passed is False
        assert result.severity == Severity.WARNING

    def test_error_count_in_details(self) -> None:
        """FAIL: error_count is surfaced in details on validation failure."""
        with pytest.raises(MltkAssertionError) as exc_info:
            assert_pydantic_schema({"age": 30}, self._Person)
        assert exc_info.value.result.details["error_count"] >= 1

    def test_importerror_when_unavailable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """FAIL: ImportError with pip-install hint when _PYDANTIC_AVAILABLE is False."""
        import mltk.domains.llm.structured_output as mod

        monkeypatch.setattr(mod, "_PYDANTIC_AVAILABLE", False)
        with pytest.raises(ImportError, match="pip install pydantic"):
            assert_pydantic_schema({"name": "x", "age": 1}, self._Person)


# ---------------------------------------------------------------------------
# Review-fix regression: wrappers must not leak raw third-party exceptions
# ---------------------------------------------------------------------------


def test_valid_json_non_str_input_fails_cleanly() -> None:
    """An already-parsed object (non-str) yields a clean fail, not a TypeError crash."""
    with pytest.raises(MltkAssertionError):
        assert_valid_json({"already": "parsed"})  # type: ignore[arg-type]


def test_valid_json_non_str_warning_does_not_raise() -> None:
    """WARNING severity on non-str input returns passed=False instead of crashing."""
    result = assert_valid_json(12345, severity=Severity.WARNING)  # type: ignore[arg-type]
    assert result.passed is False


def test_json_schema_malformed_schema_fails_cleanly() -> None:
    """A malformed *schema* is reported as a clean fail, not a raw SchemaError."""
    pytest.importorskip("jsonschema")
    with pytest.raises(MltkAssertionError) as exc_info:
        assert_json_schema({"x": 1}, {"type": 123})
    assert "schema" in exc_info.value.result.message.lower()


def test_pydantic_schema_rejects_non_basemodel() -> None:
    """A non-BaseModel `model` arg raises a clear TypeError, not AttributeError."""
    pytest.importorskip("pydantic")

    class NotAModel:
        pass

    with pytest.raises(TypeError, match="BaseModel"):
        assert_pydantic_schema({"x": 1}, NotAModel)
