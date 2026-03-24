"""Tests for mltk.inference.contract -- API schema validation."""

import pytest

from mltk.core.assertion import MltkAssertionError
from mltk.inference.contract import assert_api_contract


def _echo_func(data: dict) -> dict:
    return {"prediction": data.get("value", 0) * 2, "confidence": 0.95}


def _broken_func(data: dict) -> dict:
    raise RuntimeError("Model failed to load")


class TestAssertApiContract:
    """API contract validation tests."""

    def test_valid_contract(self) -> None:
        """PASS: Output matches expected schema."""
        output_schema = {
            "type": "object",
            "required": ["prediction"],
            "properties": {"prediction": {"type": "number"}},
        }
        result = assert_api_contract(
            _echo_func,
            {"value": 5},
            output_schema=output_schema,
        )
        assert result.passed is True

    def test_output_schema_violation(self) -> None:
        """FAIL: Output doesn't match expected schema."""
        # Expect output to be a string, but it's a dict
        output_schema = {"type": "string"}
        with pytest.raises(MltkAssertionError) as exc:
            assert_api_contract(
                _echo_func, {"value": 1}, output_schema=output_schema
            )
        assert "Contract violations" in str(exc.value)

    def test_input_schema_validation(self) -> None:
        """PASS: Input matches expected schema."""
        input_schema = {
            "type": "object",
            "required": ["value"],
            "properties": {"value": {"type": "number"}},
        }
        result = assert_api_contract(
            _echo_func, {"value": 5}, input_schema=input_schema
        )
        assert result.passed is True

    def test_input_schema_violation(self) -> None:
        """FAIL: Input doesn't match schema (wrong type)."""
        input_schema = {"type": "string"}
        with pytest.raises(MltkAssertionError):
            assert_api_contract(
                _echo_func, {"value": 1}, input_schema=input_schema
            )

    def test_function_raises_error(self) -> None:
        """FAIL: Function raises exception — caught and reported."""
        with pytest.raises(MltkAssertionError) as exc:
            assert_api_contract(_broken_func, {"value": 1})
        assert "RuntimeError" in str(exc.value)

    def test_no_schema_passes(self) -> None:
        """PASS: No schema specified — just verify function runs."""
        result = assert_api_contract(_echo_func, {"value": 1})
        assert result.passed is True
