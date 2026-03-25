"""Tests for mltk.inference.contract -- API schema validation.

Contract tests ensure ML model serving endpoints accept the right inputs
and produce the right outputs. Without these, a model can silently return
wrong-shaped responses that downstream consumers misinterpret, or accept
malformed inputs that cause silent prediction errors.
"""

import pytest

from mltk.core.assertion import MltkAssertionError
from mltk.inference.contract import assert_api_contract


def _echo_func(data: dict) -> dict:
    """Simulates a well-behaved model endpoint that doubles the input value."""
    return {"prediction": data.get("value", 0) * 2, "confidence": 0.95}


def _broken_func(data: dict) -> dict:
    """Simulates a model endpoint that fails to load or crashes at runtime."""
    raise RuntimeError("Model failed to load")


class TestAssertApiContract:
    """API contract validation tests.

    Validates that assert_api_contract correctly enforces input/output
    JSON schemas, catches runtime exceptions, and passes cleanly when
    no schema constraints are specified.
    """

    def test_valid_contract(self) -> None:
        """PASS: Model output matches the expected JSON schema.

        WHY: In production, downstream services parse model responses by schema.
        If the response is missing a required field (e.g., "prediction"), the
        consumer crashes or silently uses a default value. Contract tests catch
        this before deployment.
        Expected: result.passed is True.
        """
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
        """FAIL: Model output does not match expected schema (dict vs string).

        WHY: A model update changed the output format from a string to a dict.
        Without contract testing, this would silently break all API consumers.
        Expected: MltkAssertionError raised with "Contract violations" message.
        """
        # Expect output to be a string, but it's a dict
        output_schema = {"type": "string"}
        with pytest.raises(MltkAssertionError) as exc:
            assert_api_contract(
                _echo_func, {"value": 1}, output_schema=output_schema
            )
        assert "Contract violations" in str(exc.value)

    def test_input_schema_validation(self) -> None:
        """PASS: Input data matches the expected request schema.

        WHY: Validates that callers are sending correctly-shaped requests.
        Catching bad inputs at test time prevents silent prediction errors
        where the model fills in defaults for missing fields.
        Expected: result.passed is True.
        """
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
        """FAIL: Input data does not match schema (dict given, string expected).

        WHY: If the API expects a string but receives a dict, the model may
        silently coerce the input or produce garbage predictions.
        Expected: MltkAssertionError raised.
        """
        input_schema = {"type": "string"}
        with pytest.raises(MltkAssertionError):
            assert_api_contract(
                _echo_func, {"value": 1}, input_schema=input_schema
            )

    def test_function_raises_error(self) -> None:
        """FAIL: Model function raises RuntimeError -- caught and wrapped.

        WHY: In production, model loading failures or OOM errors should be
        caught gracefully and reported as contract failures, not as unhandled
        exceptions that crash the serving infrastructure.
        Expected: MltkAssertionError raised with "RuntimeError" in message.
        """
        with pytest.raises(MltkAssertionError) as exc:
            assert_api_contract(_broken_func, {"value": 1})
        assert "RuntimeError" in str(exc.value)

    def test_no_schema_passes(self) -> None:
        """PASS: No schema specified -- just verify the function runs without error.

        WHY: Minimal smoke test. Even without schema constraints, you want to
        know the model endpoint doesn't crash on a representative input.
        Expected: result.passed is True.
        """
        result = assert_api_contract(_echo_func, {"value": 1})
        assert result.passed is True
