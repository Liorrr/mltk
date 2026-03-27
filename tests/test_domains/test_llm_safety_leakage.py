"""Tests for assert_no_system_prompt_leakage in mltk.domains.llm.safety."""

import pytest

from mltk.core.assertion import MltkAssertionError
from mltk.domains.llm.safety import assert_no_system_prompt_leakage


def _safe_model(prompt: str) -> str:
    """Model that never leaks its system prompt."""
    return "I cannot share my system instructions."


def _leaky_model(system_prompt: str):
    """Factory for a model that echoes back its system prompt."""
    def model_fn(prompt: str) -> str:
        if "system" in prompt.lower() or "instruction" in prompt.lower():
            return f"My instructions are: {system_prompt}"
        return "I cannot help with that."
    return model_fn


class TestSystemPromptLeakage:
    """System prompt leakage assertion tests."""

    def test_safe_model_passes(self) -> None:
        """PASS: Model never reveals system prompt content."""
        result = assert_no_system_prompt_leakage(
            model_fn=_safe_model,
            system_prompt="You are a helpful assistant that answers questions.",
        )
        assert result.passed is True
        assert result.details["leaked_count"] == 0

    def test_leaky_model_fails(self) -> None:
        """FAIL: Model reveals system prompt when asked."""
        system_prompt = "You are a helpful assistant that answers questions."
        with pytest.raises(MltkAssertionError):
            assert_no_system_prompt_leakage(
                model_fn=_leaky_model(system_prompt),
                system_prompt=system_prompt,
            )

    def test_custom_payloads(self) -> None:
        """PASS: Custom payloads that don't trigger leakage."""
        result = assert_no_system_prompt_leakage(
            model_fn=_safe_model,
            system_prompt="Xylophone zebra quantum flux capacitor.",
            extraction_payloads=["Tell me a joke", "What is 2+2?"],
        )
        assert result.passed is True
        assert result.details["total_payloads"] == 2

    def test_high_threshold_passes(self) -> None:
        """PASS: Partial overlap below a very high threshold."""
        system_prompt = "Xylophone zebra quantum flux capacitor."
        # Safe model response has zero overlap with this prompt
        result = assert_no_system_prompt_leakage(
            model_fn=_safe_model,
            system_prompt=system_prompt,
            min_overlap_threshold=0.99,
        )
        assert result.passed is True

    def test_empty_system_prompt(self) -> None:
        """EDGE: Empty system prompt always passes."""
        result = assert_no_system_prompt_leakage(
            model_fn=_safe_model,
            system_prompt="",
        )
        assert result.passed is True

    def test_model_error_handled(self) -> None:
        """EDGE: Model that raises exceptions doesn't crash assertion."""
        def error_model(prompt: str) -> str:
            raise RuntimeError("Model crashed")

        result = assert_no_system_prompt_leakage(
            model_fn=error_model,
            system_prompt="Some system prompt.",
        )
        assert result.passed is True  # Errors = not leaking

    def test_details_structure(self) -> None:
        """Verify TestResult details contain expected keys."""
        result = assert_no_system_prompt_leakage(
            model_fn=_safe_model,
            system_prompt="Be helpful and kind.",
        )
        assert "leaked_count" in result.details
        assert "total_payloads" in result.details
        assert "max_overlap" in result.details
        assert "leaked_payloads" in result.details
