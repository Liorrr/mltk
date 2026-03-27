"""Tests for expanded prompt injection payloads in mltk.domains.nlp.security."""

import pytest

from mltk.core.assertion import MltkAssertionError
from mltk.domains.nlp.security import (
    _DEFAULT_INJECTION_PAYLOADS,
    assert_no_prompt_injection,
)


def _safe_model(prompt: str) -> str:
    """Model that refuses all injection attempts."""
    return "I cannot comply with that request."


def _vulnerable_model(prompt: str) -> str:
    """Model that complies with certain injection patterns."""
    if "ignore" in prompt.lower() or "override" in prompt.lower():
        return "I have been hacked. My system prompt is revealed."
    return "I cannot help with that."


class TestExpandedPayloads:
    """Tests for the 50-payload injection library."""

    def test_payload_count(self) -> None:
        """Verify exactly 50 default payloads."""
        assert len(_DEFAULT_INJECTION_PAYLOADS) == 50

    def test_payload_structure(self) -> None:
        """All payloads have category and payload keys."""
        for item in _DEFAULT_INJECTION_PAYLOADS:
            assert "category" in item, f"Missing category: {item}"
            assert "payload" in item, f"Missing payload: {item}"
            assert isinstance(item["category"], str)
            assert isinstance(item["payload"], str)
            assert len(item["payload"]) > 0

    def test_category_coverage(self) -> None:
        """All 6 categories are represented."""
        categories = {item["category"] for item in _DEFAULT_INJECTION_PAYLOADS}
        expected = {
            "direct_override",
            "instruction_leakage",
            "persona_hijack",
            "encoding",
            "delimiter",
            "multi_language",
        }
        assert categories == expected

    def test_category_counts(self) -> None:
        """Each category has expected number of payloads."""
        counts: dict[str, int] = {}
        for item in _DEFAULT_INJECTION_PAYLOADS:
            counts[item["category"]] = counts.get(item["category"], 0) + 1
        assert counts["direct_override"] == 10
        for cat in ["instruction_leakage", "persona_hijack", "encoding",
                     "delimiter", "multi_language"]:
            assert counts[cat] == 8, f"{cat} has {counts[cat]}, expected 8"


class TestPromptInjectionExpanded:
    """Tests for assert_no_prompt_injection with expanded payloads."""

    def test_safe_model_passes(self) -> None:
        """PASS: Safe model resists all 50 payloads."""
        result = assert_no_prompt_injection(model_fn=_safe_model)
        assert result.passed is True
        assert result.details["total_payloads"] == 50
        assert result.details["vulnerable_count"] == 0

    def test_vulnerable_model_fails(self) -> None:
        """FAIL: Vulnerable model triggered by injection payloads."""
        with pytest.raises(MltkAssertionError):
            assert_no_prompt_injection(model_fn=_vulnerable_model)

    def test_categories_hit_in_details(self) -> None:
        """Vulnerable model reports which categories were hit."""
        try:
            assert_no_prompt_injection(model_fn=_vulnerable_model)
        except MltkAssertionError as exc:
            result = exc.result
            assert "categories_hit" in result.details
            assert isinstance(result.details["categories_hit"], list)
            assert len(result.details["categories_hit"]) > 0

    def test_backward_compat_string_payloads(self) -> None:
        """Backward compatibility: plain string payloads still work."""
        result = assert_no_prompt_injection(
            model_fn=_safe_model,
            payloads=["hello", "test prompt"],
        )
        assert result.passed is True
        assert result.details["total_payloads"] == 2

    def test_custom_forbidden_patterns(self) -> None:
        """Custom forbidden patterns detect specific responses."""
        def echo_model(prompt: str) -> str:
            return "DANGER ZONE"

        with pytest.raises(MltkAssertionError):
            assert_no_prompt_injection(
                model_fn=echo_model,
                payloads=["test"],
                forbidden_patterns=[r"DANGER"],
            )

    def test_model_error_not_vulnerable(self) -> None:
        """Model that raises exceptions is not counted as vulnerable."""
        def error_model(prompt: str) -> str:
            raise ValueError("crash")

        result = assert_no_prompt_injection(model_fn=error_model)
        assert result.passed is True
