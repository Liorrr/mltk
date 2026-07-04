"""Tests for mltk.cost.pricing -- pricing table lookups and cost estimation."""

from __future__ import annotations

import pytest

from mltk.cost import pricing
from mltk.cost.pricing import (
    MODEL_PRICING,
    PRICING_LAST_UPDATED,
    estimate_cost,
    get_pricing,
    register_pricing,
)


@pytest.fixture(autouse=True)
def _clean_overrides():
    """Isolate _OVERRIDES between tests so register_pricing calls don't leak."""
    pricing._OVERRIDES.clear()
    yield
    pricing._OVERRIDES.clear()


class TestPricingTable:
    """Sanity checks on the built-in MODEL_PRICING table."""

    def test_pricing_last_updated(self) -> None:
        """Table carries a datestamp."""
        assert PRICING_LAST_UPDATED == "2026-06-30"

    def test_anthropic_models_present(self) -> None:
        """All six Anthropic models exist in the table."""
        expected = {
            "claude-fable-5",
            "claude-opus-4-8",
            "claude-opus-4-7",
            "claude-opus-4-6",
            "claude-sonnet-4-6",
            "claude-haiku-4-5",
        }
        assert expected.issubset(MODEL_PRICING.keys())

    def test_openai_models_present(self) -> None:
        """All nine OpenAI models exist in the table."""
        expected = {
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4.1",
            "gpt-4.1-mini",
            "o4-mini",
            "o3",
            "gpt-5.4",
            "gpt-5.4-mini",
            "gpt-5.5",
        }
        assert expected.issubset(MODEL_PRICING.keys())


class TestGetPricing:
    """get_pricing() — lookup and error handling."""

    def test_known_anthropic_model(self) -> None:
        """Returns correct prices for claude-sonnet-4-6."""
        in_p, out_p = get_pricing("claude-sonnet-4-6")
        assert in_p == pytest.approx(3.00)
        assert out_p == pytest.approx(15.00)

    def test_known_openai_model(self) -> None:
        """Returns correct prices for gpt-4o."""
        in_p, out_p = get_pricing("gpt-4o")
        assert in_p == pytest.approx(2.50)
        assert out_p == pytest.approx(10.00)

    def test_unknown_model_raises_value_error(self) -> None:
        """Raises ValueError mentioning register_pricing for an unknown model."""
        with pytest.raises(ValueError, match="register_pricing"):
            get_pricing("totally-unknown-model-xyz")

    def test_error_message_includes_model_name(self) -> None:
        """ValueError message names the bad model id."""
        with pytest.raises(ValueError, match="my-bad-model"):
            get_pricing("my-bad-model")


class TestRegisterPricing:
    """register_pricing() — runtime overrides."""

    def test_override_existing_model(self) -> None:
        """Overriding an existing model changes what get_pricing returns."""
        # Original price for claude-haiku-4-5 is (1.00, 5.00)
        register_pricing("claude-haiku-4-5", 2.00, 10.00)
        in_p, out_p = get_pricing("claude-haiku-4-5")
        assert in_p == pytest.approx(2.00)
        assert out_p == pytest.approx(10.00)

    def test_override_does_not_mutate_model_pricing(self) -> None:
        """_OVERRIDES stays separate; MODEL_PRICING is unchanged after override."""
        register_pricing("claude-haiku-4-5", 99.00, 99.00)
        # Built-in table still has original values
        assert MODEL_PRICING["claude-haiku-4-5"] == pytest.approx((1.00, 5.00))

    def test_register_brand_new_model(self) -> None:
        """A previously unknown model becomes queryable after registration."""
        register_pricing("my-custom-llm", 0.50, 2.00)
        in_p, out_p = get_pricing("my-custom-llm")
        assert in_p == pytest.approx(0.50)
        assert out_p == pytest.approx(2.00)

    def test_overrides_checked_before_table(self) -> None:
        """When a model exists in both _OVERRIDES and MODEL_PRICING, the override wins."""
        register_pricing("gpt-4o", 99.00, 99.00)
        in_p, out_p = get_pricing("gpt-4o")
        assert in_p == pytest.approx(99.00)


class TestEstimateCost:
    """estimate_cost() — cost arithmetic."""

    def test_anthropic_model_exact_cost(self) -> None:
        """claude-sonnet-4-6: 1 000 in + 500 out → $0.0105.

        Calculation: 1000/1e6 * 3.00 + 500/1e6 * 15.00 = 0.003 + 0.0075 = 0.0105
        """
        cost = estimate_cost("claude-sonnet-4-6", 1_000, 500)
        assert cost == pytest.approx(0.0105)

    def test_openai_model_exact_cost(self) -> None:
        """gpt-4o: 2 000 in + 1 000 out → $0.015.

        Calculation: 2000/1e6 * 2.50 + 1000/1e6 * 10.00 = 0.005 + 0.010 = 0.015
        """
        cost = estimate_cost("gpt-4o", 2_000, 1_000)
        assert cost == pytest.approx(0.015)

    def test_another_anthropic_model(self) -> None:
        """claude-opus-4-8: 1 000 in + 1 000 out → $0.030.

        Calculation: 1000/1e6 * 5.00 + 1000/1e6 * 25.00 = 0.005 + 0.025 = 0.030
        """
        cost = estimate_cost("claude-opus-4-8", 1_000, 1_000)
        assert cost == pytest.approx(0.030)

    def test_zero_tokens_returns_zero(self) -> None:
        """Zero input and output tokens costs exactly $0.00."""
        cost = estimate_cost("gpt-4o-mini", 0, 0)
        assert cost == pytest.approx(0.0)

    def test_unknown_model_raises_value_error(self) -> None:
        """Propagates ValueError from get_pricing on unknown model."""
        with pytest.raises(ValueError, match="register_pricing"):
            estimate_cost("ghost-model-9000", 100, 100)

    def test_estimate_after_override(self) -> None:
        """estimate_cost reflects a registered override immediately.

        Original claude-haiku-4-5: (1.00, 5.00)
        Override to: (2.00, 10.00)
        1000 in + 1000 out: 1000/1e6*2 + 1000/1e6*10 = 0.002 + 0.010 = 0.012
        """
        register_pricing("claude-haiku-4-5", 2.00, 10.00)
        cost = estimate_cost("claude-haiku-4-5", 1_000, 1_000)
        assert cost == pytest.approx(0.012)

    def test_estimate_new_model_after_registration(self) -> None:
        """Newly registered model is immediately usable in estimate_cost.

        custom-llm-v2: (1.00, 4.00)
        500 in + 250 out: 500/1e6*1 + 250/1e6*4 = 0.0005 + 0.001 = 0.0015
        """
        register_pricing("custom-llm-v2", 1.00, 4.00)
        cost = estimate_cost("custom-llm-v2", 500, 250)
        assert cost == pytest.approx(0.0015)


def test_negative_tokens_raise_value_error() -> None:
    """Negative token counts are a misconfiguration -> ValueError, not negative cost."""
    with pytest.raises(ValueError, match="non-negative"):
        estimate_cost("gpt-4o", -1_000, 500)
    with pytest.raises(ValueError, match="non-negative"):
        estimate_cost("gpt-4o", 100, -5)
