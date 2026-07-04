"""LLM token-cost pricing tables and estimation."""

from __future__ import annotations

PRICING_LAST_UPDATED = "2026-06-30"

# model_id -> (input_usd_per_1m, output_usd_per_1m)
MODEL_PRICING: dict[str, tuple[float, float]] = {
    # Anthropic
    "claude-fable-5": (10.00, 50.00),
    "claude-opus-4-8": (5.00, 25.00),
    "claude-opus-4-7": (5.00, 25.00),
    "claude-opus-4-6": (5.00, 25.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
    # OpenAI
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4.1": (2.00, 8.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "o4-mini": (1.10, 4.40),
    "o3": (2.00, 8.00),
    "gpt-5.4": (2.50, 15.00),
    "gpt-5.4-mini": (0.75, 4.50),
    "gpt-5.5": (5.00, 30.00),
}

# Runtime overrides — never modify MODEL_PRICING; add or replace entries here.
_OVERRIDES: dict[str, tuple[float, float]] = {}


def register_pricing(model: str, input_per_1m: float, output_per_1m: float) -> None:
    """Add or override a model's price at runtime.

    Prices change without notice; use this to update without editing source.

    Args:
        model: Model identifier (e.g. "gpt-4o").
        input_per_1m: Input price in USD per 1 million tokens.
        output_per_1m: Output price in USD per 1 million tokens.

    Example:
        >>> register_pricing("my-model", 1.00, 4.00)
    """
    _OVERRIDES[model] = (input_per_1m, output_per_1m)


def get_pricing(model: str) -> tuple[float, float]:
    """Return (input_usd_per_1m, output_usd_per_1m) for a model.

    Checks runtime overrides first, then the built-in table.

    Args:
        model: Model identifier.

    Returns:
        Tuple of (input_price, output_price) per 1M tokens.

    Raises:
        ValueError: If model is not found in overrides or MODEL_PRICING.

    Example:
        >>> in_p, out_p = get_pricing("gpt-4o")
    """
    if model in _OVERRIDES:
        return _OVERRIDES[model]
    if model in MODEL_PRICING:
        return MODEL_PRICING[model]
    raise ValueError(
        f"Unknown model {model!r}. "
        "Use register_pricing(model, input_per_1m, output_per_1m) to add a custom price, "
        "or verify the model id."
    )


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Compute USD cost for a single LLM call.

    cost = input_tokens / 1e6 * input_price + output_tokens / 1e6 * output_price

    Args:
        model: Model identifier.
        input_tokens: Number of input (prompt) tokens.
        output_tokens: Number of output (completion) tokens.

    Returns:
        Estimated cost in USD.

    Raises:
        ValueError: If model is not found.

    Example:
        >>> estimate_cost("gpt-4o", 1000, 500)
        0.007500000000000001
    """
    if input_tokens < 0 or output_tokens < 0:
        raise ValueError(
            "token counts must be non-negative, got "
            f"input_tokens={input_tokens}, output_tokens={output_tokens}"
        )
    in_price, out_price = get_pricing(model)
    return input_tokens / 1_000_000 * in_price + output_tokens / 1_000_000 * out_price
