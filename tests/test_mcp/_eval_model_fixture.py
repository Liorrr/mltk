"""Trusted callables for mltk_eval model_mode=module tests."""

from __future__ import annotations

# Non-callable attribute used to assert honest TypeError refuse.
not_a_callable = 42


def constant_four(prompt: str) -> str:
    """Ignore prompt; always return ``4`` (for exact_match targets)."""
    _ = prompt
    return "4"


def reverse_model(prompt: str) -> str:
    """Deterministic non-identity model for injection tests."""
    return prompt[::-1]
