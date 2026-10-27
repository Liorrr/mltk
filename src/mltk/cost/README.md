# mltk.cost

## Overview
LLM token and dollar cost estimation, tracking, and budget assertions.

## Interface
- Pricing: `MODEL_PRICING`, `PRICING_LAST_UPDATED`, `register_pricing(model, input_per_1m, output_per_1m)`, `get_pricing(model)`, `estimate_cost(model, input_tokens, output_tokens)`.
- Tracking: `UsageRecord`, `CostTracker` (`record`, `total_cost_usd`, `total_tokens`, `by_model`, `reset`).
- Assertions: `assert_cost_within(...)`, `assert_token_usage(...)`.

## Invariants
- Prices are per one million tokens as `(input, output)` tuples.
- Runtime overrides registered with `register_pricing` take effect in `get_pricing`.
- Built-in prices are a dated snapshot; check `PRICING_LAST_UPDATED`.

## Side Effects
- `register_pricing` mutates module-level state. No I/O.

## Verification
- Run: `python -m pytest tests/test_cost -q`
