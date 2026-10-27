# mltk.inference

## Overview
Inference-time assertions for latency, cold start, throughput, and API response contracts.

## Interface
- `assert_latency(...)`, `assert_cold_start(...)`, `assert_throughput(...)` (minimum requests per second), `assert_api_contract(...)` (response schema check).

## Invariants
- All assertions are wrapped with `timed_assertion` and return result objects.
- Schema checks collect all mismatches rather than stopping at the first.

## Side Effects
- Invokes the caller-supplied model or predict function repeatedly; no file or network I/O of its own.

## Verification
- Run: `python -m pytest tests/test_inference -q`
