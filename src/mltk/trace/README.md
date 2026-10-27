# mltk.trace

## Overview
Canonical, clearance-tiered capture record for model/agent runs. This package owns the record; other trace schemas are adapter outputs and are not imported here.

## Interface
- `ClearanceTier` - enum `T0`, `T1`, `T2`.
- `CapturedCall`, `CaptureRecord` - frozen dataclasses.
- `CaptureSession` - context manager with `record_round_trip`, `record_retry`, `record_tool_call`, `record_api_call`, `finish() -> CaptureRecord`.
- `capture(*, tier)` - decorator returning `(result, CaptureRecord)`.

## Invariants
- Unobserved fields stay `None`, distinct from a recorded zero.
- `total_duration_ms` is observed time only, not wall clock.
- `tier` is a typed field, not metadata.

## Side Effects
- None; in-memory only.

## Verification
- Run: `python -m pytest tests/test_trace -q`
