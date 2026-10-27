# mltk.pipeline

## Overview
Pipeline-level assertions: end-to-end runs, reproducibility, checksums, ONNX validity, stage compatibility, and fault resilience.

## Interface
- `assert_pipeline(...)`, `assert_reproducible(...)`, `assert_checksum(...)`, `assert_onnx_valid(...)`.
- `StageSpec`, `assert_pipeline_stages_compatible(...)` (dtype compatibility between stages).
- `DEFAULT_FAULTS`, `apply_fault(...)`, `assert_pipeline_resilient(...)`.

## Invariants
- Assertions use `timed_assertion` and return result objects.
- `assert_reproducible` resolves a default seed of 42 when none is given.

## Side Effects
- Runs caller-supplied pipeline callables; `assert_checksum` reads files; ONNX checks load a model file.

## Verification
- Run: `python -m pytest tests/test_pipeline -q`
