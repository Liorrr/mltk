# speech

## Overview
Speech-model assertions for recognition accuracy and performance.

## Interface
- `recognition.py`: `assert_wer`, `assert_cer`
- `performance.py`: `assert_rtf`, `assert_accent_coverage`
- `__init__.py` re-exports the four assertions via `__all__`.

## Invariants
- Assertions return `TestResult`; a failed check at `Severity.CRITICAL` (the default) raises `MltkAssertionError` via `mltk.core.assert_true`.
- WER/CER raise `ImportError` with an install hint when the optional metric dependency is missing.

## Side Effects
- None; pure computation.

## Verification
- Run: `python -m pytest tests/test_domains -q`
