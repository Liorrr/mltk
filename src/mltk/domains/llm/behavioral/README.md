# llm/behavioral

## Overview
Behavioral consistency tests for LLMs: invariance to paraphrase/format, semantic equivalence, output stability, and retrieval consistency.

## Interface
- `invariance.py`: `assert_paraphrase_invariance`, `assert_format_invariance`
- `semantic.py`: `assert_semantic_equivalence`, `assert_directional_expectation`
- `stability.py`: `assert_output_stability`
- `retrieval.py`: `assert_retrieval_consistency`
- `paraphrase.py`: `ParaphraseGenerator` (deterministic templates; LLM-backed optional)
- `__init__.py` re-exports via `__all__`.

## Invariants
- Assertions return `TestResult`; a failed check at `Severity.CRITICAL` (the default) raises `MltkAssertionError` via `mltk.core.assert_true`.
- Default paraphrasing is template-based and deterministic.

## Side Effects
- Repeatedly calls user-supplied model callables.

## Verification
- Run: `python -m pytest tests/test_domains -q`
