# domains

## Overview
Domain-specific test kits (CV, NLP, speech, tabular, LLM, multimodal) plus single-file kits for code generation, healthcare, recommendation, and RL. All expose `assert_*` functions returning `TestResult`.

## Interface
- `codegen.py`: `assert_code_executes`, `assert_code_passes_tests`, `assert_no_code_vulnerabilities`, `assert_code_complexity`
- `healthcare.py`: `assert_sensitivity`, `assert_specificity`, `assert_ppv`, `assert_npv`, `assert_clinical_agreement`
- `recommendation.py`: `assert_hit_rate`, `assert_diversity`, `assert_novelty`, `assert_coverage`, `assert_serendipity`
- `rl.py`: `assert_reward_bounded`, `assert_cumulative_reward`
- Subpackages: [cv](./cv/README.md), [llm](./llm/README.md), [multimodal](./multimodal/README.md), [nlp](./nlp/README.md), [speech](./speech/README.md), [tabular](./tabular/README.md)

## Invariants
- Assertions return `TestResult`; a failed check at `Severity.CRITICAL` (the default) raises `MltkAssertionError` via `mltk.core.assert_true`; they build on `mltk.core`.
- Each domain package re-exports its public API via `__all__`.

## Side Effects
- `codegen.py` runs generated code in a subprocess (`shell=False`, with a timeout) from a temp file.
- Other modules are pure computation.

## Verification
- Run: `python -m pytest tests/test_domains -q`
