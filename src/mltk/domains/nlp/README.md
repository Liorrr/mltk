# nlp

## Overview
Classic NLP assertions: generation-quality scores, NER, sentiment, text robustness, and prompt-injection security.

## Interface
- `generation.py`: `assert_bleu`, `assert_rouge`, `assert_chrf`
- `ner.py`: `assert_ner_f1`
- `sentiment.py`: `assert_sentiment_positive`, `assert_no_sentiment_drift`
- `robustness.py`: `TextPerturber`, `assert_text_robust`
- `security.py`: `assert_no_prompt_injection`
- `__init__.py` re-exports these via `__all__`.

## Invariants
- Assertions return `TestResult`; a failed check at `Severity.CRITICAL` (the default) raises `MltkAssertionError` via `mltk.core.assert_true`.

## Side Effects
- None beyond calling user-supplied model callables.

## Verification
- Run: `python -m pytest tests/test_domains -q`
