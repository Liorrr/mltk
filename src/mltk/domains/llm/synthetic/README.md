# llm/synthetic

## Overview
Synthetic QA-pair generation from documents for LLM evaluation datasets.

## Interface
- `generator.py`: `QuestionType`, `QAPair`, `SyntheticQAGenerator`
- `_quality.py`: `QualityFilter` (private)
- `_splitter.py`: `split_text` (word-count chunks, no dependencies)
- `_templates.py`: `build_prompt`, `parse_response`, `parse_conversational_response`, `extract_key_sentences`, `declarative_to_interrogative` (private)
- `__init__.py` exports the public generator types.

## Invariants
- Text splitting is overlap-based and dependency-free.
- Generated pairs can be scored and filtered by quality.

## Side Effects
- May call a user-supplied LLM callable when generating.

## Verification
- Run: `python -m pytest tests/test_domains -q`
