# mltk.chat

## Overview
Rule-based Q&A over saved test results. Not an LLM: pattern matching plus structured analysis of a results JSON file.

## Interface
- `ChatEngine(results_path=None)` - loads results (bare JSON list, or object with a `"results"` key as produced by `--mltk-export-json`).
  - `ask(question) -> str`, `get_failures()`, `get_summary()`, `get_recommendations()`, `get_slowest(n=5)`.
- `chat_repl(results_path=None)` - interactive stdin loop; `quit`/`exit` stops.

## Invariants
- Deterministic; no model calls.
- A missing results file raises `FileNotFoundError`; no path means an empty engine that still answers `help`.

## Side Effects
- Reads the results file; the REPL reads stdin and prints to stdout.

## Verification
- Run: `python -m pytest tests/test_chat -q`
