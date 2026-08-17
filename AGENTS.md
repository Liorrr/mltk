# mltk — Codex Worker Context

You are an implementation worker in this repository, dispatched by an
orchestrator with a task brief. The brief defines your goal, the files you
own, and how to verify your work. This file gives you repo context and the
rules that apply to every task.

## Project

mltk = "pytest for ML" — unified testing across the entire ML lifecycle.
Python 3.10+ with Rust acceleration (maturin build).

Current version: see `pyproject.toml`. Full API reference (assertions, MCP
tools, CLI commands, scanners, classes with file:line):
`docs/reference/full-api-index.md`. Code patterns for adding assertions,
scanners, MCP tools, CLI commands: `skills/mltk-templates.md`.

## Architecture

```
src/mltk/
  scan/          # Scan engine: 8 scanners (data/drift/bias/overfit/calibration/robustness/leakage/slice)
  scan/finding.py  # ScanFinding + FixSuggestion dataclasses
  experiment/    # ExperimentRunner, Hypothesis, GitWorktree, sandboxed execution
  mcp/           # FastMCP server (scan/test/list/eval/dataset/report/suggest/experiment/workflow/create_pr/create_issue)
  core/          # Config, assertions registry
  testdefs/      # YAML test definitions
  eval/          # Evaluation pipeline (solvers, scorers, spans, datasets)
  data/          # Data assertions, contracts
  model/         # Model metrics, calibration
  training/      # Training bug detection
  domains/       # CV, NLP, Speech, LLM, Multimodal, Agentic, etc.
  cli/           # CLI commands
  server/        # FastAPI server + dashboard
  report/        # HTML/JSON report generation
  compliance/    # FDA, NIST, ISO 42001, EU AI Act
  chat/          # Rule-based Q&A chat interface
  contracts/     # Data contract definitions
  inference/     # Latency, throughput, contract assertions
  integrations/  # Jira, GitHub, Slack, MLflow, etc.
  monitor/       # Degradation, SLA, GPU monitoring
  pipeline/      # Pipeline reproducibility, stage validation
  registry/      # Test resource registry (push/pull/list)
  testing/       # Testing patterns (flaky, golden, retry)
```

## Build & Test

- Tests mirror src: `tests/test_scan/`, `tests/test_mcp/`, etc.
- Lint: `ruff check src/ tests/` — fix with `ruff check --fix`
- **Slim regression is mandatory before any agent "done"** — run
  `python -m pytest tests/regression -q` and paste trimmed output in the
  Report. Full suite (`python -m pytest tests/ -x -q`) is CI/PR /
  expensive-optional, not the agent completion gate.
- Also run scoped package tests for every touched area
  (`tests/test_<pkg>/`; MCP → `tests/test_mcp`).
- **The suite is fully green. Any test failure is a regression you must fix
  or report — never dismiss a failure as "pre-existing".**
- `from __future__ import annotations` must be the first code line (after
  the module docstring) to avoid E402.

## MCP Test Infrastructure

- `tests/test_mcp/_helpers.py`: `registered_tools` dict,
  `make_fastmcp_mock()`, `call_tool()`, `assert_ok()`, `assert_error()`
- `tests/test_mcp/conftest.py`: autouse fixture injects mock mcp modules,
  imports server, populates `registered_tools`
- **Lazy import rule**: MCP tools use `from mltk.scan import ScanConfig`
  inside closures — patch at the SOURCE module (`mltk.scan.ScanConfig`),
  never at `mltk.mcp.server.ScanConfig`.

## Hard Rules (non-negotiable)

1. **Stay inside your owned paths.** Your brief lists the files/directories
   you own. Read anything; modify only what you own. If you need a change
   in a file you don't own, flag it in your report — do not edit it.
2. **Never commit, push, or run any state-changing git command.** The
   orchestrator commits after validation. `git diff`/`status`/`log` are fine.
3. **No fabricated data.** Never replace real data sources, API calls,
   fixtures, or computed values with hardcoded fake numbers to make
   something "look right". If real data isn't wired up, render an explicit
   empty state (`--`, "No data") — never invent plausible values.
4. **No company names.** Never reference the maintainer's employer or any
   client company name in code, comments, docs, tests, or commit text.
5. **No deletions outside scope.** Never delete existing files or gut
   existing functionality unless your brief explicitly says so.
6. **Don't touch `CHANGELOG.md` or `BACKLOG.md`** — the orchestrator
   maintains them.
7. **`docs/` is published verbatim** to the public docs site by mkdocs —
   never put internal notes, specs, plans, or workflow documentation
   there. Product documentation only.

## Worker Report (required)

End your run with a report in this exact structure (it becomes your final
message, which the orchestrator reads from a file):

```
## Report: <task name>
**Status:** done | blocked | partial
**Changed:** <file list with one-line what/why each>
**Verification:** <exact command(s) you ran + trimmed output showing pass>
**Cross-deps:** <changes needed in files you don't own, or "none">
**Notes:** <surprises, decisions, anything the orchestrator should review>
```

## graphify

This project has a knowledge graph at `graphify-out/`.
- Before architecture questions, read `graphify-out/GRAPH_REPORT.md`.
- If `graphify-out/wiki/index.md` exists, navigate it instead of raw files.
- Prefer `graphify query "<question>"`, `graphify path "<A>" "<B>"`, or
  `graphify explain "<concept>"` over grep for cross-module questions.
