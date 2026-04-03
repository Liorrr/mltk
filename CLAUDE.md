# mltk Repository Rules

## HARD RULES
- **Never auto-commit** — provide commit messages for manual commit.

## Project Overview
mltk = "pytest for ML" — unified testing across the entire ML lifecycle.
- Python 3.10+ with Rust acceleration (maturin build)
- v0.9.0, 230 assertions, 4225+ tests, 8 scanners, 11 MCP tools
- Phase F (Agent Integration): COMPLETE — building toward v1.0.0

## Architecture
```
src/mltk/
  scan/          # Scan engine: 8 scanners (data/drift/bias/overfit/calibration/robustness/leakage/slice)
  scan/finding.py  # ScanFinding + FixSuggestion dataclasses
  experiment/    # ExperimentRunner, Hypothesis, GitWorktree, sandboxed execution
  mcp/           # FastMCP server (11 tools: scan/test/list/eval/dataset/report/suggest/experiment/workflow/create_pr/create_issue)
  core/          # Config, assertions registry
  testdefs/      # YAML test definitions
  eval/          # Evaluation pipeline (solvers, scorers, spans, datasets)
  data/          # Data assertions, contracts
  model/         # Model metrics, calibration
  training/      # Training bug detection
  domains/       # CV, NLP, Speech, LLM, Multimodal, Agentic, etc.
  cli/           # 24+ CLI commands
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

## Testing Conventions
- Tests mirror src: `tests/test_scan/`, `tests/test_mcp/`, etc.
- Lint: `ruff check src/ tests/` — fix with `ruff check --fix`
- Run tests: `python -m pytest tests/ -x -q` (use `-x` to stop on first failure)
- 2 known pre-existing failures in leakage scanner (KeyError: 0) — don't count as regressions
- `from __future__ import annotations` must be the first code line (after docstring above it) to avoid E402

## MCP Test Infrastructure
- `tests/test_mcp/_helpers.py`: `registered_tools` dict, `make_fastmcp_mock()`, `call_tool()`, `assert_ok()`, `assert_error()`
- `tests/test_mcp/conftest.py`: autouse fixture injects mock mcp modules, imports server, populates `registered_tools`
- **Lazy import rule**: MCP tools use `from mltk.scan import ScanConfig` inside closures — patch at SOURCE module (`mltk.scan.ScanConfig`), never at `mltk.mcp.server.ScanConfig`

## Sprint Workflow (mltk-specific)
Uses sprint-executor skill: research → design plan → user approval → parallel agents → lint/test → Opus review → fix findings → CHANGELOG/BACKLOG/commit message.
- Dispatch ALL implementation agents in a single parallel batch
- Don't create module scaffolding manually — agents create their own files
- Update CHANGELOG.md + BACKLOG.md at sprint end

## Codebase Index Skill
A generated skill at `~/.claude/skills/mltk-index.md` indexes the full API surface (230 assertions, 11 MCP tools, 28 CLI commands, 8 scanners, 28 key classes with file:line pointers). Regenerate after each sprint:
```
python scripts/generate_skill_index.py
```
Detailed reference with full signatures: `docs/reference/full-api-index.md`

### Subagent usage
- **Builder agents**: Include skill index content in prompt — they need file locations
- **Test hardening agents**: Include skill index — they need assertion names and test file mapping
- **Wiring/integration agents**: Include skill index — they need module structure
- **Documentation agents**: Include skill index — they need to know what exists
- **Researchers**: Do NOT include — they search the web, not the codebase
- **Reviewers/auditors**: Include skill index — helps them navigate during review

### How to include in agent prompts
Read the skill index and paste the content into the agent's prompt context:
```
Read ~/.claude/skills/mltk-index.md and include its content below as ## Codebase Index
```

## Key Files
- `BACKLOG.md` — sprint history + backlog items
- `CHANGELOG.md` — version changelog
- `pyproject.toml` — maturin build, dependencies
- `src/mltk/__init__.py` — public API exports
- `scripts/generate_skill_index.py` — regenerates skill index from source
