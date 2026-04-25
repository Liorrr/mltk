# mltk Repository Rules

## HARD RULES
- **Never auto-commit** — provide commit messages for manual commit.

## Project Overview
mltk = "pytest for ML" — unified testing across the entire ML lifecycle.
- Python 3.10+ with Rust acceleration (maturin build)
- v0.12.0, 232 assertions, 4291+ tests, 8 scanners, 12 MCP tools
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
  cli/           # 28 CLI commands
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
- Regenerate skill index after sprint: `python scripts/generate_skill_index.py`

## Skills for Subagents
Two skills exist. The orchestrator MUST read and include them in agent prompts per the matrix below.

| Skill | Path | Content |
|-------|------|---------|
| **Index** | `~/.claude/skills/mltk-index.md` (generated) | 232 assertions, 12 MCP tools, 28 CLI, 8 scanners, 28 classes with file:line |
| **Templates** | `skills/mltk-templates.md` (repo) → `~/.claude/skills/` | Patterns for adding assertions, scanners, MCP tools, CLI commands |

Regenerate index after each sprint: `python scripts/generate_skill_index.py`
Detailed reference with full signatures: `docs/reference/full-api-index.md`

### Which agents get which skills

| Agent Type | Index | Templates | Why |
|------------|:-----:|:---------:|-----|
| Builder | Y | Y | Needs file locations + code patterns |
| Test hardening | Y | N | Needs assertion names + test file mapping |
| Wiring/integration | Y | Y | Needs module structure + export patterns |
| Documentation | Y | N | Needs to know what exists |
| Reviewer/auditor | Y | N | Needs to navigate during review |
| Researcher | N | N | Searches web, not codebase |

### How to include
Read both skill files and paste their content into the agent prompt:
```
## Codebase Index
{content of ~/.claude/skills/mltk-index.md}

## Development Templates
{content of ~/.claude/skills/mltk-templates.md}
```

## VS Code Extension (separate repo)
- **Repo**: `C:\Users\lior1\mltk-vscode` (GitHub: `Liorrr/mltk-vscode`)
- **Version**: 0.3.0, 27 TypeScript files, esbuild + vitest
- **Architecture**: subprocess-based — spawns `python -m pytest` and `python -m mltk` CLI commands, parses JSON output
- **No MCP integration** — uses CLI only, not the 12 MCP tools
- **Features**: test runner, inline gutter/hover/CodeLens, dashboard webview, YAML validation, model scan, security scan, PII scan, red team CodeLens, native Test Explorer
- **Key files**: `src/extension.ts` (entry), `src/testRunner.ts`, `src/scanRunner.ts`, `src/securityScanRunner.ts`
- **Hard rule**: same no-company-name restriction as main repo

## Key Files
- `BACKLOG.md` — sprint history + backlog items
- `CHANGELOG.md` — version changelog
- `pyproject.toml` — maturin build, dependencies
- `src/mltk/__init__.py` — public API exports
- `scripts/generate_skill_index.py` — regenerates skill index from source

## graphify

This project has a graphify knowledge graph at graphify-out/.

Rules:
- Before answering architecture or codebase questions, read graphify-out/GRAPH_REPORT.md for god nodes and community structure
- If graphify-out/wiki/index.md exists, navigate it instead of reading raw files
- For cross-module "how does X relate to Y" questions, prefer `graphify query "<question>"`, `graphify path "<A>" "<B>"`, or `graphify explain "<concept>"` over grep — these traverse the graph's EXTRACTED + INFERRED edges instead of scanning files
- After modifying code files in this session, run `graphify update .` to keep the graph current (AST-only, no API cost)
