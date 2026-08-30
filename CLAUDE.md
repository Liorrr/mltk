# mltk Repository Rules

## HARD RULES
- **Agents may commit and open PRs directly** — use clear, descriptive commit messages (Conventional Commits style, matching existing history). Still avoid destructive git operations (force-push, hard reset, history rewrite, branch deletion) without explicit per-instance confirmation — those remain governed by the general git safety protocol.

## Project Overview
mltk = "pytest for ML" — unified testing across the entire ML lifecycle.
- Python 3.10+ with Rust acceleration (maturin build)
- v0.13.0, 243 assertion definitions (238 unique names; 229 discoverable), 5004+ tests, 8 scanners, 13 MCP tools
- Phase F (Agent Integration): COMPLETE — building toward v1.0.0

## Testing Conventions
- Tests mirror src: `tests/test_scan/`, `tests/test_mcp/`, etc.
- Lint: `ruff check src/ tests/` — fix with `ruff check --fix`
- **Slim regression is mandatory before any agent "done"** — run
  `python -m pytest tests/regression -q` and paste output in the Report.
  Full suite (`python -m pytest tests/ -x -q`) is CI/PR / expensive-optional,
  not the agent completion gate.
- Scoped package tests still required for touched areas (`tests/test_<pkg>/`).
- The suite is fully green — the historical leakage-scanner failures (KeyError: 0) were fixed in the S97 review cycle; any failure is a regression
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
- **Implementation batches dispatch to Codex** via the `codex-worker` skill
  (`~/.claude/skills/codex-worker/`): Claude keeps research/design/plan/
  validation-gate/commit; Codex workers implement under file-ownership
  briefs. Workers read `AGENTS.md` — keep it in sync when conventions
  change (it holds no version/count facts by design).

## Skills for Subagents
Before dispatching ANY subagent or Codex worker in this repo, read `skills/mltk-agent-dispatch.md` — it holds the required index/templates matrix and the paste format.
Regenerate the index after each sprint: `python scripts/generate_skill_index.py`

## VS Code Extension (separate repo)
- **Repo**: `C:\Users\lior1\mltk-vscode` (GitHub: `Liorrr/mltk-vscode`) — open it for current version and file layout
- **Architecture**: subprocess-based — spawns `python -m pytest` and `python -m mltk` CLI commands and parses JSON output. Deliberately **no MCP integration**; CLI only.
- **Hard rule**: same no-company-name restriction as main repo

## graphify

This project has a graphify knowledge graph at graphify-out/.

Rules:
- Before answering architecture or codebase questions, read graphify-out/GRAPH_REPORT.md for god nodes and community structure
- If graphify-out/wiki/index.md exists, navigate it instead of reading raw files
- For cross-module "how does X relate to Y" questions, prefer `graphify query "<question>"`, `graphify path "<A>" "<B>"`, or `graphify explain "<concept>"` over grep — these traverse the graph's EXTRACTED + INFERRED edges instead of scanning files
- After modifying code files in this session, run `graphify update .` to keep the graph current (AST-only, no API cost)
