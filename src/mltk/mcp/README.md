# mltk.mcp

## Overview
MCP server exposing mltk to AI agents over FastMCP. Run with `python -m mltk.mcp`.

## Interface
- `create_server() -> FastMCP`, `run_server()`.
- Tools (registered in `server.py`): `mltk_scan`, `mltk_test`, `mltk_list`, `mltk_eval`, `mltk_dataset`, `mltk_report`, `mltk_suggest`, `mltk_experiment`, `mltk_workflow`, `mltk_create_pr`, `mltk_create_issue`, `mltk_container_scan`, `mltk_import`.

## Invariants
- Tools return JSON strings via `_ok` / `_error`; errors carry a suggested action.
- Heavy imports are lazy inside tools.
- Module-based eval model loading is gated by the `MLTK_MCP_MODEL_MODULES` environment variable.

## Side Effects
- `mltk_test` runs pytest in a subprocess.
- `mltk_experiment` can create git worktrees for sandboxed fixes.
- `mltk_create_pr` / `mltk_create_issue` act on a git repository and remote.

## Verification
- Run: `python -m pytest tests/test_mcp -q`
