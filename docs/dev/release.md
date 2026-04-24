# Release Process

See [CONTRIBUTING.md](../../CONTRIBUTING.md#release-process) for the full workflow.

## Quick reference

| Task | Command |
|------|---------|
| Refresh doc counts (pre-commit does this automatically) | `python scripts/bump.py refresh` |
| Check for count drift (CI gate) | `python scripts/bump.py verify` |
| Preview a version bump | `python scripts/bump.py release --dry-run <version>` |
| Bump version + roll CHANGELOG | `python scripts/bump.py release <version>` |
| Full release make target | `make bump-release VERSION=<version>` |

## What gets updated

`bump release` updates all of the following in one atomic operation:
- `pyproject.toml` — `version = "X.Y.Z"`
- `rust/Cargo.toml` — `version = "X.Y.Z"`
- `src/mltk/__init__.py` — `__version__` fallback literal
- `CHANGELOG.md` — rolls `[Unreleased]` → `[X.Y.Z] — YYYY-MM-DD`
- `README.md`, `docs/index.md`, `docs/roadmap.md`, and all API/guide pages — assertion/test/scanner/MCP/CLI counts
- `CLAUDE.md`, `BACKLOG.md` — version and counts

**`Since: vX.Y.Z` markers in API docs are intentionally preserved** — they are historical feature-introduction markers, not current-version claims.
