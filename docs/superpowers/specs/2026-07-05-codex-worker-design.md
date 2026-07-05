# Codex as Sprint Implementation Engine — Design Spec

**Date:** 2026-07-05
**Status:** Approved
**Owner:** Lior (project lead) / Claude (orchestrator)

## Goal

Offload sprint implementation work from Claude subagents to Codex CLI workers
(`codex exec`), so Claude usage concentrates on planning, orchestration, and
validation. Claude remains the orchestrator; Codex becomes the implementation
engine inside the existing sprint-executor pipeline.

## Locked Decisions

| # | Decision | Choice | Rationale |
|---|----------|--------|-----------|
| 1 | Codex role | Sprint implementation engine inside sprint-executor | Highest token savings; integrates with existing workflow instead of adding a parallel one |
| 2 | Worker isolation | Shared working tree + explicit file-ownership clauses per brief | Same contract Claude agent batches use today; avoids the worktree PYTHONPATH gotcha; one test run validates everything |
| 3 | Validation depth | Full gate every batch (ruff + full pytest + per-hunk diff review) | Trust is uncalibrated; Claude reviewing Codex output is inherently cross-model review |
| 4 | Protocol home | Global skill `~/.claude/skills/codex-worker/` | Project-agnostic dispatch mechanics, reusable in other repos; mltk specifics live in AGENTS.md |

## Division of Labor

**Claude keeps:** research → design → plan → user approval → validation gate →
CHANGELOG/BACKLOG updates → commit.
**Codex takes:** the implementation batch (code + tests per plan task).

Economics note: this pays off most for well-scoped work (test hardening,
coverage, refactors, template-following features). Design-heavy or exploratory
tasks stay with Claude — writing a Codex-precise spec for those costs as much
as doing them.

## Flow

```
plan tasks → Claude writes N briefs → dispatch ALL in one parallel batch:
  codex exec --full-auto --skip-git-repo-check -C <repo> \
    -o <scratchpad>/worker-N.md "<brief>"     (run_in_background)
→ wait for all workers → read worker reports
→ validation gate:
    1. ruff check src/ tests/
    2. python -m pytest tests/ -x -q
    3. full diff review: every hunk vs plan intent + guardrails
       (no mock data, no company name, scope respected)
→ failures become fix briefs via `codex exec resume <session-id>`
  (worker retains its own context — cheaper than a fresh brief)
→ after 2 failed fix rounds on the same task, Claude takes that task over
→ Claude commits (workers NEVER commit)
```

## Deliverable 1 — `AGENTS.md` rewrite (repo root, committed)

Replaces the stale untracked copy. Structural principle: **no volatile facts**
— the old copy went stale because it duplicated version numbers and test
counts. Content:

- Project identity, architecture map, testing conventions, MCP test
  infrastructure + lazy-import patching rule, E402/`from __future__` rule —
  mirrored from CLAUDE.md
- Truth corrections: suite is fully green; **any test failure is a
  regression** (removes the outdated "2 known failures" line). No version or
  count numbers — point at `pyproject.toml` and
  `docs/reference/full-api-index.md` instead
- **Guardrails**: no company name anywhere, no mock/fabricated data, never
  commit, stay inside owned paths, no deletions outside scope
- **Worker protocol**: expected report format — what changed, verification
  command run with output shown, cross-dependencies flagged (not fixed)
- Keep the graphify section (Codex benefits from graph-first navigation)
- Removed: Claude-side skill-injection matrix, sprint orchestration details,
  `~/.Codex/skills` paths — orchestrator business, not worker business

## Deliverable 2 — `~/.claude/skills/codex-worker/SKILL.md` (global)

Project-agnostic dispatch protocol:

- **Brief template** with mandatory clauses: goal + acceptance criteria,
  owned paths ("do NOT modify files outside these paths"), verify command,
  no-commit rule, report contract
- Invocation pattern: `codex exec --full-auto --skip-git-repo-check -C <repo>
  -o <out>.md "<brief>"` with background monitoring; all briefs of a batch
  dispatched in parallel
- Validation gate checklist (lint → tests → per-hunk diff review)
- Retry via `codex exec resume <session-id>`; 2-strikes escalation to
  orchestrator
- Precondition: target repo must have an `AGENTS.md`; if missing, stop and
  create one first

## Deliverable 3 — Wiring + housekeeping

- mltk `CLAUDE.md`: ~4-line note in the Sprint Workflow section —
  implementation batches dispatch to Codex via the `codex-worker` skill;
  Claude retains research/design/review/commit; keep `AGENTS.md` in sync when
  conventions (not stats) change
- Commit `AGENTS.md` + `.codex/` (config.toml is MCP-server wiring only —
  safe for the public repo)

## Error Handling

- Worker output file missing or empty → treat as failed task, re-dispatch
  once fresh, then escalate
- Gate failure attributable to one worker → fix brief via `resume` to that
  worker's session
- Cross-worker file collision detected in diff review → orchestrator resolves
  manually; note it for brief-template tightening
- 2 failed fix rounds on one task → Claude implements that task directly

## Acceptance Criteria

1. `AGENTS.md` rewritten, stale facts gone, guardrails present, committed
2. `codex-worker` skill exists globally and contains brief template, gate
   checklist, retry/escalation rules
3. CLAUDE.md wiring note added
4. **Pilot:** one small real backlog item runs the full loop end-to-end
   (brief → exec → gate → commit) successfully before the next real sprint
   relies on it

## Out of Scope

- Git-worktree isolation per worker (revisit only if shared-tree collisions
  become real)
- Codex-side skills in `~/.codex/skills/` (in-repo docs + AGENTS.md pointers
  suffice)
- Wiring other repos (bellkis/kernel, mltk-vscode) — reuse the skill later
- Replacing the Codex review skills (adversarial-review, codex-review) — they
  stay as-is
