# Codex Sprint Implementation Engine — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire Codex CLI (`codex exec`) into the mltk sprint workflow as the implementation engine: Codex-facing `AGENTS.md`, global `codex-worker` dispatch skill, CLAUDE.md wiring, and one end-to-end pilot.

**Architecture:** Three documentation/config artifacts plus a live pilot. Claude orchestrates (plan → dispatch → validate → commit); Codex workers implement in a shared working tree under file-ownership briefs. Full validation gate (ruff + pytest + per-hunk diff review) after every batch.

**Tech Stack:** Codex CLI 0.118.0 (`codex exec --full-auto`), Claude Code skills (markdown), git, ruff, pytest.

**Spec:** `docs/superpowers/specs/2026-07-05-codex-worker-design.md`

## Global Constraints

- **No company/employer name** in any mltk file, ever (hard rule; phrase the rule itself generically in AGENTS.md).
- **No volatile facts in AGENTS.md**: no version numbers, assertion counts, or test counts — point at `pyproject.toml` / `docs/reference/full-api-index.md`.
- **Workers never commit** — Claude commits after the validation gate.
- **Public-repo boundary**: everything committed to mltk is public; no internal info.
- Repo commits follow Conventional Commits style; end commit messages with the Claude co-author trailer.

---

### Task 1: Rewrite `AGENTS.md` and commit it with `.codex/`

**Files:**
- Modify (full replace): `AGENTS.md` (repo root — currently a stale untracked copy)
- Commit (already exist, untracked): `.codex/config.toml`, `.codex/hooks.json`

**Interfaces:**
- Produces: the Codex-facing context file every worker reads automatically. Task 4's pilot brief assumes AGENTS.md carries guardrails + report format, so briefs stay short.

- [ ] **Step 1: Replace AGENTS.md content entirely**

Write exactly this content to `AGENTS.md`:

````markdown
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
- Run tests: `python -m pytest tests/ -x -q` (`-x` stops on first failure)
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
````

- [ ] **Step 2: Verify no stale facts or forbidden content survived**

Run: `grep -nE "0\.12\.4|232 assertions|4291|2 known|pre-existing failures|~/.Codex|Never auto-commit" AGENTS.md`
Expected: no output (exit 1).

Run: `grep -c "Hard Rules" AGENTS.md && grep -c "Worker Report" AGENTS.md`
Expected: `1` and `1`.

Also manually verify the maintainer's employer name does not appear anywhere in AGENTS.md (the executor knows the name from memory; it must never be written into any repo file, including as a grep pattern).

- [ ] **Step 3: Commit AGENTS.md + .codex/**

```bash
git add AGENTS.md .codex/config.toml .codex/hooks.json
git commit -m "feat: add Codex worker context (AGENTS.md) and .codex config

- AGENTS.md rewritten as worker-facing context: architecture, test
  conventions, MCP test infra, hard rules, report format
- no volatile facts (version/counts point at pyproject + API index)
- .codex/config.toml wires mltk MCP server; hooks.json wires graphify hint

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: Create the global `codex-worker` skill

**Files:**
- Create: `C:\Users\lior1\.claude\skills\codex-worker\SKILL.md`

**Interfaces:**
- Consumes: AGENTS.md contract from Task 1 (workers read it automatically; briefs don't repeat repo context).
- Produces: the dispatch protocol Task 4's pilot follows verbatim.

- [ ] **Step 1: Write SKILL.md**

Write exactly this content to `C:\Users\lior1\.claude\skills\codex-worker\SKILL.md`:

````markdown
---
name: codex-worker
description: >-
  Dispatch implementation work to Codex CLI workers (codex exec) and validate
  the results. Use for sprint implementation batches or any well-scoped
  coding task being offloaded to Codex while Claude orchestrates. Triggers:
  "dispatch to codex", "codex batch", sprint implementation steps in repos
  wired for Codex (AGENTS.md present).
---

# Codex Worker Dispatch

Claude plans and validates; Codex implements. Workers share the working
tree under strict file-ownership briefs. Full gate after every batch.

## Preconditions

1. `codex --version` succeeds (CLI on PATH).
2. Target repo has an `AGENTS.md` worker context file. **If missing, STOP**
   — create one first (repo context, hard rules, worker report format).
   Briefs stay short only because AGENTS.md carries the standing rules.

## Brief Template

Every brief MUST contain all six clauses. Omitting the ownership clause is
the #1 cause of collisions.

```
## Task: <name>
<goal — what to build/fix and WHY, 2-6 sentences. Include acceptance
criteria concrete enough to test against.>

## Owned paths
You own: <explicit file/dir list>. Do NOT modify any file outside these
paths. Read anything you like. If a change is needed elsewhere, flag it in
your report under Cross-deps.

## Method
Test-driven: write the failing test first, watch it fail, implement, watch
it pass. Follow existing patterns in the module you're editing.

## Verify
Run: <exact command, e.g. python -m pytest tests/test_x/ -q>
All of it must pass. Also run: ruff check <owned paths>

## Constraints
Never commit. No fabricated data. Stay in owned paths.
<task-specific constraints, if any>

## Report
End with the Worker Report format from AGENTS.md.
```

## Dispatch

One worker per plan task, ALL dispatched in a single parallel batch.
Per worker (Bash, `run_in_background: true`):

```sh
codex exec --full-auto --skip-git-repo-check -C <repo-root> \
  -o <scratchpad>/worker-N.md "<brief>" > <scratchpad>/worker-N.log 2>&1
```

- `-o` captures the worker's final message (the report).
- The log captures the transcript; the session id appears near the top —
  extract it for the fix loop: `grep -im1 "session id" <scratchpad>/worker-N.log`
- Scratchpad = the session scratchpad dir, never the repo.

Wait for all background tasks to complete. A missing or empty `worker-N.md`
= failed worker: re-dispatch once with a fresh brief, then escalate.

## Validation Gate (every batch, no exceptions)

1. **Lint:** `ruff check src/ tests/` (or repo equivalent) — must be clean.
2. **Tests:** full suite (`python -m pytest tests/ -x -q` or repo
   equivalent) — must be green.
3. **Diff review:** read EVERY hunk of `git diff` against the plan task it
   claims to implement. Check: intent match, no fabricated data, no
   forbidden names, no files touched outside the worker's owned paths, no
   deleted functionality. Read each worker report; honor flagged Cross-deps.

## Fix Loop

Gate failure attributable to worker N → resume its session (it keeps its
context; cheaper and better than a fresh brief):

```sh
codex exec --full-auto --skip-git-repo-check -C <repo-root> \
  -o <scratchpad>/worker-N-fix.md resume <SESSION_ID> \
  "<what failed, exact error output, what to change>" \
  > <scratchpad>/worker-N-fix.log 2>&1
```

**Two strikes:** after 2 failed fix rounds on the same task, stop
dispatching — implement that task yourself.

Cross-worker file collision found in diff review → resolve it yourself,
and note which brief's ownership clause was violated.

## After the Gate

- The orchestrator (you) commits — workers never do.
- Update project bookkeeping (CHANGELOG/BACKLOG or equivalent) yourself.
- What Codex is NOT for: design-heavy or exploratory work where the brief
  would be as expensive as the implementation. Keep those.
````

- [ ] **Step 2: Verify skill file structure**

Run: `head -12 ~/.claude/skills/codex-worker/SKILL.md`
Expected: YAML frontmatter opening `---`, `name: codex-worker`, description block.

Run: `grep -c "Two strikes" ~/.claude/skills/codex-worker/SKILL.md`
Expected: `1`.

No commit — `~/.claude/skills/` is outside the repo.

---

### Task 3: Wire into mltk CLAUDE.md

**Files:**
- Modify: `C:\Users\lior1\mltk\CLAUDE.md` (Sprint Workflow section, after the "Regenerate skill index" bullet)

**Interfaces:**
- Consumes: skill name `codex-worker` from Task 2.

- [ ] **Step 1: Add wiring note**

In `CLAUDE.md`, find:

```markdown
- Update CHANGELOG.md + BACKLOG.md at sprint end
- Regenerate skill index after sprint: `python scripts/generate_skill_index.py`
```

Append directly below:

```markdown
- **Implementation batches dispatch to Codex** via the `codex-worker` skill
  (`~/.claude/skills/codex-worker/`): Claude keeps research/design/plan/
  validation-gate/commit; Codex workers implement under file-ownership
  briefs. Workers read `AGENTS.md` — keep it in sync when conventions
  change (it holds no version/count facts by design).
```

- [ ] **Step 2: Verify + commit**

Run: `grep -n "codex-worker" CLAUDE.md`
Expected: one match in the Sprint Workflow section.

```bash
git add CLAUDE.md
git commit -m "docs: wire codex-worker dispatch into sprint workflow

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: Pilot — run one real task through the full loop

**Files:**
- Modified by Codex worker: `src/mltk/pipeline/compatibility.py`, `tests/test_pipeline/test_compatibility.py`
- Modified by orchestrator after gate: `BACKLOG.md` (check off item), `CHANGELOG.md`

**Interfaces:**
- Consumes: AGENTS.md (Task 1), dispatch protocol (Task 2).

The pilot item (BACKLOG.md "First-Mover Sprint Follow-ups"):
`assert_pipeline_stages_compatible` — dtype canonicalization so `int64`/`Int64`/`int` compare equal.

- [ ] **Step 1: Dispatch the pilot worker**

Bash with `run_in_background: true` (scratchpad = session scratchpad dir):

```sh
codex exec --full-auto --skip-git-repo-check -C /c/Users/lior1/mltk \
  -o "$SCRATCH/pilot.md" "$(cat <<'BRIEF'
## Task: dtype canonicalization in assert_pipeline_stages_compatible
src/mltk/pipeline/compatibility.py defines assert_pipeline_stages_compatible
(around line 30). Today dtype comparison is exact-string match, so 'int64'
vs 'Int64' vs 'int' produce false FAILs. Canonicalize dtypes before
comparing using numpy: np.dtype(x).name, treating equivalent spellings as
equal. Pandas extension dtypes like 'Int64' (nullable) that np.dtype()
cannot parse should canonicalize via their lowercase name fallback so
'Int64' == 'int64'. Acceptance: ('int64','Int64'), ('int64','int'),
('float64','float') all compare equal; ('int64','float64') still FAILs.

## Owned paths
You own: src/mltk/pipeline/compatibility.py,
tests/test_pipeline/test_compatibility.py. Do NOT modify any file outside
these paths. Read anything. Flag needed external changes under Cross-deps.

## Method
Test-driven: add failing tests for the acceptance pairs first, watch them
fail, implement canonicalization, watch them pass. Follow existing test
patterns in tests/test_pipeline/test_compatibility.py.

## Verify
Run: python -m pytest tests/test_pipeline/ -q
All of it must pass. Also run: ruff check src/mltk/pipeline/ tests/test_pipeline/

## Constraints
Never commit. No fabricated data. Stay in owned paths. Preserve the public
signature of assert_pipeline_stages_compatible.

## Report
End with the Worker Report format from AGENTS.md.
BRIEF
)" > "$SCRATCH/pilot.log" 2>&1
```

- [ ] **Step 2: Collect report + session id**

Run: `cat "$SCRATCH/pilot.md"` — expect a Worker Report with Status: done.
Run: `grep -im1 "session id" "$SCRATCH/pilot.log"` — record the UUID.
If `pilot.md` is missing/empty → re-dispatch once fresh, then escalate per skill.

- [ ] **Step 3: Validation gate**

Run: `ruff check src/ tests/` — expected: clean.
Run: `python -m pytest tests/ -x -q` — expected: all green (suite is fully green; any failure is a regression → fix loop).
Run: `git diff` — review every hunk: only the two owned files touched; canonicalization matches intent; tests assert the acceptance pairs; no fabricated data; no forbidden names.

- [ ] **Step 4: Fix loop if needed**

```sh
codex exec --full-auto --skip-git-repo-check -C /c/Users/lior1/mltk \
  -o "$SCRATCH/pilot-fix.md" resume <SESSION_ID> \
  "<exact failing output + what to change>" > "$SCRATCH/pilot-fix.log" 2>&1
```

Two failed rounds → implement directly (per skill).

- [ ] **Step 5: Bookkeeping + commit**

- BACKLOG.md: check off the dtype canonicalization item (line ~216).
- CHANGELOG.md: add entry under Unreleased/current section: dtype canonicalization in `assert_pipeline_stages_compatible`.
- Mirror the BACKLOG update to the Obsidian MLTK backlog note per global rules.

```bash
git add src/mltk/pipeline/compatibility.py tests/test_pipeline/test_compatibility.py BACKLOG.md CHANGELOG.md
git commit -m "fix(pipeline): canonicalize dtypes in assert_pipeline_stages_compatible

- int64/Int64/int and float64/float now compare equal via np.dtype
  canonicalization with lowercase fallback for pandas extension dtypes
- closes S95 Opus-review deferred P2 item
- pilot run of the codex-worker dispatch loop (brief -> exec -> gate)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

- [ ] **Step 6: Regenerate skill index**

Run: `python scripts/generate_skill_index.py`
(Assertion behavior changed; keep the index honest. Commit if it produces a diff in-repo; the generated `~/.claude/skills/mltk-index.md` is outside the repo.)
