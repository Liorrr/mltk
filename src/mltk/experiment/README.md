# mltk.experiment

## Overview
Experiment runner that validates fix suggestions by testing hypotheses against scan findings, optionally in isolated git worktrees.

## Interface
- `Hypothesis`, `HypothesisResult` (hypothesis.py)
- `ExperimentResult` (result.py)
- `ExperimentRunner` (runner.py)
- `SandboxedExperimentRunner` (sandbox.py)
- `rank_hypotheses` (ranking.py)
- `git_available`, `find_git_root`, `GitWorktree` (worktree.py)

## Invariants
- Sandboxed runs apply changes only inside a worktree.
- Sandboxed execution requires git (`git_available`).

## Side Effects
- sandbox.py/worktree.py invoke git subprocesses and create/remove worktrees.

## Verification
- Run: `python -m pytest tests/test_experiment -q`
