# mltk.eval

## Overview
Composable evaluation pipelines for LLM testing: dataset + solvers + scorers composed into a task.

## Interface
- `EvalSample`, `EvalState`, `Score`, `EvalResult` (_types.py)
- `EvalTask`, `load_dataset` (task.py)
- `Solver`, `GenerateSolver`, `ChainOfThoughtSolver`, `FewShotSolver`, `chain` (solvers.py)
- `Scorer`, `ExactMatchScorer`, `IncludesScorer`, `PatternScorer`, `LLMJudgeScorer` (scorers.py)
- `DatasetCard`, `EvalDataset`, `DatasetDiff`, `DatasetInfo`, `DatasetRegistry`, `assert_dataset_quality` (dataset.py)

## Invariants
- Solvers and scorers implement the `Solver` / `Scorer` ABCs so they compose in `EvalTask`.

## Side Effects
- dataset.py reads/writes datasets in a local registry on disk; task.py reads dataset files.
- Model-backed solvers/scorers call the caller-supplied model.

## Verification
- Run: `python -m pytest tests/test_eval -q`
