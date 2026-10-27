# mltk.testing

## Overview
Testing patterns for ML suites: flaky detection, golden sets, retry with confidence intervals, smart test selection, impact analysis and combinatorial coverage.

## Interface
- `detect_flaky`, `FlakySummary` (flaky.py)
- `save_golden`, `load_golden`, `assert_matches_golden` (golden.py)
- `retry_until_confident`, `RetryResult` (retry.py)
- `build_test_map`, `select_affected_tests` (selection.py)
- `analyze_impact`, `assert_impact_coverage` (impact.py)
- `combinatorial_coverage`, `assert_combinatorial_coverage` (combinatorial.py)

## Invariants
- Assertions return `TestResult`; a failed check at `Severity.CRITICAL` (the default) raises `MltkAssertionError` via `mltk.core.assert_true`. They are plain functions usable inside pytest.

## Side Effects
- golden.py reads/writes golden files on disk.
- selection/impact analysis read source files.

## Verification
- Run: `python -m pytest tests/test_testing -q`
