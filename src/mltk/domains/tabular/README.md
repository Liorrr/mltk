# tabular

## Overview
Tabular/DataFrame assertions for feature drift, importance stability, and class balance.

## Interface
- `features.py`: `assert_feature_drift`, `assert_feature_importance_stable`
- `quality.py`: `assert_class_balance`
- `__init__.py` re-exports all three via `__all__`.

## Invariants
- Assertions return `TestResult`; a failed check at `Severity.CRITICAL` (the default) raises `MltkAssertionError` via `mltk.core.assert_true`.

## Side Effects
- None.

## Verification
- Run: `python -m pytest tests/test_domains -q`
