# mltk.core

## Overview
Foundation types and framework shared by mltk packages: assertion error and helpers, results, config, suite runner, plugin registry, polars bridge and input validation.

## Interface
- `MltkAssertionError`, `assert_true`, `timed_assertion` (assertion.py)
- `Severity`, `TestResult`, `TestSuite` (result.py)
- `MltkSuite`, `SuiteResult` (suite.py)
- `MltkConfig` (config.py)
- `register_assertion`, `get_registered_assertions`, `discover_plugins` (plugin.py)
- `is_polars`, `to_pandas`, `coerce_dataframe` (polars_bridge.py)
- `unknown_on_empty_result`, `empty_input_result` (empty.py)
- `require_same_length` (validation.py)

## Invariants
- Base layer: other mltk packages build on it.
- Polars support is opt-in: `to_pandas` / `@coerce_dataframe` convert a Polars frame only where explicitly applied; no assertion package applies them today, so assertions receive inputs unconverted.

## Side Effects
- `MltkConfig` reads `pyproject.toml` / `mltk.yaml` from disk.
- `register_assertion` mutates a global registry; `discover_plugins` loads plugins.

## Verification
- Run: `python -m pytest tests/test_core -q`
