# mltk.pytest_plugin

## Overview
The pytest plugin (entry point `mltk = mltk.pytest_plugin.plugin`): markers, fixtures, result collection, and report export.

## Interface
- Markers: `ml_data`, `ml_model`, `ml_drift`, `ml_inference`, `ml_slow`, `ml_nondeterministic`, `ml_smoke`, `ml_gpu`.
- Options: `--mltk-report`, `--mltk-export-json`, `--mltk-mlflow`, `--mltk-server`.
- Fixtures: `ml_config`, `ml_report`.
- Hooks (`plugin.py`): `pytest_addoption` (options), `pytest_configure` (markers), `pytest_runtest_makereport` (result collection), `pytest_sessionfinish` (report/export/MLflow/server upload).
- `MltkReportCollector` - accumulates results (`add`, `passed_count`, `failed_count`, `total`, `to_json_records`).

## Invariants
- Markers are registered in `pytest_configure`.
- Optional integrations (MLflow, server push) run only when their flag is set.

## Side Effects
- Writes the JSON export; `--mltk-mlflow` logs to MLflow; `--mltk-server` posts over HTTP (key from `MLTK_API_KEY`).

## Verification
- Run: `python -m pytest tests/test_pytest_plugin -q`
