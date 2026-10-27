# mltk.importer

## Overview
Smart Dataset Importer: load datasets from multiple sources, auto-map column roles, classify the task type, bind golden sets, register datasets behind a quality gate, and generate suites or pytest code.

## Interface
- `DatasetImporter` (loader.py)
- `ColumnRole`, `ColumnMapping`, `ImportResult` (schema.py)
- `auto_map_columns` (mapping.py)
- `TaskType`, `classify_task` (classify.py)
- `GoldenSpec`, `GoldenBindingReport`, `load_golden`, `bind_golden` (golden.py)
- `RegistrationResult`, `register_dataset` (registry.py)
- `compute_baseline_thresholds`, `build_suite` (suite_gen.py)
- `generate_pytest` (codegen.py)

## Invariants
- Column mapping and task classification are deterministic.
- `build_suite` builds suites in memory.

## Side Effects
- loader.py and golden.py read dataset/golden files.
- `register_dataset` records a dataset in the registry.
- codegen.py produces pytest source text.

## Verification
- Run: `python -m pytest tests/test_importer -q`
