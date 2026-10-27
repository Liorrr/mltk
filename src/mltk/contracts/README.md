# mltk.contracts

## Overview
Data contracts: a YAML spec that can be validated against a DataFrame or turned into a generated pytest file.

## Interface
- `Contract.from_yaml(path)`, `ColumnSpec`, `QualitySpec` - spec dataclasses.
- `validate_data(...)` - runs schema, null, and per-column checks, returning a result suite.
- `generate_tests_from_contract(contract_path, output_path, data_path="data.csv") -> Path`.

## Invariants
- Validation failures are reported as results (for example `contract.schema`), not raised.
- Columns absent from the data are skipped in per-column checks.

## Side Effects
- `Contract.from_yaml` reads a file; the generator creates parent directories and writes the test file.

## Verification
- Run: `python -m pytest tests/test_contracts -q`
