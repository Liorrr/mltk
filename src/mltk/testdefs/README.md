# mltk.testdefs

## Overview
Declarative YAML test suites. A suite names a data source and assertions (data suites) or attack configuration (red-team suites); the runner dispatches to the matching mltk assertions.

## Interface
- `load_test_suite(path)` -> `TestSuiteYaml` or `RedTeamSuiteYaml`.
- `run_test_suite(suite)`, `run_red_team_suite(suite)`.
- Schema types: `TestDef`, `TestSuiteYaml`, `RedTeamSuiteYaml`, `RedTeamDefaults`, `CustomAttack`.

## Invariants
- Suite type must be `data` or `red_team`.
- `env:NAME` references resolve from the environment at load time.
- Per-test errors become failed results rather than aborting the run.

## Side Effects
- Reads YAML and the CSV/Parquet data source; red-team runs may import a user model function and call it.

## Verification
- Run: `python -m pytest tests/test_testdefs -q`
