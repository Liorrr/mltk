# Changelog

## [Unreleased] — Sprint 1: Core + Data Quality

### Added
- **Config loading**: `MltkConfig` loads from `pyproject.toml [tool.mltk]` and `mltk.yaml` with cascade fallback
- **Data schema testing**:
  - `assert_schema()` — validate column names + dtypes
  - `assert_no_nulls()` — detect null/NaN values (all columns or subset)
  - `assert_dtypes()` — strict dtype checking for specific columns
- **Data distribution testing**:
  - `assert_range()` — numeric value bounds [min, max]
  - `assert_unique()` — duplicate detection (single or composite key)
  - `assert_no_outliers()` — IQR-based statistical outlier detection
- **Data freshness testing**:
  - `assert_freshness()` — verify data recency (max age in days)
  - `assert_row_count()` — validate dataset size (min/max bounds)
- **MkDocs documentation site** with API docs for all 8 assertions
- **47 tests** (up from 12 in Sprint 0) covering all assertion functions

### Sprint 0 (Initial)
- Project skeleton with full module structure
- Core types: MltkConfig, TestResult, TestSuite, Severity
- Base assertion framework: assert_true, MltkAssertionError
- Rust crate skeleton (PyO3 0.28 + Maturin)
- pytest plugin with ML markers (ml_data, ml_model, ml_drift, ml_inference)
- CLI skeleton with Typer
- CI/CD workflows (lint, test, release)
