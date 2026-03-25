# Changelog

## [Unreleased] — Sprint 6: Reports + Pipeline + ML Test Score

### Added
- **HTML report generation** (`generate_report`):
  - Self-contained single-file HTML with dark theme
  - Pass/fail summary, per-module breakdown, test details table
  - Auto-generated when `pytest --mltk-report` is run
- **ML Test Score** (`compute_ml_test_score`):
  - Google's 28-test rubric (data/model/infrastructure/monitoring)
  - `mltk score` CLI command
- **Pipeline testing**:
  - `assert_reproducible()` — deterministic training with seed control
  - `assert_checksum()` — SHA-256 artifact validation
  - `assert_pipeline()` — E2E pipeline execution with type checking
- **Milestone:** `pytest --mltk-report` generates HTML report
- **162 tests** (16 new)

### Sprint 5
- Inference (latency, throughput, contract), CLI (init, scan, drift). 146 tests.

### Sprint 4
- Bias (5 methods), adversarial, --mltk-report terminal. 124 tests.

### Sprint 3
- Model metrics (9), regression, slicing, calibration. 102 tests.

### Sprint 2
- Drift (4 methods), PII (11 patterns), labels, Rust KS/PSI. 78 tests.

### Sprint 1
- Config loading, 8 data quality assertions, MkDocs docs. 47 tests.

### Sprint 0
- Project skeleton, core types, Rust crate, pytest plugin, CLI, CI/CD.
