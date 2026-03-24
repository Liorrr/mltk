# Changelog

## [Unreleased] — Sprint 4: Bias + Adversarial + pytest Plugin

### Added
- **Model bias/fairness testing** (`assert_no_bias`):
  - 5 methods: demographic parity, equalized odds, predictive parity, disparate impact, equal opportunity
  - Zero dependencies (pure numpy), Fairlearn-compatible naming
  - EU AI Act compliant, US four-fifths rule support
- **Adversarial robustness** (`assert_robust`):
  - Perturbation-based stability testing (gaussian + uniform noise)
  - Configurable epsilon and stability threshold
- **pytest plugin expansion**:
  - `--mltk-report` flag generates per-module test summary
  - `ml_config` fixture loads MltkConfig automatically
  - `ml_report` fixture for report collection
  - New markers: `ml_smoke` (fast CI), `ml_gpu` (GPU runners)
  - Report shows pass/fail counts, per-module breakdown, failed assertion details
- **API documentation** for bias, adversarial, and pytest plugin
- **124 Python tests** (22 new) + **5 Rust tests**
- **Milestone:** `pytest --mltk-report` works end-to-end

### Sprint 3
- Model metrics (9 types), regression testing, slicing, calibration (ECE). 102 tests.

### Sprint 2
- Drift (4 methods), PII (11 patterns), labels, real Rust KS/PSI. 78 tests.

### Sprint 1
- Config loading, 8 data quality assertions, MkDocs docs. 47 tests.

### Sprint 0
- Project skeleton, core types, Rust crate, pytest plugin, CLI skeleton, CI/CD.
