# Changelog

## [Unreleased] — Sprint 3: Model Quality Testing

### Added
- **Model metrics** (`assert_metric`):
  - 9 metrics: accuracy, F1, precision, recall, AUC (classification) + MSE, RMSE, MAE, R2 (regression)
  - Automatic lower-is-better handling for error metrics
  - Multiclass averaging (weighted/macro/micro)
  - Optional scikit-learn dependency (`pip install mltk[sklearn]`)
- **Model regression testing**:
  - `save_baseline()` — persist metrics as JSON for future comparison
  - `assert_no_regression()` — compare current model against saved baseline with tolerance
  - Supports baseline from float, dict, or JSON file
- **Model slicing** (`assert_slice_performance`):
  - Test model on EVERY subgroup independently
  - Catches "works on average, fails for minorities" bug
- **Model calibration** (`assert_calibration`):
  - Expected Calibration Error (ECE) with per-bin breakdown
  - Catches overconfident models (says 90%, correct 60%)
- **API documentation** for all Sprint 3 functions (model-metrics, model-regression, model-slicing)
- **102 Python tests** (24 new) + **5 Rust tests**
- **ML bug research**: 51 training-specific assertions catalogued across 10 categories (unique to mltk)

### Sprint 2
- Drift (4 methods), PII (11 patterns), labels, real Rust KS/PSI. 78 tests.

### Sprint 1
- Config loading, 8 data quality assertions, MkDocs docs. 47 tests.

### Sprint 0
- Project skeleton, core types, Rust crate, pytest plugin, CLI skeleton, CI/CD.
