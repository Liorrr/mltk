# Changelog

## [Unreleased] — Sprint 5: Inference + CLI

### Added
- **Inference latency testing**:
  - `assert_latency()` — percentile validation (P50/P95/P99) with mandatory warmup
  - `assert_cold_start()` — first-call latency for model loading time
- **Inference throughput testing**:
  - `assert_throughput()` — duration-based RPS measurement with concurrent workers
  - Error tracking and goodput calculation
- **API contract testing**:
  - `assert_api_contract()` — JSON Schema validation for input/output
  - Falls back to basic type checking when jsonschema not installed
- **CLI commands** (mltk):
  - `mltk init` — scaffold mltk.yaml + example test file
  - `mltk scan <path>` — quick data quality scan on CSV files
  - `mltk drift <ref> <cur>` — drift comparison between two datasets
- **4 cloud skills installed**: vertex-ai-api-dev, monitoring-observability, kubernetes, azure-ai-ml-py
- **Cloud infrastructure research**: AWS/GCP/Azure/on-prem ML lifecycle testing patterns catalogued in BACKLOG
- **146 Python tests** (22 new) + **5 Rust tests**

### Sprint 4
- Bias (5 fairness methods), adversarial robustness, --mltk-report plugin. 124 tests.

### Sprint 3
- Model metrics (9 types), regression testing, slicing, calibration. 102 tests.

### Sprint 2
- Drift (4 methods), PII (11 patterns), labels, real Rust KS/PSI. 78 tests.

### Sprint 1
- Config loading, 8 data quality assertions, MkDocs docs. 47 tests.

### Sprint 0
- Project skeleton, core types, Rust crate, pytest plugin, CLI skeleton, CI/CD.
