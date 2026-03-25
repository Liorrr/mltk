# Changelog

## [Unreleased] — Sprint 7: CV Domain Kit

### Added
- **Computer vision domain kit** (`mltk.domains.cv`):
  - `compute_iou()` — vectorized NxM IoU matrix computation (pure numpy)
  - `assert_iou()` — minimum mean IoU between predicted and ground-truth boxes
  - `assert_map()` — mean Average Precision with per-class AP breakdown (COCO/VOC-style)
  - `assert_frame_accuracy()` — per-frame detection/classification accuracy for video
  - `assert_temporal_consistency()` — frame-to-frame tracking smoothness validation
  - `assert_topk_accuracy()` — top-K accuracy for image classification
- **Kaleidoo CV example** (`examples/kaleidoo_cv_test.py`)
- **3 research agents dispatched**: LLM/GenAI evaluation, data contracts, embedding drift
- **178 tests** (16 new)

### Sprint 6
- HTML reports, pipeline (reproducibility, checksum, E2E), ML Test Score. 162 tests.

### Sprint 5
- Inference (latency, throughput, contract), CLI (init, scan, drift). 146 tests.

### Sprint 4
- Bias (5 methods), adversarial, --mltk-report. 124 tests.

### Sprint 3
- Model metrics (9), regression, slicing, calibration. 102 tests.

### Sprint 2
- Drift (4 methods), PII (11 patterns), labels, Rust KS/PSI. 78 tests.

### Sprint 1
- Config loading, 8 data quality assertions, MkDocs docs. 47 tests.

### Sprint 0
- Project skeleton, core types, Rust crate, pytest plugin, CLI, CI/CD.
