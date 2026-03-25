# Changelog

## [Unreleased] — Sprint 16

### Added
- **CV tracking metrics** (`mltk.domains.cv.tracking`):
  - `assert_mota()` — Multi-Object Tracking Accuracy
  - `assert_motp()` — Multi-Object Tracking Precision
  - `assert_idf1()` — ID F1 score for identity-aware tracking
- **Training bug P1** (`mltk.training.gradient`, `mltk.training.numerical`):
  - Gradient pathology assertions (flow, vanishing, exploding, loss finite)
  - Numerical stability assertions (NaN/Inf detection, softmax validity)
  - Learning rate assertions (loss decreasing, no divergence)
- **Docs deployment**: Dockerfile + nginx for company server hosting
- `mltk[torch]` optional dependency group

## [0.1.0] — 2026-03-25 — First Public Release

**60+ assertion functions, 261 tests, 6 domain kits, Rust acceleration, pytest plugin, CLI, HTML reports.**

### Sprint 15
- Face recognition: assert_face_far. Wiring audit fixed 13 gaps. Examples for all domains.

### Sprint 14
- Jira integration: IssueTrackerAdapter, JiraAdapter, TicketDecisionEngine, ML ticket templates.

### Sprint 13
- PII expansion (24 patterns + Luhn). Training bug P0: train/test overlap, temporal split, target leakage.

### Sprint 12
- LLM evaluation: semantic similarity, toxicity, hallucination, TTFT/ITL.

### Sprint 11
- Data contracts engine (YAML → pytest). Drift expansion: JS, Wasserstein, auto, embedding drift.

### Sprint 10
- v0.1.0 published to PyPI. README overhaul, CONTRIBUTING.md, GitHub repo.

### Added
- **Production monitoring** (`mltk.monitor`):
  - `assert_no_degradation()` — sliding window metric decline detection
  - `assert_sla()` — latency P99 + error rate SLA compliance
- **Tabular domain kit** (`mltk.domains.tabular`):
  - `assert_feature_drift()` — per-column drift across DataFrames
  - `assert_feature_importance_stable()` — SHAP ranking stability (WARNING severity)
  - `assert_class_balance()` — convenience wrapper for DataFrame label columns
- **New backlog items**: CLI+Web chat, resource summarization (text/image/video), AI prediction chat, test resource registry, Jira integration
- **204 tests** (12 new)
- **All feature sprints complete** — ready for v0.1.0 release (Sprint 10)

### Sprint 8
- NLP (BLEU, ROUGE, NER, prompt injection), Speech (WER, CER, RTF, accent). 192 tests.

### Sprint 7
- CV (IoU, mAP, frame accuracy, temporal consistency, top-K). 178 tests.

### Sprint 6
- HTML reports, pipeline, ML Test Score. 162 tests.

### Sprint 5
- Inference (latency, throughput, contract), CLI. 146 tests.

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
