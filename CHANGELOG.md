# Changelog

## [Unreleased] — Sprint 27

### Added
- **Chat interface** (`mltk.chat`):
  - `mltk chat` CLI — interactive Q&A about test results
  - `ChatEngine` — analyzes test results, suggests fixes, recommends tests
  - Embeddable in HTML reports

### Sprint 26
- GitHub Issues, Slack notifications, plugin system. 594 tests.

## [Previous] — Sprint 26

### Added
- **GitHub Issues adapter** (`mltk.integrations.github_adapter`) — create/search/update issues
- **Slack notifications** (`mltk.integrations.slack`) — webhook-based failure alerts
- **Plugin system** (`mltk.core.plugin`) — `@register_assertion` + `discover_plugins()`
- **CLI**: `mltk notify slack`

### Sprint 25
- Test resource registry: push/pull/list. 563 tests.

## [Previous] — Sprint 25

### Added
- **Test resource registry** (`mltk.registry`):
  - `save_collection()` / `load_collection()` / `list_collections()`
  - `mltk registry push/pull/list` CLI commands
  - Collection format: manifest.json + YAML test defs + golden data

### Sprint 24
- Testing patterns (flaky, golden, retry, selection). Local docs server. 547 tests.

## [Previous Unreleased] — Sprint 24

### Added
- **Testing patterns** (`mltk.testing`):
  - `detect_flaky()` — run test N times, flag as flaky if pass rate below threshold
  - `save_golden()` / `load_golden()` / `assert_matches_golden()` — versioned baseline management
  - `retry_until_confident()` — retry with confidence intervals for non-deterministic tests
  - `select_affected_tests()` / `build_test_map()` — smart test selection from imports
- **Local docs server**: `mltk docs serve` (hot reload) + `mltk docs build` (static HTML)

## [0.3.0] — 2026-03-25

**84+ assertions, 496+ tests, PII Tier 3, Rust acceleration, Training Bug P2.**

### Sprint 23 — v0.3.0 Release
- PII Tier 3: UK NHS, UK NINO, Germany Steuer-ID, India Aadhaar, India PAN
- Version bump, README update, backlog cleanup

### Sprint 22
- Training Bug P2: augmentation, checkpoint, distributed, memory. 8 new assertions.

### Sprint 21
- Rust: KL, chi2, JS, Wasserstein, PII scanning. 13 Rust tests. Benchmarks.

### Sprint 20
- Cloud monitoring: AWS SageMaker, GCP Vertex AI, Azure ML, Prometheus/Triton. 450 tests.

### Sprint 19
- MLflow integration (MlflowLogger, --mltk-mlflow), Jupyter _repr_html_, model card generator. 416 tests.

## [0.2.0] — 2026-03-25

**66+ assertions, 361+ tests, YAML test defs, EU AI Act compliance, mltk doctor, 11 CLI commands.**

### Sprint 18 — v0.2.0 Release
- Israel PII: Teudat Zehut (Luhn checksum), Israel phone numbers
- IBAN MOD-97 checksum validation
- README overhaul, version bump, backlog cleanup

### Sprint 17
- YAML test definitions (mltk.testdefs): write YAML, run with `mltk test`
- EU AI Act compliance report (mltk.compliance): article mapping + evidence HTML
- mltk doctor: 9 diagnostic checks with fix hints
- Env var config: MLTK_* prefix (highest priority)
- CLI: mltk doctor, mltk test, mltk compliance (11 total)
- pytest: --mltk-export-json flag

### Sprint 16
- CV tracking: assert_mota, assert_motp, assert_idf1. Training P1: gradient + numerical. Docs deployment. 314 tests.

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
