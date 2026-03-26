# Changelog

## [Unreleased] — Sprint 37

### Added
- **API key auth**: generate/verify keys, protected write endpoints (Bearer mltk_...)
- **CI/CD GitHub**: post_pr_comment, create_check_run, webhook receiver
- **Webhooks**: configurable webhook dispatch on run events (failure/success/drift)
- **Run comparison**: compare_runs diff (new failures, fixed, regressions)
- `mltk server create-key` CLI command

### Sprint 36
- Server platform: FastAPI + SQLite + dashboard + Docker. 837 tests.

## [Previous] — Sprint 36

### Added
- **Server platform** (`mltk.server`):
  - FastAPI server with /api/results, /api/reports, /api/health endpoints
  - SQLite storage for test results and report history
  - Dashboard HTML with test trends and team view
  - `mltk server` CLI command
  - Docker deployment (Dockerfile + docker-compose)
- **pytest integration**: `--mltk-server` flag to auto-push results to server

### Sprint 35
- Rust SIMD cosine, BERTScore, assert_bertscore. 824 tests.

## [Previous] — Sprint 35

### Added
- **Rust SIMD cosine**: cosine_similarity, cosine_distance_matrix, centroid_cosine_distance
- **Rust BERTScore**: bertscore_precision_recall (greedy max cosine matching)
- **assert_bertscore** (`mltk.domains.llm.bertscore`) — BERTScore F1 assertion
- Embedding drift now uses Rust cosine when available

### Sprint 34
- PII: international phones, MAC, Bitcoin/Ethereum. Allowlists. Bias report. 817 tests.

## [Previous] — Sprint 34

### Added
- **PII remaining**: international phone numbers, MAC addresses, Bitcoin/Ethereum wallets
- **PII allowlists**: `scan_pii(text, allowlist=[...])` to suppress known-safe patterns
- **Bias report**: `generate_bias_report()` — demographic breakdown Markdown from test results

## [0.6.0] — 2026-03-26

**119 assertions, 800+ tests, benchmarks, v0.6.0.**

### Sprint 33 — v0.6.0 Release
- Benchmarks vs competitors, feature-label correlation shift, output drift detection

### Sprint 32

### Added
- **RAGAS composite score** (`mltk.domains.llm.ragas`):
  - compute_ragas_score, assert_ragas_score — average of faithfulness + answer relevancy + context precision + recall
- **Coherence** (`mltk.domains.llm.coherence`):
  - assert_coherence — sentence-to-sentence consistency check
- **OWASP LLM Top 10** (`mltk.compliance.owasp_llm`):
  - OWASP_LLM_MAPPING, owasp_llm_scan, assert_owasp_coverage

### Sprint 31
- Multi-turn conversation, data quality preset, sentiment analysis. 767 tests.

## [Previous] — Sprint 31

### Added
- **Multi-turn conversation** (`mltk.domains.llm.conversation`):
  - assert_knowledge_retention, assert_turn_relevancy, assert_conversation_completeness
- **Data quality preset** (`mltk.data.preset`):
  - assert_data_quality — one-call bundle of schema/nulls/range/outliers/PII
  - data_quality_report — summary dict with missing rate, duplicates, constants
- **Sentiment analysis** (`mltk.domains.nlp.sentiment`):
  - assert_sentiment_positive, assert_no_sentiment_drift

## [0.5.0] — 2026-03-26

**108+ assertions, 700+ tests, crossing the 100-assertion milestone.**

### Sprint 30 — v0.5.0 Release
- Data statistics: assert_column_mean, assert_column_median, assert_column_stdev, assert_quantiles
- Data validation: assert_datetime_format, assert_values_in_set, assert_no_conflicting_labels
- ML quality: assert_no_overfitting, assert_label_drift
- Version 0.5.0

### Sprint 29
- RAG (faithfulness, context precision/recall/relevancy, answer relevancy)
- Agentic (task completion, tool selection, tool call correctness)
- Text quality, training-serving skew. 676 tests.

## [Previous] — Sprint 29

### Added
- **RAG evaluation** (`mltk.domains.llm.rag`):
  - assert_faithfulness, assert_context_relevancy, assert_answer_relevancy
  - assert_context_precision, assert_context_recall
- **Agentic evaluation** (`mltk.domains.llm.agentic`):
  - assert_task_completion, assert_tool_selection, assert_tool_call_correctness
- **Text quality** (`mltk.domains.llm.text_quality`):
  - assert_text_length, assert_output_format, assert_readability
- **Training-serving skew** (`mltk.training.skew`):
  - assert_no_training_serving_skew

## [0.4.0] — 2026-03-25

**87+ assertions, 606+ tests, 19 CLI commands, 102 source files.**

### Sprint 28 — v0.4.0 Release
- PII Tier 4: France NIR, Italy Codice Fiscale, Spain DNI
- BACKLOG major rewrite, README update, version bump

### Sprint 27
- Chat interface (ChatEngine, mltk chat CLI). 606 tests.

### Sprint 26
- GitHub Issues, Slack notifications, plugin system. 594 tests.

### Sprint 25
- Test resource registry (push/pull/list). 563 tests.

### Sprint 24
- Testing patterns (flaky, golden, retry, selection), local docs server. 547 tests.

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
