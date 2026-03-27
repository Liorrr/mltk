# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- API key authentication for protected write endpoints (Bearer token)
- CI/CD GitHub integration: PR comments, check runs, webhook receiver
- Configurable webhook dispatch on run events (failure/success/drift)
- Run comparison with diff (new failures, fixed, regressions)
- `assert_no_system_prompt_leakage` — detect LLM system prompt extraction (34 payloads, 8 categories)
- Prompt injection payloads expanded from 8 to 50 with categorized vectors (6 categories)
- OWASP LLM06 mapping updated with system prompt leakage assertion
- HTML report visualization: pass/fail donut chart + module breakdown bar chart (pure CSS/SVG, no external deps)
- NIST AI RMF compliance mapping (Govern, Map, Measure, Manage) with `assert_nist_rmf_coverage` + maturity tiers
- ISO 42001 compliance mapping (8 Annex A controls) with `assert_iso_42001_coverage`
- `mltk compliance-gap` CLI command — unified gap analysis across all 5 frameworks (25th CLI command)
- `AgentTrace` and `ToolCall` dataclasses for representing agent execution traces (`from_dict` supports 3 formats)
- `assert_tool_chain` — verify agent tool call sequence matches expected chain
- `assert_no_forbidden_actions` — verify agent did not use forbidden tools
- `assert_step_efficiency` — verify agent completed task within step budget
- OWASP LLM07/LLM08 mappings updated with trace-based assertions
- `assert_interval_coverage` — validate prediction interval empirical coverage
- `assert_prediction_set_size` — validate prediction set cardinality/width budget
- `assert_n_rank_gradient_sync` — N-rank gradient synchronization check
- `assert_gradient_alignment` — cosine similarity between gradient vectors across ranks
- `assert_weight_divergence` — L2 distance between weight checkpoints/ranks
- `assert_gradient_clipped` — verify gradient global norm within clipping bound
- NIST AI RMF MEASURE function mapping updated with conformal + distributed assertions
- `assert_no_streaming_drift` — real-time drift detection with ADWIN and CUSUM detectors
- `ADWINDetector` — adaptive windowing with Hoeffding bound, O(log W) memory
- `CUSUMDetector` — cumulative sum change-point detection
- `assert_no_concept_drift` — P(Y|X) drift detection via chi2/fisher/proportion tests (completes drift story)
- NIST AI RMF MANAGE mapping updated with streaming + concept drift

## [0.6.0] — 2026-03-26

### Added
- Server platform: FastAPI + SQLite + dashboard + Docker deployment
- Rust SIMD cosine similarity and BERTScore assertion
- PII expansion: international phones, MAC addresses, crypto wallets, allowlists
- Bias report generator with demographic breakdown
- RAGAS composite score, coherence check, OWASP LLM Top 10 mapping
- Multi-turn conversation evaluation (knowledge retention, turn relevancy)
- Data quality preset (one-call bundle) and sentiment analysis
- Benchmarks vs competitors, feature-label correlation shift, output drift detection

### Changed
- Embedding drift now uses Rust cosine when available
- pytest integration supports `--mltk-server` flag for auto-push

## [0.5.0] — 2026-03-26

### Added
- Data statistics: assert_column_mean, assert_column_median, assert_column_stdev, assert_quantiles
- Data validation: assert_datetime_format, assert_values_in_set, assert_no_conflicting_labels
- ML quality: assert_no_overfitting, assert_label_drift
- RAG evaluation (faithfulness, context precision/recall, answer relevancy)
- Agentic evaluation (task completion, tool selection, tool call correctness)
- Text quality assertions and training-serving skew detection

## [0.4.0] — 2026-03-25

### Added
- PII Tier 4: France NIR, Italy Codice Fiscale, Spain DNI
- Chat interface (ChatEngine, `mltk chat` CLI command)
- GitHub Issues adapter, Slack webhook notifications, plugin system
- Test resource registry with push/pull/list CLI commands
- Testing patterns: flaky detection, golden baselines, retry with confidence, smart test selection
- Local docs server with hot reload

## [0.3.0] — 2026-03-25

### Added
- PII Tier 3: UK NHS, UK NINO, Germany Steuer-ID, India Aadhaar, India PAN
- Training bug detection P2: augmentation, checkpoint, distributed, memory
- Rust acceleration: KL, chi-squared, Jensen-Shannon, Wasserstein, PII scanning
- Cloud monitoring: AWS SageMaker, GCP Vertex AI, Azure ML, Prometheus/Triton
- MLflow integration with `--mltk-mlflow` flag, Jupyter rich display, model card generator

## [0.2.0] — 2026-03-25

### Added
- Israel PII: Teudat Zehut (Luhn checksum), Israel phone numbers, IBAN MOD-97
- YAML test definitions with `mltk test` runner
- EU AI Act compliance report with article mapping and evidence HTML
- `mltk doctor` with 9 diagnostic checks and fix hints
- Environment variable config (MLTK_* prefix)
- CV tracking: assert_mota, assert_motp, assert_idf1
- Training bug P1: gradient and numerical stability checks

## [0.1.0] — 2026-03-25

### Added
- 60+ assertion functions across 6 domain kits (data, model, NLP, CV, speech, inference)
- Rust-accelerated drift (KS, PSI), PII scanning (24 patterns + Luhn)
- pytest plugin with `--mltk-report` HTML report generation
- CLI with 8 commands (run, report, config, doctor, etc.)
- Production monitoring: degradation detection, SLA compliance
- Tabular domain kit: feature drift, importance stability, class balance
- LLM evaluation: semantic similarity, toxicity, hallucination, latency (TTFT/ITL)
- Data contracts engine (YAML to pytest), drift expansion (JS, Wasserstein, embedding)
- Jira integration with ML ticket templates
- Face recognition: assert_face_far
- MkDocs documentation site
