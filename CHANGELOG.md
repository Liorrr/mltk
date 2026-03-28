# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

#### Integrations (S56-S57)
- GitHub App — webhook HMAC-SHA256 verification, check run creation, app auth (JWT → installation token)
- OpenTelemetry — `MltkTracer` (real/no-op modes), `trace_result`, `trace_suite`, `export_json`
- Weights & Biases — `WandbLogger` (log_result, log_suite, W&B Tables)
- DVC — `assert_dvc_file_tracked`, `assert_dvc_data_version`
- Kubeflow — `assert_kubeflow_pipeline_success`, `assert_kubeflow_step_outputs`
- SageMaker — `assert_sagemaker_pipeline_success`, `assert_sagemaker_step_status`
- Grafana — dashboard JSON export, provisioning YAML, 4-panel dashboard template

#### Enterprise (S58)
- RBAC — role-based access control (admin/writer/reader) for mltk server
- Audit log — SOC 2 compliant action logging with CSV export + `assert_audit_log_complete`
- HIPAA compliance mapping (4 rules) with `assert_hipaa_coverage`
- Custom compliance framework builder (YAML-driven)

#### Advanced ML Testing (S59)
- `assert_counterfactual_fairness` — per-sample fairness via attribute perturbation
- `assert_ate_significant` — Average Treatment Effect significance (pure numpy t-test)
- `assert_no_confounding` — detect treatment-feature correlations
- `assert_image_text_alignment` — multimodal CLIP-style alignment check
- `assert_cross_modal_consistency` — cross-modality prediction agreement
- `assert_reward_bounded`, `assert_cumulative_reward` — RL reward validation

#### Observability (S60)
- `assert_no_test_anomaly` — Z-score/IQR/percentile anomaly detection on test metrics
- `assert_impact_coverage` — verify all impacted tests were executed
- `analyze_impact` — import dependency graph for test impact analysis
- `TestScheduler` — periodic test run scheduling with webhook notifications
- Live monitoring portal — self-contained HTML with real-time polling (no CDN deps)

#### Retrieval Metrics + Developer Experience (S61)
- `assert_ndcg`, `assert_mrr`, `assert_recall_at_k`, `assert_map_at_k` — retrieval ranking metrics completing the RAG story
- `mltk list` CLI — assertion discovery with filter and JSON output (27th CLI command)
- JUnit XML export for Jenkins, GitLab CI, Azure DevOps integration

#### Test Hardening (S57, S61)
- 73 new parametrized + edge-case tests across safety, drift, synthetic, conformal, attribution, agentic, multi-agent, GitHub App, OTEL

## [0.7.0] — 2026-03-27

### Added

#### LLM Safety & Security (S47, S53)
- `assert_no_system_prompt_leakage` — 34 extraction payloads across 8 categories
- `assert_refusal_consistency` — phrasing-dependent safety gap detection
- `assert_safety_taxonomy` — per-category safety coverage
- Prompt injection payloads expanded 8 → 50 (6 categories, backward compatible)

#### Compliance (S48)
- NIST AI RMF mapping (Govern, Map, Measure, Manage) with `assert_nist_rmf_coverage`
- ISO 42001 mapping (8 Annex A controls) with `assert_iso_42001_coverage`
- `mltk compliance-gap` CLI — unified gap analysis across 5 frameworks

#### Agent Trace Testing (S49, S54)
- `AgentTrace`/`ToolCall` dataclasses with `from_dict()` (3 input formats)
- 9 agentic assertions: tool_chain, no_forbidden_actions, step_efficiency, no_redundant_calls, no_hallucinated_tools, cost_budget, error_recovery
- 2 multi-agent assertions: no_agent_loop, agent_handoff

#### Conformal Prediction (S50, S55)
- `assert_interval_coverage`, `assert_prediction_set_size`
- `assert_conformal_calibration` — two-sided calibration check
- `assert_conditional_coverage` — per-group fairness (Mondrian)

#### Distributed Training (S50)
- `assert_n_rank_gradient_sync`, `assert_gradient_alignment`
- `assert_weight_divergence`, `assert_gradient_clipped`

#### Drift Detection (S51)
- `assert_no_streaming_drift` with ADWIN and CUSUM detectors
- `assert_no_concept_drift` — P(Y|X) drift via chi2/fisher/proportion
- Completes drift story: P(X), P(Ŷ), streaming, P(Y|X)

#### Synthetic Data & NLP Robustness (S52)
- `assert_marginal_fidelity`, `assert_correlation_preserved`, `assert_synthetic_novelty`, `assert_dcr_safe`
- `TextPerturber` (4 methods) + `assert_text_robust`

#### Attribution Stability (S53)
- `assert_top_k_stable`, `assert_attribution_cosine_stability`

#### Infrastructure
- HTML report: pass/fail donut chart + module bar chart (pure CSS/SVG)
- TestPyPI step in release workflow
- OWASP LLM02/LLM06/LLM07/LLM08 mappings updated
- NIST AI RMF function mappings wired to new assertions
- 20 new MkDocs documentation pages

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
