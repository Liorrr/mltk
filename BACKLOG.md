# MLTK Backlog

Tracked items for the ML Test Kit project. Updated after each sprint.

## Status Legend
- **DONE** -- shipped and tested
- **PLANNED** -- scheduled for a specific sprint
- **BACKLOG** -- accepted, not yet scheduled
- **IDEA** -- needs evaluation

---

## DONE (S0-S65: 201 assertions, 2187 tests, 38 Rust tests)

### Phase A: Core Library (S0-S10) -- v0.1.0
- [x] S0: Project skeleton, pyproject.toml, Cargo.toml, CI/CD
- [x] S1: Config, 8 data assertions, MkDocs docs
- [x] S2: 4 drift methods, 14 PII patterns, Rust KS/PSI
- [x] S3: 9 model metrics, regression, slicing, calibration
- [x] S4: 5 bias methods, adversarial, --mltk-report
- [x] S5: Inference (latency, throughput, contract), 5 CLI commands
- [x] S6: HTML reports, ML Test Score, pipeline reproducibility
- [x] S7: CV (IoU, mAP, frame accuracy, temporal consistency, top-K)
- [x] S8: NLP (BLEU, ROUGE, NER, prompt injection), Speech (WER, CER, RTF)
- [x] S9: Monitoring (degradation, SLA), Tabular (feature drift, SHAP)
- [x] S10: v0.1.0 on PyPI, README, CONTRIBUTING.md

### Phase B: Post-Release Expansion (S11-S18) -- v0.2.0
- [x] S11: Data contracts (YAML->tests), JS/Wasserstein/auto drift, embedding drift
- [x] S12: LLM eval (semantic similarity, toxicity, hallucination, TTFT/ITL)
- [x] S13: 10 Tier 1 PII + Luhn, training bug P0 (leakage detection)
- [x] S14: Jira integration (adapter, dedup, templates)
- [x] S15: Wiring audit (13 gaps fixed), face recognition (assert_face_far)
- [x] S16: CV tracking (MOTA/MOTP/IDF1), training bug P1 (gradient/numerical), docs deploy
- [x] S17: YAML test defs, EU AI Act compliance, mltk doctor, env vars, JSON export
- [x] S18: v0.2.0, Israel PII (Teudat Zehut), IBAN MOD-97

### Phase C: Performance + Polish (S19-S23) -- v0.3.0
- [x] S19: MLflow integration, Jupyter _repr_html_, model card generator
- [x] S20: Cloud monitoring (AWS, GCP, Azure, Prometheus/Triton)
- [x] S21: Rust acceleration (KL, chi2, JS, Wasserstein, PII scanning)
- [x] S22: Training bug P2 (augmentation, checkpoint, distributed, memory)
- [x] S23: v0.3.0, PII Tier 3 (UK NHS, NINO, Germany Steuer-ID, India Aadhaar/PAN)

### Phase D: Platform Features (S24-S27)
- [x] S24: Testing patterns (flaky, golden, retry, selection), local docs server
- [x] S25: Test resource registry (push/pull/list)
- [x] S26: GitHub Issues, Slack notifications, plugin system
- [x] S27: Chat interface (rule-based Q&A)

### Phase E: Assertion Expansion + Server (S28-S37) -- v0.4.0-v0.6.0
- [x] S28: PII Tier 4 (France NIR, Italy Codice Fiscale, Spain DNI), v0.4.0
- [x] S29: RAG evaluation (faithfulness, context precision/recall/relevancy, answer relevancy), agentic (task completion, tool selection, tool call correctness), text quality, training-serving skew
- [x] S30: Data statistics (mean, median, stdev, quantiles), data validation (datetime, set membership, conflicting labels), overfitting detection, label drift, v0.5.0
- [x] S31: Multi-turn conversation (retention, relevancy, completeness), data quality preset, sentiment analysis
- [x] S32: RAGAS composite score, coherence, OWASP LLM Top 10 mapping
- [x] S33: Benchmarks, feature-label correlation shift, output drift, v0.6.0
- [x] S34: PII remaining (international phones, MAC, crypto wallets), allowlists, bias report
- [x] S35: Rust SIMD cosine, centroid distance, BERTScore in Rust, assert_bertscore
- [x] S36: FastAPI server, SQLite storage, dashboard HTML, Docker, --mltk-server
- [x] S37: API key auth, GitHub CI (PR comments, check runs), webhooks, run comparison

### Phase F: Compliance, Integrations & Polish (S38-S41)
- [x] S38: FDA 21 CFR Part 11 audit trail, compliance PDF export, CLI commands (fda-audit, compliance-pdf)
- [x] S39: Resource summarization (trend analysis, flaky detection, recommendations), visual diff reports
- [x] S40: Linear adapter (GraphQL), Asana adapter (REST), data lineage tracking (assert_lineage_complete)
- [x] S41: VS Code extension (mltk-vscode), NLP/Speech module refactoring, 24 CLI commands total

### Phase G: Audit & Research (S42-S46)
- [x] S42-S46: Full audit (21 subagent review, 189 suggestions), P0/P1 fixes, 12-topic research offensive

### Phase H: Capability Expansion (S47+)
- [x] S47: LLM safety hardening — assert_no_system_prompt_leakage, 50 categorized injection payloads, report charts
- [x] S48: NIST AI RMF + ISO 42001 compliance, compliance-gap CLI command
- [x] S49: AgentTrace dataclasses, assert_tool_chain, assert_no_forbidden_actions, assert_step_efficiency
- [x] S50: Conformal prediction (interval coverage, set size), distributed training (N-rank sync, alignment, divergence, clipping)
- [x] S51: Streaming drift (ADWIN, CUSUM), concept drift (chi2/fisher/proportion) — completes drift story
- [x] S52: Synthetic data (marginal fidelity, correlation, novelty, DCR), text noise robustness (TextPerturber + assert_text_robust)
- [x] S53: Attribution stability (top-K + cosine), extended LLM safety (refusal consistency + taxonomy)
- [x] S54: Extended agentic (redundant, hallucinated, cost, recovery) + multi-agent (loop, handoff) — **150 ASSERTION MILESTONE**
- [x] S55: Advanced conformal (calibration + conditional coverage), v0.7.0 release, README update, TestPyPI
- [x] S56: GitHub App integration (webhook, check runs, app auth) + OpenTelemetry (MltkTracer, export_json)
- [x] S57: ML platform integrations (Kubeflow, SageMaker, W&B, DVC) + test hardening (+40 tests) + gap research
- [x] S58: Enterprise (RBAC, audit log, HIPAA, custom compliance builder)
- [x] S59: Advanced ML (counterfactual fairness, causal inference, multimodal, RL)
- [x] S60: Observability (test impact, anomaly detection, Grafana, live portal, scheduler) — **v0.8.0**
- [x] S61: Retrieval metrics (nDCG, MRR, Recall@K, MAP@K), mltk list CLI, JUnit XML export + test hardening
- [x] S62: LLM-as-Judge (score + pairwise), summarization metrics (coverage, compression, faithfulness) + test hardening
- [x] S63: Recommendation systems (5 assertions, first-mover), long-context LLM (needle/utilization/lost-in-middle) + test hardening
- [x] S64: Composable TestSuite API (MltkSuite), code generation assertions (4), test hardening
- [x] S65: Healthcare (5 assertions), SR 11-7 compliance, Polars bridge, v0.8.0 release

---

## BACKLOG (not yet scheduled)

### Integrations
- [x] GitHub App for auto-running mltk on PRs
- [ ] Create GitHub Releases (v0.1.0-v0.7.0) when stealth mode ends

### Advanced Features
- [ ] Test impact analysis (dependency graph)
- [ ] Anomaly detection on test result time series

### Monetization (Pro tier)
- [ ] Cloud dashboard: hosted report aggregation, team views
- [ ] Multi-tenant server with SSO
- [ ] Scheduled test runs with alerting

### Observability & Monitoring
- [ ] OpenTelemetry integration for test execution tracing
- [ ] Grafana plugin for mltk dashboards (Grafana OSS — free, self-hosted)
- [ ] Real-time streaming drift detection

### Monitoring Visualization Portal
- [ ] **ACTION**: Research build vs. buy — compare custom portal effort against free solutions (Grafana OSS, Apache Superset, Metabase, Redash) before committing to implementation
- [ ] Live visualization portal connected to mltk server for monitoring state and data
- [ ] Monitor connects to mltk server to transmit data/state; server provides free port or port range for streaming visualization data (consider WebSocket on existing server port as simpler alternative)
- [ ] Visualize scale, live processes, and performance metrics in real-time
- [ ] **GATE**: Security audit required before committing to portal infrastructure — no deployment on non-secured infrastructure

### ML Platform Integration
- [ ] Kubeflow pipeline assertions
- [ ] SageMaker Pipeline step validation
- [ ] Weights & Biases adapter
- [ ] DVC data version assertions

### Advanced ML Testing
- [ ] Counterfactual fairness testing
- [ ] Causal inference validation
- [ ] Federated learning test patterns
- [ ] Multi-modal (image+text) evaluation
- [ ] Reinforcement learning reward validation

### Enterprise
- [ ] RBAC for server platform
- [ ] Audit log export (SOC 2 compatible)
- [ ] Custom compliance framework builder
- [ ] HIPAA compliance report template

---

*Last updated: Sprint 47 (March 27, 2026) — counts verified from source*
