# MLTK Backlog

Tracked items for the ML Test Kit project. Updated after each sprint.

## Status Legend
- **DONE** -- shipped and tested
- **PLANNED** -- scheduled for a specific sprint
- **BACKLOG** -- accepted, not yet scheduled
- **IDEA** -- needs evaluation

---

## DONE (S0-S37: 121 assertions, 1053 tests, 13 Rust tests)

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

---

## BACKLOG (not yet scheduled)

### Integrations
- [ ] VS Code extension (inline test results)
- [ ] Linear/Asana adapters
- [ ] GitHub App for auto-running mltk on PRs

### Advanced Features
- [ ] Resource summarization + AI predictions
- [ ] Data lineage visualization (DAG rendering)
- [ ] Test impact analysis (dependency graph)
- [ ] Anomaly detection on test result time series

### Monetization (Pro tier)
- [ ] Cloud dashboard: hosted report aggregation, team views
- [ ] Multi-tenant server with SSO
- [ ] Scheduled test runs with alerting

---

## POST-v1.0 ROADMAP (ideas from capability gap research)

### Observability
- [ ] OpenTelemetry integration for test execution tracing
- [ ] Grafana plugin for mltk dashboards
- [ ] Real-time streaming drift detection

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

*Last updated: Sprint 37 (March 26, 2026)*
