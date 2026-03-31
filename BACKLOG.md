# MLTK Backlog

Tracked items for the ML Test Kit project. Updated after each sprint.

## Status Legend
- **DONE** -- shipped and tested
- **PLANNED** -- scheduled for a specific sprint
- **BACKLOG** -- accepted, not yet scheduled
- **IDEA** -- needs evaluation

---

## DONE (S0-S74: 206 assertions, 2700+ tests, 38 Rust tests) — v0.9.0

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
- [x] S66: mltk scan JSON export, all 8 scanners wired, VS Code Test Inspector, docs rewrite
- [x] S67: Multi-method dispatch for hallucination + RAG (lexical/embedding/nli/llm), unicode defense
- [x] S68: Toxicity classifier, semantic leak detection, BERTScore warnings
- [x] S69: Behavioral invariance (paraphrase, format) + output stability — 3 first-mover assertions
- [x] S70: Behavioral family complete (semantic equiv, directional, retrieval) + ParaphraseGenerator
- [x] S71a: Property-based testing (Hypothesis), E2E pipeline, backend hardening
- [x] S71b: Presentation demo script, assertion index, snapshot tests (syrupy)
- [x] S72: MMD multivariate drift, SmoothECE calibration, intersectional fairness — 3 new assertions
- [x] S73: NER PII detection (Presidio/spaCy/GLiNER/hybrid), test hardening (+22), 3 research briefs
- [x] S74: v0.9.0 release — version bump, presentation-ready docs, test hardening, 2 research briefs

---

## BACKLOG (not yet scheduled)

### URGENT — Method Fixes (S66 Audit: 3 REJECT items)
*Audit report: `docs/research/project-audit-s66.md`*
- [ ] **R-1**: Add NLI/embedding methods to `assert_no_hallucination` — token overlap is unreliable (Huang et al. 2024, Nature)
- [ ] **R-2**: Add NLI/embedding/LLM methods to RAG assertions (faithfulness, relevancy) — RAGAS itself uses LLM-judge, not keyword matching
- [ ] **R-3**: Add classifier method to `assert_no_toxicity` — 4 regex patterns are trivially bypassable (Detoxify, MIT license)

### Method Enhancements (S66 Audit: 6 items)
- [x] **E-1**: S72 — MMD multivariate drift (RBF multi-bandwidth, permutation test, pure numpy)
- [x] **E-2**: S72 — SmoothECE calibration (reflected Gaussian kernel, auto-bandwidth, ICLR 2024)
- [x] **E-3**: S68 — BERTScore limitation warnings (antonymy blindness, number blindness)
- [x] **E-4**: S72 — Intersectional fairness (Crenshaw, Cartesian product, min_subgroup_size=30)
- [x] **E-5**: S73 — NER PII detection (Presidio + spaCy + GLiNER + hybrid method dispatch)
- [ ] **E-6**: Add semantic similarity method to system prompt leak detection

### Testing Infrastructure (S66 Audit: 5 items)
- [ ] **A-1**: Add property-based testing with Hypothesis (50x more mutations — OOPSLA 2025)
- [ ] **A-2**: Add pytest-xdist for parallel execution (60-80% CI time cut)
- [ ] **A-3**: Add pytest-randomly for order independence verification
- [ ] **A-4**: Add snapshot testing (syrupy) for HTML/XML report outputs
- [ ] **A-5**: Add per-module coverage thresholds

### Integrations
- [x] GitHub App for auto-running mltk on PRs
- [ ] Create GitHub Releases (v0.1.0-v0.7.0) when stealth mode ends

### Behavioral Consistency (Research: 40+ sources, March 2026)
*Research brief: `docs/research/paraphrase-invariance-research.md`*
- [x] `assert_paraphrase_invariance` — S69: 6 methods, per-input details (DONE)
- [x] `assert_output_stability` — S69: N-run consistency detection (DONE)
- [x] `assert_format_invariance` — S69: 5 default transforms (DONE)
- [x] `assert_retrieval_consistency` — S70: Jaccard on RAG doc sets (DONE)
- [x] `assert_directional_expectation` — S70: CheckList DIR pattern (DONE)
- [x] `assert_semantic_equivalence` — S70: NLI bidirectional, catches contradictions (DONE)
- [ ] Add "semantic_equivalence" criterion to LLM-as-Judge `DEFAULT_CRITERIA`
- [x] `ParaphraseGenerator` utility — S70: template (4 techniques) + LLM-based (DONE)
- [ ] Consider upgrading default embedding model from MiniLM to mpnet (SemScore paper, Jan 2024)

### Competitive Gaps (S66 Audit: 6 critical)
- [ ] **CG-1**: Synthetic test data generation (RAGAS, DeepEval, Giskard have it) — 2-3 sprints
- [ ] **CG-2**: Dynamic multi-turn red teaming (Promptfoo: 135 plugins, Giskard: autonomous agents) — 3-4 sprints
- [ ] **CG-3**: MCP evaluation metrics (DeepEval is first mover) — 1 sprint
- [ ] **CG-4**: Multimodal LLM evaluation (image input/output) — 2 sprints
- [ ] **CG-5**: LLM observability/tracing (Arize Phoenix 13K stars) — consider integration vs build
- [ ] **CG-6**: Automated prompt optimization — 2 sprints

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

### Claude Code Skills for mltk (Persona-Based Agent Behaviors)
*Milestone: Create `.claude/skills/` that teach Claude Code how to use mltk + mltk-MCP for specific roles*
- [ ] **mltk-qa-skill** — QA engineer persona: run mltk scan, interpret findings, write tests, validate coverage, use mltk-MCP tools. Knows which assertions to use for data quality, model validation, drift detection, bias testing.
- [ ] **mltk-dev-skill** — Developer persona: use mltk + mltk-MCP + mltk-reports for TDD, test-first development, interpreting scan results, fixing test failures, generating test suites from scan findings.
- [ ] **mltk-pm-skill** — Product Manager persona: read mltk reports, interpret ML Test Score, understand compliance status (EU AI Act, FDA, SR 11-7), generate stakeholder summaries from scan/test results, track quality trends via server dashboard.
- [ ] **mltk-devops-skill** — DevOps persona: integrate mltk in CI/CD pipelines, configure pytest markers, set up mltk server, configure webhooks/alerts, manage GitHub/Jira integrations, interpret JUnit XML output, set quality gates.
- [ ] **mltk-mcp-config** — `.mcp.json` project template that auto-registers mltk-MCP server for any Claude Code session opening the repo

### Industry Patterns (S66 Audit: 7 items)
- [ ] **IP-1**: LLM-as-Judge as default for subjective metrics (toxicity, faithfulness, coherence)
- [ ] **IP-2**: YAML-first red teaming configuration (extend YAML tests for security scans)
- [ ] **IP-3**: Solver/Scorer architecture for complex eval workflows (Inspect AI pattern)
- [ ] **IP-4**: Trace-level evaluation — extend AgentTrace to span-level scoring
- [ ] **IP-5**: Versioned evaluation datasets via registry
- [ ] **IP-6**: Document OTLP export, test with Phoenix/Langfuse
- [ ] **IP-7**: Add `mltk security-scan` CLI command for continuous red teaming

---

*Last updated: Sprint 66 (March 30, 2026) — counts verified from source*
