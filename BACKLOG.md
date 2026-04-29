# MLTK Backlog

Tracked items for the ML Test Kit project. Updated after each sprint.

## Status Legend
- **DONE** -- shipped and tested
- **PLANNED** -- scheduled for a specific sprint
- **BACKLOG** -- accepted, not yet scheduled
- **IDEA** -- needs evaluation

---

## DONE (S0-S93: 232 assertions, 4291+ tests, 38 Rust tests) — v0.12.3

### S93 — Container & Kubernetes Friendliness
- [x] Multi-stage Dockerfile: `runtime-slim` (`:latest`) + `runtime-full` (`:full`, Trivy 0.60.0 bundled)
- [x] GHA `docker-publish.yml`: multi-arch (amd64 + arm64 native), GHCR push on `v*` tags to `ghcr.io/liorrr/mltk`
- [x] `src/mltk/container/` module: `TrivyAdapter`, `ContainerScanner`, `assert_container_vulnerabilities`, `assert_no_secrets_in_image`
- [x] MCP tool #12: `mltk_container_scan`
- [x] CLI: `mltk container scan <image>` with `--json`, `--junit-xml` flags
- [x] Prometheus `/metrics` endpoint on FastAPI server (opt-in: `mltk[metrics]`)
- [x] Docs: `guides/container-scanning.md`, `guides/container-deployment.md`
- [x] pyproject extras: `mltk[container]`, `mltk[metrics]` (both in `mltk[all]`)
- [ ] v0.12.0 PyPI publish — separate release PR after soak on master

### PyPI Name Claim — PEP 541 (A2-quiet, started 2026-04-25)
- [ ] **Week 0**: Send first outreach email to manan.shah.777@gmail.com (see `audit/pypi-claim/email-template.md`)
- [ ] **Week 0**: Configure Trusted Publisher on TestPyPI: project `mlspec`, owner `Liorrr`, repo `mltk`, workflow `release.yml`
- [ ] **Week 0**: Create GitHub Actions environment `test-pypi` in repo settings
- [ ] **Week 0**: Capture archive.org snapshots (github.com/mananshah99/mltk 404 + TestPyPI page)
- [ ] **Week 2**: Second outreach email
- [ ] **Week 4**: Third outreach email
- [ ] **Week 6**: File pypi-support issue (see `audit/pypi-claim/issue-body.md`)
- [ ] **Week 16 deadline (2026-08-15)**: If no PSF decision → switch to Path B (full rename to ml-test-kit everywhere)
- [ ] If claim granted → `mlspec` becomes deprecation alias for `mltk`, docs updated to `pip install mltk`
- [ ] If claim denied → `mlspec` becomes primary brand; decide whether to unify import name too (see Obsidian `MLTK/PyPI Rename Plan.md`)

### S94 — Golden Data DB Connection / Gateway (PLANNED)
- [ ] User-based DB connection or mltk gateway service for golden datasets
- [ ] Helm chart for mltk server (deferred from S93)
- [ ] CycloneDX ML-BOM SBOM export (deferred from S93)
- [ ] KinD-based container integration tests (deferred from S93)

---

## DONE (S0-S92: 232 assertions, 4291+ tests, 38 Rust tests) — v0.12.3

### Phase A: Core Library (S0-S10) -- v0.1.0
- [x] S0: Project skeleton, pyproject.toml, Cargo.toml, CI/CD
- [x] S1: Config, 8 data assertions, MkDocs docs
- [x] S2: 4 drift methods, 14 PII patterns, Rust KS/PSI
- [x] S3: 9 model metrics, regression, slicing, calibration
- [x] S4: 5 bias methods, adversarial, --mltk-report
- [x] S5: Inference (latency, throughput, contract), 28 CLI commands
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
- [x] S41: VS Code extension (mltk-vscode), NLP/Speech module refactoring, 28 CLI commands total

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
- [x] S75: MCP evaluation (5 assertions, McpTrace dataclass), test hardening (+25), synthetic data research
- [x] S76: SyntheticQAGenerator (template + LLM, 5 question types), test hardening (+20), NER PII research
- [x] S77: Red Team v1 (4 assertions, 55 payloads, 8 mutations), Synthetic v2 (multi-hop, conversational, distracting), security-scan CLI
- [x] S78: Multimodal v1 (4 assertions, ImageInput, judge-based), Red Team v2 (sessions, chains, tiers, LLM attacker)
- [x] S79: Multimodal v2 (CLIPScore, POPE, SSIM, OCR — completes CG-4)
- [x] S80: LLM-as-Judge defaults (IP-1) + Phoenix/Langfuse observability (CG-5)
- [x] S81: YAML-first red team configuration (IP-2)
- [x] S82: Solver/Scorer evaluation pipeline (IP-3)
- [x] S83: Span-level trace evaluation (IP-4)
- [x] S84: Versioned evaluation datasets (IP-5)
- [x] S85: MCP server mode (F-1) — FastMCP server with 6 tools (scan/test/list/eval/dataset/report)
- [x] S86: MCP server test debt cleanup — rewrote 93 tests (was 77 failures), split to 8 files, Opus-reviewed with 7 hardening tests
- [x] S87: Fix Suggestion Engine (F-2) — FixSuggestion dataclass, _gen_fix() on all 8 scanners, mltk_suggest MCP tool (7th), console/JSON integration, 51 new tests
- [x] S88: Experiment Runner (F-3) — ExperimentRunner + Hypothesis/Result dataclasses, rank_hypotheses (3 strategies), mltk_experiment MCP tool (8th), 58 new tests
- [x] S89: Sandboxed Execution (F-4) — GitWorktree context manager, SandboxedExperimentRunner subclass, mltk_experiment sandbox param, path traversal + injection protection, 97 new tests
- [x] S90: PR Generator + Issue Linker (F-5+F-6) — PullRequestGenerator, IssueLinker, GitHubIssuesAdapter.create_pull_request(), JiraAdapter.add_remote_link(), mltk_create_pr + mltk_create_issue MCP tools (10 total), 54 new tests
- [x] S91: Agent Protocol + E2E Pipeline Tests (F-7) — mltk_workflow tool (11th), workflow_hint metadata, severity-conditional routing, fallback_parameters, .mcp.json, 55 new E2E/workflow tests

---

## BACKLOG (not yet scheduled)

### URGENT — Method Fixes (S66 Audit: 3 REJECT items) — ALL DONE
*Audit report: `docs/research/project-audit-s66.md`*
- [x] **R-1**: S67 — Multi-method dispatch for hallucination (lexical/embedding/nli/llm)
- [x] **R-2**: S67 — Multi-method dispatch for RAG assertions (faithfulness, relevancy)
- [x] **R-3**: S68 — Toxicity classifier via toxic-bert

### Method Enhancements (S66 Audit: 6 items)
- [x] **E-1**: S72 — MMD multivariate drift (RBF multi-bandwidth, permutation test, pure numpy)
- [x] **E-2**: S72 — SmoothECE calibration (reflected Gaussian kernel, auto-bandwidth, ICLR 2024)
- [x] **E-3**: S68 — BERTScore limitation warnings (antonymy blindness, number blindness)
- [x] **E-4**: S72 — Intersectional fairness (Crenshaw, Cartesian product, min_subgroup_size=30)
- [x] **E-5**: S73 — NER PII detection (Presidio + spaCy + GLiNER + hybrid method dispatch)
- [x] **E-6**: S85 — Semantic similarity method for system prompt leak detection (already implemented)

### Testing Infrastructure (S66 Audit: 5 items) — ALL DONE
- [x] **A-1**: S71a — Hypothesis property-based testing
- [x] **A-2**: S68 — pytest-xdist for parallel execution
- [x] **A-3**: S68 — pytest-randomly for order independence
- [x] **A-4**: S71b — syrupy snapshot testing for HTML/XML reports
- [x] **A-5**: S92 — Per-module coverage thresholds (pyproject.toml config)

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
- [x] Add "semantic_equivalence" criterion to LLM-as-Judge `DEFAULT_CRITERIA` (S92)
- [x] `ParaphraseGenerator` utility — S70: template (4 techniques) + LLM-based (DONE)
- [x] S92 — Upgraded default embedding model from MiniLM to mpnet (SemScore paper, Jan 2024)

### Competitive Gaps (S66 Audit: 6 critical)
- [x] **CG-1**: S76 — SyntheticQAGenerator (template + LLM modes, 5 question types, quality filter)
- [x] **CG-2**: S77 — Red Team v1 (4 assertions, 55 payloads, 8 encoding mutations, multi-turn, security-scan CLI)
- [x] **CG-3**: S75 — MCP evaluation (5 assertions, JSON Schema validation, resource access, context window)
- [x] **CG-4**: S78 — Multimodal v1 (4 assertions: faithfulness, coherence, helpfulness, VQA)
- [x] **CG-5**: S80 — Phoenix + Langfuse adapters, assert_trace_quality, register_phoenix
- [ ] ~~**CG-6**: Automated prompt optimization~~ — REMOVED (dilutes "pytest for ML" message)

### First-Mover Assertions (Epic Plan — not yet tracked)
*Source: Obsidian v0.9.0 Epic Plan — First-Mover Opportunities*
- [ ] `assert_no_unicode_attacks` — Zero-width, homoglyph, bidi attack detection (defense exists in tokenizer, user-facing assertion missing)
- [ ] `assert_pipeline_stages_compatible` — Inter-stage schema validation (distinct from data contracts)
- [ ] `assert_pipeline_resilient` — ML chaos engineering primitives (inject faults, assert graceful degradation)
- [ ] `assert_combinatorial_coverage` — NIST Covering Arrays / SDCC input space coverage

### Structured Output & Cost Tracking
*Source: roadmap.md Tier 2-3, Competitors & Positioning*
- [ ] JSON Schema / Structured Output Validation — validate LLM outputs against JSON Schema, Pydantic models (competitors DeepEval, Promptfoo have this; jsonschema dep already present from S75)
- [ ] Cost and Token Tracking per Assertion/Suite — budget alerts, cost-per-suite reports (provider-specific adapters needed)

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

### ML Platform Integration — ALL DONE (S57)
- [x] Kubeflow pipeline assertions
- [x] SageMaker Pipeline step validation
- [x] Weights & Biases adapter
- [x] DVC data version assertions

### Advanced ML Testing — ALL DONE (S59)
- [x] Counterfactual fairness testing
- [x] Causal inference validation
- [x] Federated learning test patterns — skipped (no demand)
- [x] Multi-modal (image+text) evaluation
- [x] Reinforcement learning reward validation

### Enterprise — ALL DONE (S58)
- [x] RBAC for server platform
- [x] Audit log export (SOC 2 compatible)
- [x] Custom compliance framework builder
- [x] HIPAA compliance report template

### MCP Server Expansion
*Source: docs/research/agent-integration-research.md*
- [ ] MCP HTTP transport + OAuth 2.1 — remote/enterprise mode (stdio-only limits to local use)
- [ ] MCP registry publishing — publish to Smithery, mcp.so for discoverability

### Claude Code-Native Project Skeleton
*Source: Obsidian MLTK/Idea - Claude Code Native Project.md*
- [ ] Full `.claude/` folder shipped with repo — CLAUDE.md at root, subagents, workflow recipes, hooks, memory seed (4 persona skills done in S91-S92; surrounding infrastructure not)

### Claude Code Skills for mltk
*Skills that teach Claude Code how to use mltk for specific roles*
- [x] **mltk-index** — Codebase index skill (232 assertions, 11 MCP, 28 CLI, file:line pointers). Generated by `scripts/generate_skill_index.py`
- [x] **mltk-templates** — Development templates skill (assertion/scanner/MCP/CLI patterns). Source: `skills/mltk-templates.md`
- [x] **mltk-mcp-config** — S91 — `.mcp.json` project template for MCP server registration
- [x] **mltk-qa-skill** — S92 — QA engineer persona: scan, interpret findings, write tests, use MCP tools
- [x] **mltk-dev-skill** — S92 — Developer persona: TDD, fix test failures, generate test suites from scan findings
- [x] **mltk-pm-skill** — S92 — PM persona: interpret ML Test Score, compliance status, stakeholder summaries
- [x] **mltk-devops-skill** — S92 — DevOps persona: CI/CD integration, server setup, webhooks, quality gates

### Industry Patterns (S66 Audit: 7 items)
- [x] **IP-1**: LLM-as-Judge as default for subjective metrics (S80)
- [x] **IP-2**: YAML-first red teaming configuration (S81)
- [x] **IP-3**: Solver/Scorer architecture for complex eval workflows (S82)
- [x] **IP-4**: Trace-level evaluation — span-level scoring (S83)
- [x] **IP-5**: Versioned evaluation datasets via registry (S84)
- [x] **IP-6**: S92 — OTLP: OpenInference attributes, env var docs, workflow examples
- [x] **IP-7**: S77 — `mltk security-scan` CLI command for continuous red teaming

---

### VS Code Extension Sync
- [x] S93 — Foundation sync: 16 new commands, snippets 7→44, 6 new TS modules (v0.5.0)
- [x] S94 — Agent workflow: MCP client, fix suggestion panel, eval pipeline UI (v0.6.0)
- [x] MCP client integration: JSON-RPC 2.0 over stdio, 11 typed tool wrappers
- [x] Version alignment: extension v0.6.0 synced with mltk v0.9.0

*Last updated: Sprint 94 (April 5, 2026) — VS Code extension sync complete, roadmap in mltk-vscode/BACKLOG.md*
