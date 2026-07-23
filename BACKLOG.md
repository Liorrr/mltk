# MLTK Backlog

Tracked items for the ML Test Kit project. Updated after each sprint.

## Status Legend
- **DONE** -- shipped and tested
- **PLANNED** -- scheduled for a specific sprint
- **BACKLOG** -- accepted, not yet scheduled
- **IDEA** -- needs evaluation

---

## DONE (S0-S100: 241 assertions, 4969+ tests, 38 Rust tests) — v0.13.0

### S102 — Adversarial-review fix batch 2 (10 medium-severity findings)
- [x] Batch 2 of the cross-model adversarial review — all 10 confirmed mediums fixed. Prep landed first (`mltk.core.empty` + `mltk.core.validation` shared contracts, `feat(core)`; config finding #6 `wire seed, delete report_format/baseline_dir`, `fix(config)`); the remaining 8 shipped via a 4-worker Codex dispatch (lanes A–D, file-exclusive) + orchestrator gate: (#1) **BREAKING** paired-sequence length contract — ~20 `zip(strict=False)` truncation sites now `require_same_length` (raises `ValueError`) or `zip(strict=True)` for genuine invariants (pii NHS checksum); (#2) fairness metrics propagate undefined per-group TPR/FPR/PPV as `None` and exclude them instead of coercing to 0.0 → no more false bias alarms (equalized_odds/predictive_parity/equal_opportunity; selection-rate checks untouched); (#3) scan→PR/issue round-trip — `_finding_from_json` reads `to_json`'s flat shape so findings keep name/message through the MCP chain; (#4) overfit `_accuracy` shape check; (#5) `embedding_drift`/`bertscore` narrow swallow-all `except` to `ImportError` + seeded (order-unbiased) MMD sampling; (#7) Asana/Linear wired into the `mltk_create_issue` tracker allowlist; (#8) scan-engine model probe timeout-bounded (daemon-thread leak documented); (#9) `submit_run` webhook I/O offloaded to `BackgroundTasks`; (#10) six drifted `on_empty` helper copies deduped into `mltk.core.empty` (zero call-site churn, severity default reproduces all variants). Gate: ruff clean, suite **4923 green** (+56). Bundles with S101 toward a **0.14.0** release (multiple BREAKING changes — see CHANGELOG). One worker bug caught at the gate (undefined `n` in judge pairwise) + 19 B905/6 I001 lint fixes applied by the orchestrator.

### S101 — Adversarial-review fix batch 1 (7 high-severity findings)
- [x] Cross-model adversarial review (3 Codex reviewers: Skeptic/Architect/Minimalist + Fable lead pass) of the 128 pre-PR-era direct-to-master commits produced ~30 deduped findings; batch 1 fixed all 7 confirmed highs via 7-worker Codex dispatch + 1 fix round: (1) experiment sandbox now replays the original assertion instead of hardcoding `passed=True`; (2) MCP `mltk_scan` reports `scan_performed` honestly; (3) **BREAKING** `on_empty="fail"` default across ~25 empty-input gates (rag/retrieval/conversation/recommendation/codegen/synthetic) + zero-IDCG query exclusion; (4) `assert_calibration` rejects out-of-range/NaN probs and non-binary labels; (5) recursive assertion discovery + container/cost/eval targets (229 discoverable, was 218); (6) env/yaml config actually consumed via new `MltkConfig.explicit_fields` provenance (explicit-only, legacy defaults preserved); (7) server storage serialized behind a lock with Row factory set once (fail-first concurrency test). Suite 4867 green.


### S100 — v0.13.0 milestone release (importer epic + S95–S99 → PyPI)
- [x] Cut **v0.13.0** (first release since v0.12.7): rolled CHANGELOG (`[Unreleased]` → `[0.13.0]`, bundling S95–S99), bumped version (pyproject/Cargo/`__init__`), refreshed doc-counts + prose version, regenerated skill index. Fixed a `bump.py` history-clobber bug — `_update_backlog_header` rewrote *every* DONE bucket's version suffix; now `count=1` so historical buckets keep the version they shipped at. Ships the complete Smart Dataset Importer epic + 9 net-new assertions (S95/S96) to PyPI (`mlspec`) + Docker/GHCR via the tag-triggered Release workflow.

### S99 — Smart Importer Sprint 3: MCP tool + golden binding + registry (epic complete)
- [x] `mltk_import` MCP tool (#13, return-only + opt-in file write + golden/register params; `EXPECTED_TOOLS` 12→13), `bind_golden()`/`load_golden()` golden-set binding (key-column or row-order join; `metadata["scoring"]` exact/judge partition), `--judge` codegen fallback (`assert_llm_judge_score` one-arg contract; `judge_fn` fixture via `MLTK_JUDGE_FN`), `register_dataset()` (blocking `assert_dataset_quality` gate → `DatasetRegistry`), CLI `--golden`/`--golden-target-column`/`--golden-key`/`--golden-key-column`/`--judge`/`--register` — 42 new tests, no new deps. Implemented directly (single-session, tight golden↔codegen contract coupling) rather than via Codex batch.

### S98 — Smart Importer Sprint 2: classifier + suite generator + pytest emitter + CLI
- [x] `classify_task()` (5-type taxonomy from column roles), `build_suite()` (two-tier semantics + self-passing baseline thresholds), `generate_pytest()` (byte-deterministic committable scaffold; Tier-2 tests un-skip via `MLTK_PREDICT_FN` or one fixture edit), `mltk import <source>` CLI (emit-by-default, `--force` overwrite protection) — 46 new tests, no new deps; acceptance gate: emitted file runs green via subprocess. First multi-worker Codex dispatch batch (4 parallel workers + 1 fix round).

### S97 — Smart Importer Sprint 1: loader + auto-mapper (+ CI bring-up)
- [x] `mltk.importer` package: `DatasetImporter` (CSV/JSON/Parquet/HF Hub), token-based column-role heuristics v2, exclusive overrides, `to_eval_dataset()` conventions — 200 new tests, `mltk[importer]` extra. Review-hardened via 4-perspective PR review; CI brought up for real (15/15 green, first time ever). PR #3.

### S96 — Structured Output + Cost Tracking
- [x] 5 net-new assertions: `assert_valid_json` / `assert_json_schema` / `assert_pydantic_schema` (llm), `assert_cost_within` / `assert_token_usage` (new `mltk.cost` package) — 76 new tests, jsonschema + pydantic optional deps

### S95 — First-Mover Assertions
- [x] 4 net-new assertions: `assert_no_unicode_attacks` (llm), `assert_pipeline_stages_compatible` + `assert_pipeline_resilient` (pipeline), `assert_combinatorial_coverage` (testing) — 88 new tests (incl. 8 from Opus review fixes), no new deps

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

## DONE (S0-S92: 241 assertions, 4969+ tests, 38 Rust tests) — v0.12.4

### Phase A: Core Library (S0-S10) -- v0.1.0
- [x] S0: Project skeleton, pyproject.toml, Cargo.toml, CI/CD
- [x] S1: Config, 8 data assertions, MkDocs docs
- [x] S2: 4 drift methods, 14 PII patterns, Rust KS/PSI
- [x] S3: 9 model metrics, regression, slicing, calibration
- [x] S4: 5 bias methods, adversarial, --mltk-report
- [x] S5: Inference (latency, throughput, contract), 19 top-level + 5 groups CLI
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
- [x] S41: VS Code extension (mltk-vscode), NLP/Speech module refactoring, 19 top-level + 5 groups CLI total

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

### Adversarial-review follow-ups
- [x] ~~Dedup the ~6 cosine/jaccard/token-overlap similarity reimplementations flagged by the Minimalist (residual half of S102 finding #10).~~ **WON'T-FIX (2026-07-21):** on inspection these are similar-looking but semantically distinct, not copies — `agentic._token_overlap` returns 1.0 for two-empty vs `behavioral/retrieval._jaccard` 0.0 (documented "two empty retrievals share nothing") vs `coherence` 0.0 for either-empty; `judge_defaults` is an asymmetric overlap ratio `|A∩B|/|A|`, a different formula; the cosines differ in zero-norm handling (`== 0.0` vs `< 1e-10`) and input type (list vs ndarray), and `bertscore._cosine` runs in an O(n·m) hot loop where routing through a shared/`_rust` helper risks a perf regression. No shared helper preserves all behavior as a net win, so the sites stay separate by design.

*(All other adversarial-review batch 2 mediums shipped in S102 — see DONE.)*

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

### First-Mover Assertions (S95 — DONE)
*Source: Obsidian v0.9.0 Epic Plan — First-Mover Opportunities*
- [x] **S95** `assert_no_unicode_attacks` — Zero-width (cat Cf, legit Arabic/Syriac allowlisted), bidi override (Trojan Source CVE-2021-42574), mixed-script homoglyph detection + `detect_unicode_attacks()` (28 tests)
- [x] **S95** `assert_pipeline_stages_compatible` — Inter-stage schema validation via `StageSpec(produces/requires)`, distinct from data contracts (17 tests)
- [x] **S95** `assert_pipeline_resilient` — ML chaos engineering: 7-fault injection catalog, graceful-degradation assertion + `apply_fault()`/`DEFAULT_FAULTS` (22 tests)
- [x] **S95** `assert_combinatorial_coverage` — NIST t-way covering-array coverage measurement + `combinatorial_coverage()` (21 tests)

### First-Mover Sprint Follow-ups (S95 Opus review — deferred P2 enhancements)
- [x] `assert_pipeline_stages_compatible` — dtype canonicalization (`np.dtype(x).name` + alias map + lowercase fallback) so `int64`/`Int64`/`int` compare equal (2026-07-05, codex-worker pilot; widening-aware predicate still open below)
- [ ] `assert_pipeline_stages_compatible` — optional widening-aware predicate (e.g. `int32` satisfies `int64` requirement)
- [ ] `assert_pipeline_resilient` — optional `validate_output: Callable[[Any], bool]` so silent `None`/garbage output counts as a failure (currently "graceful == no exception raised" only)
- [ ] `assert_no_unicode_attacks` — single-script confusable detection (whole-word Cyrillic/Greek spoofs, e.g. an all-Cyrillic lookalike) + widen confusable set beyond Cyrillic/Greek (fullwidth, mathematical-alphanumeric)

### Structured Output & Cost Tracking (S96 — DONE)
*Source: roadmap.md Tier 2-3, Competitors & Positioning*
- [x] **S96** JSON Schema / Structured Output Validation — `assert_valid_json` / `assert_json_schema` / `assert_pydantic_schema` (jsonschema + pydantic optional deps, fail-clean on malformed input)
- [x] **S96** Cost and Token Tracking per Assertion/Suite — `mltk.cost` package: `MODEL_PRICING` (15 models), `estimate_cost`, `register_pricing`, `CostTracker`, `assert_cost_within` / `assert_token_usage`

### Advanced Features
- [ ] Test impact analysis (dependency graph)
- [ ] Anomaly detection on test result time series

### Smart Dataset Importer / Test-Suite Mapper (DONE — S97–S99, epic complete)
*One-click: point at a dataset + golden set → auto-generated, runnable mltk eval suite.*
- [x] `DatasetImporter` (S97) — load datasets from HuggingFace Hub (`datasets`, mocked in tests) and local CSV/Parquet/JSON; normalize to columns/dtypes/rows. URL adapter deferred (`NotImplementedError` placeholder — download locally first).
- [x] Schema/column auto-mapper (S97) — `ColumnRole`/`ColumnMapping`/`auto_map_columns()` infer field roles (input/golden/context/label/metadata) via deterministic name + dtype heuristics; `ColumnMapping.preview()`/`.override()` for a user-confirmable mapping; `ImportResult.to_eval_dataset()` materializes an `EvalDataset`. New `src/mltk/importer/` package, optional `mltk[importer]` extra (`datasets`, `pyarrow`), 4969+ tests, no network in tests.
- [x] Task-type detection → suite generation (S98) — `classify_task()` (5-type taxonomy from role presence) + `build_suite()` (two-tier: dataset-quality baselines always; judge-scored golden/context checks when a `judge_fn` is given) + `generate_pytest()` (committable, byte-deterministic, ast-gated scaffold; Tier-2 model tests skipped behind a `predict_fn` fixture / `MLTK_PREDICT_FN`). Acceptance gate: emitted file for the bundled fixture runs green via subprocess.
- [x] CLI entrypoint (S98) — `mltk import <source>`: preview → classify → suite summary → emit-by-default pytest file with `--force` overwrite protection. NOTE: descoped from the original wording — the CLI does not run the suite through the eval pipeline (solvers/scorers) at import time; running the emitted file is `pytest <file>`, and solver/scorer wiring belongs to the S99 registry/eval integration.
- [x] MCP tool + registry integration (S99) — `mltk_import` (tool #13, return-only + opt-in write); `register_dataset()` runs a blocking `assert_dataset_quality` gate before saving to the versioned `DatasetRegistry` (`~/.mltk/datasets/`, `MLTK_DATASET_DIR` override).
- [x] Golden-set binding (S99) — `bind_golden()` maps a provided golden/reference file onto imported samples (key-column or row-order join); samples with no exact golden are stamped `"judge"` and the emitted scaffold falls back to `assert_llm_judge_score` (opt-in via `--judge`).
- [ ] Pluggable source adapters — HuggingFace first; design for Kaggle, OpenML, local files, and object storage later *(deferred beyond the epic)*

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
- [x] **mltk-index** — Codebase index skill (241 assertions, 11 MCP, 28 CLI, file:line pointers). Generated by `scripts/generate_skill_index.py`
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
