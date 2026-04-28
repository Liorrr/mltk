# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

### Changed

### Fixed

## [0.12.2] — 2026-04-28
### Added

### Changed

### Fixed

## [0.12.0] — 2026-04-25
### Added

#### Container & Kubernetes Friendliness (S93)
- **`mltk.container` module** — Trivy-backed container image security scanning
  - `assert_container_vulnerabilities(image, max_critical=0, max_high=0)` — pytest-native CVE threshold assertion
  - `assert_no_secrets_in_image(image)` — pytest-native exposed-secrets assertion
  - `ContainerScanner` — sibling scanner returning `ScanFinding` objects (not a `Scanner` ABC subclass)
  - `TrivyAdapter` — subprocess wrapper for Trivy JSON SchemaVersion 2 output; supports `scan_image` and `scan_fs`
  - `_binary.py` — Trivy binary auto-discovery: `PATH` → `trivy-py` installed binary → `ImportError` with install hint
- **MCP tool #12**: `mltk_container_scan(image, max_critical, max_high)` — scans image and returns structured JSON with pass/fail + CVE + secret details
- **CLI**: `mltk container scan <image>` — with `--max-critical`, `--max-high`, `--severity-floor`, `--json`, `--junit-xml` flags; exit codes 0/1/2
- **`/metrics` endpoint** on FastAPI server — Prometheus exposition format (opt-in: `pip install mltk[metrics]`); returns HTTP 404 if `prometheus_client` not installed
  - Counters: `mltk_assertions_total{status,category}`, `mltk_container_scan_vulnerabilities_total{severity}`
  - Histogram: `mltk_assertion_duration_seconds{category}`
- **Multi-architecture Docker images** on Docker Hub (`liorrr/mltk`) and GHCR (`ghcr.io/liorrr/mltk`) (published on `v*` tags via `docker-publish.yml`):
  - `:latest` / `:<version>` — `python:3.12-slim` + `mltk[all]`, `linux/amd64` + `linux/arm64`
  - `:full` / `:<version>-full` — `:latest` + Trivy 0.60.0 bundled at `/usr/local/bin/trivy`
- New docs: `guides/container-scanning.md`, `guides/container-deployment.md`
- New pyproject extras: `mltk[container]` (`trivy-py>=0.70`), `mltk[metrics]` (`prometheus-client>=0.20`); both included in `mltk[all]`
- 61 new tests (4273+ total); 2 known pre-existing leakage scanner failures unchanged

### Changed

### Fixed

## [0.11.1] — 2026-04-24
### Added

#### Persona Skills (S92)
- `mltk-qa` skill — QA engineer persona: scan → triage → assert → report workflow (174 lines)
- `mltk-dev` skill — Developer persona: TDD, failure fixes, test generation from scans (176 lines)
- `mltk-pm` skill — PM persona: ML Test Score, compliance, risk assessment, stakeholder reports (171 lines)
- `mltk-devops` skill — DevOps persona: CI/CD gates, server setup, monitoring, MCP config (221 lines)
- Updated `scripts/generate_skill_index.py` to install all `mltk-*.md` skills from repo

### Changed

#### License
- **License changed from Apache 2.0 to Elastic License 2.0 (ELv2).** Free for internal, non-commercial, and evaluation use. Redistribution as a hosted/managed service requires a commercial license. See [LICENSE](LICENSE) and [LICENSE-COMMERCIAL.md](LICENSE-COMMERCIAL.md). Prior releases (v0.9.0 and earlier) remain under Apache 2.0.

### Fixed

#### S90-S92 Audit Fixes
- **SEC-2**: `similarity.py` now uses `_backends.embedding_cosine_pairs()` instead of direct `SentenceTransformer` — restores supply-chain revision pinning
- **SEC-2**: `_backends.py` uses `revision=` kwarg directly (not `model_kwargs`) for correct SentenceTransformer API
- `server.py`: `issue_url` field now properly stringified (was passing raw object for Jira)
- `jira_adapter.py`: `add_remote_link` and `update_issue` now log warnings on failure instead of silent swallow
- `github_adapter.py`: narrowed `except Exception` to `json.JSONDecodeError/ValueError`; moved `urllib.parse` to top-level import
- `judge.py`: empty-prompts pass case uses `Severity.INFO` (was `CRITICAL`)
- `similarity.py`: removed unused `numpy` import after refactor
- `docs/api/otel.md`: fixed all 3 `MltkTracer` method signatures (were documenting wrong API)
- `docs/api/llm.md`: fixed code-gen section (wrong module `code_gen` → `codegen`, 4 wrong function names)
- `docs/api/llm.md`: `assert_summary_conciseness` → `assert_summary_compression` (function didn't exist)
- `docs/api/llm.md`: extraction payload categories 8 → 9
- `BACKLOG.md`: header S91→S92, 228→230 assertions; footer updated
- `CHANGELOG.md`: source file count 9→7

### Added

#### OTLP / OpenInference (S92, IP-6)
- OpenInference span attributes on all assertion spans (`openinference.span.kind=EVALUATION`, `eval.name`, `eval.score`, `eval.label`)
- Phoenix displays mltk assertions in native Evaluations tab (not generic spans)
- Added to both live `trace_result()` and JSON `export_json()` code paths
- Environment variable documentation (OTEL_EXPORTER_OTLP_ENDPOINT, etc.)
- End-to-end workflow examples: local Phoenix, CI/CD JSON export, Langfuse scoring
- 2 new tests for OpenInference attribute verification

#### Embedding Model Upgrade (S92)
- Default embedding model upgraded from `all-MiniLM-L6-v2` (84-85% STS) to `all-mpnet-base-v2` (87-88% STS)
- Validated by SemScore paper (Jan 2024) as best sentence-transformer for LLM evaluation
- Pinned revision `e8c3b32edf5434bc` for supply-chain defense (SEC-2)
- MiniLM still supported — pass `embedding_model="all-MiniLM-L6-v2"` for lightweight mode
- Updated all 7 source files, 4 doc files, 2 test files, regenerated API index

#### Quick Wins (S92)
- `"semantic_equivalence"` criterion in LLM-as-Judge `DEFAULT_CRITERIA` — rubric for meaning-preserving evaluation
- Per-module coverage thresholds in `pyproject.toml` (`[tool.coverage.run]` + `[tool.coverage.report]`)
- Marked E-6 (semantic similarity for prompt leak detection) as done — already implemented in S85

#### Agent Protocol + E2E Pipeline Tests (S91, F-7)
- `mltk_workflow` MCP tool (11th) — canonical agent workflow with 5 pipeline paths and severity-based decision tree
- `workflow_hint` metadata in all tool success responses — `position` (start/middle/late/end/info) + `next_tools` list for agent routing
- Severity-conditional `suggested_next_step` in `mltk_scan` JSON-report path — critical→suggest, warning→issue, info→report
- `fallback_parameters` in `_error()` responses — mid-chain recovery guidance (e.g., PR failure → issue creation)
- `.mcp.json` sample config for Claude Code / Cursor / VS Code / Cline / OpenClaw
- ~55 new tests: 34 E2E agent simulation tests + 23 workflow/response enhancement tests

#### PR Generator + Issue Linker (S90, F-5+F-6)
- `PullRequestGenerator` — create GitHub PRs from scan findings + fix suggestions via isolated git worktrees
- `PullRequestResult` dataclass — PR URL, branch name, number, draft status
- `render_pr_body()` — structured Markdown PR body (finding/fix/code sections)
- `IssueLinker` — create tracker tickets from scan findings with dedup + template rendering
- `GitHubIssuesAdapter.create_pull_request()` — GitHub REST API PR creation with draft support and label attachment
- `JiraAdapter.add_remote_link()` — link external URLs (e.g., PRs) to Jira issues
- `mltk_create_pr` MCP tool (9th) — end-to-end PR creation from finding + fix JSON
- `mltk_create_issue` MCP tool (10th) — issue creation with GitHub/Jira backends, dedup, and optional PR linking
- `"finding_issue"` ticket template for scan-finding-based issues
- ~54 new tests across 4 test files (PR generator, issue linker, MCP tools, tool registration)

#### Sandboxed Execution (S89, F-4)
- `GitWorktree` context manager — create/cleanup git worktrees for isolated experiment execution
- `SandboxedExperimentRunner(ExperimentRunner)` — runs hypotheses in isolated git worktrees via subprocess
- `git_available()` / `find_git_root()` — git CLI detection and repo root discovery
- Path traversal protection in `write_file()` — validates relative paths stay inside worktree
- Code injection prevention in assertion scripts — scanner names escaped via `json.dumps()`
- `mltk_experiment` MCP tool gains `sandbox: bool = False` parameter for worktree-based execution
- Proper `ScanFinding` construction with baseline `TestResult` in MCP sandbox path
- ~97 new tests across 4 test files (worktree, sandbox, MCP sandbox, integration)

#### Experiment Runner (S88, F-3)
- `ExperimentRunner` — test fix hypotheses against scan findings: baseline → apply fix → re-run assertion → rank results
- `Hypothesis` / `HypothesisResult` dataclasses — pair fix suggestions with apply functions, track improvement and ranking
- `ExperimentResult` — aggregated results with `selected_fix`, `any_fix_works`, `best_result` properties
- `rank_hypotheses()` — 3 ranking strategies: `passed` (binary pass/fail), `delta` (metric improvement), `composite` (weighted score)
- `mltk_experiment` MCP tool — 8th tool, heuristic ranking of fixes by confidence/category/snippet availability
- Per-hypothesis timeout with daemon thread isolation (matches ScanEngine pattern)
- `run_batch()` for testing fixes across multiple findings with `apply_fns_map` lookup
- 58 new tests (14 dataclass + 10 runner + 10 ranking + 10 integration + 14 MCP)

#### Fix Suggestion Engine (S87, F-2)
- `FixSuggestion` dataclass — category (code/config/data/process), title, description, confidence (high/medium/low), code_snippet
- `ScanFinding.suggested_fixes` — 1-3 ranked fix suggestions per finding
- `_gen_fix()` / `_gen_null_fix()` / `_gen_pii_fix()` on all 8 scanners (drift, bias, overfit, calibration, data, leakage, robustness, slice)
- `mltk_suggest` MCP tool — 7th tool, parses finding JSON, returns ranked fixes with category/confidence filtering
- `format_fixes()` console formatter with confidence tags (+++/++/+) and code snippet display
- `format_console_output(verbose=True)` shows inline fix suggestions per finding
- `ScanReport.to_json()` serializes `suggested_fixes` array per finding
- `ScanReport.summary()` shows fix count footer
- `__post_init__` validation on FixSuggestion category and confidence values
- 51 new tests (12 dataclass + 10 engine + 15 integration + 14 MCP)

### Fixed

#### MCP Server Test Debt (S86)
- Rewrote 86 MCP server tests (77 were failing due to wrong mock targets and missing `create_server()` calls)
- Split monolithic `test_server.py` into 8 focused files with shared conftest/helpers
- Fixed mock targets: patch lazy imports at source modules (`mltk.scan.*`, `mltk.eval.task.*`, etc.) instead of non-existent module-level functions
- Added autouse fixture that creates mock server and populates tool registry before every test
- Added 7 hardening tests from Opus code review (YAML-not-dict, .yml extension, verbose .py, list error path, report FAIL items, dict results_json, 50-file cap)
- Total: 93 MCP server tests, all passing (was 77 failing / 9 passing)

#### MCP Evaluation (S75)
- `assert_mcp_tool_schema_conformance` — validate tool args against JSON Schema (first-mover, no LLM needed)
- `assert_mcp_tool_selection` — server-namespace-aware tool selection (precision/recall/F1)
- `assert_mcp_resource_access` — expected/forbidden URI access patterns (unique to mltk)
- `assert_mcp_context_window` — model-aware context utilization check
- `assert_mcp_error_recovery` — detect same-tool retry loops
- `McpTrace`, `McpToolCall`, `McpResourceAccess` dataclasses (extend AgentTrace)
- New `mcp` optional dependency group: `pip install mltk[mcp]`

#### LLM-as-Judge Defaults (S80, IP-1)
- `configure_default_judge()` — set a default judge_fn for all subjective assertions
- `resolve_judge()` — priority chain: explicit > module default > fallback method
- `assert_with_judge()` — convenience wrapper with auto-fallback to lexical
- Thread-safe module-level configuration

#### LLM Observability Adapters (S80, CG-5)
- `PhoenixAdapter` — wrap any mltk assertion as a Phoenix evaluator callable
- `register_phoenix()` — one-line OTLP endpoint configuration
- `LangfuseAdapter` — wrap mltk assertion as Langfuse score function
- `assert_trace_quality` — unified CI/CD quality gate (latency + cost + score)
- New `phoenix` and `langfuse` optional dependency groups

#### Multimodal Evaluation v2 (S79)
- `assert_clip_score` — CLIPScore via open-clip or pre-computed embeddings (dual-path, zero-dep option)
- `assert_object_hallucination` — POPE-style binary probing for VLM object hallucination
- `assert_edit_preservation` — SSIM structural similarity + pixel_diff fallback
- `assert_ocr_accuracy` — CER/WER for OCR quality (pure Python Levenshtein, zero deps)

#### Multimodal Evaluation v1 (S78)
- `assert_prompt_faithfulness` — text-to-image semantic alignment via LLM judge
- `assert_image_coherence` — image-text document coherence
- `assert_image_helpfulness` — image utility for comprehension
- `assert_vqa_accuracy` — VQA correctness (judge + exact match modes)
- `ImageInput` unified type (str path, Path, bytes) with `image_description` escape hatch
- New `multimodal` and `clip` optional dependency groups

#### Red Team v2 Enhancements (S78)
- `RedTeamSession` — stateful multi-turn attack management
- 3 built-in attack chains (trust building, roleplay escalation, context poisoning)
- Confidence tiers (COMPROMISED/LIKELY/AMBIGUOUS/RESILIENT) with indicator tracking
- `llm_attacker` parameter for LLM-generated payload variants

#### Red Team Framework (S77)
- `assert_red_team_resilient` — run 55+ attack payloads across 7 OWASP categories (closes CG-2)
- `assert_no_session_jailbreak` — multi-turn conversation attack detection
- `assert_owasp_llm_coverage` — meta-assertion for OWASP category coverage
- `assert_encoding_mutation_resilience` — 8 encoding bypass techniques (Base64, ROT13, leetspeak, etc.)
- `mltk security-scan` CLI command — run red team catalog against any model function
- 55 built-in educational attack payloads across 7 categories

#### Synthetic QA v2 Enhancements (S77)
- `generate_multi_hop()` — questions requiring cross-chunk reasoning
- `generate_conversational()` — multi-turn dialogue generation
- `generate_distracting()` — questions with misleading elements from different contexts
- New QuestionType values: CONVERSATIONAL, DISTRACTING

#### Synthetic QA Generation (S76)
- `SyntheticQAGenerator` — generate synthetic QA pairs from documents (closes CG-1)
- Template mode (zero-dep, CI-safe) + LLM mode (any `Callable[[str], str]`)
- 5 question types: factual, reasoning, multi-hop, counterfactual, out-of-scope
- `QAPair` dataclass integrates directly with RAG assertions
- `QualityFilter` for LLM-generated pair scoring
- `split_text()` zero-dep word-count text splitter

#### Test Hardening (S75-S76)
- +25 tests across behavioral stability, retrieval, paraphrase generator

#### Research (S75)
- Synthetic data generation research (RAGAS, DeepEval, Giskard comparison)

## [0.9.0] — 2026-03-31

### Added

#### NER PII Detection (S73)
- `assert_no_pii(method="ner")` — Microsoft Presidio + spaCy NER for contextual PII (names, orgs, locations)
- `assert_no_pii(method="gliner")` — GLiNER zero-shot NER for domain-specific PII (healthcare MRN, legal case numbers)
- `assert_no_pii(method="hybrid")` — regex + NER union with intelligent span deduplication
- `scan_pii_dispatch()` — unified routing function for all 4 methods
- `scan_pii_ner()`, `scan_pii_gliner()`, `scan_pii_hybrid()` — standalone NER scanning functions
- New `ner` optional dependency group: `pip install mltk[ner]`

#### Test Hardening (S73)
- +22 tests across drift (MMD), calibration (SmoothECE), fairness (intersectional), behavioral (invariance)
- High-dimensional MMD, perfectly calibrated ECE, three-attribute intersectionality, all 6 paraphrase methods

#### Research (S73)
- NER PII detection research brief (Presidio architecture, GLiNER zero-shot, hybrid approach)
- Red teaming architecture research (Promptfoo 135 plugins, Giskard GOAT, hybrid recommendation)
- MCP evaluation research (JSON Schema validation, resource access, DeepEval comparison)

## [0.8.0] — 2026-03-27

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

#### LLM-as-Judge + Summarization (S62)
- `assert_llm_judge_score` — score model outputs via any LLM (vendor-neutral judge_fn callable)
- `assert_llm_judge_pairwise` — A/B comparison via LLM judge (pairwise win rate)
- `assert_summary_coverage` — key information preservation (token recall)
- `assert_summary_compression` — compression ratio bounds
- `assert_summary_faithfulness` — no hallucinated content (token precision)
- `DEFAULT_CRITERIA` — 5 built-in rubrics (helpfulness, correctness, coherence, relevance, harmlessness)

#### Recommendation Systems (S63) — FIRST-MOVER
- `assert_hit_rate`, `assert_diversity`, `assert_novelty`, `assert_coverage`, `assert_serendipity`
- Zero competitors offer recommendation system assertions as pytest assertions

#### Long-Context LLM Testing (S63)
- `assert_needle_in_haystack` — fact retrieval at configurable context positions
- `assert_context_utilization` — verify model uses multiple facts from full window
- `assert_no_lost_in_middle` — detect accuracy degradation in middle of context

#### Composable TestSuite API (S64)
- `MltkSuite` — run assertions without pytest (notebooks, scripts, CI)
- `SuiteResult` — structured results with pass_rate, duration, counts
- Export to JSON, HTML, JUnit XML via `to_json()`, `to_html()`, `to_junit()`
- Method chaining: `suite.add(...).add(...).run()`

#### Code Generation Testing (S64)
- `assert_code_executes` — subprocess isolation with timeout
- `assert_code_passes_tests` — run generated code against test cases
- `assert_no_code_vulnerabilities` — AST scan for eval/exec/shell=True/hardcoded creds
- `assert_code_complexity` — cyclomatic complexity + line count bounds

#### Test Hardening (S57, S61, S62, S63, S64)
- 208 new parametrized + edge-case tests across safety, drift, synthetic, conformal, attribution, agentic, multi-agent, GitHub App, OTEL, kubeflow, sagemaker, dvc, hipaa, counterfactual, multimodal, anomaly, audit

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
