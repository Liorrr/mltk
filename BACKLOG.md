# MLTK Backlog

Tracked items for the ML Test Kit project. Updated after each sprint.

## Status Legend
- **DONE** -- shipped and tested
- **PLANNED** -- scheduled for a specific sprint
- **BACKLOG** -- accepted, not yet scheduled
- **IDEA** -- needs evaluation

---

## DONE (S0-S22: 84+ assertions, 496 tests, 13 Rust tests)

### Sprint 0 -- Project Skeleton
- [x] Repo structure, pyproject.toml, Cargo.toml, CI/CD, Makefile, README

### Sprint 1 -- Core + Data Quality + Docs
- [x] Config loading (TOML + YAML), assert_schema, assert_no_nulls, assert_dtypes
- [x] assert_range, assert_unique, assert_no_outliers, assert_freshness, assert_row_count
- [x] MkDocs documentation site

### Sprint 2 -- Data Drift + PII + Labels + Rust
- [x] assert_no_drift (KS, PSI, KL, chi2), assert_no_pii (14 patterns), assert_label_balance, assert_label_coverage
- [x] Rust: real KS test + PSI implementations

### Sprint 3 -- Model Quality
- [x] assert_metric (9 metrics), assert_no_regression, save_baseline, assert_slice_performance, assert_calibration

### Sprint 4 -- Bias + Adversarial + pytest Plugin
- [x] assert_no_bias (5 fairness methods), assert_robust, --mltk-report, ml_config/ml_report fixtures, ml_smoke/ml_gpu markers

### Sprint 5 -- Inference + CLI
- [x] assert_latency, assert_cold_start, assert_throughput, assert_api_contract
- [x] CLI: mltk init, scan, drift, version, score

### Sprint 6 -- Reports + Pipeline + ML Test Score
- [x] generate_report (HTML dark theme), compute_ml_test_score, assert_reproducible, assert_checksum, assert_pipeline

### Sprint 7 -- CV Domain Kit
- [x] compute_iou, assert_iou, assert_map (COCO mAP), assert_frame_accuracy, assert_temporal_consistency, assert_topk_accuracy
- [x] examples/mycompany_cv_test.py

### Sprint 8 -- NLP + Speech Domain Kits
- [x] NLP: assert_bleu, assert_rouge, assert_ner_f1, assert_no_prompt_injection
- [x] Speech: assert_wer, assert_cer, assert_rtf, assert_accent_coverage

### Sprint 9 -- Monitoring + Tabular
- [x] assert_no_degradation, assert_sla
- [x] assert_feature_drift, assert_feature_importance_stable, assert_class_balance

### Sprint 10 -- v0.1.0 Release
- [x] PyPI publish (v0.1.0 on PyPI)
- [x] Cross-platform wheels (maturin-action: Linux, macOS, Windows)
- [x] README overhaul with badges, feature matrix, install instructions
- [x] CONTRIBUTING.md with dev setup, code style, PR process
- [x] GitHub repo (Liorrr/mltk)

### Sprint 11 -- Data Contracts + Drift Expansion
- [x] Data contracts engine: YAML spec, validate_data(), generate_tests_from_contract()
- [x] CLI: mltk contract init/validate/generate-tests
- [x] Jensen-Shannon divergence (method="js"), Wasserstein (method="wasserstein"), auto-select (method="auto")
- [x] Embedding drift: assert_no_embedding_drift (cosine centroid, euclidean centroid, MMD)

### Sprint 12 -- LLM/GenAI Evaluation
- [x] assert_semantic_similarity (token-level F1, lightweight)
- [x] assert_no_toxicity (regex + keyword patterns)
- [x] assert_no_hallucination (keyword overlap)
- [x] assert_ttft, assert_itl (LLM streaming latency)

### Sprint 13 -- PII Expansion + Training Bug Detection P0
- [x] 10 Tier 1 PII patterns: IPv4, IPv6, JWT, PEM, DB conn, Stripe, Bearer, Google API, IBAN, URL auth
- [x] Luhn checksum for credit cards
- [x] assert_no_train_test_overlap, assert_temporal_split, assert_no_target_leakage

### Sprint 14 -- Jira + PM Integrations
- [x] IssueTrackerAdapter ABC (vendor-agnostic)
- [x] JiraAdapter with lazy init, custom field mapping
- [x] TicketDecisionEngine: content hash dedup + cooldown + severity filter
- [x] 5 ML ticket templates

### Sprint 15 -- Wiring Audit + Face Recognition
- [x] assert_face_far (FAR at operating point)
- [x] Wiring audit: fixed 13 gaps (deps, CLI, exports, examples)
- [x] 30+ MkDocs doc pages aligned with code
- [x] Top-level convenience imports
- [x] Examples: nlp, llm, contract, training

### Sprint 16 -- Advanced CV Tracking + Training Bug P1 + Docs Deploy
- [x] CV tracking: assert_mota, assert_motp, assert_idf1 (CLEAR-MOT)
- [x] Training P1 gradient: assert_gradient_flow, assert_no_vanishing_gradient, assert_no_exploding_gradient, assert_loss_finite
- [x] Training P1 numerical: assert_no_nan_inf, assert_loss_decreasing, assert_no_loss_divergence, assert_softmax_valid
- [x] Docs deployment: Dockerfile + nginx + deploy.sh for company server
- [x] mltk[torch] optional dependency

### Documentation Audit
- [x] All source files: Args/Returns/Example on every function
- [x] All test files: scenario + WHY + expected on every test
- [x] All doc pages: aligned with code signatures

---

## DONE (continued)

### Sprint 17 -- QA Practitioner Toolkit + EU AI Act Compliance
- [x] YAML test definitions: mltk.testdefs (schema, runner, pytest hook)
- [x] EU AI Act compliance report: mltk.compliance (article mapping, risk classification, HTML evidence)
- [x] mltk doctor: 9 diagnostic checks with fix hints
- [x] Env var config: MLTK_* prefix, highest priority in cascade
- [x] CLI expansion: mltk doctor, mltk test, mltk compliance (11 total commands)
- [x] pytest JSON export: --mltk-export-json flag

### Sprint 18 -- v0.2.0 Release + Israel PII + Polish
- [x] Israel PII: Teudat Zehut (9 digits + Luhn), Israel phone numbers
- [x] IBAN MOD-97 checksum validation
- [x] README overhaul (66+ assertions, 11 CLI commands, new features)
- [x] Version bump to v0.2.0
- [x] Backlog cleanup

---

## PLANNED

### Sprint 19 -- Production Integrations
- [x] MLflow integration (MlflowLogger, --mltk-mlflow flag)
- [x] Jupyter notebook (_repr_html_ on TestResult/TestSuite)
- [x] Model card generator (Google Model Cards from test results)

### Sprint 20 -- Cloud Monitoring
- [x] AWS SageMaker: endpoint health, latency, error rate
- [x] GCP Vertex AI: endpoint health, prediction latency
- [x] Azure ML: endpoint health, latency
- [x] Prometheus: PromQL queries, GPU utilization, Triton health

---

## PLANNED

### Sprint 21 -- Rust Acceleration + Benchmarks
- [x] Rust: KL divergence, chi-squared, JS divergence, Wasserstein
- [x] Rust: PII regex scanning (regex crate)
- [x] Python bridge + scipy/numpy fallbacks
- [x] Benchmarks: bench_drift.py + bench_pii.py

### Sprint 22 -- Training Bug P2
- [x] Augmentation: assert_no_augmentation_on_test, assert_augmentation_preserves_signal
- [x] Checkpoint: assert_checkpoint_complete, assert_resume_loss_continuous
- [x] Distributed: assert_effective_batch_size, assert_gradient_sync
- [x] Memory: assert_no_memory_leak, assert_loss_is_detached

### Sprint 23 -- v0.3.0 Release + PII Tier 3 (ACTIVE)
- [ ] PII Tier 3: UK NHS, UK NINO, Germany Steuer-ID, India Aadhaar, India PAN
- [ ] Version bump to v0.3.0
- [ ] README + BACKLOG cleanup
- [ ] Rust: KL divergence, chi-squared, Wasserstein, JS divergence
- [ ] Rust: PII regex scanning (regex crate)
- [ ] Python bridge: update _rust.py with new functions + fallbacks
- [ ] Benchmarks: drift + PII speed comparison (Rust vs scipy/numpy)

### Sprint 22 -- Test Resource Registry
- [ ] `mltk pull <collection>` -- download test definitions from registry
- [ ] `mltk push <collection>` -- share test suites with team
- [ ] `mltk list` -- browse available test collections
- [ ] Versioned test resources with dependency tracking
- [ ] Server API (REST) for hosting registry (company or SaaS)

### Sprint 23 -- Chat Interface
- [ ] `mltk chat` CLI -- interactive chat for querying test results
- [ ] Web-based chat embedded in HTML reports
- [ ] "Why did my model regress?" → analyzes test results + suggests fixes
- [ ] "What tests should I run?" → auto-generates test plan

### Sprint 24 -- Resource Summarization + AI Predictions
- [ ] Project/folder/collection summarization (text/doc, image/PNG, video/MP4)
- [ ] AI-powered predictions from prompt + resources/features
- [ ] "Predict if this model version is safe to deploy" → risk assessment
- [ ] Historical trend analysis from test result archives

---

## BACKLOG (not yet scheduled)

### Core Enhancements
- [x] ~~YAML-driven test definitions~~ (DONE S17: mltk.testdefs)
- [ ] Test registry with @register_test decorator for custom assertions
- [x] ~~Config from environment variables~~ (DONE S17: MLTK_* prefix)
- [ ] Plugin system for third-party assertion libraries

### Training Bug Detection P2
- [ ] Data augmentation: assert_no_augmentation_on_test, assert_label_augmentation_consistency, assert_augmentation_preserves_signal
- [ ] Checkpoint/resume: assert_checkpoint_complete, assert_resume_lr_matches, assert_resume_loss_continuous, assert_optimizer_state_loaded
- [ ] Distributed training: assert_distributed_sampler_used, assert_effective_batch_size, assert_gradient_sync
- [ ] Memory leaks: assert_no_memory_leak, assert_loss_is_detached, assert_grad_accumulation_correct

### Testing Patterns
- [ ] Statistical assertion primitives (tolerance bands, semantic similarity)
- [ ] Smart test selection (only re-run tests affected by specific changes)
- [ ] Golden test set management (versioned baselines)
- [ ] Non-deterministic test retry with confidence intervals
- [ ] Flaky test detection + quarantine

### PII Tier 2-4
- [ ] Israel Teudat Zehut (9 digits + Luhn + keyword anchor)
- [ ] UK NHS, UK NINO, Germany Steuer-ID, France NIR, Italy Codice Fiscale, Spain DNI
- [ ] India Aadhaar (Verhoeff checksum), India PAN
- [ ] International phone numbers, MAC addresses, crypto wallets
- [ ] Allowlists for known-safe patterns

### Regulatory / Compliance
- [x] ~~EU AI Act compliance report~~ (DONE S17: mltk.compliance)
- [ ] FDA AI device audit trail export
- [ ] Bias/fairness report with demographic breakdowns
- [ ] Regulatory compliance automation (test results → evidence docs)

### Performance / Rust
- [ ] Rust: KL divergence, chi-squared, Wasserstein, JS divergence
- [ ] Rust: fast PII scanning with regex crate
- [ ] Rust: SIMD-accelerated cosine similarity for embedding drift
- [ ] Rust: BERTScore token matching
- [ ] Benchmarks vs Great Expectations, Deepchecks, Evidently

### Integrations
- [ ] LinearAdapter, AsanaAdapter, GitHubIssuesAdapter
- [ ] VS Code extension (inline test results)
- [ ] Jupyter notebook integration (rich output for assertions)
- [ ] MLflow integration (log mltk results as MLflow metrics)

### Monetization (Pro tier)
- [ ] Cloud dashboard: hosted report aggregation, team views
- [ ] CI/CD GitHub App: auto-run mltk on PRs
- [ ] Compliance PDF exports (EU AI Act, FDA)
- [ ] Custom domain kits (healthcare, fintech, autonomous)
- [ ] Priority support with SLA

---

## IDEAS (needs evaluation)

- [x] ~~`mltk doctor`~~ (DONE S17: 9 diagnostic checks)
- [ ] Visual diff for model predictions (before/after comparison)
- [ ] Slack/Teams notifications for drift alerts
- [ ] Data lineage tracking (where does each feature come from?)
- [ ] Model card auto-generation from test results
- [ ] Cost-per-prediction tracking and optimization
- [ ] Multi-tenant test isolation for enterprise deployments

---

## Research Library (13 completed missions)

| File | Topic | Key Output |
|------|-------|------------|
| `docs/research/llm-evaluation-research.md` | LLM/GenAI evaluation | 20 assertions, BERTScore, TTFT/ITL, prompt injection |
| `docs/research/data-contracts-research.md` | Data contract spec | YAML → auto-generate tests (killer feature) |
| `docs/research/embedding-drift-research.md` | Embedding drift | 5 methods, domain classifier recommended |
| `docs/research/sprint4-bias-fairness-research.md` | Bias/fairness | 5 metrics, EU AI Act, impossibility theorem |
| `docs/research/sprint5-inference-testing-research.md` | Inference testing | SLA benchmarks, warmup patterns |
| `docs/research/jira-integration-research.md` | Jira integration | Adapter pattern, dedup engine |
| Sprint 1 agents | Competitor intel | Cleanlab acquired, DeepEval rising |
| Sprint 1 agents | QA pain points | Top 5 ranked by frequency |
| Sprint 2 agents | Drift methods | JS + Wasserstein recommended |
| Sprint 2 agents | PII gaps | 40 patterns ranked Tier 1-4 |
| Sprint 3 agents | ML training bugs | 51 assertions, 10 categories |
| Sprint 4 agents | CI/CD patterns | Test tiering, flaky detection, GPU runners |
| Sprint 5 agents | Cloud ML infra | AWS/GCP/Azure/on-prem lifecycle |
| Sprint 6 agents | HTML reports | Plotly charts, dark theme, single-file |
| Sprint 7 agents | CV testing | COCO/VOC, NIST FRVT, temporal consistency |

---

## Competitor Watch (March 2026)

| Competitor | Stars | Status | mltk Advantage |
|---|---|---|---|
| **Cleanlab** | 11.4K | Acquired by Handshake (Jan 28) | Capture orphaned users |
| **Giskard** | 5.2K | v2 deprecated, v3 unreleased | Stable alternative |
| **Deepchecks** | 4K | 1.7/4 review, LLM-only pivot | Better DX + full-spectrum |
| **Evidently** | 7.3K | $500/mo entry | Free, full-featured |
| **DeepEval** | 14.3K | LLM-only ("pytest for LLMs") | "pytest for ALL ML" |
| **Great Expectations** | 11.3K | Data-only, v1.15.1 | Complementary |

---

## Workflow Rules

1. **Docs first**: Write documentation before building
2. **Build to match docs**: Implementation must match documented API
3. **Verify alignment**: Compare docs vs code after building
4. **Research every sprint**: 2-3 background researchers per sprint
5. **Backlog in repo**: This file is the single source of truth
6. **All agents get write permission**: Every subagent dispatched with write access
7. **Commits by user**: Agent provides commit message, user handles git

---

*Last updated: Sprint 21 start (March 25, 2026)*
