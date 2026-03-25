# MLTK Backlog

Tracked items for the ML Test Kit project. Updated after each sprint.

## Status Legend
- **DONE** -- shipped and tested
- **PLANNED** -- scheduled for a specific sprint
- **BACKLOG** -- accepted, not yet scheduled
- **IDEA** -- needs evaluation

---

## DONE (S0-S9: 47 assertions, 204 tests, 5 Rust tests)

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

### Documentation Audit
- [x] All 47 source files: Args/Returns/Example on every function
- [x] All 20 test files: scenario + WHY + expected on every test
- [x] All 25 doc pages: aligned with code signatures

---

## PLANNED

### Sprint 10 -- v0.1.0 Release
- [ ] PyPI publish via trusted publishing
- [ ] Cross-platform wheels (maturin-action: Linux, macOS, Windows)
- [ ] README with badges, benchmarks, install instructions
- [ ] CONTRIBUTING.md + code of conduct
- [ ] GitHub repo creation (Liorrr/mltk)
- [ ] TestPyPI dry-run before production publish
- [ ] Final MkDocs polish + deploy to GitHub Pages

### Sprint 11 -- Data Contracts + Drift Expansion
- [ ] **Data contracts engine** (KILLER FEATURE): YAML spec → auto-generate pytest tests
  - Parse mltk contract YAML (columns, types, ranges, drift baselines, PII rules)
  - `mltk contract validate data.csv --contract contract.yaml`
  - Auto-generate pytest test file from contract
  - Source: `docs/research/data-contracts-research.md`
- [ ] **Jensen-Shannon divergence** -- `method="js"`, symmetric, bounded [0,1], threshold 0.1
- [ ] **Wasserstein distance** -- `method="wasserstein"`, scipy.stats.wasserstein_distance, threshold 0.1
- [ ] **method="auto"** -- auto-select drift method based on sample size + dtype
- [ ] **Embedding drift** -- `assert_no_embedding_drift(ref, cur, method="cosine|mmd|classifier")`
  - 5 methods: cosine centroid, euclidean centroid, MMD, PCA+KS, domain classifier
  - Source: `docs/research/embedding-drift-research.md`

### Sprint 12 -- LLM/GenAI Evaluation
- [ ] **BERTScore** -- semantic similarity scoring (lazy import sentence-transformers)
- [ ] **assert_no_toxicity** -- toxicity classification (lazy import detoxify)
- [ ] **assert_no_hallucination** -- factual consistency checking
- [ ] **TTFT + ITL metrics** -- Time to First Token, Inter-Token Latency for LLM endpoints
- [ ] **LLM-as-judge** -- configurable LLM evaluator for open-ended outputs
- [ ] Source: `docs/research/llm-evaluation-research.md` (20 assertions proposed, 4 phases)

### Sprint 13 -- PII Expansion + Training Bug Detection P0
- [ ] **10 Tier 1 PII patterns**: IPv4/IPv6, JWT, PEM keys, DB connection strings, Stripe keys, Bearer tokens, Google API keys, IBAN, URL auth tokens
- [ ] **Checksum validation**: Luhn (credit cards), MOD-97 (IBAN) -- eliminates ~100% false positives
- [ ] **Confidence scoring**: 0.0-1.0 per PII match
- [ ] **P0 Data leakage**: assert_no_train_test_overlap, assert_temporal_split, assert_group_split, assert_no_future_leakage, assert_preprocessing_after_split
- [ ] **P0 Feature leakage**: assert_no_feature_target_leakage, assert_feature_available_at_inference, assert_no_id_features, assert_target_encoding_is_oof

### Sprint 14 -- Jira + PM Integrations
- [ ] **IssueTrackerAdapter** base class (vendor-agnostic)
- [ ] **JiraAdapter**: create/search/update/close issues via REST API
- [ ] **TicketDecisionEngine**: content hash dedup, cooldown, severity filtering
- [ ] **ML ticket templates**: data quality, model regression, drift detection, bias violation
- [ ] **Slack notifications**: post failure summary + Jira link
- [ ] **GitHub Actions action**: `uses: liorrr/mltk-action@v1`
- [ ] Source: `docs/research/jira-integration-research.md`

### Sprint 15 -- Advanced CV + Face Recognition
- [ ] **assert_face_far**: false accept rate at operating point (NIST FRVT)
- [ ] **assert_face_bias**: FAR/FRR demographic differential
- [ ] **MOTA/MOTP/IDF1**: multi-object tracking metrics
- [ ] **Per-class AP breakdown**: precision-recall curves in report
- [ ] **assert_confusion_matrix**: per-class quality gates
- [ ] Source: CV research (COCO eval, NIST FRVT benchmarks)

### Sprint 16 -- Training Bug Assertions P1
- [ ] Gradient pathologies: assert_gradient_flow, assert_weight_update_ratio, assert_gradient_bounded, assert_loss_finite, assert_neuron_utilization, assert_gradient_snr
- [ ] Learning rate: assert_loss_decreasing, assert_no_loss_divergence, assert_lr_schedule_matches, assert_lr_bounded, assert_warmup_fraction
- [ ] Batch normalization: assert_eval_mode_set, assert_bn_statistics_valid, assert_bn_batch_size, assert_train_eval_consistency
- [ ] Numerical stability: assert_no_nan_inf, assert_loss_scale_effective, assert_numerical_stability, assert_softmax_valid

### Sprint 17 -- Cloud Monitoring Integration
- [ ] **AWS SageMaker**: endpoint health, CloudWatch metrics, Model Monitor integration
- [ ] **GCP Vertex AI**: prediction endpoint monitoring, TPU utilization
- [ ] **Azure ML**: managed endpoint metrics, Responsible AI Dashboard
- [ ] **On-prem**: Prometheus/Grafana integration, DCGM GPU metrics, Triton health
- [ ] Source: Cloud ML infrastructure research

### Sprint 18 -- Test Resource Registry
- [ ] `mltk pull <collection>` -- download test definitions from registry
- [ ] `mltk push <collection>` -- share test suites with team
- [ ] `mltk list` -- browse available test collections
- [ ] Versioned test resources with dependency tracking
- [ ] Server API (REST) for hosting registry (company or SaaS)

### Sprint 19 -- Chat Interface
- [ ] `mltk chat` CLI -- interactive chat for querying test results
- [ ] Web-based chat embedded in HTML reports
- [ ] "Why did my model regress?" → analyzes test results + suggests fixes
- [ ] "What tests should I run?" → auto-generates test plan

### Sprint 20 -- Resource Summarization + AI Predictions
- [ ] Project/folder/collection summarization (text/doc, image/PNG, video/MP4)
- [ ] AI-powered predictions from prompt + resources/features
- [ ] "Predict if this model version is safe to deploy" → risk assessment
- [ ] Historical trend analysis from test result archives

---

## BACKLOG (not yet scheduled)

### Core Enhancements
- [ ] YAML-driven test definitions (run tests from config, no code)
- [ ] Test registry with @register_test decorator for custom assertions
- [ ] Config from environment variables (MLTK_DRIFT_METHOD, etc.)
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
- [ ] EU AI Act compliance report template (deadline: Aug 2, 2026)
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

- [ ] `mltk doctor` -- diagnose common ML pipeline issues automatically
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

*Last updated: Post-Sprint 9 documentation audit (March 25, 2026)*
