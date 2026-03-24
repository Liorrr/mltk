# MLTK Backlog

Tracked items for the ML Test Kit project. Updated after each sprint.

## Status Legend
- **DONE** -- shipped and tested
- **IN PROGRESS** -- currently being built
- **PLANNED** -- scheduled for a specific sprint
- **BACKLOG** -- accepted, not yet scheduled
- **IDEA** -- needs evaluation

---

## DONE

### Sprint 0 -- Project Skeleton
- [x] Repo structure, pyproject.toml, Cargo.toml
- [x] Core types: MltkConfig, TestResult, TestSuite, Severity
- [x] Base assertion framework: assert_true, MltkAssertionError
- [x] Rust crate skeleton (PyO3 0.28 + Maturin)
- [x] pytest plugin with ML markers
- [x] CLI skeleton (Typer)
- [x] CI/CD workflows (lint, test, release)
- [x] 12 tests passing

### Sprint 1 -- Core + Data Quality + Docs
- [x] Config loading from pyproject.toml and mltk.yaml
- [x] assert_schema, assert_no_nulls, assert_dtypes
- [x] assert_range, assert_unique, assert_no_outliers (IQR)
- [x] assert_freshness, assert_row_count
- [x] MkDocs documentation site (8 pages, API reference)
- [x] 47 tests passing

---

## PLANNED

### Sprint 2 -- Data Drift + PII + Labels
- [ ] assert_no_drift (KS, PSI, KL, chi-squared)
- [ ] assert_no_pii, scan_pii (regex-based PII detection)
- [ ] assert_label_balance, assert_label_coverage
- [ ] Rust acceleration: drift.rs (KS test, PSI)
- [ ] _rust.py fallback bridge completion

### Sprint 3 -- Model Quality
- [ ] assert_metric (accuracy, F1, AUC, precision, recall, MSE, R2)
- [ ] assert_no_regression (compare against saved baseline)
- [ ] assert_slice_performance (subgroup evaluation)
- [ ] save_baseline utility

### Sprint 4 -- Model Bias + pytest Plugin
- [ ] assert_no_bias (demographic parity, equalized odds, predictive parity)
- [ ] assert_robust (adversarial perturbation testing)
- [ ] pytest plugin: fixtures (ml_config, ml_baseline, ml_report)
- [ ] pytest plugin: HTML report hook via --mltk-report

### Sprint 5 -- Inference + CLI
- [ ] assert_latency (P50, P95, P99 percentiles)
- [ ] assert_throughput (requests per second)
- [ ] assert_api_contract (input/output JSON schema)
- [ ] CLI: mltk init, mltk scan, mltk drift, mltk version

### Sprint 6 -- Reports + Pipeline + ML Test Score
- [ ] HTML report generator (Plotly + Jinja2)
- [ ] assert_reproducible, assert_pipeline
- [ ] Google ML Test Score rubric (28-test scoring)
- [ ] CLI: mltk score, mltk report

### Sprint 7 -- Domain Kit: Computer Vision
- [ ] assert_map, assert_iou (object detection metrics)
- [ ] assert_frame_accuracy, assert_temporal_consistency (video)
- [ ] Example: kaleidoo_cv_test.py

### Sprint 8 -- Domain Kits: NLP + Speech
- [ ] NLP: assert_ner_f1, assert_bleu, assert_rouge, assert_no_prompt_injection
- [ ] Speech: assert_wer, assert_rtf, assert_accent_coverage

### Sprint 9 -- Monitoring + Tabular + Full Docs
- [ ] drift_monitor (continuous), assert_no_degradation, assert_sla
- [ ] Tabular: assert_feature_drift, assert_calibration, assert_class_balance
- [ ] Complete MkDocs documentation site

### Sprint 10 -- v0.1.0 Release
- [ ] PyPI publish via trusted publishing
- [ ] Cross-platform wheels (maturin-action)
- [ ] README badges, benchmarks, contributor guide

---

## BACKLOG (not yet scheduled)

### Core Enhancements
- [ ] YAML-driven test definitions (run tests from config, no code)
- [ ] Test registry with @register_test decorator for custom assertions
- [ ] Config from environment variables (MLTK_DRIFT_METHOD, etc.)

### Testing Patterns (from research)
- [ ] Statistical assertion primitives (tolerance bands, semantic similarity)
- [ ] LLM-as-judge evaluation patterns for generative AI
- [ ] Smart test selection (only re-run tests affected by specific changes)
- [ ] Golden test set management (versioned baselines)
- [ ] Non-deterministic test retry with confidence intervals

### Regulatory / Compliance
- [ ] EU AI Act compliance report template
- [ ] FDA AI device audit trail export
- [ ] Bias/fairness report with demographic breakdowns

### Integrations
- [ ] GitHub Actions action (`uses: liorrr/mltk-action@v1`)
- [ ] VS Code extension (inline test results)
- [ ] Jupyter notebook integration (rich output for assertions)
- [ ] MLflow integration (log mltk results as MLflow metrics)

### Performance
- [ ] Rust: KL divergence, chi-squared, Wasserstein distance
- [ ] Rust: fast PII scanning with regex
- [ ] Rust: SIMD-accelerated distribution comparisons
- [ ] Benchmarks vs Great Expectations, Deepchecks, Evidently

### Monetization (Pro tier -- future)
- [ ] Cloud dashboard: hosted report aggregation
- [ ] CI/CD GitHub App: auto-run mltk on PRs
- [ ] Compliance PDF exports (EU AI Act, FDA)
- [ ] Custom domain kits (healthcare, fintech, autonomous)

---

## IDEAS (needs evaluation)

- [ ] `mltk doctor` -- diagnose common ML pipeline issues automatically
- [ ] Visual diff for model predictions (before/after comparison)
- [ ] Slack/Teams notifications for drift alerts
- [ ] Data contract specification format (like OpenAPI but for datasets)
- [ ] Plugin system for third-party assertion libraries

---

## Research Findings (Sprint 1)

### Competitor Intelligence (March 2026)

| Competitor | Stars | Status | mltk Opportunity |
|---|---|---|---|
| **Cleanlab** | 11.4K | **ACQUIRED by Handshake (Jan 28)** — OSS in maintenance mode, no releases since | Capture displaced users. Add label quality checks. |
| **Giskard** | 5.2K | v2 deprecated, v3 unreleased — users in limbo | Stable alternative for bias/fairness testing |
| **Deepchecks** | 4K | 1.7/4 review score, pivoting to LLM-only | Beat on DX and traditional ML coverage |
| **Evidently** | 7.3K | $500/mo entry, limited OSS support | Free, full-featured alternative |
| **Great Expectations** | 11.3K | Data-only (no model testing), v1.15.1 | Complementary, not competitive |
| **DeepEval** | 14.3K | Fastest-growing, LLM-only ("pytest for LLMs") | Own "pytest for ALL ML" (not just LLMs) |
| **MLflow 3.x** | Massive | Adding evaluation features, absorbing Giskard as plugin | Stay independent, integrate as output target |
| **Inspect AI** | New | UK govt-backed, adopted by Anthropic/DeepMind | Safety niche, not general ML testing |

### Strategic Priorities (from research)
1. **Capture Cleanlab's orphaned users** — add label quality checks (Sprint 2)
2. **Fill the Giskard v2→v3 gap** — stable bias/fairness testing (Sprint 4)
3. **Don't pivot to LLM-only** — everyone else is. Own full-spectrum ML testing.
4. **Beat Evidently on price** — fully free OSS, no $500/mo gate
5. **Beat Deepchecks on DX** — 1.7/4 review score = low bar to clear

### QA Pain Points (ranked by frequency)
1. Non-deterministic outputs break traditional assertions
2. Silent drift in production with no alerts
3. No unified framework across ML lifecycle (OUR VALUE PROP)
4. GPU costs block CI/CD testing
5. Regulatory audit trails missing from testing tools

### Key Stats
- 87% of ML models never reach production
- 46% of devs distrust AI accuracy (up from 31% in 2024)
- 89% piloting GenAI in QA, only 15% have enterprise implementations
- Cleanlab acquired = 11.4K star user base looking for alternatives
- Every major competitor pivoting to LLM-only = traditional ML testing gap widening

---

## Workflow Rules

1. **Docs first**: Write documentation before building. Docs define the contract.
2. **Build to match docs**: Implementation must match documented API signatures and behavior.
3. **Verify alignment**: After building, compare docs vs implementation. If they differ, convene team (ml-test-engineer + qa-lead + ml-engineer) to reach consensus on the better design.
4. **Research every sprint**: Dispatch 2-3 background researchers per sprint (competitor watch, gap analysis, user pain points).
5. **Backlog in repo**: This file (BACKLOG.md) is the single source of truth. Update after every sprint.
6. **All agents get write permission**: Every subagent dispatched with write access from the start.

---

*Last updated: Sprint 1 completion (March 25, 2026)*
