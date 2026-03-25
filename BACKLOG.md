# MLTK Backlog

Tracked items for the ML Test Kit project. Updated after each sprint.

## Status Legend
- **DONE** -- shipped and tested
- **PLANNED** -- scheduled for a specific sprint
- **BACKLOG** -- accepted, not yet scheduled
- **IDEA** -- needs evaluation

---

## DONE (S0-S27: 87 assertions, 606 tests, 13 Rust tests)

### Phase A: Core Library (S0-S10) — v0.1.0
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

### Phase B: Post-Release Expansion (S11-S18) — v0.2.0
- [x] S11: Data contracts (YAML→tests), JS/Wasserstein/auto drift, embedding drift
- [x] S12: LLM eval (semantic similarity, toxicity, hallucination, TTFT/ITL)
- [x] S13: 10 Tier 1 PII + Luhn, training bug P0 (leakage detection)
- [x] S14: Jira integration (adapter, dedup, templates)
- [x] S15: Wiring audit (13 gaps fixed), face recognition (assert_face_far)
- [x] S16: CV tracking (MOTA/MOTP/IDF1), training bug P1 (gradient/numerical), docs deploy
- [x] S17: YAML test defs, EU AI Act compliance, mltk doctor, env vars, JSON export
- [x] S18: v0.2.0, Israel PII (Teudat Zehut), IBAN MOD-97

### Phase C: Performance + Polish (S19-S23) — v0.3.0
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

---

## PLANNED

### Sprint 28 -- v0.4.0 Release + EU PII Tier 4
- [x] PII Tier 4: France NIR, Italy Codice Fiscale, Spain DNI
- [x] BACKLOG major rewrite, v0.4.0, competitor gap analysis

### Sprint 29 -- RAG + Agentic + Text Quality
- [x] RAG: faithfulness, context relevancy/precision/recall, answer relevancy
- [x] Agentic: task completion, tool selection, tool call correctness
- [x] Text quality: text length, output format, readability
- [x] Training-serving skew detection

### Sprint 30 -- Data Quality Quick Wins + v0.5.0
- [x] Statistics: mean, median, stdev, quantiles
- [x] Validation: datetime format, values in set, conflicting labels
- [x] ML: overfitting detection, label drift
- [x] v0.5.0 (108 assertions, 724 tests)

### Sprint 31 -- Multi-turn + Data Preset + Sentiment
- [x] Multi-turn conversation: knowledge retention, turn relevancy, completeness
- [x] Data quality preset: assert_data_quality, data_quality_report
- [x] Sentiment: assert_sentiment_positive, assert_no_sentiment_drift

### Sprint 32 -- RAGAS + Coherence + OWASP LLM (ACTIVE)
- [ ] RAGAS composite score (average of 4 RAG metrics)
- [ ] Coherence check (sentence-to-sentence consistency)
- [ ] OWASP LLM Top 10 mapping + scan entrypoint

---

## BACKLOG (not yet scheduled)

### Regulatory / Compliance
- [ ] FDA device audit trail export
- [ ] Bias/fairness report with demographic breakdowns

### Performance / Rust
- [ ] SIMD cosine similarity for embedding drift
- [ ] BERTScore token matching in Rust
- [ ] Benchmarks vs Great Expectations, Deepchecks, Evidently

### Integrations
- [ ] VS Code extension (inline test results)
- [ ] Linear/Asana adapters

### PII Remaining
- [ ] International phone numbers, MAC addresses, crypto wallets
- [ ] Allowlists for known-safe patterns

### Advanced Features
- [ ] Resource summarization + AI predictions
- [ ] Visual diff for model predictions
- [ ] Data lineage tracking

### Monetization (Pro tier)
- [ ] Cloud dashboard: hosted report aggregation, team views
- [ ] CI/CD GitHub App: auto-run mltk on PRs
- [ ] Compliance PDF exports (EU AI Act, FDA)

---

*Last updated: Sprint 28 — v0.4.0 release (March 25, 2026)*
