# Test Coverage Index

Complete index of all **1054 tests** across **95 test files** in the mltk test suite.

---

## Summary

| Module | Test Files | Test Count | Coverage |
|--------|-----------|------------|---------|
| [test_core](#test_core) | 4 | 27 | `mltk.core` — assertion, config, plugin, result |
| [test_data](#test_data) | 17 | 167 | `mltk.data` — schema, drift, PII, labels, lineage, statistics, validation, preset |
| [test_domains](#test_domains) | 17 | 176 | `mltk.domains` — CV, NLP, Speech, Tabular, LLM/RAG/Agentic |
| [test_model](#test_model) | 6 | 53 | `mltk.model` — metrics, regression, slicing, bias, adversarial, overfitting |
| [test_training](#test_training) | 8 | 88 | `mltk.training` — leakage, gradient, numerical, augmentation, checkpoint, distributed, memory, skew |
| [test_inference](#test_inference) | 3 | 18 | `mltk.inference` — latency, cold start, throughput, contract |
| [test_monitor](#test_monitor) | 6 | 40 | `mltk.monitor` — degradation, SLA, output drift, AWS/GCP/Azure, Prometheus |
| [test_pipeline](#test_pipeline) | 2 | 11 | `mltk.pipeline` — reproducibility, checksum, e2e pipeline |
| [test_compliance](#test_compliance) | 4 | 24 | `mltk.compliance` — EU AI Act, FDA, OWASP LLM, PDF export |
| [test_integrations](#test_integrations) | 6 | 42 | `mltk.integrations` — Jira, GitHub, Slack, MLflow, Asana, Linear |
| [test_server](#test_server) | 6 | 63 | `mltk.server` — routes, auth, storage, webhooks, GitHub CI, comparison |
| [test_contracts](#test_contracts) | 1 | 16 | `mltk.contracts` — YAML data contract parsing, validation, codegen |
| [test_cli](#test_cli) | 1 | 38 | `mltk.cli` — init, scan, drift, score, doctor, test, model-card, compliance, registry, notify, docs, server |
| [test_testdefs](#test_testdefs) | 1 | 16 | `mltk.testdefs` — YAML test definition loading and execution |
| [test_testing](#test_testing) | 4 | 25 | `mltk.testing` — flaky detection, golden baselines, retry, test selection |
| [test_chat](#test_chat) | 1 | 12 | `mltk.chat` — chat engine load/ask queries |
| [test_report](#test_report) | 5 | 33 | `mltk.report` — HTML generator, model card, bias report, summarizer, visual diff |
| [test_doctor](#test_doctor) | 1 | 11 | `mltk.doctor` — diagnostic checks |
| [test_registry](#test_registry) | 1 | 8 | `mltk.registry` — push/pull/list test resource collections |
| [test_pytest_plugin](#test_pytest_plugin) | 1 | 6 | `mltk.pytest_plugin` — pytest integration, markers, report flags |
| test_jupyter | *(collected via conftest)* | — | `mltk.jupyter` — notebook integration |
| **TOTAL** | **95** | **1054** | |

---

## Per-Module Sections

### test_core

Tests for the foundational types and configuration system.

| File | Tests | Covers |
|------|-------|--------|
| `test_assertion.py` | 4 | `mltk.core.assertion` — assert_true pass/fail, warning severity, detail propagation |
| `test_config.py` | 18 | `mltk.core.config` — default config, YAML/pyproject loading, env var overrides (MLTK_*), priority resolution |
| `test_plugin.py` | 10 | `mltk.core.plugin` — @register_assertion decorator, discover_plugins, custom prefix, error handling |
| `test_result.py` | 5 | `mltk.core.result` — TestResult creation, TestSuite pass/fail/warning, score computation |

### test_data

Tests for all data quality assertions (schema, drift, PII, labels, lineage, statistics, validation, preset).

| File | Tests | Covers |
|------|-------|--------|
| `test_schema.py` | 12 | `mltk.data.schema` — assert_schema, assert_no_nulls, assert_dtypes |
| `test_distribution.py` | 9 | `mltk.data.distribution` — assert_range, assert_unique, assert_no_outliers |
| `test_drift.py` | 10 | `mltk.data.drift` — assert_no_drift (KS, PSI, KL, Chi2), custom threshold, unknown method |
| `test_drift_advanced.py` | 6 | `mltk.data.drift` — Jensen-Shannon, Wasserstein, auto-select method |
| `test_embedding_drift.py` | 5 | `mltk.data.embedding_drift` — assert_no_embedding_drift (cosine, euclidean, MMD) |
| `test_freshness.py` | 8 | `mltk.data.freshness` — assert_freshness, assert_row_count |
| `test_labels.py` | 9 | `mltk.data.labels` — assert_label_balance, assert_label_coverage (auto-detect, insufficient samples) |
| `test_pii.py` | 12 | `mltk.data.pii` — scan_pii (email, phone, SSN, credit card, API keys), assert_no_pii |
| `test_pii_expanded.py` | 7 | `mltk.data.pii` — expanded patterns (IPv4, JWT, PEM, DB connection, Bearer, Google API key) |
| `test_pii_israel.py` | 16 | `mltk.data.pii` — Israel Teudat Zehut (Luhn), Israel phone, IBAN MOD-97, assert_no_pii with Israel patterns |
| `test_pii_remaining.py` | 8 | `mltk.data.pii` — international phone, MAC address, Bitcoin/Ethereum, allowlist suppression |
| `test_pii_tier3.py` | 20 | `mltk.data.pii` — UK NHS, UK NINO, Germany Steuer-ID, India Aadhaar (Verhoeff), India PAN |
| `test_pii_tier4.py` | 13 | `mltk.data.pii` — France NIR, Italy Codice Fiscale, Spain DNI, assert_no_pii_eu |
| `test_preset.py` | 13 | `mltk.data.preset` — assert_data_quality (one-call bundle), data_quality_report |
| `test_statistics.py` | 19 | `mltk.data.statistics` — assert_column_mean, assert_column_median, assert_column_stdev, assert_quantiles |
| `test_validation.py` | 15 | `mltk.data.validation` — assert_datetime_format, assert_values_in_set, assert_no_conflicting_labels |
| `test_feature_correlation.py` | 6 | `mltk.data.validation` — assert_feature_label_correlation_stable |
| `test_lineage.py` | 8 | `mltk.data.lineage` — track_lineage decorator, LineageGraph, assert_lineage_complete |

### test_domains

Tests for domain-specific kits (CV, NLP, Speech, Tabular, LLM).

| File | Tests | Covers |
|------|-------|--------|
| `test_cv.py` | 13 | `mltk.domains.cv` — assert_iou, assert_map, assert_frame_accuracy, assert_temporal_consistency, assert_topk_accuracy |
| `test_cv_face.py` | 4 | `mltk.domains.cv.face` — assert_face_far (False Accept Rate) |
| `test_cv_tracking.py` | 16 | `mltk.domains.cv.tracking` — assert_mota, assert_motp, assert_idf1, edge cases |
| `test_nlp.py` | 8 | `mltk.domains.nlp` — assert_ner_f1, assert_no_prompt_injection |
| `test_nlp_generation.py` | 8 | `mltk.domains.nlp.generation` — assert_bleu, assert_rouge |
| `test_nlp_sentiment.py` | 14 | `mltk.domains.nlp.sentiment` — assert_sentiment_positive, assert_no_sentiment_drift, score function |
| `test_speech.py` | 7 | `mltk.domains.speech` — assert_rtf, assert_accent_coverage |
| `test_speech_recognition.py` | 8 | `mltk.domains.speech.recognition` — assert_wer, assert_cer |
| `test_tabular.py` | 7 | `mltk.domains.tabular` — assert_feature_drift, assert_feature_importance_stable, assert_class_balance |
| `test_llm.py` | 16 | `mltk.domains.llm` — assert_semantic_similarity, assert_no_toxicity, assert_no_hallucination, assert_ttft, assert_itl |
| `test_llm_agentic.py` | 15 | `mltk.domains.llm.agentic` — assert_task_completion, assert_tool_selection, assert_tool_call_correctness |
| `test_llm_bertscore.py` | 7 | `mltk.domains.llm.bertscore` — assert_bertscore (identical, orthogonal, threshold, empty, metadata) |
| `test_llm_coherence.py` | 8 | `mltk.domains.llm.coherence` — assert_coherence (consistent, random, single sentence, edge cases) |
| `test_llm_conversation.py` | 16 | `mltk.domains.llm.conversation` — assert_knowledge_retention, assert_turn_relevancy, assert_conversation_completeness |
| `test_llm_rag.py` | 16 | `mltk.domains.llm.rag` — assert_faithfulness, assert_context_relevancy, assert_answer_relevancy, assert_context_precision, assert_context_recall |
| `test_llm_ragas.py` | 10 | `mltk.domains.llm.ragas` — compute_ragas_score, assert_ragas_score |
| `test_llm_text_quality.py` | 19 | `mltk.domains.llm.text_quality` — assert_text_length, assert_output_format, assert_readability |
| `test_llm_utils.py` | 4 | `mltk.domains.llm._utils` — tokenize helper |

### test_model

Tests for model quality assertions.

| File | Tests | Covers |
|------|-------|--------|
| `test_metrics.py` | varies | `mltk.model.metrics` — assert_metric (accuracy, precision, recall, F1, ROC-AUC, etc.) |
| `test_regression.py` | varies | `mltk.model.regression` — assert_no_regression (baseline comparison) |
| `test_slicing.py` | varies | `mltk.model.slicing` — assert_slice_performance, assert_calibration |
| `test_bias.py` | varies | `mltk.model.bias` — assert_no_bias (5 fairness methods) |
| `test_adversarial.py` | varies | `mltk.model.adversarial` — assert_robust (adversarial perturbation) |
| `test_overfitting.py` | varies | `mltk.model.overfitting` — assert_no_overfitting, assert_label_drift |

### test_training

Tests for training bug detection.

| File | Tests | Covers |
|------|-------|--------|
| `test_leakage.py` | 6 | `mltk.training.leakage` — assert_no_train_test_overlap, assert_temporal_split, assert_no_target_leakage |
| `test_gradient.py` | 16 | `mltk.training.gradient` — assert_gradient_flow, assert_no_vanishing_gradient, assert_no_exploding_gradient, assert_loss_finite |
| `test_numerical.py` | 18 | `mltk.training.numerical` — assert_no_nan_inf, assert_loss_decreasing, assert_no_loss_divergence, assert_softmax_valid |
| `test_augmentation.py` | 11 | `mltk.training.augmentation` — assert_no_augmentation_on_test, assert_augmentation_preserves_signal |
| `test_checkpoint.py` | 11 | `mltk.training.checkpoint` — assert_checkpoint_complete, assert_resume_loss_continuous |
| `test_distributed.py` | 11 | `mltk.training.distributed` — assert_effective_batch_size, assert_gradient_sync |
| `test_memory.py` | 12 | `mltk.training.memory` — assert_no_memory_leak, assert_loss_is_detached |
| `test_skew.py` | 7 | `mltk.training.skew` — assert_no_training_serving_skew |

### test_inference

Tests for inference performance assertions.

| File | Tests | Covers |
|------|-------|--------|
| `test_latency.py` | 7 | `mltk.inference.latency` — assert_latency (P50/P95/P99), assert_cold_start |
| `test_throughput.py` | 5 | `mltk.inference.throughput` — assert_throughput (sequential, concurrent, error tracking) |
| `test_contract.py` | 6 | `mltk.inference.contract` — assert_api_contract (input/output schema validation) |

### test_monitor

Tests for production monitoring.

| File | Tests | Covers |
|------|-------|--------|
| `test_monitor.py` | varies | `mltk.monitor.drift_monitor` — assert_no_degradation, assert_sla |
| `test_output_drift.py` | varies | `mltk.monitor.drift_monitor` — assert_no_output_drift |
| `test_aws.py` | varies | `mltk.monitor.aws` — assert_endpoint_healthy, assert_endpoint_latency, assert_endpoint_error_rate |
| `test_gcp.py` | varies | `mltk.monitor.gcp` — assert_endpoint_healthy, assert_prediction_latency |
| `test_azure.py` | varies | `mltk.monitor.azure` — assert_endpoint_healthy, assert_endpoint_latency |
| `test_prometheus.py` | varies | `mltk.monitor.prometheus` — assert_prometheus_metric, assert_gpu_utilization, assert_triton_healthy |

### test_pipeline

Tests for pipeline reproducibility.

| File | Tests | Covers |
|------|-------|--------|
| `test_reproducibility.py` | varies | `mltk.pipeline.reproducibility` — assert_reproducible, assert_checksum |
| `test_e2e.py` | varies | `mltk.pipeline.e2e` — assert_pipeline (end-to-end step execution) |

### test_compliance

Tests for regulatory compliance tooling.

| File | Tests | Covers |
|------|-------|--------|
| `test_eu_ai_act.py` | 12 | `mltk.compliance.eu_ai_act` — classify_risk, map_results_to_articles, find_gaps, generate_report |
| `test_fda.py` | 6 | `mltk.compliance.fda` — FDA 21 CFR Part 11 audit trail report generation |
| `test_owasp_llm.py` | 8 | `mltk.compliance.owasp_llm` — owasp_llm_scan, assert_owasp_coverage, gap tracking |
| `test_pdf_export.py` | 4 | `mltk.compliance.pdf_export` — print CSS injection, HTML export |

### test_integrations

Tests for external service integrations.

| File | Tests | Covers |
|------|-------|--------|
| `test_integrations.py` | 25 | `mltk.integrations` — IssueTrackerAdapter, TicketDecisionEngine (severity, dedup, cooldown, hash), ticket templates |
| `test_github.py` | 10 | `mltk.integrations.github_adapter` — create/search/update issues, token handling, API errors |
| `test_slack.py` | varies | `mltk.integrations.slack` — webhook-based Slack notifications |
| `test_mlflow.py` | varies | `mltk.integrations.mlflow_logger` — MlflowLogger result logging |
| `test_asana.py` | 4 | `mltk.integrations.asana_adapter` — create/search/update tasks, API errors |
| `test_linear.py` | 3 | `mltk.integrations.linear_adapter` — create/search/update issues |

### test_server

Tests for the mltk server platform.

| File | Tests | Covers |
|------|-------|--------|
| `test_routes.py` | varies | `mltk.server.routes` — API endpoints (/api/results, /api/reports, /api/health) |
| `test_auth.py` | varies | `mltk.server.auth` — API key generation, verification, Bearer auth |
| `test_storage.py` | varies | `mltk.server.storage` — SQLite storage for results and reports |
| `test_webhooks.py` | varies | `mltk.server.webhooks` — webhook dispatch, URL validation, auth |
| `test_github_ci.py` | varies | `mltk.server.github_ci` — PR comment posting, check run creation |
| `test_comparison.py` | varies | `mltk.server.comparison` — run comparison (new failures, fixed, regressions) |

### test_contracts

Tests for the data contract engine.

| File | Tests | Covers |
|------|-------|--------|
| `test_contracts.py` | 16 | `mltk.contracts` — YAML parsing, column properties, validation (missing/null/range), pytest codegen, markers |

### test_cli

Tests for the CLI interface.

| File | Tests | Covers |
|------|-------|--------|
| `test_cli.py` | 38 | `mltk.cli` — init, scan (CSV/Parquet), drift, score, doctor, test, model-card, compliance, contract, registry, notify slack, docs, version, server create-key |

### test_testdefs

Tests for YAML test definitions.

| File | Tests | Covers |
|------|-------|--------|
| `test_yaml_runner.py` | 16 | `mltk.testdefs` — load_suite (env vars, missing files), run_suite (schema, no_nulls, range, row_count, unique) |

### test_testing

Tests for testing utility patterns.

| File | Tests | Covers |
|------|-------|--------|
| `test_flaky.py` | 7 | `mltk.testing.flaky` — detect_flaky (stable pass/fail, flaky detection, custom threshold) |
| `test_golden.py` | 9 | `mltk.testing.golden` — save/load roundtrip, assert_matches_golden, numpy arrays, versioning |
| `test_retry.py` | 6 | `mltk.testing.retry` — retry_until_confident (always pass/fail, flaky, confidence interval) |
| `test_selection.py` | 6 | `mltk.testing.selection` — build_test_map, select_affected_tests (deduplication, empty) |

### test_chat

Tests for the natural-language chat interface.

| File | Tests | Covers |
|------|-------|--------|
| `test_engine.py` | 12 | `mltk.chat.engine` — ChatEngine load/wrapped results, ask queries (failures, summary, why, recommendations, slowest, drift, bias, help, unknown) |

### test_report

Tests for report generation.

| File | Tests | Covers |
|------|-------|--------|
| `test_generator.py` | varies | `mltk.report.generator` — HTML report generation |
| `test_model_card.py` | varies | `mltk.report.model_card` — model card Markdown generation |
| `test_bias_report.py` | varies | `mltk.report.bias_report` — demographic breakdown report |
| `test_summarizer.py` | varies | `mltk.report.summarizer` — resource summarization |
| `test_visual_diff.py` | varies | `mltk.report.visual_diff` — visual diff rendering |

### test_doctor

Tests for the diagnostic tool.

| File | Tests | Covers |
|------|-------|--------|
| `test_doctor.py` | 11 | `mltk.doctor` — diagnose (all checks, Python version, core deps, config validation, fix hints) |

### test_registry

Tests for the test resource registry.

| File | Tests | Covers |
|------|-------|--------|
| `test_registry.py` | 8 | `mltk.registry` — save_collection, load_collection, list_collections |

### test_pytest_plugin

Tests for the pytest plugin integration.

| File | Tests | Covers |
|------|-------|--------|
| `test_plugin.py` | 6 | `mltk.pytest_plugin` — pytest markers, --mltk-report flag, --mltk-export-json |

---

## Running Specific Test Groups

```bash
# By directory
pytest tests/test_core/ -q            # Core types & config
pytest tests/test_data/ -q            # Data quality (schema, drift, PII, labels)
pytest tests/test_model/ -q           # Model quality (metrics, bias, adversarial)
pytest tests/test_training/ -q        # Training bugs (leakage, gradients, memory)
pytest tests/test_inference/ -q       # Inference performance
pytest tests/test_domains/ -q         # Domain kits (CV, NLP, Speech, LLM)
pytest tests/test_monitor/ -q         # Production monitoring
pytest tests/test_pipeline/ -q        # Pipeline reproducibility
pytest tests/test_compliance/ -q      # Compliance (EU AI Act, FDA, OWASP)
pytest tests/test_integrations/ -q    # External integrations
pytest tests/test_server/ -q          # Server platform (needs fastapi)
pytest tests/test_cli/ -q             # CLI commands
pytest tests/test_testing/ -q         # Testing patterns (flaky, golden, retry)
pytest tests/test_chat/ -q            # Chat interface
pytest tests/test_report/ -q          # Report generation
pytest tests/test_contracts/ -q       # Data contracts
pytest tests/test_testdefs/ -q        # YAML test definitions

# By marker
pytest -m ml_data -q                  # Data quality assertions
pytest -m ml_model -q                 # Model quality assertions
pytest -m ml_training -q              # Training bug assertions

# Specific file
pytest tests/test_data/test_pii.py -q           # PII detection only
pytest tests/test_domains/test_llm_rag.py -q    # RAG evaluation only
pytest tests/test_training/test_gradient.py -q  # Gradient health only

# Full suite
pytest tests/ -q                      # All 1054 tests
```
