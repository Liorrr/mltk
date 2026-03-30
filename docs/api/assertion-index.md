# Assertion Reference

Complete index of all 201 assertion functions in mltk, organized by category. Every assertion is sequentially numbered with no duplicates.

---

## Data Quality

Assertions for validating DataFrame schema, distributions, freshness, and completeness.

### Schema & Structure

| # | Assertion | Module | Description | Since |
|---|-----------|--------|-------------|-------|
| 1 | `assert_schema` | `mltk.data.schema` | Verify DataFrame columns and dtypes match expected schema | v0.1.0 |
| 2 | `assert_no_nulls` | `mltk.data.schema` | Fail if any null values exist in specified columns | v0.1.0 |
| 3 | `assert_dtypes` | `mltk.data.schema` | Verify column dtypes match expected type mapping | v0.1.0 |

### Distribution

| # | Assertion | Module | Description | Since |
|---|-----------|--------|-------------|-------|
| 4 | `assert_range` | `mltk.data.distribution` | Verify all values in a Series fall within [min, max] bounds | v0.1.0 |
| 5 | `assert_unique` | `mltk.data.distribution` | Verify uniqueness of values (single column or composite key) | v0.1.0 |
| 6 | `assert_no_outliers` | `mltk.data.distribution` | Detect outliers using IQR or z-score method | v0.1.0 |

### Freshness & Row Count

| # | Assertion | Module | Description | Since |
|---|-----------|--------|-------------|-------|
| 7 | `assert_freshness` | `mltk.data.freshness` | Fail if data is older than a max_age threshold | v0.1.0 |
| 8 | `assert_row_count` | `mltk.data.freshness` | Verify DataFrame row count is within [min, max] bounds | v0.1.0 |

### Drift Detection

| # | Assertion | Module | Description | Since |
|---|-----------|--------|-------------|-------|
| 9 | `assert_no_drift` | `mltk.data.drift` | Detect distribution drift via KS, PSI, KL, Chi2, JS, Wasserstein, or auto | v0.1.0 |
| 10 | `assert_no_embedding_drift` | `mltk.data.embedding_drift` | Detect embedding drift via cosine, euclidean, or MMD distance | v0.1.0 |

### Synthetic Data Quality

| # | Assertion | Module | Description | Since |
|---|-----------|--------|-------------|-------|
| 11 | `assert_marginal_fidelity` | `mltk.data.synthetic` | Verify per-column distribution fidelity (KS/PSI) between real and synthetic | v0.7.0 |
| 12 | `assert_correlation_preserved` | `mltk.data.synthetic` | Verify cross-column correlations match between real and synthetic | v0.7.0 |
| 13 | `assert_synthetic_novelty` | `mltk.data.synthetic` | Verify synthetic rows are not exact copies of real records | v0.7.0 |
| 14 | `assert_dcr_safe` | `mltk.data.synthetic` | Verify Distance to Closest Record meets privacy threshold | v0.7.0 |

### PII Detection

| # | Assertion | Module | Description | Since |
|---|-----------|--------|-------------|-------|
| 15 | `assert_no_pii` | `mltk.data.pii` | Scan DataFrame columns for PII patterns (email, phone, SSN, credit card, API keys, and 30+ more) | v0.1.0 |

### Labels

| # | Assertion | Module | Description | Since |
|---|-----------|--------|-------------|-------|
| 16 | `assert_label_balance` | `mltk.data.labels` | Verify label class ratio does not exceed imbalance threshold | v0.1.0 |
| 17 | `assert_label_coverage` | `mltk.data.labels` | Verify all expected labels are present with minimum sample count | v0.1.0 |

### Statistics

| # | Assertion | Module | Description | Since |
|---|-----------|--------|-------------|-------|
| 18 | `assert_column_mean` | `mltk.data.statistics` | Verify column mean falls within [min, max] bounds | v0.5.0 |
| 19 | `assert_column_median` | `mltk.data.statistics` | Verify column median falls within [min, max] bounds | v0.5.0 |
| 20 | `assert_column_stdev` | `mltk.data.statistics` | Verify column standard deviation falls within [min, max] bounds | v0.5.0 |
| 21 | `assert_quantiles` | `mltk.data.statistics` | Verify column quantile values fall within expected bounds | v0.5.0 |

### Validation

| # | Assertion | Module | Description | Since |
|---|-----------|--------|-------------|-------|
| 22 | `assert_datetime_format` | `mltk.data.validation` | Verify all values in a column parse as valid datetime with given format | v0.5.0 |
| 23 | `assert_values_in_set` | `mltk.data.validation` | Verify all column values belong to an allowed set | v0.5.0 |
| 24 | `assert_no_conflicting_labels` | `mltk.data.validation` | Detect rows with identical features but different labels | v0.5.0 |
| 25 | `assert_feature_label_correlation_stable` | `mltk.data.validation` | Verify feature-label correlations have not shifted between train and production | v0.6.0 |

### Data Lineage

| # | Assertion | Module | Description | Since |
|---|-----------|--------|-------------|-------|
| 26 | `assert_lineage_complete` | `mltk.data.lineage` | Verify all required pipeline steps appear in the lineage graph | v0.1.0 |

### Preset (One-Call Bundle)

| # | Assertion | Module | Description | Since |
|---|-----------|--------|-------------|-------|
| 27 | `assert_data_quality` | `mltk.data.preset` | Run schema, nulls, range, outliers, and PII checks in a single call; returns TestSuite | v0.5.0 |

---

## Model Quality

Assertions for validating model performance, fairness, and robustness.

### Metrics

| # | Assertion | Module | Description | Since |
|---|-----------|--------|-------------|-------|
| 28 | `assert_metric` | `mltk.model.metrics` | Verify any sklearn metric (accuracy, precision, recall, F1, ROC-AUC, etc.) meets threshold | v0.1.0 |

### Regression

| # | Assertion | Module | Description | Since |
|---|-----------|--------|-------------|-------|
| 29 | `assert_no_regression` | `mltk.model.regression` | Fail if current model metric drops below baseline by more than tolerance | v0.1.0 |

### Slicing

| # | Assertion | Module | Description | Since |
|---|-----------|--------|-------------|-------|
| 30 | `assert_slice_performance` | `mltk.model.slicing` | Verify metric holds across data slices (subgroups) | v0.1.0 |
| 31 | `assert_calibration` | `mltk.model.slicing` | Verify predicted probabilities match observed frequencies | v0.1.0 |

### Bias

| # | Assertion | Module | Description | Since |
|---|-----------|--------|-------------|-------|
| 32 | `assert_no_bias` | `mltk.model.bias` | Detect bias across groups using 5 fairness methods (demographic parity, equal opportunity, predictive parity, calibration, individual fairness) | v0.1.0 |

### Adversarial

| # | Assertion | Module | Description | Since |
|---|-----------|--------|-------------|-------|
| 33 | `assert_robust` | `mltk.model.adversarial` | Verify model prediction stability under adversarial perturbation | v0.1.0 |

### Conformal Prediction

| # | Assertion | Module | Description | Since |
|---|-----------|--------|-------------|-------|
| 34 | `assert_interval_coverage` | `mltk.model.conformal` | Verify prediction interval empirical coverage meets target | v0.7.0 |
| 35 | `assert_prediction_set_size` | `mltk.model.conformal` | Verify prediction set cardinality/width within budget | v0.7.0 |
| 36 | `assert_conformal_calibration` | `mltk.model.conformal` | Two-sided calibration check (coverage matches promise) | v0.7.0 |
| 37 | `assert_conditional_coverage` | `mltk.model.conformal` | Per-group coverage fairness check (Mondrian) | v0.7.0 |

### Overfitting

| # | Assertion | Module | Description | Since |
|---|-----------|--------|-------------|-------|
| 38 | `assert_no_overfitting` | `mltk.model.overfitting` | Fail if train-test score gap exceeds threshold | v0.5.0 |
| 39 | `assert_label_drift` | `mltk.model.overfitting` | Detect label distribution drift between train and serving | v0.5.0 |

### A/B Testing

| # | Assertion | Module | Description | Since |
|---|-----------|--------|-------------|-------|
| 40 | `assert_ab_significance` | `mltk.model.ab_test` | A/B test statistical significance | v0.8.0 |

### Attribution Stability

| # | Assertion | Module | Description | Since |
|---|-----------|--------|-------------|-------|
| 41 | `assert_top_k_stable` | `mltk.model.attribution` | Verify top-K feature attribution overlap across runs (Jaccard) | v0.7.0 |
| 42 | `assert_attribution_cosine_stability` | `mltk.model.attribution` | Verify attribution vector direction consistency (cosine similarity) | v0.7.0 |

### Counterfactual Fairness

| # | Assertion | Module | Description | Since |
|---|-----------|--------|-------------|-------|
| 43 | `assert_counterfactual_fairness` | `mltk.model.counterfactual` | Per-sample fairness via protected attribute perturbation | v0.8.0 |

### Causal Inference

| # | Assertion | Module | Description | Since |
|---|-----------|--------|-------------|-------|
| 44 | `assert_ate_significant` | `mltk.model.causal` | Average Treatment Effect statistical significance | v0.8.0 |
| 45 | `assert_no_confounding` | `mltk.model.causal` | Detect treatment-feature correlations (confounders) | v0.8.0 |

---

## Training

Assertions for catching training bugs and infrastructure issues.

### Data Leakage

| # | Assertion | Module | Description | Since |
|---|-----------|--------|-------------|-------|
| 46 | `assert_no_train_test_overlap` | `mltk.training.leakage` | Fail if train and test DataFrames share any rows | v0.1.0 |
| 47 | `assert_temporal_split` | `mltk.training.leakage` | Verify all train dates precede all test dates | v0.1.0 |
| 48 | `assert_no_target_leakage` | `mltk.training.leakage` | Detect features with suspiciously high correlation to target | v0.1.0 |

### Gradient Health

| # | Assertion | Module | Description | Since |
|---|-----------|--------|-------------|-------|
| 49 | `assert_gradient_flow` | `mltk.training.gradient` | Verify gradients are non-zero across layers (no dead layers) | v0.2.0 |
| 50 | `assert_no_vanishing_gradient` | `mltk.training.gradient` | Fail if gradient norms drop below threshold | v0.2.0 |
| 51 | `assert_no_exploding_gradient` | `mltk.training.gradient` | Fail if gradient norms exceed max threshold | v0.2.0 |
| 52 | `assert_loss_finite` | `mltk.training.gradient` | Verify all loss values are finite (no NaN/Inf) | v0.2.0 |

### Numerical Stability

| # | Assertion | Module | Description | Since |
|---|-----------|--------|-------------|-------|
| 53 | `assert_no_nan_inf` | `mltk.training.numerical` | Verify arrays contain no NaN or Inf values | v0.2.0 |
| 54 | `assert_loss_decreasing` | `mltk.training.numerical` | Verify training loss trend is decreasing over time | v0.2.0 |
| 55 | `assert_no_loss_divergence` | `mltk.training.numerical` | Fail if loss values diverge (spike beyond threshold) | v0.2.0 |
| 56 | `assert_softmax_valid` | `mltk.training.numerical` | Verify softmax output rows sum to 1.0 with no negatives | v0.2.0 |

### Augmentation

| # | Assertion | Module | Description | Since |
|---|-----------|--------|-------------|-------|
| 57 | `assert_no_augmentation_on_test` | `mltk.training.augmentation` | Verify test data has not been augmented (near-duplicate detection) | v0.3.0 |
| 58 | `assert_augmentation_preserves_signal` | `mltk.training.augmentation` | Verify augmentation does not alter label distribution | v0.3.0 |

### Checkpoint

| # | Assertion | Module | Description | Since |
|---|-----------|--------|-------------|-------|
| 59 | `assert_checkpoint_complete` | `mltk.training.checkpoint` | Verify checkpoint file exists and contains all required keys | v0.3.0 |
| 60 | `assert_resume_loss_continuous` | `mltk.training.checkpoint` | Verify loss continuity when resuming from checkpoint | v0.3.0 |

### Distributed Training

| # | Assertion | Module | Description | Since |
|---|-----------|--------|-------------|-------|
| 61 | `assert_effective_batch_size` | `mltk.training.distributed` | Verify effective batch size equals local_batch * num_gpus * grad_accum | v0.3.0 |
| 62 | `assert_gradient_sync` | `mltk.training.distributed` | Verify gradients are synchronized across GPU ranks | v0.3.0 |
| 63 | `assert_n_rank_gradient_sync` | `mltk.training.distributed` | Verify gradient sync across N ranks (all pairs) | v0.7.0 |
| 64 | `assert_gradient_alignment` | `mltk.training.distributed` | Verify gradient direction consistency via cosine similarity | v0.7.0 |
| 65 | `assert_weight_divergence` | `mltk.training.distributed` | Verify model weights haven't diverged across ranks/checkpoints | v0.7.0 |
| 66 | `assert_gradient_clipped` | `mltk.training.distributed` | Verify gradient global norm is within clipping bound | v0.7.0 |

### Memory

| # | Assertion | Module | Description | Since |
|---|-----------|--------|-------------|-------|
| 67 | `assert_no_memory_leak` | `mltk.training.memory` | Detect monotonically increasing memory usage over training steps | v0.3.0 |
| 68 | `assert_loss_is_detached` | `mltk.training.memory` | Verify loss tensor is properly detached (no graph retention leak) | v0.3.0 |

### Training-Serving Skew

| # | Assertion | Module | Description | Since |
|---|-----------|--------|-------------|-------|
| 69 | `assert_no_training_serving_skew` | `mltk.training.skew` | Verify train and serving outputs are numerically close | v0.5.0 |

---

## Inference

Assertions for validating inference latency, throughput, and API contracts.

| # | Assertion | Module | Description | Since |
|---|-----------|--------|-------------|-------|
| 70 | `assert_latency` | `mltk.inference.latency` | Verify function latency P50/P95/P99 meets threshold (with warmup exclusion) | v0.1.0 |
| 71 | `assert_cold_start` | `mltk.inference.latency` | Verify first-call latency meets cold start threshold | v0.1.0 |
| 72 | `assert_throughput` | `mltk.inference.throughput` | Verify requests-per-second meets minimum (sequential or concurrent) | v0.1.0 |
| 73 | `assert_api_contract` | `mltk.inference.contract` | Verify function input/output schemas match JSON Schema contract | v0.1.0 |

---

## Computer Vision

| # | Assertion | Module | Description | Since |
|---|-----------|--------|-------------|-------|
| 74 | `assert_iou` | `mltk.domains.cv.detection` | Verify mean IoU between predicted and ground-truth bounding boxes | v0.1.0 |
| 75 | `assert_map` | `mltk.domains.cv.detection` | Verify mean Average Precision for object detection | v0.1.0 |
| 76 | `assert_frame_accuracy` | `mltk.domains.cv.video` | Verify per-frame classification accuracy for video | v0.1.0 |
| 77 | `assert_temporal_consistency` | `mltk.domains.cv.video` | Verify smooth bounding box tracking across video frames | v0.1.0 |
| 78 | `assert_topk_accuracy` | `mltk.domains.cv.classification` | Verify top-K classification accuracy | v0.1.0 |
| 79 | `assert_face_far` | `mltk.domains.cv.face` | Verify False Accept Rate for face recognition systems | v0.1.0 |
| 80 | `assert_mota` | `mltk.domains.cv.tracking` | Verify Multiple Object Tracking Accuracy (misses, FP, ID switches) | v0.2.0 |
| 81 | `assert_motp` | `mltk.domains.cv.tracking` | Verify Multiple Object Tracking Precision (localization quality) | v0.2.0 |
| 82 | `assert_idf1` | `mltk.domains.cv.tracking` | Verify IDF1 score for identity-aware tracking consistency | v0.2.0 |

---

## NLP

| # | Assertion | Module | Description | Since |
|---|-----------|--------|-------------|-------|
| 83 | `assert_bleu` | `mltk.domains.nlp.generation` | Verify BLEU score for text generation quality | v0.1.0 |
| 84 | `assert_rouge` | `mltk.domains.nlp.generation` | Verify ROUGE score (rouge1/rouge2/rougeL) for summarization | v0.1.0 |
| 85 | `assert_ner_f1` | `mltk.domains.nlp.ner` | Verify named entity recognition F1 score | v0.1.0 |
| 86 | `assert_no_prompt_injection` | `mltk.domains.nlp.security` | Test model resilience against prompt injection payloads | v0.1.0 |
| 87 | `assert_sentiment_positive` | `mltk.domains.nlp.sentiment` | Verify positive sentiment ratio meets threshold | v0.5.0 |
| 88 | `assert_no_sentiment_drift` | `mltk.domains.nlp.sentiment` | Detect sentiment distribution shift between reference and current texts | v0.5.0 |
| 89 | `assert_text_robust` | `mltk.domains.nlp.robustness` | Verify NLP model prediction stability under text perturbations | v0.7.0 |

---

## Speech

| # | Assertion | Module | Description | Since |
|---|-----------|--------|-------------|-------|
| 90 | `assert_wer` | `mltk.domains.speech.recognition` | Verify Word Error Rate is below threshold | v0.1.0 |
| 91 | `assert_cer` | `mltk.domains.speech.recognition` | Verify Character Error Rate is below threshold | v0.1.0 |
| 92 | `assert_rtf` | `mltk.domains.speech.performance` | Verify Real-Time Factor (processing speed vs audio duration) | v0.1.0 |
| 93 | `assert_accent_coverage` | `mltk.domains.speech.performance` | Verify WER gap across accents stays below bias threshold | v0.1.0 |

---

## LLM / RAG / Agentic

### Safety & Similarity

| # | Assertion | Module | Description | Since |
|---|-----------|--------|-------------|-------|
| 94 | `assert_semantic_similarity` | `mltk.domains.llm.similarity` | Verify semantic similarity between reference and candidate texts | v0.1.0 |
| 95 | `assert_no_toxicity` | `mltk.domains.llm.safety` | Detect toxic content in generated texts via keyword scoring | v0.1.0 |
| 96 | `assert_no_hallucination` | `mltk.domains.llm.safety` | Verify generated claims are grounded in source documents | v0.1.0 |
| 97 | `assert_no_system_prompt_leakage` | `mltk.domains.llm.safety` | Detect system prompt leakage via adversarial extraction payloads | v0.7.0 |
| 98 | `assert_refusal_consistency` | `mltk.domains.llm.safety` | Verify LLM consistently refuses unsafe prompts across phrasings | v0.7.0 |
| 99 | `assert_safety_taxonomy` | `mltk.domains.llm.safety` | Verify per-category safety coverage (violence, self-harm, etc.) | v0.7.0 |

### LLM Latency

| # | Assertion | Module | Description | Since |
|---|-----------|--------|-------------|-------|
| 100 | `assert_ttft` | `mltk.domains.llm.latency` | Verify Time To First Token meets latency threshold | v0.1.0 |
| 101 | `assert_itl` | `mltk.domains.llm.latency` | Verify Inter-Token Latency meets threshold | v0.1.0 |

### RAG Evaluation

| # | Assertion | Module | Description | Since |
|---|-----------|--------|-------------|-------|
| 102 | `assert_faithfulness` | `mltk.domains.llm.rag` | Verify answer is grounded in retrieved context (no hallucination) | v0.5.0 |
| 103 | `assert_context_relevancy` | `mltk.domains.llm.rag` | Verify retrieved context is relevant to the question | v0.5.0 |
| 104 | `assert_answer_relevancy` | `mltk.domains.llm.rag` | Verify answer is relevant to the original question | v0.5.0 |
| 105 | `assert_context_precision` | `mltk.domains.llm.rag` | Verify relevant documents are ranked higher in context | v0.5.0 |
| 106 | `assert_context_recall` | `mltk.domains.llm.rag` | Verify all relevant documents are retrieved | v0.5.0 |
| 107 | `assert_ragas_score` | `mltk.domains.llm.ragas` | Verify composite RAGAS score (average of faithfulness, answer relevancy, context precision, recall) | v0.6.0 |

### Retrieval Ranking

| # | Assertion | Module | Description | Since |
|---|-----------|--------|-------------|-------|
| 108 | `assert_ndcg` | `mltk.domains.llm.retrieval` | Verify nDCG@K retrieval ranking quality (graded relevance) | v0.8.0 |
| 109 | `assert_mrr` | `mltk.domains.llm.retrieval` | Verify Mean Reciprocal Rank (first relevant result position) | v0.8.0 |
| 110 | `assert_recall_at_k` | `mltk.domains.llm.retrieval` | Verify Recall@K (coverage of relevant documents in top K) | v0.8.0 |
| 111 | `assert_map_at_k` | `mltk.domains.llm.retrieval` | Verify MAP@K (ranking quality with precision at each relevant position) | v0.8.0 |

### LLM-as-Judge

| # | Assertion | Module | Description | Since |
|---|-----------|--------|-------------|-------|
| 112 | `assert_llm_judge_score` | `mltk.domains.llm.judge` | Score model responses via LLM judge on quality criteria | v0.8.0 |
| 113 | `assert_llm_judge_pairwise` | `mltk.domains.llm.judge` | Pairwise comparison of responses via LLM judge (A/B testing) | v0.8.0 |

### Summarization

| # | Assertion | Module | Description | Since |
|---|-----------|--------|-------------|-------|
| 114 | `assert_summary_coverage` | `mltk.domains.llm.summarization` | Verify summary preserves key source content (token recall) | v0.8.0 |
| 115 | `assert_summary_compression` | `mltk.domains.llm.summarization` | Verify summary achieves target compression ratio | v0.8.0 |
| 116 | `assert_summary_faithfulness` | `mltk.domains.llm.summarization` | Verify summary doesn't hallucinate (token precision) | v0.8.0 |

### Coherence & BERTScore

| # | Assertion | Module | Description | Since |
|---|-----------|--------|-------------|-------|
| 117 | `assert_coherence` | `mltk.domains.llm.coherence` | Verify sentence-to-sentence consistency within generated text | v0.6.0 |
| 118 | `assert_bertscore` | `mltk.domains.llm.bertscore` | Verify BERTScore F1 (embedding-based precision/recall) meets threshold | v0.6.0 |

### Long-Context LLM

| # | Assertion | Module | Description | Since |
|---|-----------|--------|-------------|-------|
| 119 | `assert_needle_in_haystack` | `mltk.domains.llm.long_context` | Verify model retrieves facts at various context positions | v0.8.0 |
| 120 | `assert_context_utilization` | `mltk.domains.llm.long_context` | Verify model uses multiple facts from full context window | v0.8.0 |
| 121 | `assert_no_lost_in_middle` | `mltk.domains.llm.long_context` | Verify model accuracy is uniform across context positions | v0.8.0 |

### Agentic Evaluation

| # | Assertion | Module | Description | Since |
|---|-----------|--------|-------------|-------|
| 122 | `assert_task_completion` | `mltk.domains.llm.agentic` | Verify agent output matches expected task result | v0.5.0 |
| 123 | `assert_tool_selection` | `mltk.domains.llm.agentic` | Verify agent selected the correct tools (no missing, no extra) | v0.5.0 |
| 124 | `assert_tool_call_correctness` | `mltk.domains.llm.agentic` | Verify agent tool call arguments match expected values | v0.5.0 |
| 125 | `assert_tool_chain` | `mltk.domains.llm.agentic` | Verify agent tool call sequence matches expected chain | v0.7.0 |
| 126 | `assert_no_forbidden_actions` | `mltk.domains.llm.agentic` | Verify agent did not use forbidden tools | v0.7.0 |
| 127 | `assert_step_efficiency` | `mltk.domains.llm.agentic` | Verify agent completed task within step budget | v0.7.0 |
| 128 | `assert_no_redundant_calls` | `mltk.domains.llm.agentic` | Detect stuck agent loops from consecutive repeated tool calls | v0.7.0 |
| 129 | `assert_no_hallucinated_tools` | `mltk.domains.llm.agentic` | Verify agent only calls tools that actually exist | v0.7.0 |
| 130 | `assert_cost_budget` | `mltk.domains.llm.agentic` | Enforce token and duration budget on agent traces | v0.7.0 |
| 131 | `assert_error_recovery` | `mltk.domains.llm.agentic` | Verify agent recovers from errors without infinite retry | v0.7.0 |

### Multi-Agent Coordination

| # | Assertion | Module | Description | Since |
|---|-----------|--------|-------------|-------|
| 132 | `assert_no_agent_loop` | `mltk.domains.llm.multi_agent` | Detect circular delegation in multi-agent systems | v0.7.0 |
| 133 | `assert_agent_handoff` | `mltk.domains.llm.multi_agent` | Verify agent handoff sequence matches expected flow | v0.7.0 |

### Text Quality

| # | Assertion | Module | Description | Since |
|---|-----------|--------|-------------|-------|
| 134 | `assert_text_length` | `mltk.domains.llm.text_quality` | Verify generated text length falls within [min, max] word bounds | v0.5.0 |
| 135 | `assert_output_format` | `mltk.domains.llm.text_quality` | Verify text matches a regex pattern (JSON, UUID, date, etc.) | v0.5.0 |
| 136 | `assert_readability` | `mltk.domains.llm.text_quality` | Verify text readability score (Flesch-Kincaid grade level) | v0.5.0 |

### Multi-Turn Conversation

| # | Assertion | Module | Description | Since |
|---|-----------|--------|-------------|-------|
| 137 | `assert_knowledge_retention` | `mltk.domains.llm.conversation` | Verify assistant retains information across conversation turns | v0.5.0 |
| 138 | `assert_turn_relevancy` | `mltk.domains.llm.conversation` | Verify each assistant response is relevant to the preceding user turn | v0.5.0 |
| 139 | `assert_conversation_completeness` | `mltk.domains.llm.conversation` | Verify conversation covers all expected topics | v0.5.0 |

---

## Recommendation Systems

| # | Assertion | Module | Description | Since |
|---|-----------|--------|-------------|-------|
| 140 | `assert_hit_rate` | `mltk.domains.recommendation` | Verify fraction of users with at least 1 relevant recommendation | v0.8.0 |
| 141 | `assert_diversity` | `mltk.domains.recommendation` | Verify category diversity within recommendation lists | v0.8.0 |
| 142 | `assert_novelty` | `mltk.domains.recommendation` | Verify recommendations include non-obvious items (inverse popularity) | v0.8.0 |
| 143 | `assert_coverage` | `mltk.domains.recommendation` | Verify catalog utilization across all recommendations | v0.8.0 |
| 144 | `assert_serendipity` | `mltk.domains.recommendation` | Verify unexpected but relevant recommendations | v0.8.0 |

---

## Healthcare

| # | Assertion | Module | Description | Since |
|---|-----------|--------|-------------|-------|
| 145 | `assert_sensitivity` | `mltk.domains.healthcare` | Verify minimum true positive rate (recall) for clinical screening | v0.8.0 |
| 146 | `assert_specificity` | `mltk.domains.healthcare` | Verify minimum true negative rate for diagnostic tests | v0.8.0 |
| 147 | `assert_ppv` | `mltk.domains.healthcare` | Verify positive predictive value (precision) meets threshold | v0.8.0 |
| 148 | `assert_npv` | `mltk.domains.healthcare` | Verify negative predictive value meets threshold | v0.8.0 |
| 149 | `assert_clinical_agreement` | `mltk.domains.healthcare` | Verify Cohen's Kappa agreement beyond random chance | v0.8.0 |

---

## Code Generation

| # | Assertion | Module | Description | Since |
|---|-----------|--------|-------------|-------|
| 150 | `assert_code_executes` | `mltk.domains.codegen` | Verify generated code runs without errors (subprocess isolation) | v0.8.0 |
| 151 | `assert_code_passes_tests` | `mltk.domains.codegen` | Verify generated code passes provided test cases | v0.8.0 |
| 152 | `assert_no_code_vulnerabilities` | `mltk.domains.codegen` | Scan generated code for eval/exec/shell=True/hardcoded creds | v0.8.0 |
| 153 | `assert_code_complexity` | `mltk.domains.codegen` | Verify cyclomatic complexity and line count within bounds | v0.8.0 |

---

## Multimodal

| # | Assertion | Module | Description | Since |
|---|-----------|--------|-------------|-------|
| 154 | `assert_image_text_alignment` | `mltk.domains.multimodal` | Verify CLIP-style image-text embedding alignment | v0.8.0 |
| 155 | `assert_cross_modal_consistency` | `mltk.domains.multimodal` | Verify cross-modality prediction agreement | v0.8.0 |

---

## Reinforcement Learning

| # | Assertion | Module | Description | Since |
|---|-----------|--------|-------------|-------|
| 156 | `assert_reward_bounded` | `mltk.domains.rl` | Verify RL rewards within expected bounds | v0.8.0 |
| 157 | `assert_cumulative_reward` | `mltk.domains.rl` | Verify RL episode cumulative reward meets threshold | v0.8.0 |

---

## Tabular

| # | Assertion | Module | Description | Since |
|---|-----------|--------|-------------|-------|
| 158 | `assert_feature_drift` | `mltk.domains.tabular.features` | Detect per-column drift across two DataFrames | v0.1.0 |
| 159 | `assert_feature_importance_stable` | `mltk.domains.tabular.features` | Verify SHAP feature importance ranking stability (WARNING severity) | v0.1.0 |
| 160 | `assert_class_balance` | `mltk.domains.tabular.quality` | Convenience wrapper for label balance check on DataFrame column | v0.1.0 |

---

## Monitoring

Assertions for production model monitoring and cloud provider health checks.

### Drift & SLA

| # | Assertion | Module | Description | Since |
|---|-----------|--------|-------------|-------|
| 161 | `assert_no_degradation` | `mltk.monitor.drift_monitor` | Detect sliding-window metric decline in production | v0.1.0 |
| 162 | `assert_sla` | `mltk.monitor.drift_monitor` | Verify latency P99 and error rate meet SLA targets | v0.1.0 |
| 163 | `assert_no_output_drift` | `mltk.monitor.drift_monitor` | Detect output distribution shift between reference and current predictions | v0.6.0 |
| 164 | `assert_no_streaming_drift` | `mltk.monitor.streaming_drift` | Detect real-time distribution shifts using ADWIN or CUSUM | v0.7.0 |
| 165 | `assert_no_concept_drift` | `mltk.monitor.concept_drift` | Detect P(Y|X) drift via error rate comparison (chi2/fisher/proportion) | v0.7.0 |

### GPU (Local)

| # | Assertion | Module | Description | Since |
|---|-----------|--------|-------------|-------|
| 166 | `assert_gpu_utilization_local` | `mltk.monitor.gpu` | Verify local GPU utilization is below threshold | v0.8.0 |
| 167 | `assert_gpu_memory_local` | `mltk.monitor.gpu` | Verify local GPU memory usage is below threshold | v0.8.0 |

### AWS SageMaker

| # | Assertion | Module | Description | Since |
|---|-----------|--------|-------------|-------|
| 168 | `assert_endpoint_healthy` | `mltk.monitor.aws` | Verify SageMaker endpoint status is InService | v0.3.0 |
| 169 | `assert_endpoint_latency` | `mltk.monitor.aws` | Verify SageMaker endpoint invocation latency from CloudWatch | v0.3.0 |
| 170 | `assert_endpoint_error_rate` | `mltk.monitor.aws` | Verify SageMaker endpoint error rate from CloudWatch | v0.3.0 |

### GCP Vertex AI

| # | Assertion | Module | Description | Since |
|---|-----------|--------|-------------|-------|
| 171 | `assert_endpoint_healthy` | `mltk.monitor.gcp` | Verify Vertex AI endpoint is deployed and serving | v0.3.0 |
| 172 | `assert_prediction_latency` | `mltk.monitor.gcp` | Verify Vertex AI prediction latency from Cloud Monitoring | v0.3.0 |

### Azure ML

| # | Assertion | Module | Description | Since |
|---|-----------|--------|-------------|-------|
| 173 | `assert_endpoint_healthy` | `mltk.monitor.azure` | Verify Azure ML managed endpoint is healthy | v0.3.0 |
| 174 | `assert_endpoint_latency` | `mltk.monitor.azure` | Verify Azure ML endpoint response latency | v0.3.0 |

### Prometheus / Triton

| # | Assertion | Module | Description | Since |
|---|-----------|--------|-------------|-------|
| 175 | `assert_prometheus_metric` | `mltk.monitor.prometheus` | Query Prometheus and verify a metric meets threshold | v0.3.0 |
| 176 | `assert_gpu_utilization` | `mltk.monitor.prometheus` | Verify GPU utilization from Prometheus DCGM exporter | v0.3.0 |
| 177 | `assert_triton_healthy` | `mltk.monitor.prometheus` | Verify NVIDIA Triton Inference Server is ready | v0.3.0 |

### Anomaly Detection

| # | Assertion | Module | Description | Since |
|---|-----------|--------|-------------|-------|
| 178 | `assert_no_test_anomaly` | `mltk.monitor.anomaly` | Detect anomalous test metrics (Z-score/IQR/percentile) | v0.8.0 |

---

## Pipeline

Assertions for reproducibility and end-to-end pipeline validation.

| # | Assertion | Module | Description | Since |
|---|-----------|--------|-------------|-------|
| 179 | `assert_reproducible` | `mltk.pipeline.reproducibility` | Verify function produces identical output across multiple runs | v0.1.0 |
| 180 | `assert_checksum` | `mltk.pipeline.reproducibility` | Verify file SHA-256 checksum matches expected value | v0.1.0 |
| 181 | `assert_pipeline` | `mltk.pipeline.e2e` | Run a sequence of pipeline steps and verify all succeed | v0.1.0 |
| 182 | `assert_onnx_valid` | `mltk.pipeline.onnx` | Verify ONNX model file is valid and loadable | v0.8.0 |

---

## Compliance

| # | Assertion | Module | Description | Since |
|---|-----------|--------|-------------|-------|
| 183 | `assert_owasp_coverage` | `mltk.compliance.owasp_llm` | Verify test results cover all OWASP LLM Top 10 risk categories | v0.6.0 |
| 184 | `assert_nist_rmf_coverage` | `mltk.compliance.nist_ai_rmf` | Verify test results cover NIST AI RMF functions (Govern, Map, Measure, Manage) | v0.7.0 |
| 185 | `assert_iso_42001_coverage` | `mltk.compliance.iso_42001` | Verify test results cover ISO 42001 Annex A controls | v0.7.0 |
| 186 | `assert_hipaa_coverage` | `mltk.compliance.hipaa` | Verify test results cover HIPAA rules | v0.8.0 |
| 187 | `assert_custom_coverage` | `mltk.compliance.custom` | Verify results cover a custom YAML compliance framework | v0.8.0 |
| 188 | `assert_sr_11_7_coverage` | `mltk.compliance.sr_11_7` | Verify test results cover SR 11-7 (Federal Reserve model risk) controls | v0.8.0 |

---

## ML Platform Integrations

| # | Assertion | Module | Description | Since |
|---|-----------|--------|-------------|-------|
| 189 | `assert_kubeflow_pipeline_success` | `mltk.integrations.kubeflow` | Verify KFP pipeline run succeeded | v0.8.0 |
| 190 | `assert_kubeflow_step_outputs` | `mltk.integrations.kubeflow` | Verify pipeline step produced expected artifacts | v0.8.0 |
| 191 | `assert_sagemaker_pipeline_success` | `mltk.integrations.sagemaker_pipeline` | Verify SageMaker pipeline execution succeeded | v0.8.0 |
| 192 | `assert_sagemaker_step_status` | `mltk.integrations.sagemaker_pipeline` | Verify SageMaker pipeline step status | v0.8.0 |
| 193 | `assert_dvc_file_tracked` | `mltk.integrations.dvc` | Verify file is tracked by DVC | v0.8.0 |
| 194 | `assert_dvc_data_version` | `mltk.integrations.dvc` | Verify DVC file hash matches expected | v0.8.0 |

---

## Enterprise

| # | Assertion | Module | Description | Since |
|---|-----------|--------|-------------|-------|
| 195 | `assert_audit_log_complete` | `mltk.server.audit_log` | Verify required actions appear in audit trail | v0.8.0 |

---

## Testing Utilities

| # | Assertion | Module | Description | Since |
|---|-----------|--------|-------------|-------|
| 196 | `assert_matches_golden` | `mltk.testing.golden` | Verify current output matches a versioned golden baseline (dict, list, or numpy array) | v0.4.0 |
| 197 | `assert_impact_coverage` | `mltk.testing.impact` | Verify all impacted tests were executed after code changes | v0.8.0 |

---

## Core

| # | Assertion | Module | Description | Since |
|---|-----------|--------|-------------|-------|
| -- | `assert_true` | `mltk.core.assertion` | Primitive assertion -- wrap any boolean condition into a TestResult | v0.1.0 |

---

## Import Quick Reference

All public assertions are importable from their module path:

```python
# Data quality
from mltk.data.schema import assert_schema, assert_no_nulls, assert_dtypes
from mltk.data.distribution import assert_range, assert_unique, assert_no_outliers
from mltk.data.freshness import assert_freshness, assert_row_count
from mltk.data.drift import assert_no_drift
from mltk.data.embedding_drift import assert_no_embedding_drift
from mltk.data.synthetic import assert_marginal_fidelity, assert_correlation_preserved, assert_synthetic_novelty, assert_dcr_safe
from mltk.data.pii import assert_no_pii
from mltk.data.labels import assert_label_balance, assert_label_coverage
from mltk.data.statistics import assert_column_mean, assert_column_median, assert_column_stdev, assert_quantiles
from mltk.data.validation import assert_datetime_format, assert_values_in_set, assert_no_conflicting_labels, assert_feature_label_correlation_stable
from mltk.data.lineage import assert_lineage_complete
from mltk.data.preset import assert_data_quality

# Model quality
from mltk.model.metrics import assert_metric
from mltk.model.regression import assert_no_regression
from mltk.model.slicing import assert_slice_performance, assert_calibration
from mltk.model.bias import assert_no_bias
from mltk.model.adversarial import assert_robust
from mltk.model.conformal import assert_interval_coverage, assert_prediction_set_size, assert_conformal_calibration, assert_conditional_coverage
from mltk.model.overfitting import assert_no_overfitting, assert_label_drift
from mltk.model.ab_test import assert_ab_significance
from mltk.model.attribution import assert_top_k_stable, assert_attribution_cosine_stability
from mltk.model.counterfactual import assert_counterfactual_fairness
from mltk.model.causal import assert_ate_significant, assert_no_confounding

# Training bugs
from mltk.training.leakage import assert_no_train_test_overlap, assert_temporal_split, assert_no_target_leakage
from mltk.training.gradient import assert_gradient_flow, assert_no_vanishing_gradient, assert_no_exploding_gradient, assert_loss_finite
from mltk.training.numerical import assert_no_nan_inf, assert_loss_decreasing, assert_no_loss_divergence, assert_softmax_valid
from mltk.training.augmentation import assert_no_augmentation_on_test, assert_augmentation_preserves_signal
from mltk.training.checkpoint import assert_checkpoint_complete, assert_resume_loss_continuous
from mltk.training.distributed import assert_effective_batch_size, assert_gradient_sync, assert_n_rank_gradient_sync, assert_gradient_alignment, assert_weight_divergence, assert_gradient_clipped
from mltk.training.memory import assert_no_memory_leak, assert_loss_is_detached
from mltk.training.skew import assert_no_training_serving_skew

# Inference
from mltk.inference.latency import assert_latency, assert_cold_start
from mltk.inference.throughput import assert_throughput
from mltk.inference.contract import assert_api_contract

# CV
from mltk.domains.cv.detection import assert_iou, assert_map
from mltk.domains.cv.video import assert_frame_accuracy, assert_temporal_consistency
from mltk.domains.cv.classification import assert_topk_accuracy
from mltk.domains.cv.face import assert_face_far
from mltk.domains.cv.tracking import assert_mota, assert_motp, assert_idf1

# NLP
from mltk.domains.nlp.generation import assert_bleu, assert_rouge
from mltk.domains.nlp.ner import assert_ner_f1
from mltk.domains.nlp.security import assert_no_prompt_injection
from mltk.domains.nlp.sentiment import assert_sentiment_positive, assert_no_sentiment_drift
from mltk.domains.nlp.robustness import assert_text_robust

# Speech
from mltk.domains.speech.recognition import assert_wer, assert_cer
from mltk.domains.speech.performance import assert_rtf, assert_accent_coverage

# LLM / RAG / Agentic
from mltk.domains.llm.similarity import assert_semantic_similarity
from mltk.domains.llm.safety import assert_no_toxicity, assert_no_hallucination, assert_no_system_prompt_leakage, assert_refusal_consistency, assert_safety_taxonomy
from mltk.domains.llm.latency import assert_ttft, assert_itl
from mltk.domains.llm.rag import assert_faithfulness, assert_context_relevancy, assert_answer_relevancy, assert_context_precision, assert_context_recall
from mltk.domains.llm.ragas import assert_ragas_score
from mltk.domains.llm.retrieval import assert_ndcg, assert_mrr, assert_recall_at_k, assert_map_at_k
from mltk.domains.llm.judge import assert_llm_judge_score, assert_llm_judge_pairwise
from mltk.domains.llm.summarization import assert_summary_coverage, assert_summary_compression, assert_summary_faithfulness
from mltk.domains.llm.coherence import assert_coherence
from mltk.domains.llm.bertscore import assert_bertscore
from mltk.domains.llm.long_context import assert_needle_in_haystack, assert_context_utilization, assert_no_lost_in_middle
from mltk.domains.llm.agentic import assert_task_completion, assert_tool_selection, assert_tool_call_correctness, assert_tool_chain, assert_no_forbidden_actions, assert_step_efficiency, assert_no_redundant_calls, assert_no_hallucinated_tools, assert_cost_budget, assert_error_recovery
from mltk.domains.llm.multi_agent import assert_no_agent_loop, assert_agent_handoff
from mltk.domains.llm.text_quality import assert_text_length, assert_output_format, assert_readability
from mltk.domains.llm.conversation import assert_knowledge_retention, assert_turn_relevancy, assert_conversation_completeness

# Recommendation
from mltk.domains.recommendation import assert_hit_rate, assert_diversity, assert_novelty, assert_coverage, assert_serendipity

# Healthcare
from mltk.domains.healthcare import assert_sensitivity, assert_specificity, assert_ppv, assert_npv, assert_clinical_agreement

# Code Generation
from mltk.domains.codegen import assert_code_executes, assert_code_passes_tests, assert_no_code_vulnerabilities, assert_code_complexity

# Multimodal & RL
from mltk.domains.multimodal import assert_image_text_alignment, assert_cross_modal_consistency
from mltk.domains.rl import assert_reward_bounded, assert_cumulative_reward

# Tabular
from mltk.domains.tabular.features import assert_feature_drift, assert_feature_importance_stable
from mltk.domains.tabular.quality import assert_class_balance

# Monitoring
from mltk.monitor.drift_monitor import assert_no_degradation, assert_sla, assert_no_output_drift
from mltk.monitor.streaming_drift import assert_no_streaming_drift
from mltk.monitor.concept_drift import assert_no_concept_drift
from mltk.monitor.gpu import assert_gpu_utilization_local, assert_gpu_memory_local
from mltk.monitor.aws import assert_endpoint_healthy, assert_endpoint_latency, assert_endpoint_error_rate
from mltk.monitor.gcp import assert_endpoint_healthy, assert_prediction_latency
from mltk.monitor.azure import assert_endpoint_healthy, assert_endpoint_latency
from mltk.monitor.prometheus import assert_prometheus_metric, assert_gpu_utilization, assert_triton_healthy
from mltk.monitor.anomaly import assert_no_test_anomaly

# Pipeline
from mltk.pipeline.reproducibility import assert_reproducible, assert_checksum
from mltk.pipeline.e2e import assert_pipeline
from mltk.pipeline.onnx import assert_onnx_valid

# Compliance
from mltk.compliance.owasp_llm import assert_owasp_coverage
from mltk.compliance.nist_ai_rmf import assert_nist_rmf_coverage
from mltk.compliance.iso_42001 import assert_iso_42001_coverage
from mltk.compliance.hipaa import assert_hipaa_coverage
from mltk.compliance.custom import assert_custom_coverage
from mltk.compliance.sr_11_7 import assert_sr_11_7_coverage

# Integrations
from mltk.integrations.kubeflow import assert_kubeflow_pipeline_success, assert_kubeflow_step_outputs
from mltk.integrations.sagemaker_pipeline import assert_sagemaker_pipeline_success, assert_sagemaker_step_status
from mltk.integrations.dvc import assert_dvc_file_tracked, assert_dvc_data_version

# Enterprise
from mltk.server.audit_log import assert_audit_log_complete

# Testing utilities
from mltk.testing.golden import assert_matches_golden
from mltk.testing.impact import assert_impact_coverage

# Core
from mltk.core.assertion import assert_true
```
