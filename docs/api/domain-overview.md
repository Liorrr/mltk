# Domain Overview

A single-page map of every domain testing kit in mltk. Each kit provides specialized `assert_*` functions for a specific ML domain or application type. All assertions follow the same pattern: call with your data, get a `TestResult` back, integrate with pytest.

For the complete assertion-by-assertion index (every parameter, every module path), see the [Assertion Index](assertion-index.md).

---

## At a Glance

| Domain Kit | Assertions | Module | Key Use Cases |
|-----------|-----------|--------|---------------|
| [Data Quality](#data-quality) | 23 | `mltk.data.*` | Schema, nulls, range, drift, PII, lineage, synthetic data |
| [Model Quality](#model-quality) | 15 | `mltk.model.*` | Metrics, regression, slicing, bias, adversarial, conformal, overfitting |
| [LLM / GenAI](#llm--genai) | 30+ | `mltk.domains.llm.*` | Safety, RAG, agentic, judge, summarization, retrieval, long-context |
| [NLP](#nlp) | 8 | `mltk.domains.nlp.*` | BLEU, ROUGE, NER, prompt injection, sentiment, robustness |
| [Computer Vision](#computer-vision) | 9 | `mltk.domains.cv.*` | IoU, mAP, video, tracking, face recognition, top-K |
| [Speech](#speech) | 4 | `mltk.domains.speech.*` | WER, CER, real-time factor, accent coverage |
| [Tabular](#tabular) | 3 | `mltk.domains.tabular.*` | Feature drift, importance stability, class balance |
| [Multimodal](#multimodal) | 2 | `mltk.domains.multimodal` | Image-text alignment, cross-modal consistency |
| [Reinforcement Learning](#reinforcement-learning) | 2 | `mltk.domains.rl` | Reward bounds, cumulative reward |
| [Recommendation](#recommendation) | 5 | `mltk.domains.recommendation` | Hit rate, nDCG, coverage, diversity, novelty |
| [Healthcare](#healthcare) | 5 | `mltk.domains.healthcare` | Sensitivity, specificity, calibration, subgroup, regulatory |
| [Code Generation](#code-generation) | 4 | `mltk.domains.code_gen` | Syntax validity, test pass rate, security scan, functional correctness |

**Total: ~190+ assertions** across 12 domain kits, plus training, inference, monitoring, pipeline, and compliance modules.

---

## Data Quality

Validate raw data before it reaches your model. Schema enforcement, null detection, range checks, distribution analysis, drift detection, PII scanning, label quality, data lineage, and synthetic data validation.

**23 assertions** across 10 submodules.

| Category | Assertions | What they catch |
|----------|-----------|-----------------|
| Schema & Structure | `assert_schema`, `assert_no_nulls`, `assert_dtypes` | Wrong column types, missing columns, unexpected nulls |
| Distribution | `assert_range`, `assert_unique`, `assert_no_outliers` | Out-of-range values, duplicate keys, statistical outliers |
| Freshness | `assert_freshness`, `assert_row_count` | Stale data, unexpected row counts |
| Drift | `assert_no_drift`, `assert_no_embedding_drift` | Feature distribution shift between training and serving |
| Synthetic Data | `assert_marginal_fidelity`, `assert_correlation_preserved`, `assert_synthetic_novelty`, `assert_dcr_safe` | Poor synthetic data quality, privacy leaks |
| PII | `assert_no_pii` | Emails, phone numbers, SSNs, API keys in training data |
| Labels | `assert_label_balance`, `assert_label_coverage` | Class imbalance, missing label categories |
| Statistics | `assert_column_mean`, `assert_column_median`, `assert_column_stdev`, `assert_quantiles` | Statistical property violations |
| Validation | `assert_datetime_format`, `assert_values_in_set`, `assert_no_conflicting_labels`, `assert_feature_label_correlation_stable` | Format errors, invalid values, contradictory labels |
| Lineage | `assert_lineage_complete` | Missing pipeline steps in data provenance |

:point_right: [Data Schema](data-schema.md) | [Distribution](data-distribution.md) | [Drift](data-drift.md) | [PII](data-pii.md) | [Labels](data-labels.md) | [Statistics](data-statistics.md) | [Synthetic Data](synthetic-data.md)

---

## Model Quality

Validate model performance, fairness, and robustness after training and before deployment.

**15 assertions** across 7 submodules.

| Category | Assertions | What they catch |
|----------|-----------|-----------------|
| Metrics | `assert_metric` | Any sklearn metric below threshold (accuracy, F1, AUC, etc.) |
| Regression | `assert_no_regression` | Performance drop from baseline |
| Slicing | `assert_slice_performance`, `assert_calibration` | Poor performance on subgroups, uncalibrated probabilities |
| Bias | `assert_no_bias` | Demographic parity, equal opportunity, predictive parity violations |
| Adversarial | `assert_robust` | Prediction instability under perturbation |
| Conformal | `assert_interval_coverage`, `assert_prediction_set_size`, `assert_conformal_calibration`, `assert_conditional_coverage` | Unreliable uncertainty estimates |
| Overfitting | `assert_no_overfitting`, `assert_label_drift` | Train-test gap, label distribution shift |

:point_right: [Metrics](model-metrics.md) | [Regression](model-regression.md) | [Slicing](model-slicing.md) | [Bias](model-bias.md) | [Adversarial](model-adversarial.md) | [Conformal](conformal.md)

---

## LLM / GenAI

The largest domain kit. Covers safety, hallucination detection, RAG evaluation, agentic tool-use testing, LLM-as-judge scoring, summarization metrics, retrieval ranking, multi-agent coordination, conversation quality, and long-context window evaluation.

**30+ assertions** across 15+ submodules. No external model dependencies for most assertions -- keyword overlap and regex heuristics run in-process.

| Category | Assertions | What they catch |
|----------|-----------|-----------------|
| Safety | `assert_no_toxicity`, `assert_no_hallucination`, `assert_no_system_prompt_leakage`, `assert_refusal_consistency`, `assert_safety_taxonomy` | Toxic output, hallucinated facts, prompt leakage, inconsistent refusal, category-level safety gaps |
| Similarity | `assert_semantic_similarity` | Poor semantic match between reference and generated text |
| LLM Latency | `assert_ttft`, `assert_itl` | Slow time-to-first-token, high inter-token latency |
| RAG | `assert_faithfulness`, `assert_context_relevancy`, `assert_answer_relevancy`, `assert_context_precision`, `assert_context_recall`, `assert_ragas_score` | Hallucinated answers, irrelevant context, poor retrieval |
| Retrieval Ranking | `assert_ndcg`, `assert_mrr`, `assert_recall_at_k`, `assert_map_at_k` | Bad document ranking, low recall in search results |
| LLM-as-Judge | `assert_llm_judge_score`, `assert_llm_judge_pairwise` | Subjective quality below threshold, A/B comparison |
| Summarization | `assert_summary_faithfulness`, `assert_summary_coverage`, `assert_summary_conciseness` | Unfaithful, incomplete, or verbose summaries |
| Coherence | `assert_coherence`, `assert_bertscore` | Incoherent text, low semantic similarity (BERTScore) |
| Agentic | `assert_task_completion`, `assert_tool_selection`, `assert_tool_call_correctness`, `assert_tool_chain`, `assert_no_forbidden_actions`, `assert_step_efficiency`, `assert_no_redundant_calls`, `assert_no_hallucinated_tools`, `assert_cost_budget`, `assert_error_recovery` | Wrong tools, hallucinated tools, stuck loops, budget overruns |
| Multi-Agent | `assert_no_agent_loop`, `assert_agent_handoff` | Circular delegation, broken handoff sequences |
| Text Quality | `assert_text_length`, `assert_output_format`, `assert_readability` | Wrong length, format violations, poor readability |
| Conversation | `assert_knowledge_retention`, `assert_turn_relevancy`, `assert_conversation_completeness` | Lost context across turns, off-topic responses |
| Long-Context | `assert_needle_in_haystack`, `assert_context_utilization`, `assert_no_lost_in_middle` | Failure to use full context window, middle-of-context blind spots |
| Attribution | `assert_top_k_stable`, `assert_attribution_cosine_stability` | Unstable feature attributions across runs |

:point_right: [LLM Evaluation](llm.md) | [RAG & Agentic](rag-evaluation.md) | [LLM-as-Judge](llm-judge.md) | [Summarization](summarization-metrics.md) | [Retrieval](retrieval-metrics.md) | [Long-Context](long-context.md) | [Agent Trace](agentic-trace.md) | [Multi-Agent](multi-agent.md)

---

## NLP

Text generation quality, named entity recognition, prompt injection detection, sentiment analysis, and text robustness under perturbation.

**8 assertions** across 5 submodules.

| Assertion | What it tests |
|-----------|--------------|
| `assert_bleu` | Translation/generation quality (BLEU score) |
| `assert_rouge` | Summarization quality (ROUGE-1/2/L) |
| `assert_ner_f1` | Named entity recognition accuracy |
| `assert_no_prompt_injection` | Resilience against prompt injection attacks |
| `assert_sentiment_positive` | Positive sentiment ratio in generated text |
| `assert_no_sentiment_drift` | Sentiment distribution stability over time |
| `assert_text_robust` | Prediction stability under text perturbations (typos, case changes, synonyms) |

:point_right: [NLP Testing](nlp.md) | [Text Noise Robustness](nlp-robustness.md)

---

## Computer Vision

Object detection, video analytics, multi-object tracking, face recognition, and image classification.

**9 assertions** across 5 submodules.

| Assertion | What it tests |
|-----------|--------------|
| `assert_iou` | Bounding box overlap (Intersection over Union) |
| `assert_map` | Object detection mean Average Precision |
| `assert_frame_accuracy` | Per-frame classification accuracy for video |
| `assert_temporal_consistency` | Smooth bounding box tracking across frames |
| `assert_topk_accuracy` | Top-K image classification accuracy |
| `assert_face_far` | False Accept Rate for face recognition |
| `assert_mota` | Multiple Object Tracking Accuracy |
| `assert_motp` | Multiple Object Tracking Precision |
| `assert_idf1` | Identity-aware tracking consistency |

:point_right: [Computer Vision Testing](cv.md)

---

## Speech

Automatic speech recognition quality, processing speed, and accent fairness.

**4 assertions** across 2 submodules.

| Assertion | What it tests |
|-----------|--------------|
| `assert_wer` | Word Error Rate below threshold |
| `assert_cer` | Character Error Rate below threshold |
| `assert_rtf` | Real-Time Factor (processing speed vs. audio duration) |
| `assert_accent_coverage` | WER gap across accents stays below bias threshold |

:point_right: [Speech Testing](speech.md)

---

## Tabular

Feature-level drift detection, SHAP importance stability, and class balance for structured/tabular ML.

**3 assertions** in 2 submodules.

| Assertion | What it tests |
|-----------|--------------|
| `assert_feature_drift` | Per-column distribution drift between two DataFrames |
| `assert_feature_importance_stable` | SHAP feature importance ranking consistency |
| `assert_class_balance` | Label class distribution balance |

:point_right: [Tabular Testing](tabular.md)

---

## Multimodal

Validate alignment between modalities (image-text, audio-text) in CLIP-style models and cross-modal prediction consistency.

**2 assertions** in 1 module.

| Assertion | What it tests |
|-----------|--------------|
| `assert_image_text_alignment` | Cosine similarity between paired image and text embeddings |
| `assert_cross_modal_consistency` | Prediction agreement across different input modalities |

:point_right: [Multimodal & RL Testing](multimodal-rl.md)

---

## Reinforcement Learning

Validate RL reward functions and episode quality.

**2 assertions** in 1 module.

| Assertion | What it tests |
|-----------|--------------|
| `assert_reward_bounded` | Per-step rewards fall within expected bounds |
| `assert_cumulative_reward` | Episode cumulative reward meets minimum threshold |

:point_right: [Multimodal & RL Testing](multimodal-rl.md)

---

## Recommendation

Validate recommendation system quality: ranking accuracy, catalog coverage, result diversity, and novelty. Catches the common failure modes of recommenders -- high accuracy but only recommending popular items, or technically relevant but repetitively similar suggestions.

**5 assertions** in `mltk.domains.recommendation`.

| Assertion | What it tests |
|-----------|--------------|
| `assert_hit_rate` | Fraction of users whose relevant items appear in top-K recommendations |
| `assert_ndcg` | Ranking quality -- relevant items should appear higher in the list |
| `assert_coverage` | Fraction of the item catalog that appears in recommendations (catalog utilization) |
| `assert_diversity` | Intra-list diversity -- recommended items should not all be near-duplicates |
| `assert_novelty` | Degree to which recommendations surface non-obvious, long-tail items |

**Why these five?** A recommender can score high on accuracy (hit rate, nDCG) while only recommending the same 50 popular items to every user. Coverage, diversity, and novelty catch this pathology -- they ensure the system is useful, not just technically correct.

```python
from mltk.domains.recommendation import (
    assert_hit_rate,
    assert_ndcg,
    assert_coverage,
    assert_diversity,
    assert_novelty,
)

# user_recs: dict mapping user_id -> list of recommended item_ids (ranked)
# user_relevant: dict mapping user_id -> set of actually relevant item_ids
# all_items: set of every item in the catalog

assert_hit_rate(user_recs, user_relevant, k=10, min_rate=0.3)
assert_ndcg(user_recs, user_relevant, k=10, min_score=0.25)
assert_coverage(user_recs, all_items, min_coverage=0.15)
assert_diversity(user_recs, item_embeddings, min_diversity=0.4)
assert_novelty(user_recs, item_popularity, min_novelty=3.0)
```

:point_right: [Recommendation Systems](recommendation.md)

---

## Healthcare

Validate clinical ML models with domain-specific thresholds and regulatory awareness. These assertions enforce the stricter performance requirements of medical AI -- where a false negative can mean a missed diagnosis and a miscalibrated model can cause inappropriate treatment decisions.

**5 assertions** in `mltk.domains.healthcare`.

| Assertion | What it tests |
|-----------|--------------|
| `assert_clinical_sensitivity` | Minimum true positive rate (recall) for clinical screening -- missing a positive case is the critical failure mode |
| `assert_clinical_specificity` | Minimum true negative rate -- excessive false positives waste resources and cause patient anxiety |
| `assert_calibration_clinical` | Predicted probabilities match observed outcomes in clinical ranges (e.g., 0-10%, 10-30%, 30-50%, 50%+) |
| `assert_subgroup_performance` | Model performance meets threshold across demographic and clinical subgroups (age, sex, comorbidities) |
| `assert_regulatory_threshold` | Model metrics meet predefined regulatory floor values (e.g., FDA 510(k) substantially equivalent thresholds) |

```python
from mltk.domains.healthcare import (
    assert_clinical_sensitivity,
    assert_clinical_specificity,
    assert_subgroup_performance,
)

assert_clinical_sensitivity(y_true, y_pred, min_sensitivity=0.95)
assert_clinical_specificity(y_true, y_pred, min_specificity=0.80)
assert_subgroup_performance(
    y_true, y_pred, groups=patient_demographics,
    metric="sensitivity", min_score=0.90,
)
```

---

## Code Generation

Validate LLM-generated code for syntax correctness, functional accuracy, security, and test coverage. Catches the common failure modes of code-generating models -- syntactically plausible but non-functional code, security vulnerabilities, and code that passes no tests.

**4 assertions** in `mltk.domains.code_gen`.

| Assertion | What it tests |
|-----------|--------------|
| `assert_syntax_valid` | Generated code parses without syntax errors (language-aware: Python, JavaScript, TypeScript, Rust, Go) |
| `assert_test_pass_rate` | Fraction of test cases the generated code passes |
| `assert_no_security_issues` | Generated code contains no known vulnerability patterns (hardcoded secrets, SQL injection, path traversal, command injection) |
| `assert_functional_correctness` | Generated code produces correct output for a set of input/output examples (pass@k metric) |

```python
from mltk.domains.code_gen import (
    assert_syntax_valid,
    assert_test_pass_rate,
    assert_no_security_issues,
    assert_functional_correctness,
)

generated_code = model.generate("Write a function to merge two sorted lists.")

assert_syntax_valid(generated_code, language="python")
assert_no_security_issues(generated_code, language="python")
assert_functional_correctness(
    generated_code,
    test_cases=[
        {"input": {"a": [1, 3, 5], "b": [2, 4, 6]}, "expected": [1, 2, 3, 4, 5, 6]},
        {"input": {"a": [], "b": [1, 2]}, "expected": [1, 2]},
        {"input": {"a": [1], "b": []}, "expected": [1]},
    ],
    min_pass_rate=1.0,
)
```

---

## What Else?

Beyond domain kits, mltk provides testing infrastructure for the full ML lifecycle:

| Module | Purpose | Assertions |
|--------|---------|-----------|
| **Training** | Gradient health, leakage detection, checkpoints, distributed training, memory | 21 |
| **Inference** | Latency (P50/P95/P99), throughput, API contracts | 4 |
| **Monitoring** | Production drift, SLA enforcement, AWS/GCP/Azure/Prometheus health | 15 |
| **Pipeline** | Reproducibility, checksums, end-to-end pipeline validation | 3 |
| **Compliance** | EU AI Act, FDA, OWASP LLM, NIST AI RMF, ISO 42001, HIPAA, custom frameworks | 7 |
| **Integrations** | Kubeflow, SageMaker, DVC, GitHub, MLflow, W&B, Grafana | 6 |

:point_right: [Training Bugs](training-bugs.md) | [Inference Latency](inference-latency.md) | [Cloud Monitoring](cloud-monitoring.md) | [Pipeline](pipeline.md) | [EU AI Act](eu-ai-act.md) | [Assertion Index](assertion-index.md)
