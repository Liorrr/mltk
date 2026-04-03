# mltk Full API Index
> Generated 2026-04-04 by scripts/generate_skill_index.py

**230** assertions | **11** MCP tools | **28** CLI commands | **8** scanners

---

## Assertion Signatures (230)

### compliance

**`assert_custom_coverage`** (compliance/custom.py:307)
```python
def assert_custom_coverage(results: list[dict], framework_yaml: str, min_coverage: float=0.8)
```
> Assert that test results cover a custom compliance framework.

**`assert_hipaa_coverage`** (compliance/hipaa.py:233)
```python
def assert_hipaa_coverage(results: list[dict], min_coverage: float=0.8)
```
> Assert minimum HIPAA rule coverage.

**`assert_iso_42001_coverage`** (compliance/iso_42001.py:209)
```python
def assert_iso_42001_coverage(results: list[dict], min_coverage: float=0.8)
```
> Assert minimum ISO 42001 Annex A coverage.

**`assert_nist_rmf_coverage`** (compliance/nist_ai_rmf.py:294)
```python
def assert_nist_rmf_coverage(results: list[dict], min_coverage: float=0.8)
```
> Assert minimum NIST AI RMF function coverage.

**`assert_owasp_coverage`** (compliance/owasp_llm.py:261)
```python
def assert_owasp_coverage(results: list[dict], min_coverage: float=0.5)
```
> Assert minimum OWASP LLM Top 10 coverage.

**`assert_sr_11_7_coverage`** (compliance/sr_11_7.py:288)
```python
def assert_sr_11_7_coverage(results: list[dict], min_coverage: float=0.8)
```
> Assert minimum SR 11-7 section coverage.

### core

**`assert_true`** (core/assertion.py:20)
```python
def assert_true(condition: bool, name: str, message: str, severity: Severity=Severity.CRITICAL, **details: Any)
```
> Base assertion. Raises MltkAssertionError if condition is False.

### cv

**`assert_face_far`** (domains/cv/face.py:14)
```python
def assert_face_far(similarities: Any, labels: Any, max_far: float=0.001)
```
> Assert False Accept Rate is below threshold.

**`assert_frame_accuracy`** (domains/cv/video.py:18)
```python
def assert_frame_accuracy(frame_preds: Any, frame_labels: Any, threshold: float=0.8)
```
> Assert per-frame classification/detection accuracy.

**`assert_idf1`** (domains/cv/tracking.py:236)
```python
def assert_idf1(gt_tracks: list[dict[str, Any]], pred_tracks: list[dict[str, Any]], min_idf1: float=0.5)
```
> Assert ID F1 score (IDF1) meets threshold.

**`assert_iou`** (domains/cv/detection.py:53)
```python
def assert_iou(pred_boxes: Any, gt_boxes: Any, threshold: float=0.5)
```
> Assert minimum mean IoU between predictions and ground truth.

**`assert_map`** (domains/cv/detection.py:110)
```python
def assert_map(predictions: list[dict[str, Any]], ground_truth: list[dict[str, Any]], iou_threshold: float=0.5, min_map: float=0.5)
```
> Assert mean Average Precision meets threshold.

**`assert_mota`** (domains/cv/tracking.py:76)
```python
def assert_mota(gt_tracks: list[dict[str, Any]], pred_tracks: list[dict[str, Any]], min_mota: float=0.5)
```
> Assert Multi-Object Tracking Accuracy (MOTA) meets threshold.

**`assert_motp`** (domains/cv/tracking.py:165)
```python
def assert_motp(gt_tracks: list[dict[str, Any]], pred_tracks: list[dict[str, Any]], min_motp: float=0.5)
```
> Assert Multi-Object Tracking Precision (MOTP) meets threshold.

**`assert_temporal_consistency`** (domains/cv/video.py:64)
```python
def assert_temporal_consistency(tracked_boxes: list[Any], min_smoothness: float=0.7)
```
> Assert frame-to-frame IoU stability for tracked objects.

**`assert_topk_accuracy`** (domains/cv/classification.py:14)
```python
def assert_topk_accuracy(y_true: Any, y_probs: Any, k: int=5, threshold: float=0.9)
```
> Assert top-K accuracy meets threshold.

### data

**`assert_column_mean`** (data/statistics.py:22)
```python
def assert_column_mean(df: pd.DataFrame, column: str, min_val: float | None=None, max_val: float | None=None)
```
> Assert column mean is within [min_val, max_val]. At least one bound required.

**`assert_column_median`** (data/statistics.py:78)
```python
def assert_column_median(df: pd.DataFrame, column: str, min_val: float | None=None, max_val: float | None=None)
```
> Assert column median is within [min_val, max_val]. At least one bound required.

**`assert_column_stdev`** (data/statistics.py:136)
```python
def assert_column_stdev(df: pd.DataFrame, column: str, min_val: float | None=None, max_val: float | None=None)
```
> Assert column standard deviation is within [min_val, max_val].

**`assert_correlation_preserved`** (data/synthetic.py:141)
```python
def assert_correlation_preserved(real_df: pd.DataFrame, synthetic_df: pd.DataFrame, max_delta: float=0.1, columns: list[str] | None=None, severity: Severity=Severity.CRITICAL)
```
> Assert that pairwise column correlations in synthetic data match real data.

**`assert_data_quality`** (data/preset.py:24)
```python
def assert_data_quality(df: pd.DataFrame, config: dict | None=None)
```
> Run comprehensive data quality checks in one call.

**`assert_datetime_format`** (data/validation.py:21)
```python
def assert_datetime_format(df: pd.DataFrame, column: str, fmt: str='%Y-%m-%d')
```
> Assert all values in column match the specified datetime format.

**`assert_dcr_safe`** (data/synthetic.py:366)
```python
def assert_dcr_safe(real_df: pd.DataFrame, synthetic_df: pd.DataFrame, min_dcr: float=0.05, sample_size: int=2000, columns: list[str] | None=None, severity: Severity=Severity.CRITICAL)
```
> Assert that synthetic records are not dangerously close to real records.

**`assert_dtypes`** (data/schema.py:123)
```python
def assert_dtypes(df: pd.DataFrame, expected: dict[str, str], severity: Severity=Severity.CRITICAL)
```
> Assert exact dtype match for specified columns.

**`assert_feature_label_correlation_stable`** (data/validation.py:210)
```python
def assert_feature_label_correlation_stable(train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: list[str], label_col: str, max_shift: float=0.1)
```
> Assert feature-label correlations haven't shifted between train and test.

**`assert_freshness`** (data/freshness.py:24)
```python
def assert_freshness(df: pd.DataFrame, date_column: str, max_age_days: int, reference_date: datetime | None=None)
```
> Assert the most recent date in a column is within max_age_days.

**`assert_label_balance`** (data/labels.py:18)
```python
def assert_label_balance(labels: pd.Series, max_ratio: float=10.0)
```
> Assert class distribution is not too imbalanced.

**`assert_label_coverage`** (data/labels.py:74)
```python
def assert_label_coverage(labels: pd.Series, expected_labels: set[str] | None=None, min_samples: int=1)
```
> Assert all expected label classes are present with sufficient samples.

**`assert_lineage_complete`** (data/lineage.py:123)
```python
def assert_lineage_complete(graph: LineageGraph, expected_steps: int, severity: Severity=Severity.CRITICAL)
```
> Assert lineage graph has the expected number of transformation steps.

**`assert_marginal_fidelity`** (data/synthetic.py:35)
```python
def assert_marginal_fidelity(real: pd.Series, synthetic: pd.Series, method: str='ks', max_divergence: float=0.1, severity: Severity=Severity.CRITICAL)
```
> Assert that a synthetic column follows the same distribution as the real column.

**`assert_no_conflicting_labels`** (data/validation.py:136)
```python
def assert_no_conflicting_labels(df: pd.DataFrame, feature_cols: list[str], label_col: str)
```
> Assert no rows have identical features but different labels.

**`assert_no_drift`** (data/drift.py:38)
```python
def assert_no_drift(reference: pd.Series, current: pd.Series, method: str='ks', threshold: float | None=None, severity: Severity=Severity.CRITICAL)
```
> Assert no significant distribution drift between reference and current data.

**`assert_no_embedding_drift`** (data/embedding_drift.py:18)
```python
def assert_no_embedding_drift(reference: Any, current: Any, method: str='cosine', threshold: float=0.1)
```
> Assert no significant drift in embedding space.

**`assert_no_multivariate_drift`** (data/drift.py:514)
```python
def assert_no_multivariate_drift(reference: np.ndarray | pd.DataFrame, current: np.ndarray | pd.DataFrame, threshold: float=0.05, n_permutations: int=200, max_samples: int=500, kernel: str='rbf', sigma: float | None=None, severity: Severity=Severity.CRITICAL)
```
> Assert no multivariate distribution drift (MMD test).

**`assert_no_nulls`** (data/schema.py:74)
```python
def assert_no_nulls(df: pd.DataFrame, columns: list[str] | None=None, severity: Severity=Severity.CRITICAL)
```
> Assert no null/NaN values in specified columns (or all columns).

**`assert_no_outliers`** (data/distribution.py:109)
```python
def assert_no_outliers(series: pd.Series, method: str='iqr', threshold: float=1.5)
```
> Assert no statistical outliers in a numeric Series.

**`assert_no_pii`** (data/pii.py:621)
```python
def assert_no_pii(df: pd.DataFrame, columns: list[str] | None=None, patterns: list[str] | None=None, allowlist: list[str] | None=None, method: str='regex', entity_types: list[str] | None=None, score_threshold: float | None=None, severity: Severity=Severity.CRITICAL)
```
> Assert no PII detected in DataFrame text columns.

**`assert_quantiles`** (data/statistics.py:199)
```python
def assert_quantiles(df: pd.DataFrame, column: str, quantiles: dict[float, tuple[float, float]])
```
> Assert column quantile values are within specified bounds.

**`assert_range`** (data/distribution.py:19)
```python
def assert_range(series: pd.Series, min_val: float, max_val: float)
```
> Assert all values in a Series fall within [min_val, max_val].

**`assert_row_count`** (data/freshness.py:90)
```python
def assert_row_count(df: pd.DataFrame, min_rows: int | None=None, max_rows: int | None=None)
```
> Assert DataFrame row count is within bounds.

**`assert_schema`** (data/schema.py:19)
```python
def assert_schema(df: pd.DataFrame, expected: dict[str, str], allow_extra_columns: bool=True, severity: Severity=Severity.CRITICAL)
```
> Assert DataFrame columns and dtypes match expected schema.

**`assert_synthetic_novelty`** (data/synthetic.py:256)
```python
def assert_synthetic_novelty(real_df: pd.DataFrame, synthetic_df: pd.DataFrame, max_copy_rate: float=0.05, columns: list[str] | None=None, severity: Severity=Severity.CRITICAL)
```
> Assert that synthetic data is not just a copy of the real data.

**`assert_unique`** (data/distribution.py:66)
```python
def assert_unique(df: pd.DataFrame, columns: list[str])
```
> Assert no duplicate values across specified columns.

**`assert_values_in_set`** (data/validation.py:77)
```python
def assert_values_in_set(df: pd.DataFrame, column: str, allowed_values: set | list)
```
> Assert all values in column are members of the allowed set.

### domains

**`assert_clinical_agreement`** (domains/healthcare.py:517)
```python
def assert_clinical_agreement(y_true: np.ndarray, y_pred: np.ndarray, min_kappa: float=0.6, severity: Severity=Severity.CRITICAL)
```
> Assert that clinical agreement (Cohen's Kappa) meets a threshold.

**`assert_code_complexity`** (domains/codegen.py:475)
```python
def assert_code_complexity(code: str, max_cyclomatic: int=10, max_lines: int=200)
```
> Assert that generated code complexity stays within bounds.

**`assert_code_executes`** (domains/codegen.py:204)
```python
def assert_code_executes(code: str, timeout_seconds: float=10.0, language: str='python')
```
> Assert that generated code executes without errors.

**`assert_code_passes_tests`** (domains/codegen.py:306)
```python
def assert_code_passes_tests(code: str, test_code: str, timeout_seconds: float=30.0)
```
> Assert that generated code passes a test suite.

**`assert_coverage`** (domains/recommendation.py:319)
```python
def assert_coverage(recommended: list[list], catalog_size: int, min_coverage: float=0.1)
```
> Assert that catalog coverage meets a minimum threshold.

**`assert_cumulative_reward`** (domains/rl.py:138)
```python
def assert_cumulative_reward(rewards: np.ndarray | list, min_cumulative: float)
```
> Assert that cumulative episode reward meets a minimum threshold.

**`assert_diversity`** (domains/recommendation.py:124)
```python
def assert_diversity(recommended: list[list], item_categories: dict, min_diversity: float=0.5)
```
> Assert that recommendation diversity meets a minimum threshold.

**`assert_hit_rate`** (domains/recommendation.py:45)
```python
def assert_hit_rate(recommended: list[list], relevant: list[set], min_rate: float=0.5)
```
> Assert that hit rate meets a minimum threshold.

**`assert_no_code_vulnerabilities`** (domains/codegen.py:391)
```python
def assert_no_code_vulnerabilities(code: str, rules: list[str] | None=None)
```
> Assert that generated code contains no security vulnerabilities.

**`assert_novelty`** (domains/recommendation.py:225)
```python
def assert_novelty(recommended: list[list], popularity: dict, min_novelty: float=0.3)
```
> Assert that recommendation novelty meets a minimum threshold.

**`assert_npv`** (domains/healthcare.py:422)
```python
def assert_npv(y_true: np.ndarray, y_pred: np.ndarray, min_npv: float=0.9, severity: Severity=Severity.CRITICAL)
```
> Assert that negative predictive value meets a minimum threshold.

**`assert_ppv`** (domains/healthcare.py:326)
```python
def assert_ppv(y_true: np.ndarray, y_pred: np.ndarray, min_ppv: float=0.8, severity: Severity=Severity.CRITICAL)
```
> Assert that positive predictive value meets a minimum threshold.

**`assert_reward_bounded`** (domains/rl.py:40)
```python
def assert_reward_bounded(rewards: np.ndarray | list, min_reward: float | None=None, max_reward: float | None=None)
```
> Assert that all rewards fall within specified bounds.

**`assert_sensitivity`** (domains/healthcare.py:134)
```python
def assert_sensitivity(y_true: np.ndarray, y_pred: np.ndarray, min_sensitivity: float=0.9, severity: Severity=Severity.CRITICAL)
```
> Assert that model sensitivity meets a minimum threshold.

**`assert_serendipity`** (domains/recommendation.py:393)
```python
def assert_serendipity(recommended: list[list], expected: list[list], relevant: list[set], min_serendipity: float=0.1)
```
> Assert that recommendation serendipity meets a minimum threshold.

**`assert_specificity`** (domains/healthcare.py:231)
```python
def assert_specificity(y_true: np.ndarray, y_pred: np.ndarray, min_specificity: float=0.9, severity: Severity=Severity.CRITICAL)
```
> Assert that model specificity meets a minimum threshold.

### eval

**`assert_dataset_quality`** (eval/dataset.py:1148)
```python
def assert_dataset_quality(dataset: EvalDataset, *, min_samples: int=50, min_target_coverage: float=0.9, max_duplicate_rate: float=0.01, min_categories: int | None=None)
```
> Assert evaluation dataset meets quality standards.

### inference

**`assert_api_contract`** (inference/contract.py:80)
```python
def assert_api_contract(func: Callable[..., Any], input_data: Any, input_schema: dict[str, Any] | None=None, output_schema: dict[str, Any] | None=None)
```
> Assert inference function input/output matches schema.

**`assert_cold_start`** (inference/latency.py:109)
```python
def assert_cold_start(func: Callable[..., Any], *args: Any, max_ms: float=2000.0, severity: Severity=Severity.CRITICAL)
```
> Assert first-call latency (cold start) is within bounds.

**`assert_latency`** (inference/latency.py:20)
```python
def assert_latency(func: Callable[..., Any], *args: Any, p50: float | None=None, p95: float | None=None, p99: float | None=None, iterations: int=100, warmup: int=5, severity: Severity=Severity.CRITICAL)
```
> Assert inference latency percentiles are within bounds.

**`assert_throughput`** (inference/throughput.py:19)
```python
def assert_throughput(func: Callable[..., Any], *args: Any, min_rps: float=100.0, duration: float=5.0, concurrency: int=1)
```
> Assert model serves at least min_rps requests per second.

### integrations

**`assert_dvc_data_version`** (integrations/dvc.py:223)
```python
def assert_dvc_data_version(file_path: str, expected_md5: str | None=None, dvc_root: str='.', severity: Severity=Severity.CRITICAL)
```
> Assert that a DVC-tracked file has the expected content hash.

**`assert_dvc_file_tracked`** (integrations/dvc.py:131)
```python
def assert_dvc_file_tracked(file_path: str, dvc_root: str='.', severity: Severity=Severity.CRITICAL)
```
> Assert that a data file is properly tracked by DVC.

**`assert_kubeflow_pipeline_success`** (integrations/kubeflow.py:78)
```python
def assert_kubeflow_pipeline_success(run_id: str, host: str='http://localhost:8080', namespace: str='kubeflow', timeout_seconds: int=60)
```
> Assert that a Kubeflow Pipeline run completed successfully.

**`assert_kubeflow_step_outputs`** (integrations/kubeflow.py:174)
```python
def assert_kubeflow_step_outputs(run_id: str, step_name: str, expected_artifacts: list[str], host: str='http://localhost:8080', timeout_seconds: int=60)
```
> Assert that a Kubeflow pipeline step produced expected output artifacts.

**`assert_sagemaker_pipeline_success`** (integrations/sagemaker_pipeline.py:80)
```python
def assert_sagemaker_pipeline_success(pipeline_name: str, execution_arn: str | None=None, region: str='us-east-1')
```
> Assert that a SageMaker Pipeline execution completed successfully.

**`assert_sagemaker_step_status`** (integrations/sagemaker_pipeline.py:209)
```python
def assert_sagemaker_step_status(execution_arn: str, step_name: str, expected_status: str='Succeeded', region: str='us-east-1')
```
> Assert that a specific SageMaker Pipeline step has the expected status.

**`assert_trace_quality`** (integrations/trace_quality.py:73)
```python
def assert_trace_quality(trace: dict[str, Any], *, max_latency_ms: float | None=None, max_cost_usd: float | None=None, min_score: float | None=None, judge_fn: Callable[[dict[str, Any]], float] | None=None)
```
> Assert that an LLM trace meets production quality thresholds.

### llm

**`assert_agent_handoff`** (domains/llm/multi_agent.py:147)
```python
def assert_agent_handoff(agent_names: list[str], expected_flow: list[str], strict: bool=False)
```
> Assert that agent handoffs follow an expected flow.

**`assert_answer_relevancy`** (domains/llm/rag.py:277)
```python
def assert_answer_relevancy(question: str, answer: str, min_score: float=0.5, method: str='lexical', embedding_model: str='all-MiniLM-L6-v2', nli_model: str='cross-encoder/nli-deberta-v3-base', judge_fn: Callable[[str, str], float] | None=None)
```
> Assert answer addresses the question.

**`assert_bertscore`** (domains/llm/bertscore.py:15)
```python
def assert_bertscore(reference_embeddings: Any, hypothesis_embeddings: Any, min_f1: float=0.5, suppress_warnings: bool=False)
```
> Assert BERTScore F1 meets threshold.

**`assert_coherence`** (domains/llm/coherence.py:18)
```python
def assert_coherence(text: str, min_score: float=0.3)
```
> Assert text is internally coherent.

**`assert_context_precision`** (domains/llm/rag.py:405)
```python
def assert_context_precision(relevant_ids: list[str], retrieved_ids: list[str], min_precision: float=0.5)
```
> Assert precision of retrieval: |relevant ∩ retrieved| / |retrieved|.

**`assert_context_recall`** (domains/llm/rag.py:465)
```python
def assert_context_recall(relevant_ids: list[str], retrieved_ids: list[str], min_recall: float=0.5)
```
> Assert recall of retrieval: |relevant ∩ retrieved| / |relevant|.

**`assert_context_relevancy`** (domains/llm/rag.py:149)
```python
def assert_context_relevancy(question: str, context: str | list[str], min_score: float=0.5, method: str='lexical', embedding_model: str='all-MiniLM-L6-v2', nli_model: str='cross-encoder/nli-deberta-v3-base', judge_fn: Callable[[str, str], float] | None=None)
```
> Assert retrieved context is relevant to the question.

**`assert_context_utilization`** (domains/llm/long_context.py:149)
```python
def assert_context_utilization(model_fn: Callable[[str], str], facts: list[str], question: str, min_facts_used: int=3)
```
> Assert that an LLM uses facts from across its full context window.

**`assert_conversation_completeness`** (domains/llm/conversation.py:192)
```python
def assert_conversation_completeness(turns: list[dict[str, str]], expected_topics: list[str], min_coverage: float=0.8)
```
> Assert conversation covers all expected topics.

**`assert_cost_budget`** (domains/llm/agentic.py:581)
```python
def assert_cost_budget(trace: AgentTrace, max_total_tokens: int | None=None, max_duration_ms: float | None=None)
```
> Assert an agent's resource consumption stays within budget.

**`assert_error_recovery`** (domains/llm/agentic.py:678)
```python
def assert_error_recovery(trace: AgentTrace, max_consecutive_errors: int=3)
```
> Assert the agent does not produce long streaks of consecutive errors.

**`assert_faithfulness`** (domains/llm/rag.py:22)
```python
def assert_faithfulness(answer: str, context: str | list[str], min_score: float=0.7, method: str='lexical', embedding_model: str='all-MiniLM-L6-v2', nli_model: str='cross-encoder/nli-deberta-v3-base', judge_fn: Callable[[str, str], float] | None=None)
```
> Assert answer is grounded in the provided context.

**`assert_itl`** (domains/llm/latency.py:67)
```python
def assert_itl(func: Callable[..., Any], *args: Any, max_ms: float=50.0, num_tokens: int=10)
```
> Assert Inter-Token Latency is within bounds.

**`assert_knowledge_retention`** (domains/llm/conversation.py:16)
```python
def assert_knowledge_retention(turns: list[dict[str, str]], min_score: float=0.7)
```
> Assert bot retains factual knowledge across conversation turns.

**`assert_llm_judge_pairwise`** (domains/llm/judge.py:347)
```python
def assert_llm_judge_pairwise(judge_fn: Callable[[str], str], prompts: list[str], responses_a: list[str], responses_b: list[str], expected_winner: str='a', min_win_rate: float=0.6, criterion: str='helpfulness', rubric: str | None=None)
```
> Assert that one set of responses is preferred over another by a judge LLM.

**`assert_llm_judge_score`** (domains/llm/judge.py:206)
```python
def assert_llm_judge_score(judge_fn: Callable[[str], float], prompts: list[str], responses: list[str], criterion: str='helpfulness', min_score: float=3.0, max_score: float=5.0, rubric: str | None=None)
```
> Assert that LLM responses meet a minimum quality score via judge evaluation.

**`assert_map_at_k`** (domains/llm/retrieval.py:331)
```python
def assert_map_at_k(relevant: list[set], retrieved: list[list], k: int=10, min_map: float=0.5)
```
> Assert that Mean Average Precision@K meets a minimum threshold.

**`assert_mcp_context_window`** (domains/llm/mcp.py:553)
```python
def assert_mcp_context_window(trace: McpTrace, model_context_limit: int | None=None, max_utilization: float=0.9)
```
> Assert that context window utilization stays within budget.

**`assert_mcp_error_recovery`** (domains/llm/mcp.py:634)
```python
def assert_mcp_error_recovery(trace: McpTrace, max_same_tool_retries: int=3)
```
> Assert the agent does not retry the same tool with same args.

**`assert_mcp_resource_access`** (domains/llm/mcp.py:433)
```python
def assert_mcp_resource_access(trace: McpTrace, expected_uris: Sequence[str] | None=None, forbidden_uris: Sequence[str] | None=None, max_reads: int | None=None)
```
> Assert correct resource access patterns in an MCP session.

**`assert_mcp_tool_schema_conformance`** (domains/llm/mcp.py:189)
```python
def assert_mcp_tool_schema_conformance(tool_schema: dict[str, Any], actual_args: dict[str, Any], tool_name: str='')
```
> Assert that tool arguments conform to the tool's JSON Schema.

**`assert_mcp_tool_selection`** (domains/llm/mcp.py:300)
```python
def assert_mcp_tool_selection(trace: McpTrace, expected_tools: Sequence[str], server: str | None=None)
```
> Assert that an MCP trace contains the expected tools.

**`assert_mrr`** (domains/llm/retrieval.py:175)
```python
def assert_mrr(queries_results: list[list[bool]], min_mrr: float=0.5)
```
> Assert that Mean Reciprocal Rank meets a minimum threshold.

**`assert_ndcg`** (domains/llm/retrieval.py:95)
```python
def assert_ndcg(y_true: list[list[int]], y_scores: list[list[float]], k: int=10, min_ndcg: float=0.8)
```
> Assert that mean nDCG@k meets a minimum threshold.

**`assert_needle_in_haystack`** (domains/llm/long_context.py:52)
```python
def assert_needle_in_haystack(model_fn: Callable[[str], str], needle: str, haystack: str, positions: list[float] | None=None, min_recall: float=0.8)
```
> Assert that an LLM can retrieve a fact inserted at various context positions.

**`assert_no_agent_loop`** (domains/llm/multi_agent.py:63)
```python
def assert_no_agent_loop(agent_names: list[str], max_cycles: int=2)
```
> Assert that a multi-agent delegation sequence contains no runaway loops.

**`assert_no_forbidden_actions`** (domains/llm/agentic.py:319)
```python
def assert_no_forbidden_actions(trace: AgentTrace, forbidden_tools: list[str])
```
> Assert that no tool call in the trace used a forbidden tool.

**`assert_no_hallucinated_tools`** (domains/llm/agentic.py:518)
```python
def assert_no_hallucinated_tools(trace: AgentTrace, known_tools: list[str])
```
> Assert every tool call in the trace targets an actually-available tool.

**`assert_no_hallucination`** (domains/llm/safety.py:277)
```python
def assert_no_hallucination(claims: list[str], sources: list[str], method: str='lexical', min_coverage: float=0.3, embedding_model: str='all-MiniLM-L6-v2', nli_model: str='cross-encoder/nli-deberta-v3-base', judge_fn: Callable[[str, str], float] | None=None)
```
> Assert LLM claims are supported by source documents.

**`assert_no_lost_in_middle`** (domains/llm/long_context.py:254)
```python
def assert_no_lost_in_middle(model_fn: Callable[[str], str], facts: list[str], questions: list[str], min_accuracy: float=0.7)
```
> Assert that an LLM does not lose information in the middle of its context.

**`assert_no_redundant_calls`** (domains/llm/agentic.py:418)
```python
def assert_no_redundant_calls(trace: AgentTrace, max_repeat: int=2, ignore_tools: list[str] | None=None)
```
> Assert no tool is called more than *max_repeat* times consecutively.

**`assert_no_system_prompt_leakage`** (domains/llm/safety.py:491)
```python
def assert_no_system_prompt_leakage(model_fn: Callable[[str], str], system_prompt: str, extraction_payloads: list[str] | None=None, min_overlap_threshold: float=0.3, method: str='lexical', embedding_model: str='all-MiniLM-L6-v2')
```
> Assert that the model does not leak its system prompt.

**`assert_no_toxicity`** (domains/llm/safety.py:55)
```python
def assert_no_toxicity(texts: list[str], max_toxic_pct: float=0.01, method: str='regex', patterns: list[str] | None=None, classifier_threshold: float=0.5, classifier_model: str='unitary/toxic-bert', classifier_revision: str='4d6c22e74ba2fdd26bc4f7238f50766b045a0d94')
```
> Assert LLM outputs are not toxic.

**`assert_output_format`** (domains/llm/text_quality.py:82)
```python
def assert_output_format(text: str, pattern: str, description: str | None=None)
```
> Assert text matches a regex pattern.

**`assert_ragas_score`** (domains/llm/ragas.py:97)
```python
def assert_ragas_score(answer: str, question: str, context: str | list[str], relevant_ids: list[str] | None=None, retrieved_ids: list[str] | None=None, min_score: float=0.5)
```
> Assert RAGAS composite score meets threshold.

**`assert_readability`** (domains/llm/text_quality.py:131)
```python
def assert_readability(text: str, max_grade_level: float=12.0)
```
> Assert text readability using Flesch-Kincaid grade level.

**`assert_recall_at_k`** (domains/llm/retrieval.py:253)
```python
def assert_recall_at_k(relevant: list[set], retrieved: list[list], k: int=10, min_recall: float=0.8)
```
> Assert that mean Recall@K meets a minimum threshold.

**`assert_refusal_consistency`** (domains/llm/safety.py:707)
```python
def assert_refusal_consistency(model_fn: Callable[[str], str], unsafe_prompts: list[str], min_refusal_rate: float=0.9, refusal_patterns: list[str] | None=None)
```
> Assert that a model consistently refuses known-unsafe prompts.

**`assert_safety_taxonomy`** (domains/llm/safety.py:802)
```python
def assert_safety_taxonomy(model_fn: Callable[[str], str], prompts_by_category: dict[str, list[str]], min_refusal_per_category: float=0.8, refusal_patterns: list[str] | None=None)
```
> Assert that a model refuses unsafe prompts consistently across safety categories.

**`assert_semantic_similarity`** (domains/llm/similarity.py:70)
```python
def assert_semantic_similarity(references: list[str], hypotheses: list[str], min_score: float=0.7, method: str='token')
```
> Assert semantic similarity between reference and generated texts.

**`assert_span_budget`** (domains/llm/span_eval.py:209)
```python
def assert_span_budget(trace: SpanTrace, max_total_tokens: int | None=None, max_total_cost_usd: float | None=None, max_spans: int | None=None, max_errors: int=0)
```
> Assert trace stays within resource budget.

**`assert_span_latency`** (domains/llm/span_eval.py:117)
```python
def assert_span_latency(trace: SpanTrace, max_latency_ms: float | None=None, by_kind: dict[SpanKind, float] | None=None)
```
> Assert spans meet latency thresholds.

**`assert_span_quality`** (domains/llm/span_eval.py:34)
```python
def assert_span_quality(trace: SpanTrace, max_error_rate: float=0.0, judge_fn: Callable[[Span], float] | None=None, min_score: float=0.5, span_kinds: list[SpanKind] | None=None)
```
> Assert quality across spans in a trace.

**`assert_span_sequence`** (domains/llm/span_eval.py:303)
```python
def assert_span_sequence(trace: SpanTrace, required_kinds: list[SpanKind] | None=None, required_names: list[str] | None=None, forbidden_kinds: list[SpanKind] | None=None, min_spans: int=1)
```
> Assert trace contains expected span structure.

**`assert_step_efficiency`** (domains/llm/agentic.py:371)
```python
def assert_step_efficiency(trace: AgentTrace, max_steps: int)
```
> Assert that the agent completed its task within a step budget.

**`assert_summary_compression`** (domains/llm/summarization.py:88)
```python
def assert_summary_compression(source: str, summary: str, min_ratio: float=0.1, max_ratio: float=0.5)
```
> Assert that a summary has a reasonable compression ratio.

**`assert_summary_coverage`** (domains/llm/summarization.py:19)
```python
def assert_summary_coverage(source: str, summary: str, min_coverage: float=0.3)
```
> Assert that a summary preserves key content from the source.

**`assert_summary_faithfulness`** (domains/llm/summarization.py:165)
```python
def assert_summary_faithfulness(source: str, summary: str, min_faithfulness: float=0.5)
```
> Assert that a summary does not introduce content absent from the source.

**`assert_task_completion`** (domains/llm/agentic.py:32)
```python
def assert_task_completion(expected_output: str, actual_output: str, min_score: float=0.7)
```
> Assert agent completed the task via token overlap between expected and actual.

**`assert_text_length`** (domains/llm/text_quality.py:26)
```python
def assert_text_length(text: str, min_words: int | None=None, max_words: int | None=None)
```
> Assert text word count is within bounds.

**`assert_tool_call_correctness`** (domains/llm/agentic.py:156)
```python
def assert_tool_call_correctness(expected_args: dict, actual_args: dict, tolerance: float=0.01)
```
> Assert tool was called with correct arguments.

**`assert_tool_chain`** (domains/llm/agentic.py:244)
```python
def assert_tool_chain(trace: AgentTrace, expected_tools: list[str], strict_order: bool=False)
```
> Assert that an agent trace contains the expected sequence of tool calls.

**`assert_tool_selection`** (domains/llm/agentic.py:85)
```python
def assert_tool_selection(expected_tools: list[str], actual_tools: list[str])
```
> Assert agent selected the correct tools.

**`assert_ttft`** (domains/llm/latency.py:14)
```python
def assert_ttft(func: Callable[..., Any], *args: Any, max_ms: float=1000.0, iterations: int=5)
```
> Assert Time to First Token is within bounds.

**`assert_turn_relevancy`** (domains/llm/conversation.py:105)
```python
def assert_turn_relevancy(turns: list[dict[str, str]], min_score: float=0.5)
```
> Assert each assistant turn is relevant to the preceding user turn.

**`assert_with_judge`** (domains/llm/judge_defaults.py:156)
```python
def assert_with_judge(assertion_name: str, text_a: str, text_b: str, judge_fn: Callable[[str, str], float] | None=None, fallback_method: str='lexical', min_score: float=0.5)
```
> Generic judge-or-fallback assertion wrapper.

### llm.behavioral

**`assert_directional_expectation`** (domains/llm/behavioral/semantic.py:209)
```python
def assert_directional_expectation(model_fn: Callable[[str], str], input_text: str, perturbation: Callable[[str], str], direction_fn: Callable[[str, str], bool], perturbation_name: str | None=None)
```
> Assert a perturbation changes output in an expected direction.

**`assert_format_invariance`** (domains/llm/behavioral/invariance.py:389)
```python
def assert_format_invariance(model_fn: Callable[[str], Any], input_text: str, transforms: list[Callable[[str], str]] | None=None, equivalence_method: str='token_f1', min_invariance: float=0.9, similarity_threshold: float | None=None, embedding_model: str='all-MiniLM-L6-v2')
```
> Assert that formatting changes do not alter model output.

**`assert_output_stability`** (domains/llm/behavioral/stability.py:75)
```python
def assert_output_stability(model_fn: Callable[[str], Any], inputs: list[str], n_runs: int=5, equivalence_method: str='token_f1', min_stability: float=0.9, similarity_threshold: float | None=None, embedding_model: str='all-MiniLM-L6-v2')
```
> Assert a model produces consistent outputs across repeated runs.

**`assert_paraphrase_invariance`** (domains/llm/behavioral/invariance.py:207)
```python
def assert_paraphrase_invariance(model_fn: Callable[[str], Any], paraphrases: list[str], equivalence_method: str='token_f1', min_invariance: float=0.8, similarity_threshold: float | None=None, embedding_model: str='all-MiniLM-L6-v2', nli_model: str='cross-encoder/nli-deberta-v3-base', judge_fn: Callable[[str, str], float] | None=None)
```
> Assert that paraphrased inputs produce equivalent outputs.

**`assert_retrieval_consistency`** (domains/llm/behavioral/retrieval.py:32)
```python
def assert_retrieval_consistency(retriever_fn: Callable[[str], list[str]], paraphrases: list[str], min_overlap: float=0.7)
```
> Assert that paraphrased queries retrieve consistent docs.

**`assert_semantic_equivalence`** (domains/llm/behavioral/semantic.py:33)
```python
def assert_semantic_equivalence(text_a: str, text_b: str, method: str='nli', min_score: float=0.7, nli_model: str='cross-encoder/nli-deberta-v3-base', embedding_model: str='all-MiniLM-L6-v2')
```
> Assert two texts are semantically equivalent.

### llm.red_team

**`assert_encoding_mutation_resilience`** (domains/llm/red_team/assertions.py:460)
```python
def assert_encoding_mutation_resilience(model_fn: Callable[[str], str], payloads: list[AttackPayload] | None=None, techniques: list[MutationTechnique] | None=None, threshold: float=0.9)
```
> Assert model resists encoded variants of attacks.

**`assert_no_session_jailbreak`** (domains/llm/red_team/assertions.py:281)
```python
def assert_no_session_jailbreak(model_fn: Callable[[str], str], messages: list[str], threshold: float=1.0)
```
> Assert model is not jailbroken by a multi-turn sequence.

**`assert_owasp_llm_coverage`** (domains/llm/red_team/assertions.py:386)
```python
def assert_owasp_llm_coverage(categories: list[AttackCategory], min_categories: int=5)
```
> Assert test suite covers enough OWASP LLM categories.

**`assert_red_team_resilient`** (domains/llm/red_team/assertions.py:124)
```python
def assert_red_team_resilient(model_fn: Callable[[str], str], categories: list[AttackCategory] | None=None, threshold: float=0.8, llm_attacker: Callable[[str], str] | None=None, purpose: str='general-purpose', custom_payloads: list[AttackPayload] | None=None)
```
> Assert model resists adversarial red team attacks.

### model

**`assert_ab_significance`** (model/ab_test.py:20)
```python
def assert_ab_significance(scores_a: list[float] | np.ndarray, scores_b: list[float] | np.ndarray, method: str='bootstrap', alpha: float=0.05, n_bootstrap: int=1000, severity: Severity=Severity.CRITICAL)
```
> Assert model B is significantly better than model A.

**`assert_ate_significant`** (model/causal.py:239)
```python
def assert_ate_significant(treatment: np.ndarray, outcome: np.ndarray, alpha: float=0.05, severity: Severity=Severity.CRITICAL)
```
> Assert that the Average Treatment Effect is statistically significant.

**`assert_attribution_cosine_stability`** (model/attribution.py:120)
```python
def assert_attribution_cosine_stability(attributions_a: np.ndarray, attributions_b: np.ndarray, min_cosine: float=0.9)
```
> Assert attribution vectors are directionally stable via cosine similarity.

**`assert_calibration`** (model/slicing.py:196)
```python
def assert_calibration(y_true: Any, y_prob: Any, max_error: float=0.05, n_bins: int=10, method: str='ece')
```
> Assert prediction probabilities are well-calibrated.

**`assert_conditional_coverage`** (model/conformal.py:325)
```python
def assert_conditional_coverage(y_true: np.ndarray, y_lower: np.ndarray, y_upper: np.ndarray, groups: np.ndarray | list, nominal_coverage: float=0.9, min_group_coverage: float=0.8, min_group_size: int=10, severity: Severity=Severity.CRITICAL)
```
> Assert that prediction intervals achieve adequate coverage per group.

**`assert_conformal_calibration`** (model/conformal.py:215)
```python
def assert_conformal_calibration(y_true: np.ndarray, y_lower: np.ndarray, y_upper: np.ndarray, nominal_coverage: float=0.9, tolerance: float=0.02, severity: Severity=Severity.CRITICAL)
```
> Assert that empirical coverage is close to the promised nominal level.

**`assert_counterfactual_fairness`** (model/counterfactual.py:78)
```python
def assert_counterfactual_fairness(model_fn: Callable[..., Any], X: np.ndarray, sensitive_col: int, perturbation_fn: Callable[..., np.ndarray] | None=None, max_flip_rate: float=0.05, severity: Severity=Severity.CRITICAL)
```
> Assert that a model's predictions do not change when the protected

**`assert_intersectional_fairness`** (model/bias.py:399)
```python
def assert_intersectional_fairness(y_true: Any, y_pred: Any, sensitive_features: dict[str, Any], method: str='demographic_parity', threshold: float | None=None, min_subgroup_size: int=30, severity: Severity=Severity.CRITICAL)
```
> Assert fairness across ALL intersectional subgroups.

**`assert_interval_coverage`** (model/conformal.py:20)
```python
def assert_interval_coverage(y_true: np.ndarray, y_lower: np.ndarray, y_upper: np.ndarray, target_coverage: float=0.9, tolerance: float=0.05, severity: Severity=Severity.CRITICAL)
```
> Assert that prediction intervals achieve target coverage.

**`assert_label_drift`** (model/overfitting.py:66)
```python
def assert_label_drift(train_labels: list | np.ndarray, test_labels: list | np.ndarray, max_drift: float=0.1)
```
> Assert label distribution hasn't shifted between train and test sets.

**`assert_metric`** (model/metrics.py:84)
```python
def assert_metric(y_true: Any, y_pred: Any, metric: str='accuracy', threshold: float=0.8, average: str='weighted', severity: Severity=Severity.CRITICAL)
```
> Assert a model metric meets a minimum threshold.

**`assert_no_bias`** (model/bias.py:79)
```python
def assert_no_bias(y_true: Any, y_pred: Any, sensitive_feature: Any, method: str='demographic_parity', threshold: float | None=None, severity: Severity=Severity.CRITICAL)
```
> Assert no bias across demographic groups.

**`assert_no_confounding`** (model/causal.py:343)
```python
def assert_no_confounding(X: np.ndarray, treatment: np.ndarray, max_correlation: float=0.1, severity: Severity=Severity.CRITICAL)
```
> Assert that no features are correlated with treatment assignment.

**`assert_no_overfitting`** (model/overfitting.py:17)
```python
def assert_no_overfitting(train_score: float, test_score: float, max_gap: float=0.1, metric_name: str='accuracy')
```
> Assert the gap between training and test metrics is bounded.

**`assert_no_regression`** (model/regression.py:101)
```python
def assert_no_regression(y_true: Any, y_pred: Any, baseline: float | dict[str, Any] | str | Path, metric: str='accuracy', tolerance: float=0.02, average: str='weighted')
```
> Assert current model metrics have not regressed from baseline.

**`assert_prediction_set_size`** (model/conformal.py:115)
```python
def assert_prediction_set_size(prediction_sets: list[list[Any]] | list[set[Any]] | np.ndarray, max_avg_size: float, max_empty_frac: float=0.1, severity: Severity=Severity.CRITICAL)
```
> Assert that prediction sets are informatively sized.

**`assert_robust`** (model/adversarial.py:19)
```python
def assert_robust(model_fn: Callable[..., Any], inputs: Any, perturbation: str='gaussian', epsilon: float=0.01, stability: float=0.95)
```
> Assert model predictions are stable under input perturbations.

**`assert_slice_performance`** (model/slicing.py:124)
```python
def assert_slice_performance(y_true: Any, y_pred: Any, slices: dict[str, Any], metric: str='accuracy', min_threshold: float=0.7, average: str='weighted')
```
> Assert model meets minimum performance on EVERY data slice.

**`assert_top_k_stable`** (model/attribution.py:34)
```python
def assert_top_k_stable(attributions_a: np.ndarray, attributions_b: np.ndarray, k: int=5, min_overlap: float=0.8)
```
> Assert the top-K most important features are consistent across runs.

### monitor

**`assert_endpoint_error_rate`** (monitor/aws.py:159)
```python
def assert_endpoint_error_rate(endpoint_name: str, max_rate: float=0.01, region: str | None=None, period: int=300)
```
> Assert CloudWatch error rate (4XX + 5XX) is within threshold.

**`assert_endpoint_healthy`** (monitor/aws.py:39)
```python
def assert_endpoint_healthy(endpoint_name: str, region: str | None=None)
```
> Assert a SageMaker endpoint is InService.

**`assert_endpoint_healthy`** (monitor/azure.py:72)
```python
def assert_endpoint_healthy(endpoint_name: str, resource_group: str | None=None, subscription_id: str | None=None, workspace_name: str | None=None)
```
> Assert an Azure managed online endpoint is healthy.

**`assert_endpoint_healthy`** (monitor/gcp.py:50)
```python
def assert_endpoint_healthy(endpoint_name: str, project: str | None=None, location: str | None=None)
```
> Assert a Vertex AI endpoint is deployed and serving.

**`assert_endpoint_latency`** (monitor/aws.py:79)
```python
def assert_endpoint_latency(endpoint_name: str, max_p99_ms: float=500.0, region: str | None=None, period: int=300)
```
> Assert CloudWatch ModelLatency P99 is within threshold.

**`assert_endpoint_latency`** (monitor/azure.py:143)
```python
def assert_endpoint_latency(endpoint_name: str, max_p99_ms: float=500.0, resource_group: str | None=None, subscription_id: str | None=None, workspace_name: str | None=None, minutes: int=5)
```
> Assert Azure Monitor request latency for a managed endpoint is within threshold.

**`assert_gpu_memory_local`** (monitor/gpu.py:104)
```python
def assert_gpu_memory_local(max_util: float=0.9)
```
> Assert GPU memory usage is below threshold using nvidia-smi.

**`assert_gpu_utilization`** (monitor/prometheus.py:156)
```python
def assert_gpu_utilization(url: str, max_util: float=0.95, gpu_id: str | None=None)
```
> Assert GPU utilization is below threshold via DCGM Prometheus metrics.

**`assert_gpu_utilization_local`** (monitor/gpu.py:37)
```python
def assert_gpu_utilization_local(max_util: float=0.95)
```
> Assert GPU utilization is below threshold using nvidia-smi.

**`assert_no_concept_drift`** (monitor/concept_drift.py:133)
```python
def assert_no_concept_drift(y_true_ref: np.ndarray | list, y_pred_ref: np.ndarray | list, y_true_cur: np.ndarray | list, y_pred_cur: np.ndarray | list, method: str='chi2', alpha: float=0.05)
```
> Assert no concept drift (P(Y|X)) between reference and current windows.

**`assert_no_degradation`** (monitor/drift_monitor.py:18)
```python
def assert_no_degradation(metric_history: list[float], window: int=7, max_decline: float=0.05)
```
> Assert metric has not degraded over a sliding window.

**`assert_no_output_drift`** (monitor/drift_monitor.py:123)
```python
def assert_no_output_drift(ref_outputs: list[float] | np.ndarray, cur_outputs: list[float] | np.ndarray, method: str='ks', threshold: float=0.05)
```
> Assert model output distribution hasn't drifted.

**`assert_no_streaming_drift`** (monitor/streaming_drift.py:380)
```python
def assert_no_streaming_drift(observations: list[float] | np.ndarray, method: str='adwin', **kwargs: Any)
```
> Assert that a stream of observations shows no distributional drift.

**`assert_no_test_anomaly`** (monitor/anomaly.py:185)
```python
def assert_no_test_anomaly(history: list[float], current: float, method: str='zscore', threshold: float=3.0)
```
> Assert that the current test metric is not anomalous compared to history.

**`assert_prediction_latency`** (monitor/gcp.py:112)
```python
def assert_prediction_latency(endpoint_name: str, max_p99_ms: float=500.0, project: str | None=None, location: str | None=None, minutes: int=5)
```
> Assert prediction latency via Cloud Monitoring is within threshold.

**`assert_prometheus_metric`** (monitor/prometheus.py:52)
```python
def assert_prometheus_metric(url: str, query: str, threshold: float, comparison: str='lte')
```
> Assert a Prometheus metric meets threshold via PromQL query.

**`assert_sla`** (monitor/drift_monitor.py:70)
```python
def assert_sla(latency_p99: float | None=None, error_rate: float | None=None, thresholds: dict[str, float] | None=None)
```
> Assert SLA compliance for latency and error rate.

**`assert_triton_healthy`** (monitor/prometheus.py:245)
```python
def assert_triton_healthy(url: str)
```
> Assert NVIDIA Triton Inference Server is ready to serve requests.

### multimodal

**`assert_clip_score`** (domains/multimodal/metrics.py:191)
```python
def assert_clip_score(image: ImageInput | None=None, text: str | None=None, image_embedding: np.ndarray | None=None, text_embedding: np.ndarray | None=None, min_score: float=0.25, model_name: str='ViT-B-32')
```
> Assert that image-text CLIPScore meets a minimum threshold.

**`assert_cross_modal_consistency`** (domains/multimodal/alignment.py:175)
```python
def assert_cross_modal_consistency(predictions_a: np.ndarray | list, predictions_b: np.ndarray | list, min_agreement: float=0.8)
```
> Assert that predictions from two modalities agree on the same content.

**`assert_edit_preservation`** (domains/multimodal/metrics.py:304)
```python
def assert_edit_preservation(original: ImageInput, edited: ImageInput, method: str='ssim', threshold: float=0.8, max_image_size: int=512)
```
> Assert that an edited image preserves enough of the original.

**`assert_image_coherence`** (domains/multimodal/alignment.py:385)
```python
def assert_image_coherence(text: str, image: ImageInput | None, judge_fn: Callable[[str], str], min_score: float=0.7, image_description: str | None=None)
```
> Assert that an image is coherent with surrounding text context.

**`assert_image_helpfulness`** (domains/multimodal/vlm.py:68)
```python
def assert_image_helpfulness(question: str, image: ImageInput | None, answer: str, judge_fn: Callable[[str], str], min_score: float=0.7, image_description: str | None=None)
```
> Assert that an image helps understand or answer a question.

**`assert_image_text_alignment`** (domains/multimodal/alignment.py:89)
```python
def assert_image_text_alignment(image_embeddings: np.ndarray, text_embeddings: np.ndarray, min_cosine: float=0.5)
```
> Assert that image and text embeddings are aligned in shared space.

**`assert_object_hallucination`** (domains/multimodal/hallucination.py:74)
```python
def assert_object_hallucination(vqa_fn: Callable[[str, ImageInput | None, str | None], str], image: ImageInput | None, objects_present: list[str], objects_absent: list[str], threshold: float=0.8, image_description: str | None=None)
```
> Assert that a VLM does not hallucinate objects in an image.

**`assert_ocr_accuracy`** (domains/multimodal/vlm.py:355)
```python
def assert_ocr_accuracy(expected_text: str, actual_text: str, method: str='cer', threshold: float=0.1)
```
> Assert that OCR output is accurate enough vs ground truth.

**`assert_prompt_faithfulness`** (domains/multimodal/alignment.py:275)
```python
def assert_prompt_faithfulness(prompt: str, image: ImageInput | None, judge_fn: Callable[[str], str], min_score: float=0.7, image_description: str | None=None)
```
> Assert that an image faithfully represents a text prompt.

**`assert_vqa_accuracy`** (domains/multimodal/vlm.py:179)
```python
def assert_vqa_accuracy(question: str, image: ImageInput | None, expected_answer: str, actual_answer: str, judge_fn: Callable[[str], str] | None=None, min_score: float=0.7, image_description: str | None=None)
```
> Assert that a VQA answer is correct.

### nlp

**`assert_bleu`** (domains/nlp/generation.py:10)
```python
def assert_bleu(references: list[str], hypotheses: list[str], min_score: float=0.3)
```
> Assert BLEU score meets minimum threshold.

**`assert_ner_f1`** (domains/nlp/ner.py:10)
```python
def assert_ner_f1(y_true_entities: list[list[tuple[str, int, int]]], y_pred_entities: list[list[tuple[str, int, int]]], min_f1: float=0.8)
```
> Assert entity-level F1 score meets threshold.

**`assert_no_prompt_injection`** (domains/nlp/security.py:321)
```python
def assert_no_prompt_injection(model_fn: Callable[..., Any], payloads: list[str] | list[dict[str, str]] | None=None, forbidden_patterns: list[str] | None=None)
```
> Assert model doesn't comply with prompt injection attempts.

**`assert_no_sentiment_drift`** (domains/nlp/sentiment.py:109)
```python
def assert_no_sentiment_drift(ref_texts: list[str], cur_texts: list[str], max_drift: float=0.1)
```
> Assert sentiment distribution hasn't shifted between datasets.

**`assert_rouge`** (domains/nlp/generation.py:62)
```python
def assert_rouge(references: list[str], hypotheses: list[str], variant: str='rougeL', min_score: float=0.3)
```
> Assert ROUGE score meets minimum threshold.

**`assert_sentiment_positive`** (domains/nlp/sentiment.py:50)
```python
def assert_sentiment_positive(texts: list[str], min_ratio: float=0.5)
```
> Assert at least min_ratio of texts have positive sentiment.

**`assert_text_robust`** (domains/nlp/robustness.py:322)
```python
def assert_text_robust(model_fn: Callable[[str], Any], texts: list[str], perturbation: str='keyboard_proximity', n_perturbations: int=5, min_stability: float=0.8, rate: float=0.1, seed: int | None=42)
```
> Assert that an NLP model produces stable predictions under text noise.

### pipeline

**`assert_checksum`** (pipeline/reproducibility.py:107)
```python
def assert_checksum(path: str | Path, expected_hash: str)
```
> Assert file matches expected SHA-256 hash.

**`assert_onnx_valid`** (pipeline/onnx.py:23)
```python
def assert_onnx_valid(model_path: str | Path, test_input: np.ndarray, expected_output: np.ndarray | None=None, tolerance: float=0.01, severity: Severity=Severity.CRITICAL)
```
> Assert ONNX model loads, accepts input, and produces expected output.

**`assert_pipeline`** (pipeline/e2e.py:13)
```python
def assert_pipeline(steps: list[Callable[..., Any]], input_data: Any, expected_output_type: type | None=None)
```
> Assert an end-to-end pipeline runs without errors.

**`assert_reproducible`** (pipeline/reproducibility.py:18)
```python
def assert_reproducible(func: Callable[..., Any], *args: Any, seed: int=42, runs: int=3, tolerance: float=0.001)
```
> Assert function produces identical output across runs with same seed.

### server

**`assert_audit_log_complete`** (server/audit_log.py:275)
```python
def assert_audit_log_complete(audit_entries: list[dict[str, Any]], expected_actions: list[str])
```
> Assert that required actions appear in the audit log.

### speech

**`assert_accent_coverage`** (domains/speech/performance.py:74)
```python
def assert_accent_coverage(wer_by_accent: dict[str, float], max_gap: float=0.05)
```
> Assert WER difference across accents is within bounds.

**`assert_cer`** (domains/speech/recognition.py:56)
```python
def assert_cer(references: list[str], hypotheses: list[str], max_cer: float=0.05)
```
> Assert Character Error Rate is below threshold.

**`assert_rtf`** (domains/speech/performance.py:14)
```python
def assert_rtf(process_fn: Callable[..., Any], audio_durations: list[float], max_rtf: float=1.0)
```
> Assert Real-Time Factor is below threshold.

**`assert_wer`** (domains/speech/recognition.py:10)
```python
def assert_wer(references: list[str], hypotheses: list[str], max_wer: float=0.1)
```
> Assert Word Error Rate is below threshold.

### tabular

**`assert_class_balance`** (domains/tabular/quality.py:13)
```python
def assert_class_balance(df: pd.DataFrame, label_col: str, max_ratio: float=10.0)
```
> Assert class balance in a DataFrame column.

**`assert_feature_drift`** (domains/tabular/features.py:12)
```python
def assert_feature_drift(ref_df: pd.DataFrame, cur_df: pd.DataFrame, method: str='psi', threshold: float=0.1, columns: list[str] | None=None)
```
> Assert no drift across DataFrame columns.

**`assert_feature_importance_stable`** (domains/tabular/features.py:72)
```python
def assert_feature_importance_stable(shap_ref: dict[str, float], shap_cur: dict[str, float], max_rank_change: int=3)
```
> Assert SHAP feature importance rankings are stable.

### testing

**`assert_impact_coverage`** (testing/impact.py:297)
```python
def assert_impact_coverage(changed_files: list[str], executed_tests: list[str], project_root: str='.', test_dir: str='tests')
```
> Assert that all impacted tests were actually executed.

**`assert_matches_golden`** (testing/golden.py:119)
```python
def assert_matches_golden(current: dict | list | np.ndarray, golden_path: str | Path, tolerance: float=0.01)
```
> Assert that *current* data matches the saved golden baseline.

### training

**`assert_augmentation_preserves_signal`** (training/augmentation.py:97)
```python
def assert_augmentation_preserves_signal(original: pd.DataFrame, augmented: pd.DataFrame, label_col: str, max_distribution_shift: float=0.1)
```
> Assert augmented data preserves label distribution.

**`assert_checkpoint_complete`** (training/checkpoint.py:23)
```python
def assert_checkpoint_complete(path: str | Path, required_keys: list[str] | None=None)
```
> Assert checkpoint file exists and contains required keys.

**`assert_effective_batch_size`** (training/distributed.py:18)
```python
def assert_effective_batch_size(local_batch_size: int, world_size: int, expected_batch_size: int)
```
> Assert effective batch size equals local_batch_size * world_size.

**`assert_gradient_alignment`** (training/distributed.py:273)
```python
def assert_gradient_alignment(grads_a: list[np.ndarray], grads_b: list[np.ndarray], min_cosine: float=0.9)
```
> Assert gradient vectors from two ranks are directionally aligned.

**`assert_gradient_clipped`** (training/distributed.py:386)
```python
def assert_gradient_clipped(gradients: list[np.ndarray], max_norm: float)
```
> Assert the global gradient norm is within the clipping threshold.

**`assert_gradient_flow`** (training/gradient.py:18)
```python
def assert_gradient_flow(gradients: list[np.ndarray], min_mean_grad: float=1e-07)
```
> Assert that gradients are flowing through all layers (no dead layers).

**`assert_gradient_sync`** (training/distributed.py:66)
```python
def assert_gradient_sync(grads_rank0: list[np.ndarray], grads_rank1: list[np.ndarray], tolerance: float=1e-05)
```
> Assert gradients are synchronized across ranks after all-reduce.

**`assert_loss_decreasing`** (training/numerical.py:74)
```python
def assert_loss_decreasing(losses: np.ndarray, window: int=10, min_decrease: float=0.0)
```
> Assert that loss is generally decreasing over training.

**`assert_loss_finite`** (training/gradient.py:178)
```python
def assert_loss_finite(losses: np.ndarray)
```
> Assert that all loss values are finite (no NaN or Inf).

**`assert_loss_is_detached`** (training/memory.py:86)
```python
def assert_loss_is_detached(memory_per_step_mb: list[float], max_growth_per_step_mb: float=1.0)
```
> Assert loss tensor isn't accumulating the computation graph.

**`assert_n_rank_gradient_sync`** (training/distributed.py:160)
```python
def assert_n_rank_gradient_sync(grads_by_rank: list[list[np.ndarray]], tolerance: float=1e-05)
```
> Assert gradients are synchronized across N ranks after all-reduce.

**`assert_no_augmentation_on_test`** (training/augmentation.py:19)
```python
def assert_no_augmentation_on_test(train_df: pd.DataFrame, test_df: pd.DataFrame, key_cols: list[str] | None=None, max_dup_ratio: float=0.01)
```
> Assert test set was NOT augmented (no suspicious duplicate patterns).

**`assert_no_exploding_gradient`** (training/gradient.py:125)
```python
def assert_no_exploding_gradient(gradients: list[np.ndarray], max_grad_norm: float=1000.0)
```
> Assert that no layer has exploding gradients (L2 norm too large).

**`assert_no_loss_divergence`** (training/numerical.py:138)
```python
def assert_no_loss_divergence(losses: np.ndarray, max_increase_ratio: float=10.0)
```
> Assert that loss has not diverged (no catastrophic spike).

**`assert_no_memory_leak`** (training/memory.py:19)
```python
def assert_no_memory_leak(memory_readings_mb: list[float], max_growth_mb: float=100.0, window: int=10)
```
> Assert memory doesn't grow unbounded during training.

**`assert_no_nan_inf`** (training/numerical.py:17)
```python
def assert_no_nan_inf(arrays: list[np.ndarray], names: list[str] | None=None)
```
> Assert that none of the provided arrays contain NaN or Inf values.

**`assert_no_target_leakage`** (training/leakage.py:94)
```python
def assert_no_target_leakage(df: pd.DataFrame, target_col: str, feature_cols: list[str] | None=None, corr_threshold: float=0.95)
```
> Assert no feature is suspiciously correlated with the target.

**`assert_no_train_test_overlap`** (training/leakage.py:18)
```python
def assert_no_train_test_overlap(train_df: pd.DataFrame, test_df: pd.DataFrame, key_cols: list[str])
```
> Assert zero row overlap between train and test sets on key columns.

**`assert_no_training_serving_skew`** (training/skew.py:19)
```python
def assert_no_training_serving_skew(train_output: np.ndarray | list, serve_output: np.ndarray | list, tolerance: float=0.01)
```
> Assert training and serving pipelines produce the same output for the same input.

**`assert_no_vanishing_gradient`** (training/gradient.py:72)
```python
def assert_no_vanishing_gradient(gradients: list[np.ndarray], min_grad_norm: float=1e-08)
```
> Assert that no layer has vanishing gradients (L2 norm too small).

**`assert_resume_loss_continuous`** (training/checkpoint.py:117)
```python
def assert_resume_loss_continuous(pre_losses: list[float], post_losses: list[float], max_gap: float=0.5)
```
> Assert loss continuity after checkpoint resume.

**`assert_softmax_valid`** (training/numerical.py:209)
```python
def assert_softmax_valid(probabilities: np.ndarray)
```
> Assert that softmax outputs are valid probability distributions.

**`assert_temporal_split`** (training/leakage.py:58)
```python
def assert_temporal_split(train_df: pd.DataFrame, test_df: pd.DataFrame, time_col: str)
```
> Assert train data is strictly before test data (no temporal leakage).

**`assert_weight_divergence`** (training/distributed.py:331)
```python
def assert_weight_divergence(weights_a: list[np.ndarray], weights_b: list[np.ndarray], max_l2_distance: float=0.01)
```
> Assert weight vectors from two checkpoints or ranks are close in L2 space.


---

## MCP Tools (11)

### `mltk_scan` (server.py:173)

> Scan an ML project for quality issues, drift, bias, and security vulnerabilities.

| Param | Type | Default |
|-------|------|---------|
| path | `str` | *required* |
| scanners | `str` | `'all'` |

### `mltk_test` (server.py:237)

> Run an mltk test suite and return pass/fail results.

| Param | Type | Default |
|-------|------|---------|
| suite_path | `str` | *required* |
| verbose | `bool` | `False` |

### `mltk_list` (server.py:328)

> List available mltk assertions for ML testing.

| Param | Type | Default |
|-------|------|---------|
| filter_text | `str` | `''` |
| domain | `str` | `''` |

### `mltk_eval` (server.py:373)

> Run an evaluation pipeline on a dataset with configurable solvers and scorers.

| Param | Type | Default |
|-------|------|---------|
| dataset_path | `str` | *required* |
| scorer | `str` | `'exact_match'` |
| solver | `str` | `'generate'` |

### `mltk_dataset` (server.py:452)

> Get info about a registered evaluation dataset with quality metrics.

| Param | Type | Default |
|-------|------|---------|
| name | `str` | *required* |
| version | `str` | `''` |

### `mltk_report` (server.py:505)

> Generate a formatted ML test report from scan or test results.

| Param | Type | Default |
|-------|------|---------|
| title | `str` | *required* |
| description | `str` | `''` |
| results_json | `str` | `''` |

### `mltk_suggest` (server.py:578)

> Get fix suggestions for a scan finding.

| Param | Type | Default |
|-------|------|---------|
| finding_json | `str` | *required* |
| category | `str` | `''` |
| max_results | `int` | `5` |

### `mltk_experiment` (server.py:661)

> Rank fix suggestions for a finding using heuristic scoring.

| Param | Type | Default |
|-------|------|---------|
| finding_json | `str` | *required* |
| rank_by | `str` | `'passed'` |
| max_results | `int` | `5` |
| sandbox | `bool` | `False` |

### `mltk_workflow` (server.py:810)

> Return the canonical mltk agent workflow.

*No parameters.*

### `mltk_create_pr` (server.py:860)

> Create a GitHub PR with a fix for a scan finding.

| Param | Type | Default |
|-------|------|---------|
| finding_json | `str` | *required* |
| fix_json | `str` | *required* |
| repo | `str` | *required* |
| base_branch | `str` | `'main'` |
| draft | `bool` | `True` |

### `mltk_create_issue` (server.py:903)

> Create an issue ticket from a scan finding.

| Param | Type | Default |
|-------|------|---------|
| finding_json | `str` | *required* |
| tracker | `str` | `'github'` |
| project | `str` | `''` |
| config_json | `str` | `'{}'` |
| pr_url | `str` | `''` |


---

## CLI Commands (28)

| # | Command | Line | Description |
|---|---------|------|-------------|
| 1 | `mltk version` | 35 | Show mltk version. |
| 2 | `mltk init` | 42 | Scaffold mltk.yaml + example test file. |
| 3 | `mltk scan` | 80 | Quick data quality scan on a CSV/Parquet file. |
| 4 | `mltk drift` | 123 | Compare two datasets for distribution drift. |
| 5 | `mltk score` | 187 | Show the ML Test Score rubric and how to generate scores. |
| 6 | `mltk doctor` | 204 | Diagnose ML testing environment. |
| 7 | `mltk test` | 245 | Run YAML-defined test suite. |
| 8 | `mltk model-card` | 283 | Generate a Google Model Card from test results JSON. |
| 9 | `mltk compliance` | 317 | Generate EU AI Act compliance report. |
| 10 | `mltk contract init` | 352 | Scaffold an example data contract YAML file. |
| 11 | `mltk contract validate` | 378 | Validate a data file against a contract. |
| 12 | `mltk contract generate-tests` | 401 | Generate pytest test file from a data contract. |
| 13 | `mltk docs serve` | 422 | Serve documentation locally with hot reload. |
| 14 | `mltk docs build` | 451 | Build static HTML documentation. |
| 15 | `mltk docs open` | 480 | Build docs, start a local server, and open in browser. |
| 16 | `mltk registry push` | 539 | Push a directory of test files to the registry as a named collection. |
| 17 | `mltk registry pull` | 560 | Pull a named collection from the registry into a local directory. |
| 18 | `mltk registry list` | 575 | List all collections in the registry. |
| 19 | `mltk notify slack` | 597 | Send test results (or a custom message) to Slack. |
| 20 | `mltk chat` | 667 | Interactive Q&A about test results. |
| 21 | `mltk server` | 679 | Start the mltk server platform. |
| 22 | `mltk server-create-key` | 700 | Generate an API key for the mltk server. |
| 23 | `mltk fda-audit` | 727 | Generate FDA 21 CFR Part 11 audit trail. |
| 24 | `mltk compliance-pdf` | 750 | Convert HTML compliance report to print-ready PDF. |
| 25 | `mltk compliance-gap` | 766 | Run compliance gap analysis across frameworks. |
| 26 | `mltk grafana-export` | 975 | Export a Grafana dashboard JSON for mltk metrics. |
| 27 | `mltk scan-model` | 1000 | Scan a model for issues and generate tests. |
| 28 | `mltk list` | 1182 | List all available mltk assertions. |

---

## Scanners (8)

| Scanner | File | Line |
|---------|------|------|
| BiasScanner | scan/scanners/bias.py | 65 |
| CalibrationScanner | scan/scanners/calibration.py | 32 |
| DataScanner | scan/scanners/data.py | 35 |
| DriftScanner | scan/scanners/drift.py | 31 |
| LeakageScanner | scan/scanners/leakage.py | 39 |
| OverfitScanner | scan/scanners/overfit.py | 33 |
| RobustnessScanner | scan/scanners/robustness.py | 34 |
| SliceScanner | scan/scanners/slice.py | 40 |

---

## Key Classes (28)

### `ColumnSpec` (contracts/schema.py:11)
> Specification for a single column in a data contract.

| Field | Type | Default |
|-------|------|---------|
| name | `str` |  |
| type | `str` | `'object'` |
| nullable | `bool` | `True` |
| unique | `bool` | `False` |
| range | `tuple[float, float] | None` | `None` |
| pii_class | `str | None` | `None` |

### `QualitySpec` (contracts/schema.py:23)
> Quality requirements for the dataset.

| Field | Type | Default |
|-------|------|---------|
| min_rows | `int | None` | `None` |
| max_rows | `int | None` | `None` |
| max_nulls_pct | `float | None` | `None` |
| freshness_days | `int | None` | `None` |
| freshness_column | `str | None` | `None` |

### `Contract` (contracts/schema.py:34)
> Parsed data contract specification.

| Field | Type | Default |
|-------|------|---------|
| name | `str` | `'unnamed'` |
| version | `str` | `'1.0'` |
| columns | `list[ColumnSpec]` | `field(default_factory=list)` |
| quality | `QualitySpec` | `field(default_factory=QualitySpec)` |

### `TestResult` (core/result.py:20)
> Result of a single mltk assertion.

| Field | Type | Default |
|-------|------|---------|
| name | `str` |  |
| passed | `bool` |  |
| severity | `Severity` |  |
| message | `str` |  |
| details | `dict[str, Any]` | `field(default_factory=dict)` |
| duration_ms | `float` | `0.0` |
| timestamp | `datetime` | `field(default_factory=datetime.now)` |

### `TestSuite` (core/result.py:104)
> Collection of test results from a run.

| Field | Type | Default |
|-------|------|---------|
| results | `list[TestResult]` | `field(default_factory=list)` |

### `SuiteResult` (core/suite.py:39)
> Aggregated outcome of running an :class:`MltkSuite`.

| Field | Type | Default |
|-------|------|---------|
| name | `str` |  |
| results | `list[TestResult]` | `field(default_factory=list)` |
| total | `int` | `0` |
| passed_count | `int` | `0` |
| failed_count | `int` | `0` |
| duration_ms | `float` | `0.0` |

### `MltkSuite` (core/suite.py:99)
> Composable test suite for running mltk assertions without pytest.

*(no dataclass fields)*

### `ParaphraseGenerator` (domains/llm/behavioral/paraphrase.py:99)
> Generate paraphrases using deterministic templates.

*(no dataclass fields)*

### `McpToolCall` (domains/llm/mcp.py:58)
> MCP-specific tool call with server namespace and schema.

| Field | Type | Default |
|-------|------|---------|
| server | `str` | `''` |
| schema | `dict[str, Any]` | `field(default_factory=dict)` |
| context_tokens | `int` | `0` |

### `McpResourceAccess` (domains/llm/mcp.py:77)
> A resource read within an MCP session.

| Field | Type | Default |
|-------|------|---------|
| uri | `str` | `''` |
| server | `str` | `''` |
| content_tokens | `int` | `0` |
| result | `str | None` | `None` |
| error | `str | None` | `None` |
| duration_ms | `float` | `0.0` |

### `McpTrace` (domains/llm/mcp.py:103)
> MCP-aware agent trace extending :class:`AgentTrace`.

| Field | Type | Default |
|-------|------|---------|
| resource_accesses | `list[McpResourceAccess]` | `field(default_factory=list)` |
| model_context_limit | `int` | `0` |

### `ToolCall` (domains/llm/trace.py:11)
> A single tool/function call within an agent trace.

| Field | Type | Default |
|-------|------|---------|
| name | `str` |  |
| arguments | `dict[str, Any]` | `field(default_factory=dict)` |
| result | `str | None` | `None` |
| error | `str | None` | `None` |
| duration_ms | `float` | `0.0` |

### `AgentTrace` (domains/llm/trace.py:69)
> A complete execution trace of an AI agent.

| Field | Type | Default |
|-------|------|---------|
| tool_calls | `list[ToolCall]` | `field(default_factory=list)` |
| total_tokens | `int` | `0` |
| total_duration_ms | `float` | `0.0` |
| metadata | `dict[str, Any]` | `field(default_factory=dict)` |

### `DatasetCard` (eval/dataset.py:125)
> Metadata card for an evaluation dataset.

| Field | Type | Default |
|-------|------|---------|
| description | `str` | `''` |
| task | `str` | `''` |
| source | `str` | `''` |
| license | `str` | `''` |
| tags | `list[str]` | `field(default_factory=list)` |
| created | `str` | `''` |
| author | `str` | `''` |

### `EvalDataset` (eval/dataset.py:219)
> Versioned evaluation dataset with metadata card.

| Field | Type | Default |
|-------|------|---------|
| name | `str` |  |
| version | `str` |  |
| samples | `list[EvalSample]` |  |
| card | `DatasetCard` | `field(default_factory=DatasetCard)` |
| fingerprint | `str` | `''` |

### `DatasetDiff` (eval/dataset.py:613)
> Comparison between two dataset versions.

| Field | Type | Default |
|-------|------|---------|
| old_version | `str` |  |
| new_version | `str` |  |
| added_samples | `list[EvalSample]` |  |
| removed_samples | `list[EvalSample]` |  |
| unchanged_samples | `list[EvalSample]` |  |
| schema_changes | `list[str]` |  |
| suggested_bump | `str` |  |

### `DatasetInfo` (eval/dataset.py:653)
> Summary info for a registered dataset (for listing).

| Field | Type | Default |
|-------|------|---------|
| name | `str` |  |
| versions | `list[str]` |  |
| latest_version | `str` |  |
| sample_count | `int` |  |
| card | `DatasetCard` |  |

### `EvalTask` (eval/task.py:92)
> Composable evaluation task.

*(no dataclass fields)*

### `Hypothesis` (experiment/hypothesis.py:21)
> A single fix hypothesis to test against a finding.

| Field | Type | Default |
|-------|------|---------|
| fix | `FixSuggestion` |  |
| apply_fn | `Callable` |  |
| description | `str` | `''` |

### `HypothesisResult` (experiment/hypothesis.py:40)
> Result of testing one hypothesis.

| Field | Type | Default |
|-------|------|---------|
| hypothesis | `Hypothesis` |  |
| baseline_result | `TestResult` |  |
| fixed_result | `TestResult` |  |
| improvement | `float` | `0.0` |
| rank | `int` | `0` |

### `ExperimentResult` (experiment/result.py:22)
> Aggregated result from testing all hypotheses for one finding.

| Field | Type | Default |
|-------|------|---------|
| finding | `ScanFinding` |  |
| baseline_result | `TestResult` |  |
| hypothesis_results | `list[HypothesisResult]` | `field(default_factory=list)` |
| selected_fix | `FixSuggestion | None` | `None` |
| duration_ms | `float` | `0.0` |

### `GitWorktree` (experiment/worktree.py:137)
> Context manager for temporary git worktrees.

*(no dataclass fields)*

### `ScanConfig` (scan/config.py:49)
> User-facing configuration for ``mltk scan``.

| Field | Type | Default |
|-------|------|---------|
| max_scan_rows | `int` | `10000` |
| sample_strategy | `str` | `'stratified'` |
| time_budget_seconds | `float` | `60.0` |
| per_scanner_timeout | `float` | `30.0` |
| seed | `int` | `42` |
| categorical_threshold | `int` | `20` |
| max_slices_per_column | `int` | `50` |
| min_slice_samples | `int` | `30` |
| critical_drop | `float` | `0.2` |
| warning_drop | `float` | `0.1` |
| scanner_config | `dict[str, dict[str, Any]]` | `field(default_factory=dict)` |
| enabled_scanners | `list[str] | None` | `None` |
| disabled_scanners | `list[str] | None` | `None` |

### `ScanContext` (scan/config.py:109)
> Everything a scanner might need, pre-computed by the engine.

| Field | Type | Default |
|-------|------|---------|
| model_fn | `Callable[..., Any] | None` |  |
| predict_proba_fn | `Callable[..., Any] | None` |  |
| X | `pd.DataFrame` |  |
| y | `np.ndarray | None` |  |
| y_train | `np.ndarray | None` |  |
| X_train | `pd.DataFrame | None` |  |
| sensitive_columns | `list[str]` |  |
| numeric_columns | `list[str]` |  |
| categorical_columns | `list[str]` |  |
| model_type | `str` |  |
| config | `ScanConfig` |  |
| seed | `int` |  |

### `ScanReport` (scan/engine.py:71)
> Aggregated output of a scan run.

| Field | Type | Default |
|-------|------|---------|
| findings | `list[ScanFinding]` | `field(default_factory=list)` |
| scanners_run | `list[str]` | `field(default_factory=list)` |
| scanners_skipped | `list[str]` | `field(default_factory=list)` |
| scanners_errored | `dict[str, str]` | `field(default_factory=dict)` |
| duration_ms | `float` | `0.0` |
| model_type | `str` | `'unknown'` |
| n_samples | `int` | `0` |
| n_features | `int` | `0` |
| config | `ScanConfig` | `field(default_factory=ScanConfig)` |

### `ScanEngine` (scan/engine.py:455)
> Orchestrates scanners and produces a ScanReport.

*(no dataclass fields)*

### `FixSuggestion` (scan/finding.py:44)
> A concrete remediation step for a scan finding.

| Field | Type | Default |
|-------|------|---------|
| category | `str` |  |
| title | `str` |  |
| description | `str` |  |
| confidence | `str` |  |
| code_snippet | `str` | `''` |

### `ScanFinding` (scan/finding.py:87)
> A single issue discovered by a scanner.

| Field | Type | Default |
|-------|------|---------|
| result | `TestResult` |  |
| assertion_fn | `Callable[..., TestResult]` |  |
| assertion_args | `tuple[Any, ...]` | `()` |
| assertion_kwargs | `dict[str, Any]` | `field(default_factory=dict)` |
| suggested_test | `str` | `''` |
| scanner_name | `str` | `''` |
| suggested_fixes | `list[FixSuggestion]` | `field(default_factory=list)` |

---

## Test Layout

| Test Directory | Source Module |
|----------------|---------------|
| test_chat/ | src/mltk/chat/ |
| test_cli/ | src/mltk/cli/ |
| test_compliance/ | src/mltk/compliance/ |
| test_contracts/ | src/mltk/contracts/ |
| test_core/ | src/mltk/core/ |
| test_data/ | src/mltk/data/ |
| test_domains/ | src/mltk/domains/ |
| test_eval/ | src/mltk/eval/ |
| test_experiment/ | src/mltk/experiment/ |
| test_inference/ | src/mltk/inference/ |
| test_integrations/ | src/mltk/integrations/ |
| test_mcp/ | src/mltk/mcp/ |
| test_model/ | src/mltk/model/ |
| test_monitor/ | src/mltk/monitor/ |
| test_pipeline/ | src/mltk/pipeline/ |
| test_property/ | src/mltk/property/ |
| test_pytest_plugin/ | src/mltk/pytest_plugin/ |
| test_registry/ | src/mltk/registry/ |
| test_report/ | src/mltk/report/ |
| test_rust/ | src/mltk/rust/ |
| test_scan/ | src/mltk/scan/ |
| test_server/ | src/mltk/server/ |
| test_testdefs/ | src/mltk/testdefs/ |
| test_testing/ | src/mltk/testing/ |
| test_training/ | src/mltk/training/ |
