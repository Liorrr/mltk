# llm

## Overview
LLM/GenAI evaluation assertions: RAG, retrieval ranking, agentic/MCP traces, judge-based scoring, safety, text quality, long context, structured output, and span traces. Assertions return `TestResult` via `mltk.core.assertion`.

## Interface
- RAG/retrieval: `rag.py` (`assert_faithfulness`, `assert_context_relevancy`, `assert_answer_relevancy`, `assert_context_precision`, `assert_context_recall`), `ragas.py` (`compute_ragas_score`, `assert_ragas_score`), `retrieval.py` (`assert_ndcg`, `assert_mrr`, `assert_recall_at_k`, `assert_map_at_k`)
- Agentic: `agentic.py` (`assert_task_completion`, `assert_tool_selection`, `assert_tool_call_correctness`, `assert_tool_chain`, `assert_no_forbidden_actions`, `assert_step_efficiency`, `assert_no_redundant_calls`, `assert_no_hallucinated_tools`, `assert_cost_budget`, `assert_error_recovery`), `multi_agent.py` (`assert_no_agent_loop`, `assert_agent_handoff`), `mcp.py` (`McpToolCall`, `McpResourceAccess`, `McpTrace`, `assert_mcp_tool_schema_conformance`, `assert_mcp_tool_selection`, `assert_mcp_resource_access`, `assert_mcp_context_window`, `assert_mcp_error_recovery`)
- Traces: `trace.py` (`ToolCall`, `AgentTrace`), `span.py` (`SpanKind`, `Span`, `SpanTrace`), `span_eval.py` (`assert_span_quality`, `assert_span_latency`, `assert_span_budget`, `assert_span_sequence`), `from_capture.py` (`to_agent_trace`, `to_span_trace`)
- Judge: `judge.py` (`format_judge_prompt`, `assert_llm_judge_score`, `assert_llm_judge_pairwise`), `judge_defaults.py` (`configure_default_judge`, `get_default_judge`, `resolve_judge`, `assert_with_judge`)
- Safety: `safety.py` (`assert_no_toxicity`, `assert_no_hallucination`, `assert_no_system_prompt_leakage`, `assert_refusal_consistency`, `assert_safety_taxonomy`), `unicode_attacks.py` (`detect_unicode_attacks`, `assert_no_unicode_attacks`)
- Quality: `similarity.py` (`assert_semantic_similarity`), `bertscore.py` (`assert_bertscore`), `coherence.py` (`assert_coherence`), `text_quality.py` (`assert_text_length`, `assert_output_format`, `assert_readability`), `summarization.py` (`assert_summary_coverage`, `assert_summary_compression`, `assert_summary_faithfulness`), `conversation.py` (`assert_knowledge_retention`, `assert_turn_relevancy`, `assert_conversation_completeness`), `long_context.py` (`assert_needle_in_haystack`, `assert_context_utilization`, `assert_no_lost_in_middle`), `latency.py` (`assert_ttft`, `assert_itl`), `structured_output.py` (`assert_valid_json`, `assert_json_schema`, `assert_pydantic_schema`)
- Private helpers: `_backends.py` (embedding/NLI helpers), `_utils.py`
- Subpackages: [behavioral](./behavioral/README.md), [red_team](./red_team/README.md), [synthetic](./synthetic/README.md)
- `__init__.py` re-exports the public assertions via `__all__`, including child-package symbols importable from `mltk.domains.llm`: `assert_red_team_resilient`, `assert_no_session_jailbreak`, `assert_owasp_llm_coverage`, `assert_encoding_mutation_resilience`, `AttackCategory`, `AttackPayload` (red_team) and `QAPair`, `QuestionType`, `SyntheticQAGenerator` (synthetic).

## Invariants
- Public assertions are `assert_*` and return `TestResult`; a failed check at `Severity.CRITICAL` (the default) raises `MltkAssertionError` via `mltk.core.assert_true`.
- Judge-based assertions use a caller-supplied or configured default judge function.
- Underscore-prefixed modules are private.

## Side Effects
- Calls user-supplied model/judge callables; `configure_default_judge` sets module-level state.
- Embedding/NLI backends may load optional ML models.

## Verification
- Run: `python -m pytest tests/test_domains -q`
