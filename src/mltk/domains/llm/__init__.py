"""LLM/GenAI evaluation — RAG, agentic, text quality, similarity, toxicity, TTFT/ITL."""

from mltk.domains.llm.agentic import (
    assert_cost_budget,
    assert_error_recovery,
    assert_no_forbidden_actions,
    assert_no_hallucinated_tools,
    assert_no_redundant_calls,
    assert_step_efficiency,
    assert_task_completion,
    assert_tool_call_correctness,
    assert_tool_chain,
    assert_tool_selection,
)
from mltk.domains.llm.bertscore import assert_bertscore
from mltk.domains.llm.coherence import assert_coherence
from mltk.domains.llm.conversation import (
    assert_conversation_completeness,
    assert_knowledge_retention,
    assert_turn_relevancy,
)
from mltk.domains.llm.judge import assert_llm_judge_pairwise, assert_llm_judge_score
from mltk.domains.llm.latency import assert_itl, assert_ttft
from mltk.domains.llm.long_context import (
    assert_context_utilization,
    assert_needle_in_haystack,
    assert_no_lost_in_middle,
)
from mltk.domains.llm.mcp import (
    McpResourceAccess,
    McpToolCall,
    McpTrace,
    assert_mcp_context_window,
    assert_mcp_error_recovery,
    assert_mcp_resource_access,
    assert_mcp_tool_schema_conformance,
    assert_mcp_tool_selection,
)
from mltk.domains.llm.multi_agent import assert_agent_handoff, assert_no_agent_loop
from mltk.domains.llm.rag import (
    assert_answer_relevancy,
    assert_context_precision,
    assert_context_recall,
    assert_context_relevancy,
    assert_faithfulness,
)
from mltk.domains.llm.ragas import assert_ragas_score, compute_ragas_score
from mltk.domains.llm.red_team import (
    AttackCategory,
    AttackPayload,
    assert_encoding_mutation_resilience,
    assert_no_session_jailbreak,
    assert_owasp_llm_coverage,
    assert_red_team_resilient,
)
from mltk.domains.llm.retrieval import (
    assert_map_at_k,
    assert_mrr,
    assert_ndcg,
    assert_recall_at_k,
)
from mltk.domains.llm.safety import (
    assert_no_hallucination,
    assert_no_system_prompt_leakage,
    assert_no_toxicity,
    assert_refusal_consistency,
    assert_safety_taxonomy,
)
from mltk.domains.llm.similarity import assert_semantic_similarity
from mltk.domains.llm.span import Span, SpanKind, SpanTrace
from mltk.domains.llm.span_eval import (
    assert_span_budget,
    assert_span_latency,
    assert_span_quality,
    assert_span_sequence,
)
from mltk.domains.llm.summarization import (
    assert_summary_compression,
    assert_summary_coverage,
    assert_summary_faithfulness,
)
from mltk.domains.llm.synthetic import QAPair, QuestionType, SyntheticQAGenerator
from mltk.domains.llm.text_quality import (
    assert_output_format,
    assert_readability,
    assert_text_length,
)

__all__ = [
    # bertscore
    "assert_bertscore",
    # similarity
    "assert_semantic_similarity",
    # safety
    "assert_no_toxicity",
    "assert_no_hallucination",
    "assert_no_system_prompt_leakage",
    "assert_refusal_consistency",
    "assert_safety_taxonomy",
    # latency
    "assert_ttft",
    "assert_itl",
    # text quality
    "assert_text_length",
    "assert_output_format",
    "assert_readability",
    # RAG evaluation
    "assert_faithfulness",
    "assert_context_relevancy",
    "assert_answer_relevancy",
    "assert_context_precision",
    "assert_context_recall",
    # RAGAS composite
    "compute_ragas_score",
    "assert_ragas_score",
    # coherence
    "assert_coherence",
    # agentic evaluation
    "assert_task_completion",
    "assert_tool_selection",
    "assert_tool_call_correctness",
    # trace-based agentic evaluation
    "assert_tool_chain",
    "assert_no_forbidden_actions",
    "assert_step_efficiency",
    # extended agentic evaluation
    "assert_no_redundant_calls",
    "assert_no_hallucinated_tools",
    "assert_cost_budget",
    "assert_error_recovery",
    # multi-agent coordination
    "assert_no_agent_loop",
    "assert_agent_handoff",
    # LLM-as-Judge
    "assert_llm_judge_score",
    "assert_llm_judge_pairwise",
    # summarization
    "assert_summary_coverage",
    "assert_summary_compression",
    "assert_summary_faithfulness",
    # long-context LLM
    "assert_needle_in_haystack",
    "assert_context_utilization",
    "assert_no_lost_in_middle",
    # retrieval ranking metrics
    "assert_ndcg",
    "assert_mrr",
    "assert_recall_at_k",
    "assert_map_at_k",
    # multi-turn conversation evaluation
    "assert_knowledge_retention",
    "assert_turn_relevancy",
    "assert_conversation_completeness",
    # MCP evaluation
    "McpToolCall",
    "McpResourceAccess",
    "McpTrace",
    "assert_mcp_tool_schema_conformance",
    "assert_mcp_tool_selection",
    "assert_mcp_resource_access",
    "assert_mcp_context_window",
    "assert_mcp_error_recovery",
    # Synthetic QA generation
    "QAPair",
    "QuestionType",
    "SyntheticQAGenerator",
    # Red team evaluation
    "AttackCategory",
    "AttackPayload",
    "assert_red_team_resilient",
    "assert_no_session_jailbreak",
    "assert_owasp_llm_coverage",
    "assert_encoding_mutation_resilience",
]
