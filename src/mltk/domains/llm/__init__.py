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
from mltk.domains.llm.latency import assert_itl, assert_ttft
from mltk.domains.llm.multi_agent import assert_agent_handoff, assert_no_agent_loop
from mltk.domains.llm.rag import (
    assert_answer_relevancy,
    assert_context_precision,
    assert_context_recall,
    assert_context_relevancy,
    assert_faithfulness,
)
from mltk.domains.llm.ragas import assert_ragas_score, compute_ragas_score
from mltk.domains.llm.safety import (
    assert_no_hallucination,
    assert_no_system_prompt_leakage,
    assert_no_toxicity,
    assert_refusal_consistency,
    assert_safety_taxonomy,
)
from mltk.domains.llm.similarity import assert_semantic_similarity
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
    # multi-turn conversation evaluation
    "assert_knowledge_retention",
    "assert_turn_relevancy",
    "assert_conversation_completeness",
]
