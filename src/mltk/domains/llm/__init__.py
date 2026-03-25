"""LLM/GenAI evaluation — RAG, agentic, text quality, similarity, toxicity, TTFT/ITL."""

from mltk.domains.llm.agentic import (
    assert_task_completion,
    assert_tool_call_correctness,
    assert_tool_selection,
)
from mltk.domains.llm.latency import assert_itl, assert_ttft
from mltk.domains.llm.rag import (
    assert_answer_relevancy,
    assert_context_precision,
    assert_context_recall,
    assert_context_relevancy,
    assert_faithfulness,
)
from mltk.domains.llm.safety import assert_no_hallucination, assert_no_toxicity
from mltk.domains.llm.similarity import assert_semantic_similarity
from mltk.domains.llm.text_quality import (
    assert_output_format,
    assert_readability,
    assert_text_length,
)

__all__ = [
    # similarity
    "assert_semantic_similarity",
    # safety
    "assert_no_toxicity",
    "assert_no_hallucination",
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
    # agentic evaluation
    "assert_task_completion",
    "assert_tool_selection",
    "assert_tool_call_correctness",
]
