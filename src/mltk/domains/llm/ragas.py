"""RAGAS composite score — single metric averaging RAG evaluation components."""

from __future__ import annotations

import re

from mltk.core.assertion import assert_true, timed_assertion
from mltk.core.result import Severity, TestResult


def _tokenize(text: str) -> set[str]:
    """Lowercase, strip punctuation, split on whitespace."""
    return set(re.sub(r"[^\w\s]", "", text.lower()).split())


def _flatten_context(context: str | list[str]) -> str:
    """Join list context into a single string for token analysis."""
    if isinstance(context, list):
        return " ".join(context)
    return context


def compute_ragas_score(
    answer: str,
    question: str,
    context: str | list[str],
    relevant_ids: list[str] | None = None,
    retrieved_ids: list[str] | None = None,
) -> dict:
    """Compute RAGAS composite score from individual RAG metrics.

    Returns dict with: faithfulness, answer_relevancy, context_precision,
    context_recall, composite (average of available metrics).

    Only computes metrics for which inputs are provided:
    - faithfulness: always (needs answer + context)
    - answer_relevancy: always (needs question + answer)
    - context_precision: only if relevant_ids + retrieved_ids given
    - context_recall: only if relevant_ids + retrieved_ids given

    Args:
        answer: The LLM-generated answer.
        question: The user question that triggered retrieval.
        context: Retrieved context — string or list of chunks.
        relevant_ids: Ground-truth relevant document IDs (optional).
        retrieved_ids: IDs actually retrieved (optional).

    Returns:
        Dict with individual scores and ``composite`` (mean of available scores).

    Example:
        >>> ctx = "Python is a high-level programming language."
        >>> scores = compute_ragas_score("Python is high-level.", "What is Python?", ctx)
        >>> 0.0 <= scores["composite"] <= 1.0
        True
    """
    scores: dict[str, float] = {}

    # -- faithfulness: ratio of answer tokens found in context tokens ----------
    answer_tokens = _tokenize(answer)
    context_tokens = _tokenize(_flatten_context(context))
    if not answer_tokens:
        scores["faithfulness"] = 1.0
    elif not context_tokens:
        scores["faithfulness"] = 0.0
    else:
        overlap = len(answer_tokens & context_tokens)
        scores["faithfulness"] = overlap / len(answer_tokens)

    # -- answer_relevancy: ratio of question tokens found in answer tokens -----
    question_tokens = _tokenize(question)
    if not question_tokens:
        scores["answer_relevancy"] = 1.0
    elif not answer_tokens:
        scores["answer_relevancy"] = 0.0
    else:
        overlap = len(question_tokens & answer_tokens)
        scores["answer_relevancy"] = overlap / len(question_tokens)

    # -- context_precision + context_recall: only when IDs are supplied --------
    if relevant_ids is not None and retrieved_ids is not None:
        relevant_set = set(relevant_ids)
        retrieved_set = set(retrieved_ids)

        if not retrieved_set:
            scores["context_precision"] = 1.0
        else:
            tp = len(relevant_set & retrieved_set)
            scores["context_precision"] = tp / len(retrieved_set)

        if not relevant_set:
            scores["context_recall"] = 1.0
        else:
            tp = len(relevant_set & retrieved_set)
            scores["context_recall"] = tp / len(relevant_set)

    # -- composite: mean of all available scores --------------------------------
    scores["composite"] = sum(scores.values()) / len(scores)
    return scores


@timed_assertion
def assert_ragas_score(
    answer: str,
    question: str,
    context: str | list[str],
    relevant_ids: list[str] | None = None,
    retrieved_ids: list[str] | None = None,
    min_score: float = 0.5,
) -> TestResult:
    """Assert RAGAS composite score meets threshold.

    Computes a lightweight RAGAS-style composite score by averaging the
    individual RAG metrics available from the provided inputs:
    - faithfulness (always)
    - answer_relevancy (always)
    - context_precision (only if relevant_ids + retrieved_ids provided)
    - context_recall (only if relevant_ids + retrieved_ids provided)

    Args:
        answer: The LLM-generated answer.
        question: The user question that triggered retrieval.
        context: Retrieved context — string or list of chunks.
        relevant_ids: Ground-truth relevant document IDs (optional).
        retrieved_ids: IDs actually retrieved by the retriever (optional).
        min_score: Minimum composite score required (default 0.5).

    Returns:
        TestResult with composite score and all individual metric scores.

    Example:
        >>> ctx = "The speed of light is approximately 299,792 km/s."
        >>> assert_ragas_score(
        ...     answer="Light travels at 299792 km per second.",
        ...     question="How fast does light travel?",
        ...     context=ctx,
        ...     min_score=0.3,
        ... )
    """
    scores = compute_ragas_score(answer, question, context, relevant_ids, retrieved_ids)
    composite = scores["composite"]
    passed = composite >= min_score

    metrics_used = [k for k in scores if k != "composite"]
    message = (
        f"RAGAS composite: {composite:.4f} >= {min_score} "
        f"(metrics: {', '.join(f'{k}={v:.3f}' for k, v in scores.items() if k != 'composite')})"
        if passed
        else f"Low RAGAS composite: {composite:.4f} < {min_score} "
        f"(metrics: {', '.join(f'{k}={v:.3f}' for k, v in scores.items() if k != 'composite')})"
    )

    return assert_true(
        passed,
        name="llm.ragas.composite",
        message=message,
        severity=Severity.CRITICAL,
        composite=composite,
        min_score=min_score,
        metrics_used=metrics_used,
        **{k: v for k, v in scores.items() if k != "composite"},
    )
