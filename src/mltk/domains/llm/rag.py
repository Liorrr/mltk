"""RAG (Retrieval-Augmented Generation) evaluation assertions."""

from __future__ import annotations

import re

from mltk.core.assertion import assert_true, timed_assertion
from mltk.core.result import Severity, TestResult


def _tokenize(text: str) -> set[str]:
    """Lowercase, strip punctuation, split on whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return set(text.split())


def _flatten_context(context: str | list[str]) -> str:
    """Join list context into a single string for token analysis."""
    if isinstance(context, list):
        return " ".join(context)
    return context


@timed_assertion
def assert_faithfulness(
    answer: str,
    context: str | list[str],
    min_score: float = 0.7,
) -> TestResult:
    """Assert answer is grounded in the provided context.

    Score = ratio of answer tokens that appear in context tokens.
    A high score means the answer uses information from the context rather
    than hallucinating content not present in the retrieved documents.

    Args:
        answer: The LLM-generated answer to evaluate.
        context: Retrieved context — either a single string or list of chunks.
        min_score: Minimum grounding ratio required (default 0.7).

    Returns:
        TestResult with faithfulness score.

    Example:
        >>> ctx = "The Eiffel Tower is in Paris, France. It was built in 1889."
        >>> assert_faithfulness("The Eiffel Tower was built in 1889.", ctx, min_score=0.7)
    """
    answer_tokens = _tokenize(answer)
    context_tokens = _tokenize(_flatten_context(context))

    if not answer_tokens:
        return assert_true(
            True, name="llm.rag.faithfulness",
            message="Empty answer — trivially faithful (score=1.0)",
            severity=Severity.CRITICAL,
            score=1.0, min_score=min_score,
        )

    if not context_tokens:
        return assert_true(
            False, name="llm.rag.faithfulness",
            message="Empty context — cannot evaluate faithfulness (score=0.0)",
            severity=Severity.CRITICAL,
            score=0.0, min_score=min_score,
        )

    overlap = len(answer_tokens & context_tokens)
    score = overlap / len(answer_tokens)
    passed = score >= min_score

    message = (
        f"Faithfulness: {score:.4f} >= {min_score} "
        f"({overlap}/{len(answer_tokens)} answer tokens grounded)"
        if passed
        else f"Low faithfulness: {score:.4f} < {min_score} "
        f"({overlap}/{len(answer_tokens)} answer tokens grounded)"
    )

    return assert_true(
        passed, name="llm.rag.faithfulness", message=message,
        severity=Severity.CRITICAL,
        score=score, min_score=min_score,
        answer_tokens=len(answer_tokens),
        context_tokens=len(context_tokens),
        grounded_tokens=overlap,
    )


@timed_assertion
def assert_context_relevancy(
    question: str,
    context: str | list[str],
    min_score: float = 0.5,
) -> TestResult:
    """Assert retrieved context is relevant to the question.

    Score = ratio of question tokens found in context tokens.
    A low score indicates the retriever returned documents that do not
    address the question — a retrieval quality problem.

    Args:
        question: The user question that triggered retrieval.
        context: Retrieved context — either a single string or list of chunks.
        min_score: Minimum relevancy ratio required (default 0.5).

    Returns:
        TestResult with context relevancy score.

    Example:
        >>> ctx = "Paris is the capital of France and a major European city."
        >>> assert_context_relevancy("What is the capital of France?", ctx, min_score=0.5)
    """
    question_tokens = _tokenize(question)
    context_tokens = _tokenize(_flatten_context(context))

    if not question_tokens:
        return assert_true(
            True, name="llm.rag.context_relevancy",
            message="Empty question — trivially relevant (score=1.0)",
            severity=Severity.CRITICAL,
            score=1.0, min_score=min_score,
        )

    if not context_tokens:
        return assert_true(
            False, name="llm.rag.context_relevancy",
            message="Empty context — cannot be relevant (score=0.0)",
            severity=Severity.CRITICAL,
            score=0.0, min_score=min_score,
        )

    overlap = len(question_tokens & context_tokens)
    score = overlap / len(question_tokens)
    passed = score >= min_score

    message = (
        f"Context relevancy: {score:.4f} >= {min_score} "
        f"({overlap}/{len(question_tokens)} question tokens found in context)"
        if passed
        else f"Low context relevancy: {score:.4f} < {min_score} "
        f"({overlap}/{len(question_tokens)} question tokens found in context)"
    )

    return assert_true(
        passed, name="llm.rag.context_relevancy", message=message,
        severity=Severity.CRITICAL,
        score=score, min_score=min_score,
        question_tokens=len(question_tokens),
        context_tokens=len(context_tokens),
        matched_tokens=overlap,
    )


@timed_assertion
def assert_answer_relevancy(
    question: str,
    answer: str,
    min_score: float = 0.5,
) -> TestResult:
    """Assert answer addresses the question.

    Score = ratio of question keyword tokens found in the answer.
    A low score means the answer veers off-topic and does not address
    what was asked.

    Args:
        question: The user question.
        answer: The LLM-generated answer.
        min_score: Minimum relevancy ratio required (default 0.5).

    Returns:
        TestResult with answer relevancy score.

    Example:
        >>> assert_answer_relevancy(
        ...     "What is machine learning?",
        ...     "Machine learning is a subset of artificial intelligence.",
        ...     min_score=0.5,
        ... )
    """
    question_tokens = _tokenize(question)
    answer_tokens = _tokenize(answer)

    if not question_tokens:
        return assert_true(
            True, name="llm.rag.answer_relevancy",
            message="Empty question — trivially relevant (score=1.0)",
            severity=Severity.CRITICAL,
            score=1.0, min_score=min_score,
        )

    if not answer_tokens:
        return assert_true(
            False, name="llm.rag.answer_relevancy",
            message="Empty answer — cannot be relevant (score=0.0)",
            severity=Severity.CRITICAL,
            score=0.0, min_score=min_score,
        )

    overlap = len(question_tokens & answer_tokens)
    score = overlap / len(question_tokens)
    passed = score >= min_score

    message = (
        f"Answer relevancy: {score:.4f} >= {min_score} "
        f"({overlap}/{len(question_tokens)} question tokens found in answer)"
        if passed
        else f"Low answer relevancy: {score:.4f} < {min_score} "
        f"({overlap}/{len(question_tokens)} question tokens found in answer)"
    )

    return assert_true(
        passed, name="llm.rag.answer_relevancy", message=message,
        severity=Severity.CRITICAL,
        score=score, min_score=min_score,
        question_tokens=len(question_tokens),
        answer_tokens=len(answer_tokens),
        matched_tokens=overlap,
    )


@timed_assertion
def assert_context_precision(
    relevant_ids: list[str],
    retrieved_ids: list[str],
    min_precision: float = 0.5,
) -> TestResult:
    """Assert precision of retrieval: |relevant ∩ retrieved| / |retrieved|.

    Precision measures how many of the retrieved documents were actually
    relevant. Low precision means the retriever returns many irrelevant
    documents alongside the useful ones (noisy retrieval).

    Args:
        relevant_ids: Ground-truth set of relevant document IDs.
        retrieved_ids: IDs of documents actually retrieved by the retriever.
        min_precision: Minimum precision required (default 0.5).

    Returns:
        TestResult with precision score and counts.

    Example:
        >>> assert_context_precision(
        ...     relevant_ids=["doc1", "doc2", "doc3"],
        ...     retrieved_ids=["doc1", "doc2", "doc4", "doc5"],
        ...     min_precision=0.5,
        ... )
    """
    if not retrieved_ids:
        return assert_true(
            True, name="llm.rag.context_precision",
            message="No documents retrieved — precision undefined (trivially 1.0)",
            severity=Severity.CRITICAL,
            precision=1.0, min_precision=min_precision,
            true_positives=0, retrieved=0,
        )

    relevant_set = set(relevant_ids)
    retrieved_set = set(retrieved_ids)
    true_positives = len(relevant_set & retrieved_set)
    precision = true_positives / len(retrieved_set)
    passed = precision >= min_precision

    message = (
        f"Retrieval precision: {precision:.4f} >= {min_precision} "
        f"({true_positives}/{len(retrieved_set)} retrieved docs are relevant)"
        if passed
        else f"Low retrieval precision: {precision:.4f} < {min_precision} "
        f"({true_positives}/{len(retrieved_set)} retrieved docs are relevant)"
    )

    return assert_true(
        passed, name="llm.rag.context_precision", message=message,
        severity=Severity.CRITICAL,
        precision=precision, min_precision=min_precision,
        true_positives=true_positives,
        retrieved=len(retrieved_set),
        relevant=len(relevant_set),
    )


@timed_assertion
def assert_context_recall(
    relevant_ids: list[str],
    retrieved_ids: list[str],
    min_recall: float = 0.5,
) -> TestResult:
    """Assert recall of retrieval: |relevant ∩ retrieved| / |relevant|.

    Recall measures how many of the truly relevant documents were actually
    retrieved. Low recall means the retriever is missing important documents
    (under-retrieval).

    Args:
        relevant_ids: Ground-truth set of relevant document IDs.
        retrieved_ids: IDs of documents actually retrieved by the retriever.
        min_recall: Minimum recall required (default 0.5).

    Returns:
        TestResult with recall score and counts.

    Example:
        >>> assert_context_recall(
        ...     relevant_ids=["doc1", "doc2", "doc3", "doc4"],
        ...     retrieved_ids=["doc1", "doc2", "doc3"],
        ...     min_recall=0.75,
        ... )
    """
    if not relevant_ids:
        return assert_true(
            True, name="llm.rag.context_recall",
            message="No relevant documents defined — recall undefined (trivially 1.0)",
            severity=Severity.CRITICAL,
            recall=1.0, min_recall=min_recall,
            true_positives=0, relevant=0,
        )

    relevant_set = set(relevant_ids)
    retrieved_set = set(retrieved_ids)
    true_positives = len(relevant_set & retrieved_set)
    recall = true_positives / len(relevant_set)
    passed = recall >= min_recall

    message = (
        f"Retrieval recall: {recall:.4f} >= {min_recall} "
        f"({true_positives}/{len(relevant_set)} relevant docs retrieved)"
        if passed
        else f"Low retrieval recall: {recall:.4f} < {min_recall} "
        f"({true_positives}/{len(relevant_set)} relevant docs retrieved)"
    )

    return assert_true(
        passed, name="llm.rag.context_recall", message=message,
        severity=Severity.CRITICAL,
        recall=recall, min_recall=min_recall,
        true_positives=true_positives,
        retrieved=len(retrieved_set),
        relevant=len(relevant_set),
    )
