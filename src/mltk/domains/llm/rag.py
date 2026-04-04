"""RAG (Retrieval-Augmented Generation) evaluation assertions."""

from __future__ import annotations

from collections.abc import Callable

from mltk.core.assertion import assert_true, timed_assertion
from mltk.core.result import Severity, TestResult
from mltk.domains.llm._utils import _normalize, _tokenize

_SUPPORTED_METHODS = ("lexical", "embedding", "nli", "llm")


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
    method: str = "lexical",
    embedding_model: str = "all-mpnet-base-v2",
    nli_model: str = "cross-encoder/nli-deberta-v3-base",
    judge_fn: Callable[[str, str], float] | None = None,
) -> TestResult:
    """Assert answer is grounded in the provided context.

    Supports multiple scoring methods:

    - ``"lexical"`` -- token overlap ratio (default, zero-dep).
    - ``"embedding"`` -- cosine similarity via sentence-transformers.
    - ``"nli"`` -- entailment probability via cross-encoder NLI model.
    - ``"llm"`` -- custom LLM judge function returning 0-1 score.

    Args:
        answer: The LLM-generated answer to evaluate.
        context: Retrieved context -- single string or list of chunks.
        min_score: Minimum grounding score required (default 0.7).
        method: Scoring method (default ``"lexical"``).
        embedding_model: Model name for ``method="embedding"``.
        nli_model: Model name for ``method="nli"``.
        judge_fn: Callable for ``method="llm"``; receives
            ``(answer, context_text)`` and returns a 0-1 float.

    Returns:
        TestResult with faithfulness score.

    Example:
        >>> ctx = "The Eiffel Tower is in Paris, France."
        >>> assert_faithfulness(
        ...     "The Eiffel Tower was built in 1889.", ctx,
        ...     min_score=0.7,
        ... )
    """
    if method not in _SUPPORTED_METHODS:
        return assert_true(
            False, name="llm.rag.faithfulness",
            message=(
                f"Unknown method: '{method}'. "
                f"Supported: {', '.join(_SUPPORTED_METHODS)}"
            ),
            severity=Severity.CRITICAL,
            method=method,
        )

    context_text = _normalize(_flatten_context(context))
    answer = _normalize(answer)

    # -- edge cases (method-independent) ---------------------------------
    if not _tokenize(answer):
        return assert_true(
            True, name="llm.rag.faithfulness",
            message="Empty answer -- trivially faithful (score=1.0)",
            severity=Severity.CRITICAL,
            score=1.0, min_score=min_score, method=method,
        )

    if not _tokenize(context_text):
        return assert_true(
            False, name="llm.rag.faithfulness",
            message=(
                "Empty context -- cannot evaluate faithfulness "
                "(score=0.0)"
            ),
            severity=Severity.CRITICAL,
            score=0.0, min_score=min_score, method=method,
        )

    # -- scoring ---------------------------------------------------------
    details: dict[str, object] = {"method": method}

    if method == "lexical":
        answer_tokens = _tokenize(answer)
        context_tokens = _tokenize(context_text)
        overlap = len(answer_tokens & context_tokens)
        score = overlap / len(answer_tokens)
        details.update(
            answer_tokens=len(answer_tokens),
            context_tokens=len(context_tokens),
            grounded_tokens=overlap,
        )
    elif method == "embedding":
        from mltk.domains.llm._backends import embedding_cosine_single
        score = embedding_cosine_single(
            answer, context_text, model_name=embedding_model,
        )
        details["embedding_model"] = embedding_model
    elif method == "nli":
        from mltk.domains.llm._backends import nli_entailment_score
        nli_result = nli_entailment_score(
            context_text, answer, model_name=nli_model,
        )
        score = nli_result["entailment"]
        details["nli_model"] = nli_model
        details["nli_label"] = nli_result["label"]
    else:  # method == "llm"
        if judge_fn is None:
            return assert_true(
                False, name="llm.rag.faithfulness",
                message=(
                    "method='llm' requires a judge_fn callable"
                ),
                severity=Severity.CRITICAL,
                method=method,
            )
        score = float(judge_fn(answer, context_text))

    passed = score >= min_score
    label = "Faithfulness" if passed else "Low faithfulness"
    cmp = ">=" if passed else "<"
    message = (
        f"{label} ({method}): {score:.4f} {cmp} {min_score}"
    )

    return assert_true(
        passed, name="llm.rag.faithfulness", message=message,
        severity=Severity.CRITICAL,
        score=score, min_score=min_score,
        **details,
    )


@timed_assertion
def assert_context_relevancy(
    question: str,
    context: str | list[str],
    min_score: float = 0.5,
    method: str = "lexical",
    embedding_model: str = "all-mpnet-base-v2",
    nli_model: str = "cross-encoder/nli-deberta-v3-base",
    judge_fn: Callable[[str, str], float] | None = None,
) -> TestResult:
    """Assert retrieved context is relevant to the question.

    Supports multiple scoring methods:

    - ``"lexical"`` -- token overlap ratio (default, zero-dep).
    - ``"embedding"`` -- cosine similarity via sentence-transformers.
    - ``"nli"`` -- entailment: does context address the question?
    - ``"llm"`` -- custom LLM judge function returning 0-1 score.

    Args:
        question: The user question that triggered retrieval.
        context: Retrieved context -- single string or list of chunks.
        min_score: Minimum relevancy ratio required (default 0.5).
        method: Scoring method (default ``"lexical"``).
        embedding_model: Model name for ``method="embedding"``.
        nli_model: Model name for ``method="nli"``.
        judge_fn: Callable for ``method="llm"``; receives
            ``(question, context_text)`` and returns a 0-1 float.

    Returns:
        TestResult with context relevancy score.

    Example:
        >>> ctx = "Paris is the capital of France."
        >>> assert_context_relevancy(
        ...     "What is the capital of France?", ctx,
        ...     min_score=0.5,
        ... )
    """
    if method not in _SUPPORTED_METHODS:
        return assert_true(
            False, name="llm.rag.context_relevancy",
            message=(
                f"Unknown method: '{method}'. "
                f"Supported: {', '.join(_SUPPORTED_METHODS)}"
            ),
            severity=Severity.CRITICAL,
            method=method,
        )

    context_text = _normalize(_flatten_context(context))
    question = _normalize(question)

    # -- edge cases (method-independent) ---------------------------------
    if not _tokenize(question):
        return assert_true(
            True, name="llm.rag.context_relevancy",
            message=(
                "Empty question -- trivially relevant (score=1.0)"
            ),
            severity=Severity.CRITICAL,
            score=1.0, min_score=min_score, method=method,
        )

    if not _tokenize(context_text):
        return assert_true(
            False, name="llm.rag.context_relevancy",
            message=(
                "Empty context -- cannot be relevant (score=0.0)"
            ),
            severity=Severity.CRITICAL,
            score=0.0, min_score=min_score, method=method,
        )

    # -- scoring ---------------------------------------------------------
    details: dict[str, object] = {"method": method}

    if method == "lexical":
        question_tokens = _tokenize(question)
        context_tokens = _tokenize(context_text)
        overlap = len(question_tokens & context_tokens)
        score = overlap / len(question_tokens)
        details.update(
            question_tokens=len(question_tokens),
            context_tokens=len(context_tokens),
            matched_tokens=overlap,
        )
    elif method == "embedding":
        from mltk.domains.llm._backends import embedding_cosine_single
        score = embedding_cosine_single(
            question, context_text, model_name=embedding_model,
        )
        details["embedding_model"] = embedding_model
    elif method == "nli":
        from mltk.domains.llm._backends import nli_entailment_score
        nli_result = nli_entailment_score(
            context_text, question, model_name=nli_model,
        )
        score = nli_result["entailment"]
        details["nli_model"] = nli_model
        details["nli_label"] = nli_result["label"]
    else:  # method == "llm"
        if judge_fn is None:
            return assert_true(
                False, name="llm.rag.context_relevancy",
                message=(
                    "method='llm' requires a judge_fn callable"
                ),
                severity=Severity.CRITICAL,
                method=method,
            )
        score = float(judge_fn(question, context_text))

    passed = score >= min_score
    label = "Context relevancy" if passed else "Low context relevancy"
    cmp = ">=" if passed else "<"
    message = (
        f"{label} ({method}): {score:.4f} {cmp} {min_score}"
    )

    return assert_true(
        passed, name="llm.rag.context_relevancy", message=message,
        severity=Severity.CRITICAL,
        score=score, min_score=min_score,
        **details,
    )


@timed_assertion
def assert_answer_relevancy(
    question: str,
    answer: str,
    min_score: float = 0.5,
    method: str = "lexical",
    embedding_model: str = "all-mpnet-base-v2",
    nli_model: str = "cross-encoder/nli-deberta-v3-base",
    judge_fn: Callable[[str, str], float] | None = None,
) -> TestResult:
    """Assert answer addresses the question.

    Supports multiple scoring methods:

    - ``"lexical"`` -- token overlap ratio (default, zero-dep).
    - ``"embedding"`` -- cosine similarity via sentence-transformers.
    - ``"nli"`` -- entailment: does the answer address the question?
    - ``"llm"`` -- custom LLM judge function returning 0-1 score.

    Args:
        question: The user question.
        answer: The LLM-generated answer.
        min_score: Minimum relevancy ratio required (default 0.5).
        method: Scoring method (default ``"lexical"``).
        embedding_model: Model name for ``method="embedding"``.
        nli_model: Model name for ``method="nli"``.
        judge_fn: Callable for ``method="llm"``; receives
            ``(question, answer)`` and returns a 0-1 float.

    Returns:
        TestResult with answer relevancy score.

    Example:
        >>> assert_answer_relevancy(
        ...     "What is machine learning?",
        ...     "Machine learning is a subset of AI.",
        ...     min_score=0.5,
        ... )
    """
    if method not in _SUPPORTED_METHODS:
        return assert_true(
            False, name="llm.rag.answer_relevancy",
            message=(
                f"Unknown method: '{method}'. "
                f"Supported: {', '.join(_SUPPORTED_METHODS)}"
            ),
            severity=Severity.CRITICAL,
            method=method,
        )

    question = _normalize(question)
    answer = _normalize(answer)

    # -- edge cases (method-independent) ---------------------------------
    if not _tokenize(question):
        return assert_true(
            True, name="llm.rag.answer_relevancy",
            message=(
                "Empty question -- trivially relevant (score=1.0)"
            ),
            severity=Severity.CRITICAL,
            score=1.0, min_score=min_score, method=method,
        )

    if not _tokenize(answer):
        return assert_true(
            False, name="llm.rag.answer_relevancy",
            message=(
                "Empty answer -- cannot be relevant (score=0.0)"
            ),
            severity=Severity.CRITICAL,
            score=0.0, min_score=min_score, method=method,
        )

    # -- scoring ---------------------------------------------------------
    details: dict[str, object] = {"method": method}

    if method == "lexical":
        question_tokens = _tokenize(question)
        answer_tokens = _tokenize(answer)
        overlap = len(question_tokens & answer_tokens)
        score = overlap / len(question_tokens)
        details.update(
            question_tokens=len(question_tokens),
            answer_tokens=len(answer_tokens),
            matched_tokens=overlap,
        )
    elif method == "embedding":
        from mltk.domains.llm._backends import embedding_cosine_single
        score = embedding_cosine_single(
            question, answer, model_name=embedding_model,
        )
        details["embedding_model"] = embedding_model
    elif method == "nli":
        from mltk.domains.llm._backends import nli_entailment_score
        nli_result = nli_entailment_score(
            answer, question, model_name=nli_model,
        )
        score = nli_result["entailment"]
        details["nli_model"] = nli_model
        details["nli_label"] = nli_result["label"]
    else:  # method == "llm"
        if judge_fn is None:
            return assert_true(
                False, name="llm.rag.answer_relevancy",
                message=(
                    "method='llm' requires a judge_fn callable"
                ),
                severity=Severity.CRITICAL,
                method=method,
            )
        score = float(judge_fn(question, answer))

    passed = score >= min_score
    label = "Answer relevancy" if passed else "Low answer relevancy"
    cmp = ">=" if passed else "<"
    message = (
        f"{label} ({method}): {score:.4f} {cmp} {min_score}"
    )

    return assert_true(
        passed, name="llm.rag.answer_relevancy", message=message,
        severity=Severity.CRITICAL,
        score=score, min_score=min_score,
        **details,
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
