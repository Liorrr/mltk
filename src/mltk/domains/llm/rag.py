"""RAG (Retrieval-Augmented Generation) evaluation assertions."""

from __future__ import annotations

from collections.abc import Callable

from mltk.core.assertion import assert_true, timed_assertion
from mltk.core.empty import (
    ON_EMPTY_OPTIONS as _ON_EMPTY_OPTIONS,
)
from mltk.core.empty import (
    empty_input_result as _empty_input_result,
)
from mltk.core.empty import (
    unknown_on_empty_result as _unknown_on_empty_result,
)
from mltk.core.error_policy import (
    EVAL_ON_ERROR,
    EvalOnError,
    check_zero_threshold,
    unknown_on_error_result,
)
from mltk.core.judge_parse import call_judge
from mltk.core.result import Severity, TestResult
from mltk.domains.llm._utils import _normalize, _tokenize

_SUPPORTED_METHODS = ("lexical", "embedding", "nli", "llm")


def _flatten_context(context: str | list[str]) -> str:
    """Join list context into a single string for token analysis."""
    if isinstance(context, list):
        return " ".join(context)
    return context


def _llm_score(
    judge_fn: Callable[[str, str], float],
    first: str,
    second: str,
    on_error: str,
    details: dict[str, object],
) -> float | None:
    """Call the judge once and record the error contract in *details*.

    Returns the valid score, ``0.0`` for an errored call under
    ``on_error="zero"``, or ``None`` for an errored call that fails.
    """
    score, error = call_judge(judge_fn, first, second)
    details.update(
        errored_count=0 if error is None else 1,
        n_valid=1 if error is None else 0,
        on_error=on_error,
        errors=[] if error is None else [{"error": error}],
    )
    if error is None:
        return score
    return 0.0 if on_error == "zero" else None


def _judge_error_result(
    name: str, min_score: float, details: dict[str, object],
) -> TestResult:
    """Fail a single-call judge path whose judge errored."""
    error = details["errors"][0]["error"]  # type: ignore[index]
    return assert_true(
        False, name=name,
        message=f"Judge error (llm): {error} (on_error='{details['on_error']}')",
        severity=Severity.CRITICAL,
        score=None, min_score=min_score,
        **details,
    )


def _check_policies(
    name: str, on_empty: str, on_error: str, min_score: float,
) -> TestResult | None:
    """Validate the input policies; raise for a zero-threshold conflict."""
    if on_empty not in _ON_EMPTY_OPTIONS:
        return _unknown_on_empty_result(name, on_empty)
    if on_error not in EVAL_ON_ERROR:
        return unknown_on_error_result(name, on_error, EVAL_ON_ERROR)
    check_zero_threshold(on_error, min_score=min_score)
    return None


@timed_assertion
def assert_faithfulness(
    answer: str,
    context: str | list[str],
    min_score: float = 0.7,
    method: str = "lexical",
    embedding_model: str = "all-mpnet-base-v2",
    nli_model: str = "cross-encoder/nli-deberta-v3-base",
    judge_fn: Callable[[str, str], float] | None = None,
    on_empty: str = "fail",
    *,
    on_error: EvalOnError = "fail_if_all",
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
            ``(answer, context_text)`` and returns a score in ``[0, 1]``.
        on_empty: Policy for empty answer input: ``"fail"`` (default),
            ``"skip"``, or ``"pass"`` for legacy behavior.
        on_error: Policy for a judge error (``method="llm"`` only): the
            judge raised, or returned ``None``, a boolean, a non-finite
            number or a value outside ``[0, 1]``. ``"fail_if_all"``
            (default) and ``"strict"`` fail the result (there is one judge
            call); ``"zero"`` scores the call ``0.0``. ``"zero"`` with
            ``min_score <= 0`` raises ``ValueError``. Details then include
            ``errored_count``, ``n_valid``, ``on_error`` and ``errors``.

    Returns:
        TestResult with faithfulness score.

    Raises:
        ValueError: If ``on_error="zero"`` and ``min_score <= 0``.

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
    invalid = _check_policies("llm.rag.faithfulness", on_empty, on_error, min_score)
    if invalid is not None:
        return invalid

    context_text = _normalize(_flatten_context(context))
    answer = _normalize(answer)

    # -- edge cases (method-independent) ---------------------------------
    if not _tokenize(answer):
        return _empty_input_result(
            name="llm.rag.faithfulness",
            reason="Empty answer",
            on_empty=on_empty,
            severity=Severity.CRITICAL,
            legacy_message="Empty answer -- trivially faithful (score=1.0)",
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
        llm_score = _llm_score(judge_fn, answer, context_text, on_error, details)
        if llm_score is None:
            return _judge_error_result("llm.rag.faithfulness", min_score, details)
        score = llm_score

    passed = score >= min_score
    label = "Faithfulness" if passed else "Low faithfulness"
    cmp = ">=" if passed else "<"
    message = (
        f"{label} ({method}): {score:.4f} {cmp} {min_score}"
    )
    if details.get("errored_count"):
        message += " (judge error scored 0.0, on_error='zero')"

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
    on_empty: str = "fail",
    *,
    on_error: EvalOnError = "fail_if_all",
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
            ``(question, context_text)`` and returns a score in ``[0, 1]``.
        on_empty: Policy for empty question input: ``"fail"`` (default),
            ``"skip"``, or ``"pass"`` for legacy behavior.
        on_error: Policy for a judge error (``method="llm"`` only): the
            judge raised, or returned ``None``, a boolean, a non-finite
            number or a value outside ``[0, 1]``. ``"fail_if_all"``
            (default) and ``"strict"`` fail the result (there is one judge
            call); ``"zero"`` scores the call ``0.0``. ``"zero"`` with
            ``min_score <= 0`` raises ``ValueError``. Details then include
            ``errored_count``, ``n_valid``, ``on_error`` and ``errors``.

    Returns:
        TestResult with context relevancy score.

    Raises:
        ValueError: If ``on_error="zero"`` and ``min_score <= 0``.

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
    invalid = _check_policies("llm.rag.context_relevancy", on_empty, on_error, min_score)
    if invalid is not None:
        return invalid

    context_text = _normalize(_flatten_context(context))
    question = _normalize(question)

    # -- edge cases (method-independent) ---------------------------------
    if not _tokenize(question):
        return _empty_input_result(
            name="llm.rag.context_relevancy",
            reason="Empty question",
            on_empty=on_empty,
            severity=Severity.CRITICAL,
            legacy_message="Empty question -- trivially relevant (score=1.0)",
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
        llm_score = _llm_score(judge_fn, question, context_text, on_error, details)
        if llm_score is None:
            return _judge_error_result("llm.rag.context_relevancy", min_score, details)
        score = llm_score

    passed = score >= min_score
    label = "Context relevancy" if passed else "Low context relevancy"
    cmp = ">=" if passed else "<"
    message = (
        f"{label} ({method}): {score:.4f} {cmp} {min_score}"
    )
    if details.get("errored_count"):
        message += " (judge error scored 0.0, on_error='zero')"

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
    on_empty: str = "fail",
    *,
    on_error: EvalOnError = "fail_if_all",
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
            ``(question, answer)`` and returns a score in ``[0, 1]``.
        on_empty: Policy for empty question input: ``"fail"`` (default),
            ``"skip"``, or ``"pass"`` for legacy behavior.
        on_error: Policy for a judge error (``method="llm"`` only): the
            judge raised, or returned ``None``, a boolean, a non-finite
            number or a value outside ``[0, 1]``. ``"fail_if_all"``
            (default) and ``"strict"`` fail the result (there is one judge
            call); ``"zero"`` scores the call ``0.0``. ``"zero"`` with
            ``min_score <= 0`` raises ``ValueError``. Details then include
            ``errored_count``, ``n_valid``, ``on_error`` and ``errors``.

    Returns:
        TestResult with answer relevancy score.

    Raises:
        ValueError: If ``on_error="zero"`` and ``min_score <= 0``.

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
    invalid = _check_policies("llm.rag.answer_relevancy", on_empty, on_error, min_score)
    if invalid is not None:
        return invalid

    question = _normalize(question)
    answer = _normalize(answer)

    # -- edge cases (method-independent) ---------------------------------
    if not _tokenize(question):
        return _empty_input_result(
            name="llm.rag.answer_relevancy",
            reason="Empty question",
            on_empty=on_empty,
            severity=Severity.CRITICAL,
            legacy_message="Empty question -- trivially relevant (score=1.0)",
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
        llm_score = _llm_score(judge_fn, question, answer, on_error, details)
        if llm_score is None:
            return _judge_error_result("llm.rag.answer_relevancy", min_score, details)
        score = llm_score

    passed = score >= min_score
    label = "Answer relevancy" if passed else "Low answer relevancy"
    cmp = ">=" if passed else "<"
    message = (
        f"{label} ({method}): {score:.4f} {cmp} {min_score}"
    )
    if details.get("errored_count"):
        message += " (judge error scored 0.0, on_error='zero')"

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
    on_empty: str = "fail",
) -> TestResult:
    """Assert precision of retrieval: |relevant ∩ retrieved| / |retrieved|.

    Precision measures how many of the retrieved documents were actually
    relevant. Low precision means the retriever returns many irrelevant
    documents alongside the useful ones (noisy retrieval).

    Args:
        relevant_ids: Ground-truth set of relevant document IDs.
        retrieved_ids: IDs of documents actually retrieved by the retriever.
        min_precision: Minimum precision required (default 0.5).
        on_empty: Policy for empty retrieved IDs: ``"fail"`` (default),
            ``"skip"``, or ``"pass"`` for legacy behavior.

    Returns:
        TestResult with precision score and counts.

    Example:
        >>> assert_context_precision(
        ...     relevant_ids=["doc1", "doc2", "doc3"],
        ...     retrieved_ids=["doc1", "doc2", "doc4", "doc5"],
        ...     min_precision=0.5,
        ... )
    """
    if on_empty not in _ON_EMPTY_OPTIONS:
        return _unknown_on_empty_result("llm.rag.context_precision", on_empty)

    if not retrieved_ids:
        return _empty_input_result(
            name="llm.rag.context_precision",
            reason="No documents retrieved",
            on_empty=on_empty,
            severity=Severity.CRITICAL,
            legacy_message="No documents retrieved — precision undefined (trivially 1.0)",
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
    on_empty: str = "fail",
) -> TestResult:
    """Assert recall of retrieval: |relevant ∩ retrieved| / |relevant|.

    Recall measures how many of the truly relevant documents were actually
    retrieved. Low recall means the retriever is missing important documents
    (under-retrieval).

    Args:
        relevant_ids: Ground-truth set of relevant document IDs.
        retrieved_ids: IDs of documents actually retrieved by the retriever.
        min_recall: Minimum recall required (default 0.5).
        on_empty: Policy for empty relevant IDs: ``"fail"`` (default),
            ``"skip"``, or ``"pass"`` for legacy behavior.

    Returns:
        TestResult with recall score and counts.

    Example:
        >>> assert_context_recall(
        ...     relevant_ids=["doc1", "doc2", "doc3", "doc4"],
        ...     retrieved_ids=["doc1", "doc2", "doc3"],
        ...     min_recall=0.75,
        ... )
    """
    if on_empty not in _ON_EMPTY_OPTIONS:
        return _unknown_on_empty_result("llm.rag.context_recall", on_empty)

    if not relevant_ids:
        return _empty_input_result(
            name="llm.rag.context_recall",
            reason="No relevant documents defined",
            on_empty=on_empty,
            severity=Severity.CRITICAL,
            legacy_message="No relevant documents defined — recall undefined (trivially 1.0)",
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
