"""Retrieval ranking metrics for evaluating search and RAG retrievers.

These assertions measure how well a retrieval system ranks documents.
While ``rag.py`` evaluates the generator (faithfulness, answer relevancy),
this module evaluates the retriever -- the component that selects and
ranks documents before the LLM ever sees them.

Four standard information-retrieval metrics are provided:

- **nDCG@K** -- position-weighted relevance with graded labels
- **MRR** -- reciprocal rank of the first relevant result
- **Recall@K** -- fraction of relevant documents retrieved in top K
- **MAP@K** -- mean average precision at K

All computations are pure Python (no external dependencies).
"""

from __future__ import annotations

import math

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
from mltk.core.result import Severity, TestResult
from mltk.core.validation import require_same_length

__all__ = [
    "assert_ndcg",
    "assert_mrr",
    "assert_recall_at_k",
    "assert_map_at_k",
]


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------


def _dcg_at_k(relevances: list[int], k: int) -> float:
    """Compute Discounted Cumulative Gain at position k.

    DCG@k = sum_{i=1}^{k} (2^rel_i - 1) / log2(i + 1)

    Args:
        relevances: Relevance labels in ranked order.
        k: Number of positions to consider.

    Returns:
        DCG value (float).
    """
    dcg = 0.0
    for i, rel in enumerate(relevances[:k]):
        dcg += (2.0 ** rel - 1.0) / math.log2(i + 2)
    return dcg

def _ndcg_single(
    y_true: list[int],
    y_scores: list[float],
    k: int,
) -> float | None:
    """Compute nDCG@k for a single query.

    Sorts documents by predicted score (descending), then computes
    DCG@k divided by the ideal DCG@k (IDCG).

    Args:
        y_true: Relevance labels for each document.
        y_scores: Predicted scores for each document.
        k: Cutoff position.

    Returns:
        nDCG value in [0, 1], or ``None`` when IDCG is zero
        (no relevant documents) and the query should be excluded
        from the mean.
    """
    require_same_length(
        "llm.retrieval.ndcg",
        y_scores=y_scores,
        y_true=y_true,
    )
    # Sort documents by score descending; ties broken by original order
    paired = sorted(
        zip(y_scores, y_true, strict=True),
        key=lambda x: -x[0],
    )
    ranked_relevances = [rel for _, rel in paired]

    dcg = _dcg_at_k(ranked_relevances, k)

    # Ideal: sort relevance labels descending
    ideal_relevances = sorted(y_true, reverse=True)
    idcg = _dcg_at_k(ideal_relevances, k)

    if idcg == 0.0:
        return None
    return dcg / idcg

# ------------------------------------------------------------------
# Public assertions
# ------------------------------------------------------------------

@timed_assertion
def assert_ndcg(
    y_true: list[list[int]],
    y_scores: list[list[float]],
    k: int = 10,
    min_ndcg: float = 0.8,
    on_empty: str = "fail",
) -> TestResult:
    """Assert that mean nDCG@k meets a minimum threshold.

    Normalized Discounted Cumulative Gain measures retrieval quality
    with position-weighted relevance.  A document ranked #1
    contributes more than one ranked #10.  This is the standard
    metric for evaluating search engines and RAG retrievers when
    graded relevance labels are available.

    Formula per query:
        DCG@k  = sum_{i=1}^{k} (2^rel_i - 1) / log2(i + 1)
        IDCG@k = DCG@k of the ideal (perfectly sorted) ranking
        nDCG@k = DCG@k / IDCG@k

    The final score is the mean nDCG@k across queries with non-zero
    IDCG. Queries with no relevant documents are excluded; if no
    evaluable queries remain, ``on_empty`` controls the result.

    Args:
        y_true: Relevance labels per query.  ``y_true[i]`` is a list
            of integer relevance grades (e.g. [3, 2, 0, 1, 0]) for
            the documents associated with query *i*.
        y_scores: Model-predicted scores per query.  ``y_scores[i]``
            has the same length as ``y_true[i]``.
        k: Cutoff position (default 10).
        min_ndcg: Minimum acceptable mean nDCG@k (default 0.8).
        on_empty: Policy for no evaluable queries: ``"fail"`` (default),
            ``"skip"``, or ``"pass"`` for legacy behavior.

    Returns:
        TestResult with ``ndcg``, ``min_ndcg``, ``k``,
        ``num_queries``, and per-query scores in details.

    Example:
        >>> y_true = [[3, 2, 0, 1], [1, 0, 0, 1]]
        >>> y_scores = [[0.9, 0.8, 0.2, 0.5], [0.8, 0.1, 0.3, 0.7]]
        >>> assert_ndcg(y_true, y_scores, k=4, min_ndcg=0.5)
    """
    require_same_length(
        "llm.retrieval.ndcg",
        y_true=y_true,
        y_scores=y_scores,
    )
    if on_empty not in _ON_EMPTY_OPTIONS:
        return _unknown_on_empty_result("llm.retrieval.ndcg", on_empty)

    if not y_true:
        return _empty_input_result(
            name="llm.retrieval.ndcg",
            reason="No queries provided",
            on_empty=on_empty,
            legacy_message=(
                "No queries provided -- trivially passing (ndcg=1.0)"
            ),
            ndcg=1.0,
            min_ndcg=min_ndcg,
            k=k,
            num_queries=0,
        )

    per_query: list[float] = []
    for yt, ys in zip(y_true, y_scores, strict=True):
        score = _ndcg_single(yt, ys, k)
        if score is not None:
            per_query.append(score)

    if not per_query:
        return _empty_input_result(
            name="llm.retrieval.ndcg",
            reason="No queries with relevant documents",
            on_empty=on_empty,
            legacy_message=(
                "No queries with relevant documents -- trivially passing "
                "(ndcg=1.0)"
            ),
            ndcg=1.0,
            min_ndcg=min_ndcg,
            k=k,
            num_queries=0,
        )

    mean_ndcg = sum(per_query) / len(per_query)
    passed = mean_ndcg >= min_ndcg

    message = (
        f"nDCG@{k}: {mean_ndcg:.4f} >= {min_ndcg} "
        f"(averaged over {len(per_query)} queries)"
        if passed
        else f"Low nDCG@{k}: {mean_ndcg:.4f} < {min_ndcg} "
        f"(averaged over {len(per_query)} queries)"
    )

    return assert_true(
        passed,
        name="llm.retrieval.ndcg",
        message=message,
        severity=Severity.CRITICAL,
        ndcg=mean_ndcg,
        min_ndcg=min_ndcg,
        k=k,
        num_queries=len(per_query),
        per_query_ndcg=per_query,
    )

@timed_assertion
def assert_mrr(
    queries_results: list[list[bool]],
    min_mrr: float = 0.5,
    on_empty: str = "fail",
) -> TestResult:
    """Assert that Mean Reciprocal Rank meets a minimum threshold.

    For each query, MRR looks at the rank of the *first* relevant
    result and takes the reciprocal (1/rank).  The overall MRR is
    the average across all queries.

    Formula:
        MRR = (1 / |Q|) * sum_{i=1}^{|Q|} 1 / rank_i

    where rank_i is the position of the first ``True`` in
    ``queries_results[i]``. Queries with no relevant result are
    excluded; if no evaluable queries remain, ``on_empty`` controls
    the result.

    Args:
        queries_results: Boolean relevance per result per query.
            ``queries_results[i]`` is a list of booleans indicating
            whether each retrieved result is relevant for query *i*.
        min_mrr: Minimum acceptable MRR (default 0.5).
        on_empty: Policy for no evaluable queries: ``"fail"`` (default),
            ``"skip"``, or ``"pass"`` for legacy behavior.

    Returns:
        TestResult with ``mrr``, ``min_mrr``, ``num_queries``,
        and per-query reciprocal ranks in details.

    Example:
        >>> results = [
        ...     [False, True, False],   # first relevant at rank 2
        ...     [True, False, False],    # first relevant at rank 1
        ... ]
        >>> assert_mrr(results, min_mrr=0.5)
    """
    if on_empty not in _ON_EMPTY_OPTIONS:
        return _unknown_on_empty_result("llm.retrieval.mrr", on_empty)

    if not queries_results:
        return _empty_input_result(
            name="llm.retrieval.mrr",
            reason="No queries provided",
            on_empty=on_empty,
            legacy_message=(
                "No queries provided -- trivially passing (mrr=1.0)"
            ),
            mrr=1.0,
            min_mrr=min_mrr,
            num_queries=0,
        )

    reciprocals: list[float] = []
    for results in queries_results:
        if not any(results):
            continue
        rr = 0.0
        for rank, is_relevant in enumerate(results, start=1):
            if is_relevant:
                rr = 1.0 / rank
                break
        reciprocals.append(rr)

    if not reciprocals:
        return _empty_input_result(
            name="llm.retrieval.mrr",
            reason="No queries with relevant documents",
            on_empty=on_empty,
            legacy_message=(
                "No queries with relevant documents -- trivially passing "
                "(mrr=1.0)"
            ),
            mrr=1.0,
            min_mrr=min_mrr,
            num_queries=0,
        )

    mrr = sum(reciprocals) / len(reciprocals)
    passed = mrr >= min_mrr

    message = (
        f"MRR: {mrr:.4f} >= {min_mrr} "
        f"(averaged over {len(reciprocals)} queries)"
        if passed
        else f"Low MRR: {mrr:.4f} < {min_mrr} "
        f"(averaged over {len(reciprocals)} queries)"
    )

    return assert_true(
        passed,
        name="llm.retrieval.mrr",
        message=message,
        severity=Severity.CRITICAL,
        mrr=mrr,
        min_mrr=min_mrr,
        num_queries=len(reciprocals),
        per_query_rr=reciprocals,
    )

@timed_assertion
def assert_recall_at_k(
    relevant: list[set],
    retrieved: list[list],
    k: int = 10,
    min_recall: float = 0.8,
    on_empty: str = "fail",
) -> TestResult:
    """Assert that mean Recall@K meets a minimum threshold.

    For each query, Recall@K is the fraction of all relevant
    documents that appear in the top K retrieved results.  Low
    recall means the retriever is *missing* important documents.
    Queries with no relevant documents are excluded from the mean;
    if no evaluable queries remain, ``on_empty`` controls the result.

    Formula per query:
        Recall@k = |relevant intersect retrieved[:k]| / |relevant|

    Args:
        relevant: Relevant document IDs per query.
            ``relevant[i]`` is a set of IDs for query *i*.
        retrieved: Retrieved document IDs per query, in ranked order.
            ``retrieved[i]`` is an ordered list of IDs for query *i*.
        k: Cutoff position (default 10).
        min_recall: Minimum acceptable mean Recall@K (default 0.8).
        on_empty: Policy for no evaluable queries: ``"fail"`` (default),
            ``"skip"``, or ``"pass"`` for legacy behavior.

    Returns:
        TestResult with ``recall``, ``min_recall``, ``k``,
        ``num_queries``, and per-query recall scores.

    Example:
        >>> relevant = [{"d1", "d2", "d3"}, {"d4", "d5"}]
        >>> retrieved = [["d1", "d3", "d6", "d2"], ["d5", "d7"]]
        >>> assert_recall_at_k(relevant, retrieved, k=3, min_recall=0.5)
    """
    require_same_length(
        "llm.retrieval.recall_at_k",
        relevant=relevant,
        retrieved=retrieved,
    )
    if on_empty not in _ON_EMPTY_OPTIONS:
        return _unknown_on_empty_result("llm.retrieval.recall_at_k", on_empty)

    if not relevant:
        return _empty_input_result(
            name="llm.retrieval.recall_at_k",
            reason="No queries provided",
            on_empty=on_empty,
            legacy_message=(
                "No queries provided -- trivially passing (recall=1.0)"
            ),
            recall=1.0,
            min_recall=min_recall,
            k=k,
            num_queries=0,
        )

    per_query: list[float] = []
    for rel_set, ret_list in zip(relevant, retrieved, strict=True):
        if not rel_set:
            continue
        top_k = set(ret_list[:k])
        hits = len(rel_set & top_k)
        per_query.append(hits / len(rel_set))

    if not per_query:
        return _empty_input_result(
            name="llm.retrieval.recall_at_k",
            reason="No queries with relevant documents",
            on_empty=on_empty,
            legacy_message=(
                "No queries with relevant documents -- trivially passing "
                "(recall=1.0)"
            ),
            recall=1.0,
            min_recall=min_recall,
            k=k,
            num_queries=0,
        )

    mean_recall = sum(per_query) / len(per_query)
    passed = mean_recall >= min_recall

    message = (
        f"Recall@{k}: {mean_recall:.4f} >= {min_recall} "
        f"(averaged over {len(per_query)} queries)"
        if passed
        else f"Low Recall@{k}: {mean_recall:.4f} < {min_recall} "
        f"(averaged over {len(per_query)} queries)"
    )

    return assert_true(
        passed,
        name="llm.retrieval.recall_at_k",
        message=message,
        severity=Severity.CRITICAL,
        recall=mean_recall,
        min_recall=min_recall,
        k=k,
        num_queries=len(per_query),
        per_query_recall=per_query,
    )

@timed_assertion
def assert_map_at_k(
    relevant: list[set],
    retrieved: list[list],
    k: int = 10,
    min_map: float = 0.5,
    on_empty: str = "fail",
) -> TestResult:
    """Assert that Mean Average Precision@K meets a minimum threshold.

    For each query, Average Precision@K considers precision at every
    position where a relevant document is found, then averages those
    precision values.  This rewards systems that rank relevant
    documents *higher*, not just retrieve them somewhere in the list.
    Queries with no relevant documents are excluded from the mean;
    if no evaluable queries remain, ``on_empty`` controls the result.

    Formula per query:
        AP@k = (1 / |relevant|)
               * sum_{j=1}^{k} Precision@j * rel(j)

    where rel(j) = 1 if the j-th retrieved document is relevant.

    Args:
        relevant: Relevant document IDs per query.
            ``relevant[i]`` is a set of IDs for query *i*.
        retrieved: Retrieved document IDs per query, in ranked order.
            ``retrieved[i]`` is an ordered list of IDs for query *i*.
        k: Cutoff position (default 10).
        min_map: Minimum acceptable MAP@K (default 0.5).
        on_empty: Policy for no evaluable queries: ``"fail"`` (default),
            ``"skip"``, or ``"pass"`` for legacy behavior.

    Returns:
        TestResult with ``map_score``, ``min_map``, ``k``,
        ``num_queries``, and per-query AP scores.

    Example:
        >>> relevant = [{"d1", "d3"}, {"d2"}]
        >>> retrieved = [["d1", "d2", "d3"], ["d1", "d2", "d3"]]
        >>> assert_map_at_k(relevant, retrieved, k=3, min_map=0.5)
    """
    require_same_length(
        "llm.retrieval.map_at_k",
        relevant=relevant,
        retrieved=retrieved,
    )
    if on_empty not in _ON_EMPTY_OPTIONS:
        return _unknown_on_empty_result("llm.retrieval.map_at_k", on_empty)

    if not relevant:
        return _empty_input_result(
            name="llm.retrieval.map_at_k",
            reason="No queries provided",
            on_empty=on_empty,
            legacy_message=(
                "No queries provided -- trivially passing (map=1.0)"
            ),
            map_score=1.0,
            min_map=min_map,
            k=k,
            num_queries=0,
        )

    per_query: list[float] = []
    for rel_set, ret_list in zip(relevant, retrieved, strict=True):
        if not rel_set:
            continue

        hits = 0
        sum_precision = 0.0
        for rank, doc_id in enumerate(ret_list[:k], start=1):
            if doc_id in rel_set:
                hits += 1
                sum_precision += hits / rank

        ap = sum_precision / len(rel_set)
        per_query.append(ap)

    if not per_query:
        return _empty_input_result(
            name="llm.retrieval.map_at_k",
            reason="No queries with relevant documents",
            on_empty=on_empty,
            legacy_message=(
                "No queries with relevant documents -- trivially passing "
                "(map=1.0)"
            ),
            map_score=1.0,
            min_map=min_map,
            k=k,
            num_queries=0,
        )

    mean_ap = sum(per_query) / len(per_query)
    passed = mean_ap >= min_map

    message = (
        f"MAP@{k}: {mean_ap:.4f} >= {min_map} "
        f"(averaged over {len(per_query)} queries)"
        if passed
        else f"Low MAP@{k}: {mean_ap:.4f} < {min_map} "
        f"(averaged over {len(per_query)} queries)"
    )

    return assert_true(
        passed,
        name="llm.retrieval.map_at_k",
        message=message,
        severity=Severity.CRITICAL,
        map_score=mean_ap,
        min_map=min_map,
        k=k,
        num_queries=len(per_query),
        per_query_ap=per_query,
    )
