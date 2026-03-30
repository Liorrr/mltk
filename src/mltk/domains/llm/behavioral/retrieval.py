"""Retrieval consistency testing for RAG pipelines.

Verifies that semantically equivalent queries retrieve the same
documents.  Uses Jaccard similarity on document ID sets — the
standard set-overlap metric — to measure retrieval consistency
across paraphrased queries.
"""

from __future__ import annotations

from collections.abc import Callable
from itertools import combinations
from typing import Any

from mltk.core.assertion import assert_true, timed_assertion
from mltk.core.result import Severity, TestResult


def _jaccard(a: set[str], b: set[str]) -> float:
    """Jaccard similarity: |A & B| / |A | B|.

    Returns 0.0 when both sets are empty (no overlap by
    definition — two empty retrievals share nothing).
    """
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


@timed_assertion
def assert_retrieval_consistency(
    retriever_fn: Callable[[str], list[str]],
    paraphrases: list[str],
    min_overlap: float = 0.7,
) -> TestResult:
    """Assert that paraphrased queries retrieve consistent docs.

    Calls *retriever_fn* on each paraphrase, collects the
    returned document IDs, then computes the mean Jaccard
    similarity across all query pairs.

    Args:
        retriever_fn: ``str -> list[str]`` that returns
            document IDs for a query.
        paraphrases: Semantically equivalent queries
            (>= 2 required).
        min_overlap: Minimum mean Jaccard similarity to
            pass (0-1).

    Returns:
        TestResult with overlap stats and per-pair details.

    Example:
        >>> def stub(q):
        ...     return ["doc1", "doc2", "doc3"]
        >>> assert_retrieval_consistency(
        ...     stub,
        ...     ["What is X?", "Explain X"],
        ... )
    """
    name = "llm.behavioral.retrieval_consistency"

    if len(paraphrases) < 2:
        return assert_true(
            False,
            name=name,
            message=(
                "Need >= 2 paraphrases, "
                f"got {len(paraphrases)}"
            ),
            severity=Severity.CRITICAL,
        )

    from mltk.domains.llm._backends import (
        normalize_unicode,
    )

    normalized = [
        normalize_unicode(p) for p in paraphrases
    ]

    # Retrieve doc IDs for each query
    doc_sets: list[set[str]] = []
    doc_lists: list[list[str]] = []
    for query in normalized:
        docs = retriever_fn(query)
        doc_lists.append(docs)
        doc_sets.append(set(docs))

    # All pairs
    pairs = list(
        combinations(range(len(normalized)), 2)
    )
    n_pairs = len(pairs)

    pair_details: list[dict[str, Any]] = []
    worst_jaccard = 1.0
    worst_pair: dict[str, Any] | None = None
    total_jaccard = 0.0

    for i, j in pairs:
        intersection = doc_sets[i] & doc_sets[j]
        union = doc_sets[i] | doc_sets[j]
        jac = _jaccard(doc_sets[i], doc_sets[j])
        total_jaccard += jac

        detail: dict[str, Any] = {
            "query_a": paraphrases[i],
            "query_b": paraphrases[j],
            "jaccard": round(jac, 4),
            "docs_a": sorted(doc_lists[i]),
            "docs_b": sorted(doc_lists[j]),
            "intersection": sorted(intersection),
            "union": sorted(union),
        }
        pair_details.append(detail)

        if jac < worst_jaccard:
            worst_jaccard = jac
            worst_pair = detail

    avg_overlap = (
        total_jaccard / n_pairs if n_pairs > 0
        else 0.0
    )
    passed = avg_overlap >= min_overlap

    message = (
        f"Retrieval consistency: "
        f"{avg_overlap:.4f} >= {min_overlap} "
        f"({n_pairs} pairs)"
        if passed
        else f"Retrieval consistency too low: "
        f"{avg_overlap:.4f} < {min_overlap} "
        f"({n_pairs} pairs)"
    )

    return assert_true(
        passed,
        name=name,
        message=message,
        severity=Severity.CRITICAL,
        method="jaccard",
        avg_overlap=round(avg_overlap, 4),
        min_overlap=min_overlap,
        n_queries=len(paraphrases),
        n_pairs=n_pairs,
        per_pair=pair_details,
        worst_pair=worst_pair,
    )
