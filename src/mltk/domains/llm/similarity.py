"""Semantic similarity -- token-level F1 and embedding cosine for LLM output comparison."""

from __future__ import annotations

from mltk.core.assertion import assert_true, timed_assertion
from mltk.core.result import Severity, TestResult


def _token_f1(reference: str, hypothesis: str) -> float:
    """Compute token-level F1 between two strings.

    Splits on whitespace, computes precision/recall/F1 on token sets.
    Lightweight alternative to BERTScore — no model needed.
    """
    ref_tokens = set(reference.lower().split())
    hyp_tokens = set(hypothesis.lower().split())

    if not ref_tokens and not hyp_tokens:
        return 1.0
    if not ref_tokens or not hyp_tokens:
        return 0.0

    common = ref_tokens & hyp_tokens
    precision = len(common) / len(hyp_tokens)
    recall = len(common) / len(ref_tokens)

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _embedding_cosine(
    references: list[str],
    hypotheses: list[str],
    model_name: str = "all-mpnet-base-v2",
) -> list[float]:
    """Compute cosine similarity between reference and hypothesis embeddings.

    Uses sentence-transformers to encode texts, then computes pairwise
    cosine similarity. Raises ImportError if sentence-transformers is missing.
    """
    from mltk.domains.llm._backends import embedding_cosine_pairs

    return embedding_cosine_pairs(references, hypotheses, model_name)


@timed_assertion
def assert_semantic_similarity(
    references: list[str],
    hypotheses: list[str],
    min_score: float = 0.7,
    method: str = "token",
) -> TestResult:
    """Assert semantic similarity between reference and generated texts.

    Args:
        references: Reference texts (ground truth).
        hypotheses: Model-generated texts.
        min_score: Minimum required average similarity (0-1).
        method: Similarity method -- ``"token"`` (F1 on token overlap) or
            ``"embedding"`` (cosine similarity via sentence-transformers).

    Returns:
        TestResult with average similarity score.

    Example:
        >>> refs = ["The cat sat on the mat"]
        >>> hyps = ["A cat is sitting on a mat"]
        >>> assert_semantic_similarity(refs, hyps, min_score=0.3)
        >>> # With embeddings (requires sentence-transformers):
        >>> assert_semantic_similarity(refs, hyps, min_score=0.7, method="embedding")
    """
    if method == "token":
        scores = [
            _token_f1(ref, hyp)
            for ref, hyp in zip(references, hypotheses, strict=False)
        ]
    elif method == "embedding":
        if not references or not hypotheses:
            scores = []
        else:
            try:
                scores = _embedding_cosine(references, hypotheses)
            except ImportError as exc:
                return assert_true(
                    False, name="llm.similarity",
                    message=str(exc),
                    severity=Severity.CRITICAL,
                )
    else:
        return assert_true(
            False, name="llm.similarity",
            message=f"Unknown method: '{method}'. Supported: 'token', 'embedding'",
            severity=Severity.CRITICAL,
        )

    avg_score = sum(scores) / len(scores) if scores else 0.0

    passed = avg_score >= min_score
    message = (
        f"Similarity ({method}): {avg_score:.4f} >= {min_score}"
        if passed
        else f"Similarity too low: {avg_score:.4f} < {min_score}"
    )

    return assert_true(
        passed, name="llm.similarity", message=message,
        severity=Severity.CRITICAL,
        score=avg_score, min_score=min_score, method=method,
        num_pairs=len(scores),
    )
