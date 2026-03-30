"""BERTScore assertion — token-level semantic similarity via embeddings."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np

from mltk.core.assertion import assert_true, timed_assertion
from mltk.core.result import Severity, TestResult


@timed_assertion
def assert_bertscore(
    reference_embeddings: Any,
    hypothesis_embeddings: Any,
    min_f1: float = 0.5,
    suppress_warnings: bool = False,
) -> TestResult:
    """Assert BERTScore F1 meets threshold.

    Takes pre-computed token embeddings (not raw text -- user
    handles tokenization/encoding). Uses Rust backend when
    available for 10-100x speedup over the pure-numpy fallback.

    .. warning::
       BERTScore has known blind spots: (1) antonyms score
       nearly identical to correct matches, (2) numerical
       values are essentially random (~48% accuracy),
       (3) entity swaps receive minimal penalty.  A warning
       is emitted when F1 >= 0.95 to flag possible false
       confidence.  Suppress with ``suppress_warnings=True``.

    Args:
        reference_embeddings: (N, D) array of reference token
            embeddings.
        hypothesis_embeddings: (M, D) array of hypothesis
            token embeddings.
        min_f1: Minimum required F1 score (0--1). Default 0.5.
        suppress_warnings: If ``True``, suppress the high-F1
            limitation warning.

    Returns:
        TestResult with precision, recall, f1, and threshold
        details.

    Raises:
        MltkAssertionError: When f1 < min_f1 (CRITICAL).

    Example:
        >>> import numpy as np
        >>> ref = np.eye(4)   # 4 orthonormal token embeddings
        >>> hyp = np.eye(4)   # identical hypothesis
        >>> result = assert_bertscore(ref, hyp, min_f1=0.9)
        >>> assert result.passed
    """
    ref = np.asarray(
        reference_embeddings, dtype=np.float64,
    )
    hyp = np.asarray(
        hypothesis_embeddings, dtype=np.float64,
    )

    if ref.size == 0 or hyp.size == 0:
        return assert_true(
            False,
            name="llm.bertscore",
            message=(
                "Cannot compute BERTScore on empty "
                "embeddings"
            ),
            severity=Severity.CRITICAL,
        )

    # Ensure 2-D
    if ref.ndim == 1:
        ref = ref.reshape(1, -1)
    if hyp.ndim == 1:
        hyp = hyp.reshape(1, -1)

    try:
        from mltk._rust import (
            bertscore_precision_recall as _bertscore_pr,
        )

        precision, recall, f1 = _bertscore_pr(
            ref.tolist(), hyp.tolist(),
        )
    except (ImportError, Exception):
        precision, recall, f1 = _bertscore_numpy(ref, hyp)

    if f1 >= 0.95 and not suppress_warnings:
        warnings.warn(
            "BERTScore F1 >= 0.95 may indicate false "
            "confidence. Known limitations: "
            "(1) antonyms score nearly identical to "
            "correct matches ('best' vs 'worst'), "
            "(2) numerical values are essentially "
            "random (48% accuracy), "
            "(3) entity swaps receive minimal penalty."
            " Consider using NLI-based evaluation for"
            " factual content. "
            "Suppress with suppress_warnings=True.",
            UserWarning,
            stacklevel=3,
        )

    passed = f1 >= min_f1
    message = (
        f"BERTScore F1: {f1:.4f} >= {min_f1} "
        f"(P={precision:.4f}, R={recall:.4f})"
        if passed
        else f"BERTScore F1 too low: "
        f"{f1:.4f} < {min_f1} "
        f"(P={precision:.4f}, R={recall:.4f})"
    )

    return assert_true(
        passed,
        name="llm.bertscore",
        message=message,
        severity=Severity.CRITICAL,
        precision=precision,
        recall=recall,
        f1=f1,
        min_f1=min_f1,
        ref_tokens=ref.shape[0],
        hyp_tokens=hyp.shape[0],
        embedding_dim=ref.shape[1],
    )


def _bertscore_numpy(ref: np.ndarray, hyp: np.ndarray) -> tuple[float, float, float]:
    """Pure-numpy BERTScore computation used when Rust is unavailable."""

    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a < 1e-10 or norm_b < 1e-10:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    # Precision: for each hyp token, max sim over all ref tokens
    precision_scores = [
        max(_cosine(hyp_tok, ref_tok) for ref_tok in ref) for hyp_tok in hyp
    ]
    precision = float(np.mean(precision_scores)) if precision_scores else 0.0

    # Recall: for each ref token, max sim over all hyp tokens
    recall_scores = [
        max(_cosine(ref_tok, hyp_tok) for hyp_tok in hyp) for ref_tok in ref
    ]
    recall = float(np.mean(recall_scores)) if recall_scores else 0.0

    f1 = (
        (2.0 * precision * recall / (precision + recall))
        if (precision + recall) > 0
        else 0.0
    )
    return (precision, recall, f1)
