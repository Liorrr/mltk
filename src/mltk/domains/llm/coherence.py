"""Coherence checking — internal consistency of generated text."""

from __future__ import annotations

import re

from mltk.core.assertion import assert_true, timed_assertion
from mltk.core.result import Severity, TestResult
from mltk.domains.llm._utils import _tokenize


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences on .!? boundaries."""
    return [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]


@timed_assertion
def assert_coherence(
    text: str,
    min_score: float = 0.3,
) -> TestResult:
    """Assert text is internally coherent.

    Score = mean token overlap between consecutive sentence pairs.
    High score = sentences are related to each other (coherent).
    Low score = random unrelated sentences (incoherent).

    For single-sentence text (or empty after splitting), returns score 1.0
    because there are no consecutive pairs to be incoherent about.

    Args:
        text: The generated text to evaluate.
        min_score: Minimum coherence score required (default 0.3).

    Returns:
        TestResult with coherence score and sentence count.

    Example:
        >>> para = (
        ...     "Neural networks are inspired by the brain. "
        ...     "They learn patterns from training data. "
        ...     "Training data is collected from real-world examples."
        ... )
        >>> assert_coherence(para, min_score=0.1)
    """
    sentences = _split_sentences(text)

    # Single sentence (or empty) is trivially coherent — no pair to compare.
    if len(sentences) <= 1:
        score = 1.0
        message = (
            f"Coherence: {score:.4f} >= {min_score} "
            f"(single sentence — trivially coherent)"
        )
        return assert_true(
            True,
            name="llm.coherence",
            message=message,
            severity=Severity.CRITICAL,
            score=score,
            min_score=min_score,
            sentence_count=len(sentences),
        )

    # Compute mean Jaccard-style overlap between every consecutive pair.
    pair_scores: list[float] = []
    for i in range(len(sentences) - 1):
        tok_a = _tokenize(sentences[i])
        tok_b = _tokenize(sentences[i + 1])
        if not tok_a or not tok_b:
            # At least one empty sentence — count as 0 overlap.
            pair_scores.append(0.0)
            continue
        union = tok_a | tok_b
        overlap = tok_a & tok_b
        # Jaccard similarity: |A ∩ B| / |A ∪ B|
        pair_scores.append(len(overlap) / len(union))

    score = sum(pair_scores) / len(pair_scores) if pair_scores else 1.0
    passed = score >= min_score

    message = (
        f"Coherence: {score:.4f} >= {min_score} "
        f"({len(sentences)} sentences, {len(pair_scores)} pairs evaluated)"
        if passed
        else f"Low coherence: {score:.4f} < {min_score} "
        f"({len(sentences)} sentences, {len(pair_scores)} pairs evaluated)"
    )

    return assert_true(
        passed,
        name="llm.coherence",
        message=message,
        severity=Severity.CRITICAL,
        score=score,
        min_score=min_score,
        sentence_count=len(sentences),
        pairs_evaluated=len(pair_scores),
    )
