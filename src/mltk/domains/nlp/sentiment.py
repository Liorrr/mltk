"""Sentiment analysis assertions -- polarity checks and sentiment drift."""
from __future__ import annotations

import re

from mltk.core.assertion import assert_true, timed_assertion
from mltk.core.result import Severity, TestResult

# Simple keyword-based sentiment scorer
_POSITIVE_WORDS = {
    "good", "great", "excellent", "amazing", "wonderful", "fantastic",
    "happy", "love", "best", "perfect", "awesome", "beautiful",
    "brilliant", "outstanding", "superb", "positive", "nice", "helpful",
    "thank", "thanks", "pleased", "enjoy", "impressive", "recommend",
}

_NEGATIVE_WORDS = {
    "bad", "terrible", "awful", "horrible", "worst", "hate", "poor",
    "disappointing", "negative", "angry", "sad", "fail", "failed",
    "broken", "useless", "waste", "annoying", "frustrating", "ugly",
    "boring", "slow", "expensive", "wrong", "error", "problem",
}


def _sentiment_score(text: str) -> float:
    """Compute simple sentiment score: (pos - neg) / total words.

    Returns float in roughly [-1, 1]. 0 = neutral.
    """
    words = set(re.sub(r"[^\w\s]", "", text.lower()).split())
    pos = len(words & _POSITIVE_WORDS)
    neg = len(words & _NEGATIVE_WORDS)
    total = len(words)
    if total == 0:
        return 0.0
    return (pos - neg) / total


def _classify_sentiment(text: str) -> str:
    """Classify as 'positive', 'negative', or 'neutral'."""
    score = _sentiment_score(text)
    if score > 0.02:
        return "positive"
    elif score < -0.02:
        return "negative"
    return "neutral"


@timed_assertion
def assert_sentiment_positive(
    texts: list[str],
    min_ratio: float = 0.5,
) -> TestResult:
    """Assert at least min_ratio of texts have positive sentiment.

    Uses keyword-based sentiment analysis (no external model).

    Args:
        texts: List of text strings to evaluate.
        min_ratio: Minimum fraction of texts that must be positive (0-1).

    Returns:
        TestResult with sentiment distribution details.

    Example:
        >>> texts = ["Great product!", "Love it!", "Not bad."]
        >>> assert_sentiment_positive(texts, min_ratio=0.5)
    """
    if not texts:
        return assert_true(
            False,
            name="nlp.sentiment_positive",
            message="No texts provided",
            severity=Severity.CRITICAL,
            num_texts=0,
            positive_count=0,
            positive_ratio=0.0,
            min_ratio=min_ratio,
        )

    labels = [_classify_sentiment(t) for t in texts]
    positive_count = labels.count("positive")
    positive_ratio = positive_count / len(texts)

    passed = positive_ratio >= min_ratio
    message = (
        f"Positive ratio: {positive_ratio:.3f} >= {min_ratio} "
        f"({positive_count}/{len(texts)} positive)"
        if passed
        else f"Positive ratio: {positive_ratio:.3f} < {min_ratio} "
        f"({positive_count}/{len(texts)} positive)"
    )

    return assert_true(
        passed,
        name="nlp.sentiment_positive",
        message=message,
        severity=Severity.CRITICAL,
        num_texts=len(texts),
        positive_count=positive_count,
        negative_count=labels.count("negative"),
        neutral_count=labels.count("neutral"),
        positive_ratio=positive_ratio,
        min_ratio=min_ratio,
    )


@timed_assertion
def assert_no_sentiment_drift(
    ref_texts: list[str],
    cur_texts: list[str],
    max_drift: float = 0.1,
) -> TestResult:
    """Assert sentiment distribution hasn't shifted between datasets.

    Compares positive/negative/neutral ratios between ref and cur.
    Drift = max absolute difference across categories.

    Args:
        ref_texts: Reference (baseline) texts.
        cur_texts: Current texts to compare against reference.
        max_drift: Maximum allowed absolute ratio shift (0-1).

    Returns:
        TestResult with per-category ratios and drift details.

    Example:
        >>> ref = ["Great service!", "Love this product."]
        >>> cur = ["Great service!", "Amazing quality."]
        >>> assert_no_sentiment_drift(ref, cur, max_drift=0.1)
    """
    def _distribution(texts: list[str]) -> dict[str, float]:
        if not texts:
            return {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
        labels = [_classify_sentiment(t) for t in texts]
        n = len(texts)
        return {
            "positive": labels.count("positive") / n,
            "negative": labels.count("negative") / n,
            "neutral": labels.count("neutral") / n,
        }

    ref_dist = _distribution(ref_texts)
    cur_dist = _distribution(cur_texts)

    diffs = {cat: abs(cur_dist[cat] - ref_dist[cat]) for cat in ref_dist}
    max_observed_drift = max(diffs.values()) if diffs else 0.0

    passed = max_observed_drift <= max_drift
    message = (
        f"Sentiment drift: {max_observed_drift:.3f} <= {max_drift} (stable)"
        if passed
        else f"Sentiment drift: {max_observed_drift:.3f} > {max_drift} (shifted)"
    )

    return assert_true(
        passed,
        name="nlp.sentiment_drift",
        message=message,
        severity=Severity.CRITICAL,
        max_drift=max_drift,
        observed_drift=max_observed_drift,
        ref_distribution=ref_dist,
        cur_distribution=cur_dist,
        category_diffs=diffs,
        num_ref=len(ref_texts),
        num_cur=len(cur_texts),
    )
