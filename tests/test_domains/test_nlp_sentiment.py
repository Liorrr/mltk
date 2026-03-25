"""Tests for mltk.domains.nlp.sentiment -- sentiment analysis assertions.

Covers two public assertions:
1. assert_sentiment_positive -- gates on minimum positive-sentiment ratio
2. assert_no_sentiment_drift -- detects distribution shift across datasets

Both use a pure-Python keyword-based scorer; no external dependencies.
"""

import pytest

from mltk.core.assertion import MltkAssertionError
from mltk.domains.nlp.sentiment import (
    _sentiment_score,
    assert_no_sentiment_drift,
    assert_sentiment_positive,
)


class TestAssertSentimentPositive:
    """Tests for assert_sentiment_positive."""

    def test_sentiment_positive_pass(self) -> None:
        """SCENARIO: Majority of texts carry clear positive keywords.
        WHY: A product review set that is mostly positive should pass
        the default 0.5 threshold without any manual tuning.
        EXPECTED: result.passed is True, positive_count == 4.
        """
        texts = [
            "This is a great product, I love it!",
            "Absolutely amazing and wonderful experience.",
            "Fantastic quality, highly recommend.",
            "Outstanding and brilliant service.",
            "It was okay.",  # neutral -- 4/5 positive = 0.8 >= 0.5
        ]
        result = assert_sentiment_positive(texts, min_ratio=0.5)
        assert result.passed is True
        assert result.details["positive_count"] == 4
        assert result.details["positive_ratio"] >= 0.5

    def test_sentiment_positive_fail(self) -> None:
        """SCENARIO: Texts are dominated by negative keywords.
        WHY: A batch of customer complaints should fail a positive-sentiment
        gate, triggering an alert that output quality has degraded.
        EXPECTED: MltkAssertionError raised.
        """
        texts = [
            "This is terrible and awful.",
            "Horrible experience, worst product ever.",
            "Broken and completely useless.",
            "Very bad, frustrating and disappointing.",
            "Great job.",  # one positive among four negatives
        ]
        with pytest.raises(MltkAssertionError) as exc:
            assert_sentiment_positive(texts, min_ratio=0.5)
        assert "positive" in str(exc.value).lower()

    def test_sentiment_mixed_boundary(self) -> None:
        """SCENARIO: Exactly 50% positive texts at a min_ratio=0.5 threshold.
        WHY: Boundary conditions (ratio == threshold) must pass, not fail.
        A strict > comparison would incorrectly reject equal-to-threshold inputs.
        EXPECTED: result.passed is True (0.5 >= 0.5).
        """
        texts = [
            "Great product, I love it.",
            "Amazing and wonderful.",
            "Terrible and awful.",
            "Horrible and broken.",
        ]
        result = assert_sentiment_positive(texts, min_ratio=0.5)
        assert result.passed is True
        assert result.details["positive_ratio"] == pytest.approx(0.5)

    def test_empty_texts_fails(self) -> None:
        """SCENARIO: An empty list is passed as input.
        WHY: Empty inputs are a common upstream data pipeline bug. The assertion
        must fail loudly (not silently pass with ratio=0) so the caller is alerted.
        EXPECTED: MltkAssertionError raised.
        """
        with pytest.raises(MltkAssertionError):
            assert_sentiment_positive([], min_ratio=0.0)

    def test_neutral_texts(self) -> None:
        """SCENARIO: All texts lack any sentiment keywords (purely neutral).
        WHY: Neutral text should score 0 positive, which fails a 0.5 min_ratio gate.
        This validates that absent keywords do not inflate the positive count.
        EXPECTED: MltkAssertionError raised, positive_count == 0.
        """
        texts = [
            "The file was uploaded.",
            "The document contains three pages.",
            "A request was submitted at noon.",
        ]
        with pytest.raises(MltkAssertionError) as exc:
            assert_sentiment_positive(texts, min_ratio=0.5)
        result = exc.value.result
        assert result.details["positive_count"] == 0

    def test_details_metadata(self) -> None:
        """SCENARIO: Result carries full breakdown of positive/negative/neutral counts.
        WHY: Downstream monitoring dashboards need per-category counts, not just pass/fail.
        Verifying detail keys ensures the contract is stable across versions.
        EXPECTED: details contains positive_count, negative_count, neutral_count, num_texts.
        """
        texts = ["Great!", "Terrible.", "Okay."]
        try:
            result = assert_sentiment_positive(texts, min_ratio=0.0)
        except MltkAssertionError as exc:
            result = exc.result
        for key in ("positive_count", "negative_count", "neutral_count", "num_texts"):
            assert key in result.details


class TestAssertNoSentimentDrift:
    """Tests for assert_no_sentiment_drift."""

    def test_no_sentiment_drift_stable(self) -> None:
        """SCENARIO: Reference and current datasets have the same sentiment distribution.
        WHY: A model that stays consistent over time should not trigger drift alerts.
        Using identical datasets guarantees zero drift across all categories.
        EXPECTED: result.passed is True, observed_drift == 0.0.
        """
        texts = [
            "Great experience, highly recommend.",
            "Terrible product, completely broken.",
            "The item arrived on time.",
        ]
        result = assert_no_sentiment_drift(texts, texts, max_drift=0.1)
        assert result.passed is True
        assert result.details["observed_drift"] == pytest.approx(0.0)

    def test_sentiment_drift_detected(self) -> None:
        """SCENARIO: Production outputs shift from mostly positive to mostly negative.
        WHY: A degraded model might start generating more negative responses over time.
        The assertion must catch this shift and raise before it affects end users.
        EXPECTED: MltkAssertionError raised, observed_drift > max_drift.
        """
        ref_texts = [
            "Great service, I love it!",
            "Amazing product, highly recommend.",
            "Fantastic quality, absolutely brilliant.",
            "Outstanding and wonderful.",
        ]
        cur_texts = [
            "Terrible service, I hate it.",
            "Horrible product, very bad.",
            "Awful quality, completely broken.",
            "Worst experience, very frustrating.",
        ]
        with pytest.raises(MltkAssertionError) as exc:
            assert_no_sentiment_drift(ref_texts, cur_texts, max_drift=0.1)
        result = exc.value.result
        assert result.details["observed_drift"] > 0.1

    def test_drift_details_contain_distributions(self) -> None:
        """SCENARIO: The result exposes both reference and current distributions.
        WHY: Operators diagnosing a drift alert need to see the before/after breakdown,
        not just the magnitude of drift. This verifies the contract is complete.
        EXPECTED: result.details has ref_distribution and cur_distribution dicts.
        """
        ref = ["Great product.", "Wonderful service."]
        cur = ["Bad product.", "Terrible service."]
        try:
            result = assert_no_sentiment_drift(ref, cur, max_drift=0.99)
        except MltkAssertionError as exc:
            result = exc.result
        assert "ref_distribution" in result.details
        assert "cur_distribution" in result.details
        dist = result.details["ref_distribution"]
        assert set(dist.keys()) == {"positive", "negative", "neutral"}

    def test_drift_empty_cur(self) -> None:
        """SCENARIO: Current dataset is empty while reference is non-empty.
        WHY: A data pipeline failure could deliver an empty batch. The assertion
        must detect the resulting distribution shift (all categories differ).
        EXPECTED: MltkAssertionError raised due to drift.
        """
        ref = ["Great!", "Amazing!", "Wonderful!"]
        cur: list[str] = []
        with pytest.raises(MltkAssertionError):
            assert_no_sentiment_drift(ref, cur, max_drift=0.05)


class TestSentimentScoreFunction:
    """Unit tests for the internal _sentiment_score helper."""

    def test_score_function_positive(self) -> None:
        """SCENARIO: Text with multiple positive keywords yields a positive score.
        WHY: The scorer is the foundation of both public assertions. If it returns
        wrong polarity on obvious input, all downstream results are wrong.
        EXPECTED: score > 0.
        """
        score = _sentiment_score("This is a great and amazing product I love.")
        assert score > 0

    def test_score_function_negative(self) -> None:
        """SCENARIO: Text with multiple negative keywords yields a negative score.
        WHY: Symmetric to the positive test -- the scorer must distinguish polarity
        direction, not just detect the presence of sentiment keywords.
        EXPECTED: score < 0.
        """
        score = _sentiment_score("This is terrible, awful, and completely broken.")
        assert score < 0

    def test_score_function_empty_string(self) -> None:
        """SCENARIO: Empty string input to the scorer.
        WHY: Division by zero protection -- an empty text has no words, so
        total=0. The scorer must guard this and return 0.0, not raise.
        EXPECTED: score == 0.0 with no exception.
        """
        score = _sentiment_score("")
        assert score == 0.0

    def test_score_function_neutral(self) -> None:
        """SCENARIO: Text contains only stopwords and filler with no sentiment keywords.
        WHY: Purely neutral text (timestamps, filenames, IDs) should not be
        miscategorised. A score near 0 confirms the keyword lists are not over-broad.
        EXPECTED: abs(score) <= 0.02 (within neutral band).
        """
        score = _sentiment_score("The file was processed at three in the afternoon.")
        assert abs(score) <= 0.02
