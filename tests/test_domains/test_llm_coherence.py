"""Tests for mltk.domains.llm.coherence — internal text coherence assertions."""

import pytest

from mltk.core.assertion import MltkAssertionError
from mltk.domains.llm.coherence import assert_coherence


class TestCoherenceConsistent:
    """Coherent multi-sentence text — related topics across sentences."""

    def test_coherence_consistent(self) -> None:
        # SCENARIO: A well-structured paragraph where each sentence builds on the last.
        # WHY: Consecutive sentences share vocabulary (brain/neurons/information) but
        #      Jaccard divides by union — a large vocabulary yields a modest score (~0.09).
        # EXPECTED: score > 0 (coherent) and passes min_score=0.05.
        text = (
            "Neural networks are computational models inspired by the human brain. "
            "The brain uses neurons to process and transmit information. "
            "Information is encoded as patterns of activation across neuron layers."
        )
        result = assert_coherence(text, min_score=0.05)
        assert result.passed is True
        assert result.details["score"] >= 0.05

    def test_coherence_score_in_details(self) -> None:
        # SCENARIO: Confirm result carries the numeric score for downstream use.
        # WHY: Callers log scores to dashboards; must be in result.details.
        # EXPECTED: result.details["score"] is a float in [0, 1].
        text = (
            "Python is a high-level programming language. "
            "The language is known for its simple and readable syntax. "
            "Readable syntax makes Python great for beginners."
        )
        result = assert_coherence(text, min_score=0.05)
        assert "score" in result.details
        assert isinstance(result.details["score"], float)
        assert 0.0 <= result.details["score"] <= 1.0

    def test_coherence_result_has_timing(self) -> None:
        # SCENARIO: Every assertion is wrapped with @timed_assertion.
        # WHY: duration_ms must be populated for performance tracking.
        # EXPECTED: duration_ms > 0.
        text = "Machine learning improves with more data. Data quality matters too."
        result = assert_coherence(text, min_score=0.0)
        assert result.duration_ms > 0


class TestCoherenceRandom:
    """Incoherent text — unrelated sentences with no shared vocabulary."""

    def test_coherence_random(self) -> None:
        # SCENARIO: Sentences drawn from completely different domains with no shared words.
        # WHY: Zero token overlap between consecutive pairs → mean Jaccard = 0.0 < 0.3.
        # EXPECTED: raises MltkAssertionError.
        text = (
            "Quantum entanglement is a phenomenon in particle physics. "
            "The Renaissance period transformed European art and culture. "
            "Submarine volcanoes form new ocean floor through lava extrusion."
        )
        with pytest.raises(MltkAssertionError) as exc_info:
            assert_coherence(text, min_score=0.3)
        assert exc_info.value.result.details["score"] < 0.3


class TestCoherenceSingleSentence:
    """Single-sentence and trivially coherent edge cases."""

    def test_coherence_single_sentence(self) -> None:
        # SCENARIO: Text is a single sentence with no consecutive pair to evaluate.
        # WHY: No pairs → trivially coherent; must not crash or return 0.
        # EXPECTED: score == 1.0 and passes.
        text = "The sky is blue on a clear day."
        result = assert_coherence(text, min_score=0.5)
        assert result.passed is True
        assert result.details["score"] == 1.0
        assert result.details["sentence_count"] == 1

    def test_coherence_two_related_sentences(self) -> None:
        # SCENARIO: Minimum multi-sentence case — exactly one consecutive pair.
        # WHY: Score computed from a single pair; related sentences should pass.
        # EXPECTED: passes with sentence_count == 2 and pairs_evaluated == 1.
        text = "Dogs are loyal animals. Animals like dogs make great companions."
        result = assert_coherence(text, min_score=0.1)
        assert result.passed is True
        assert result.details["sentence_count"] == 2
        assert result.details["pairs_evaluated"] == 1


class TestCoherenceEdgeCases:
    """Empty and boundary inputs."""

    def test_coherence_empty(self) -> None:
        # SCENARIO: Empty string passed as text.
        # WHY: No sentences after splitting → handled as single/zero-sentence case.
        # EXPECTED: Does not raise TypeError or ZeroDivisionError; returns a TestResult.
        result = assert_coherence("", min_score=0.0)
        assert result.passed is True  # trivially coherent

    def test_coherence_score_range(self) -> None:
        # SCENARIO: Several realistic paragraphs — check score is always in [0.0, 1.0].
        # WHY: Jaccard similarity is bounded [0, 1]; the mean must also be in range.
        # EXPECTED: score between 0.0 and 1.0 inclusive for all inputs.
        samples = [
            "A. B. C.",  # single-letter sentences — minimal overlap
            "The cat sat on the mat. The mat was on the floor.",
            "Science studies the natural world. The natural world includes physics.",
            "Red cars go fast. Fast food is popular. Popular culture is everywhere.",
        ]
        for text in samples:
            result = assert_coherence(text, min_score=0.0)
            score = result.details["score"]
            assert 0.0 <= score <= 1.0, f"Score out of range for: {text!r}"
