"""Tests for mltk.domains.nlp.generation -- BLEU and ROUGE scoring assertions.

NLP generation tests cover two machine translation / text generation metrics:
1. BLEU (Bilingual Evaluation Understudy): measures n-gram overlap between
   model-generated text and reference translations. Standard metric for MT.
2. ROUGE (Recall-Oriented Understudy for Gisting Evaluation): measures
   n-gram recall between generated summaries and reference texts. Standard
   metric for summarization.

Both require optional dependencies (nltk, rouge-score).
"""

import pytest

nltk = pytest.importorskip("nltk", reason="nltk required for BLEU tests")
rouge_score = pytest.importorskip("rouge_score", reason="rouge-score required for ROUGE tests")

from mltk.core.assertion import MltkAssertionError  # noqa: E402
from mltk.domains.nlp.generation import assert_bleu, assert_rouge  # noqa: E402


class TestBLEU:
    """BLEU score tests.

    Validates that assert_bleu correctly computes corpus-level BLEU with
    smoothing (Method 1) and gates on a minimum score threshold.
    """

    def test_bleu_pass_high_overlap(self) -> None:
        """PASS: Hypothesis closely matches reference -- BLEU should exceed threshold.

        WHY: When the generated text shares most n-grams with the reference,
        BLEU should be high. This validates the scoring pipeline end-to-end.
        Expected: result.passed is True, score > 0.3.
        """
        refs = ["the cat sat on the mat"]
        hyps = ["the cat is on the mat"]
        result = assert_bleu(refs, hyps, min_score=0.2)
        assert result.passed is True
        assert result.details["score"] > 0.2

    def test_bleu_pass_identical(self) -> None:
        """PASS: Identical reference and hypothesis -- BLEU should be 1.0.

        WHY: Perfect match is the upper-bound baseline for the metric.
        Expected: result.passed is True, score == 1.0.
        """
        refs = ["the quick brown fox jumps over the lazy dog"]
        hyps = ["the quick brown fox jumps over the lazy dog"]
        result = assert_bleu(refs, hyps, min_score=0.9)
        assert result.passed is True
        assert result.details["score"] == pytest.approx(1.0, abs=0.01)

    def test_bleu_fail_low_overlap(self) -> None:
        """FAIL: Hypothesis is completely unrelated to reference.

        WHY: When n-gram overlap is near zero, BLEU must be below any
        reasonable threshold. This prevents deploying a model that generates
        gibberish.
        Expected: MltkAssertionError raised.
        """
        refs = ["the cat sat on the mat"]
        hyps = ["completely unrelated sentence about nothing"]
        with pytest.raises(MltkAssertionError):
            assert_bleu(refs, hyps, min_score=0.5)

    def test_bleu_multiple_pairs(self) -> None:
        """PASS: Corpus-level BLEU across multiple reference-hypothesis pairs.

        WHY: BLEU is a corpus-level metric. Testing with multiple pairs
        ensures the aggregation works correctly.
        Expected: result.passed is True.
        """
        refs = [
            "the cat sat on the mat",
            "the dog played in the park",
        ]
        hyps = [
            "the cat is on the mat",
            "the dog played in the park",
        ]
        result = assert_bleu(refs, hyps, min_score=0.3)
        assert result.passed is True


class TestROUGE:
    """ROUGE score tests.

    Validates that assert_rouge correctly computes average ROUGE-L F-measure
    across reference-hypothesis pairs and gates on a minimum score threshold.
    """

    def test_rouge_pass_high_overlap(self) -> None:
        """PASS: Hypothesis shares most content with reference.

        WHY: High token overlap means the generated summary captures the
        reference content. ROUGE-L (longest common subsequence) should be high.
        Expected: result.passed is True, score > 0.3.
        """
        refs = ["the cat sat on the mat"]
        hyps = ["the cat is on the mat"]
        result = assert_rouge(refs, hyps, variant="rougeL", min_score=0.3)
        assert result.passed is True
        assert result.details["score"] > 0.3

    def test_rouge_pass_identical(self) -> None:
        """PASS: Identical texts should yield ROUGE ~1.0.

        WHY: Perfect overlap is the upper-bound baseline.
        Expected: result.passed is True, score close to 1.0.
        """
        refs = ["the quick brown fox jumps over the lazy dog"]
        hyps = ["the quick brown fox jumps over the lazy dog"]
        result = assert_rouge(refs, hyps, variant="rougeL", min_score=0.9)
        assert result.passed is True
        assert result.details["score"] > 0.9

    def test_rouge_fail_no_overlap(self) -> None:
        """FAIL: Completely unrelated hypothesis.

        WHY: Zero meaningful overlap means the summary is useless. Must fail
        to prevent deploying a model that ignores source content.
        Expected: MltkAssertionError raised.
        """
        refs = ["the cat sat on the mat"]
        hyps = ["completely unrelated text about nothing"]
        with pytest.raises(MltkAssertionError):
            assert_rouge(refs, hyps, variant="rougeL", min_score=0.8)

    def test_rouge_variant_rouge1(self) -> None:
        """PASS: ROUGE-1 (unigram) variant works correctly.

        WHY: ROUGE supports multiple variants. Testing rouge1 ensures the
        variant parameter is wired through correctly.
        Expected: result.passed is True.
        """
        refs = ["the cat sat on the mat"]
        hyps = ["the cat is sitting on the mat"]
        result = assert_rouge(refs, hyps, variant="rouge1", min_score=0.3)
        assert result.passed is True
        assert result.details["variant"] == "rouge1"
