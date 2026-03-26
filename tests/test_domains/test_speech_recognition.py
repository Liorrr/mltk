"""Tests for mltk.domains.speech.recognition -- WER and CER assertions.

Speech recognition tests cover two error rate metrics:
1. WER (Word Error Rate): fraction of words incorrectly transcribed
   (substitutions + insertions + deletions) / total reference words.
   Standard metric for ASR system evaluation.
2. CER (Character Error Rate): same formula but at character level.
   More granular -- useful for morphologically rich languages and
   detecting near-miss transcription errors.

Both require the optional jiwer dependency.
"""

import pytest

jiwer = pytest.importorskip("jiwer", reason="jiwer required for WER/CER tests")

from mltk.core.assertion import MltkAssertionError  # noqa: E402
from mltk.domains.speech.recognition import assert_cer, assert_wer  # noqa: E402


class TestWER:
    """Word Error Rate tests.

    Validates that assert_wer correctly computes corpus-level WER via jiwer
    and gates on a maximum error rate threshold.
    """

    def test_wer_pass_low_errors(self) -> None:
        """PASS: Transcription has one word wrong out of two -- WER 0.5.

        WHY: A single word substitution in a short utterance produces 50%
        WER. With a generous threshold of 0.5, this should pass.
        Expected: result.passed is True, wer == 0.5.
        """
        refs = ["hello world"]
        hyps = ["hello word"]
        result = assert_wer(refs, hyps, max_wer=0.5)
        assert result.passed is True
        assert result.details["wer"] == pytest.approx(0.5, abs=0.01)

    def test_wer_pass_perfect(self) -> None:
        """PASS: Perfect transcription -- WER 0.0.

        WHY: When ASR output is identical to reference, WER must be zero.
        This is the baseline test for the metric computation.
        Expected: result.passed is True, wer == 0.0.
        """
        refs = ["the quick brown fox"]
        hyps = ["the quick brown fox"]
        result = assert_wer(refs, hyps, max_wer=0.1)
        assert result.passed is True
        assert result.details["wer"] == pytest.approx(0.0, abs=0.001)

    def test_wer_fail_high_errors(self) -> None:
        """FAIL: Transcription is completely wrong.

        WHY: When every word is wrong, WER approaches or exceeds 1.0.
        This must fail to prevent deploying a broken ASR model.
        Expected: MltkAssertionError raised.
        """
        refs = ["hello world"]
        hyps = ["goodbye moon"]
        with pytest.raises(MltkAssertionError):
            assert_wer(refs, hyps, max_wer=0.1)

    def test_wer_multiple_utterances(self) -> None:
        """PASS: Corpus-level WER across multiple utterances.

        WHY: jiwer computes WER at the corpus level (concatenating all
        references and hypotheses). Testing with multiple samples ensures
        the assertion handles lists correctly.
        Expected: result.passed is True.
        """
        refs = ["hello world", "good morning"]
        hyps = ["hello world", "good morning"]
        result = assert_wer(refs, hyps, max_wer=0.1)
        assert result.passed is True
        assert result.details["num_samples"] == 2


class TestCER:
    """Character Error Rate tests.

    Validates that assert_cer correctly computes corpus-level CER via jiwer
    and gates on a maximum error rate threshold.
    """

    def test_cer_pass_low_errors(self) -> None:
        """PASS: One character wrong in 'helo world' -- low CER.

        WHY: A single missing character in an 11-character reference produces
        a small CER. With threshold 0.1, this should pass.
        Expected: result.passed is True.
        """
        refs = ["hello world"]
        hyps = ["helo world"]
        result = assert_cer(refs, hyps, max_cer=0.15)
        assert result.passed is True
        assert result.details["cer"] < 0.15

    def test_cer_pass_perfect(self) -> None:
        """PASS: Perfect transcription -- CER 0.0.

        WHY: Identical strings must produce zero character error rate.
        Expected: result.passed is True, cer == 0.0.
        """
        refs = ["the quick brown fox"]
        hyps = ["the quick brown fox"]
        result = assert_cer(refs, hyps, max_cer=0.05)
        assert result.passed is True
        assert result.details["cer"] == pytest.approx(0.0, abs=0.001)

    def test_cer_fail_high_errors(self) -> None:
        """FAIL: Completely different text at character level.

        WHY: When nearly every character is wrong, CER is very high.
        Must fail to catch catastrophically broken transcription.
        Expected: MltkAssertionError raised.
        """
        refs = ["hello world"]
        hyps = ["zzzzz zzzzz"]
        with pytest.raises(MltkAssertionError):
            assert_cer(refs, hyps, max_cer=0.05)

    def test_cer_multiple_utterances(self) -> None:
        """PASS: Corpus-level CER across multiple utterances.

        WHY: Like WER, CER is computed at corpus level. Testing multiple
        samples verifies list handling.
        Expected: result.passed is True.
        """
        refs = ["hello", "world"]
        hyps = ["hello", "world"]
        result = assert_cer(refs, hyps, max_cer=0.05)
        assert result.passed is True
        assert result.details["num_samples"] == 2
