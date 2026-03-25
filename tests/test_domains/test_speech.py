"""Tests for mltk.domains.speech -- speech testing assertions.

Speech domain tests cover two critical areas:
1. Real-Time Factor (RTF): ensures ASR/TTS processing is fast enough for
   live applications. RTF > 1.0 means slower than real-time (unusable for
   live transcription or voice assistants).
2. Accent coverage: detects bias in ASR models that work well for some
   accents but poorly for others. This is both a fairness issue and a
   quality issue for global products.
"""

import time

import pytest

from mltk.core.assertion import MltkAssertionError
from mltk.domains.speech.performance import assert_accent_coverage, assert_rtf


class TestRTF:
    """Real-Time Factor tests.

    RTF = processing_time / audio_duration. RTF < 1.0 means the system
    processes audio faster than real-time (required for live streaming).
    """

    def test_fast_processing(self) -> None:
        """PASS: Processing at 1% of real-time (RTF ~0.01).

        WHY: A speech model used for live transcription must process audio
        faster than it arrives. This simulates a fast model that processes
        each second of audio in ~10ms.
        Expected: result.passed is True, RTF < 1.0.
        """
        def fast_process(duration: float) -> None:
            time.sleep(duration * 0.01)

        result = assert_rtf(fast_process, [1.0, 2.0, 3.0], max_rtf=1.0)
        assert result.passed is True
        assert result.details["rtf"] < 1.0

    def test_slow_processing_fails(self) -> None:
        """FAIL: Processing slower than real-time.

        WHY: A model that takes 100ms to process 10ms of audio (RTF=10)
        cannot be used for live applications. It would fall further behind
        with each second of audio, creating unbounded latency.
        Expected: MltkAssertionError raised.
        """
        def slow_process(duration: float) -> None:
            time.sleep(0.1)

        with pytest.raises(MltkAssertionError):
            assert_rtf(slow_process, [0.01], max_rtf=0.5)


class TestAccentCoverage:
    """Accent fairness tests.

    Validates that ASR word error rate (WER) is consistent across accents.
    A large gap between best and worst accent indicates the model was
    trained primarily on one accent and will fail for underrepresented users.
    """

    def test_equal_performance(self) -> None:
        """PASS: WER difference across accents is within 2% max gap.

        WHY: US (5%), British (6%), Australian (5.5%) are all within 1%
        of each other. The model treats all accents fairly. This is the
        target state for inclusive speech products.
        Expected: result.passed is True.
        """
        wer_by_accent = {
            "US_English": 0.05,
            "British_English": 0.06,
            "Australian": 0.055,
        }
        result = assert_accent_coverage(wer_by_accent, max_gap=0.02)
        assert result.passed is True

    def test_accent_bias_detected(self) -> None:
        """FAIL: 22% WER gap between US English (3%) and Non-Native (25%).

        WHY: Non-native speakers get 8x higher error rate than native.
        This is a severe fairness violation -- the model is essentially
        unusable for non-native speakers. Common in models trained only
        on native English data.
        Expected: MltkAssertionError with "bias" in message.
        """
        wer_by_accent = {
            "US_English": 0.03,
            "Non_Native": 0.25,
        }
        with pytest.raises(MltkAssertionError) as exc:
            assert_accent_coverage(wer_by_accent, max_gap=0.05)
        assert "bias" in str(exc.value).lower()

    def test_single_accent(self) -> None:
        """PASS: Only one accent in data -- gap comparison not applicable.

        WHY: A single-accent evaluation cannot compute between-accent
        differences. This should pass vacuously rather than error.
        Expected: result.passed is True.
        """
        result = assert_accent_coverage({"US_English": 0.05}, max_gap=0.05)
        assert result.passed is True

    def test_gap_details(self) -> None:
        """PASS: Result details include best and worst accent identifiers.

        WHY: When investigating accent bias, engineers need to know WHICH
        accents are best/worst performing. The details dict must include
        these identifiers for the HTML report and debugging.
        Expected: best_accent="A" (WER 0.02), worst_accent="B" (WER 0.04).
        """
        wer_by_accent = {"A": 0.02, "B": 0.04, "C": 0.03}
        result = assert_accent_coverage(wer_by_accent, max_gap=0.05)
        assert result.details["best_accent"] == "A"
        assert result.details["worst_accent"] == "B"
