"""Tests for mltk.domains.llm.summarization -- summarization evaluation."""

import pytest

from mltk.core.assertion import MltkAssertionError
from mltk.domains.llm.summarization import (
    assert_summary_compression,
    assert_summary_coverage,
    assert_summary_faithfulness,
)

# ── Coverage ─────────────────────────────────────────────────────────


class TestCoveragePass:
    """Good summaries that preserve key source content."""

    def test_good_summary_passes(self) -> None:
        # SCENARIO: Summary retains most source vocabulary.
        # WHY: Token overlap is high when key terms are preserved.
        #   _tokenize uses exact-match sets, so "train" != "trains".
        # EXPECTED: coverage >= 0.3 and assertion passes.
        source = (
            "Machine learning uses data to train predictive models. "
            "These models generalize patterns from training examples."
        )
        summary = (
            "Machine learning uses data to train models."
        )
        result = assert_summary_coverage(
            source, summary, min_coverage=0.3,
        )
        assert result.passed is True
        assert result.details["coverage"] >= 0.3
        assert result.name == "llm.summarization.coverage"

    def test_identical_text_gives_full_coverage(self) -> None:
        # SCENARIO: Summary is identical to source.
        # WHY: All source tokens are in the summary -> coverage = 1.0.
        # EXPECTED: coverage == 1.0.
        text = "The quick brown fox jumps over the lazy dog."
        result = assert_summary_coverage(text, text, min_coverage=0.5)
        assert result.passed is True
        assert result.details["coverage"] == 1.0

    def test_result_has_timing(self) -> None:
        # SCENARIO: @timed_assertion sets duration_ms.
        # EXPECTED: duration_ms > 0.
        result = assert_summary_coverage("abc", "abc", min_coverage=0.0)
        assert result.duration_ms > 0


class TestCoverageFail:
    """Summaries that miss too much source content."""

    def test_empty_summary_fails(self) -> None:
        # SCENARIO: Summary is empty so no source tokens are covered.
        # WHY: coverage = 0/N = 0.0 < min_coverage.
        # EXPECTED: raises MltkAssertionError.
        source = "Neural networks process information in layers."
        with pytest.raises(MltkAssertionError) as exc_info:
            assert_summary_coverage(source, "", min_coverage=0.3)
        assert exc_info.value.result.details["coverage"] == 0.0
        assert exc_info.value.result.details["common_tokens"] == 0

    def test_unrelated_summary_fails(self) -> None:
        # SCENARIO: Summary uses completely different vocabulary.
        # WHY: Zero token overlap between source and summary.
        # EXPECTED: coverage == 0.0 and assertion fails.
        source = "Photosynthesis converts sunlight into chemical energy."
        summary = "The stock market rallied yesterday."
        with pytest.raises(MltkAssertionError):
            assert_summary_coverage(source, summary, min_coverage=0.3)


class TestCoverageEdge:
    """Edge cases for coverage."""

    def test_empty_source_passes(self) -> None:
        # SCENARIO: Source is empty -- nothing to cover.
        # WHY: Trivially passes; there is no content to miss.
        # EXPECTED: coverage = 1.0 and passes.
        result = assert_summary_coverage("", "some summary", min_coverage=0.5)
        assert result.passed is True
        assert result.details["coverage"] == 1.0

    def test_both_empty_passes(self) -> None:
        # SCENARIO: Both source and summary are empty.
        # EXPECTED: Trivially passes.
        result = assert_summary_coverage("", "", min_coverage=0.0)
        assert result.passed is True

    def test_single_word_source(self) -> None:
        # SCENARIO: Source is a single word repeated in summary.
        # WHY: One token in source, same token in summary -> 1.0.
        # EXPECTED: coverage = 1.0.
        result = assert_summary_coverage("hello", "hello", min_coverage=0.5)
        assert result.passed is True
        assert result.details["coverage"] == 1.0
        assert result.details["source_tokens"] == 1


# ── Compression ──────────────────────────────────────────────────────


class TestCompressionPass:
    """Summaries with appropriate compression ratios."""

    def test_good_compression_passes(self) -> None:
        # SCENARIO: Summary is roughly 30% of source length.
        # WHY: Falls within default [0.1, 0.5] range.
        # EXPECTED: passes.
        source = "word " * 100  # 500 chars
        summary = "word " * 25  # 125 chars
        result = assert_summary_compression(source, summary)
        assert result.passed is True
        assert result.name == "llm.summarization.compression"
        ratio = result.details["compression_ratio"]
        assert 0.1 <= ratio <= 0.5

    def test_custom_range_passes(self) -> None:
        # SCENARIO: Custom range [0.2, 0.4] with ratio = 0.3.
        # EXPECTED: passes within the custom range.
        source = "a" * 100
        summary = "a" * 30
        result = assert_summary_compression(
            source, summary, min_ratio=0.2, max_ratio=0.4,
        )
        assert result.passed is True

    def test_result_has_timing(self) -> None:
        # SCENARIO: @timed_assertion sets duration_ms.
        result = assert_summary_compression(
            "x" * 100, "x" * 25,
        )
        assert result.duration_ms > 0


class TestCompressionFail:
    """Summaries with bad compression ratios."""

    def test_too_long_summary_fails(self) -> None:
        # SCENARIO: Summary is 90% of source length -- barely shorter.
        # WHY: ratio 0.9 > max_ratio 0.5.
        # EXPECTED: raises MltkAssertionError.
        source = "word " * 100
        summary = "word " * 90
        with pytest.raises(MltkAssertionError) as exc_info:
            assert_summary_compression(source, summary)
        ratio = exc_info.value.result.details["compression_ratio"]
        assert ratio > 0.5

    def test_too_short_summary_fails(self) -> None:
        # SCENARIO: Summary is 2% of source length -- too aggressive.
        # WHY: ratio 0.02 < min_ratio 0.1.
        # EXPECTED: raises MltkAssertionError.
        source = "word " * 100
        summary = "w"
        with pytest.raises(MltkAssertionError):
            assert_summary_compression(source, summary)


class TestCompressionEdge:
    """Edge cases for compression."""

    def test_empty_source_empty_summary(self) -> None:
        # SCENARIO: Both empty -- ratio is 0.0.
        # WHY: 0.0 < min_ratio 0.1 so this fails with defaults,
        #      but passes if min_ratio=0.0.
        result = assert_summary_compression(
            "", "", min_ratio=0.0, max_ratio=1.0,
        )
        assert result.passed is True

    def test_empty_source_nonempty_summary(self) -> None:
        # SCENARIO: Source is empty but summary is not.
        # WHY: ratio = 1.0 (can't compress nothing).
        # EXPECTED: fails with default max_ratio=0.5.
        with pytest.raises(MltkAssertionError):
            assert_summary_compression("", "some text")

    def test_details_include_lengths(self) -> None:
        # SCENARIO: Verify details contain source and summary lengths.
        source = "hello world"
        summary = "hi"
        result = assert_summary_compression(
            source, summary, min_ratio=0.0, max_ratio=1.0,
        )
        assert result.details["source_length"] == len(source)
        assert result.details["summary_length"] == len(summary)


# ── Faithfulness ─────────────────────────────────────────────────────


class TestFaithfulnessPass:
    """Faithful summaries that stay grounded in the source."""

    def test_faithful_summary_passes(self) -> None:
        # SCENARIO: All summary tokens exist in the source.
        # WHY: faithfulness = |common| / |summary_tokens| = 1.0.
        # EXPECTED: passes.
        source = (
            "Python is a high-level programming language "
            "known for its readability and versatility."
        )
        summary = "Python is a programming language."
        result = assert_summary_faithfulness(
            source, summary, min_faithfulness=0.5,
        )
        assert result.passed is True
        assert result.details["faithfulness"] >= 0.5
        assert result.name == "llm.summarization.faithfulness"

    def test_identical_text_perfect_faithfulness(self) -> None:
        # SCENARIO: Summary identical to source.
        # WHY: Every summary token is in the source -> 1.0.
        # EXPECTED: faithfulness = 1.0.
        text = "Data science combines statistics and programming."
        result = assert_summary_faithfulness(text, text)
        assert result.details["faithfulness"] == 1.0
        assert result.details["novel_tokens"] == 0

    def test_result_has_timing(self) -> None:
        # SCENARIO: @timed_assertion sets duration_ms.
        result = assert_summary_faithfulness("a b c", "a b", min_faithfulness=0.0)
        assert result.duration_ms > 0


class TestFaithfulnessFail:
    """Summaries that hallucinate content not in the source."""

    def test_hallucinated_content_fails(self) -> None:
        # SCENARIO: Summary introduces many words not in the source.
        # WHY: Most summary tokens are novel -> low faithfulness.
        # EXPECTED: raises MltkAssertionError.
        source = "The cat sat on the mat."
        summary = (
            "Quantum computing will revolutionize cryptography "
            "and artificial intelligence research."
        )
        with pytest.raises(MltkAssertionError) as exc_info:
            assert_summary_faithfulness(
                source, summary, min_faithfulness=0.5,
            )
        details = exc_info.value.result.details
        assert details["faithfulness"] < 0.5
        assert details["novel_tokens"] > 0

    def test_mostly_novel_fails(self) -> None:
        # SCENARIO: Summary shares one word but adds many new ones.
        # WHY: 1 common / 5 total = 0.2 < 0.5.
        # EXPECTED: fails.
        source = "Apples are red."
        summary = "Apples taste wonderful in autumn pies."
        with pytest.raises(MltkAssertionError):
            assert_summary_faithfulness(
                source, summary, min_faithfulness=0.5,
            )


class TestFaithfulnessEdge:
    """Edge cases for faithfulness."""

    def test_empty_summary_passes(self) -> None:
        # SCENARIO: Empty summary -- nothing to be unfaithful about.
        # WHY: Trivially faithful; no novel tokens.
        # EXPECTED: faithfulness = 1.0.
        result = assert_summary_faithfulness(
            "Some source text.", "", min_faithfulness=0.5,
        )
        assert result.passed is True
        assert result.details["faithfulness"] == 1.0

    def test_empty_source_empty_summary(self) -> None:
        # SCENARIO: Both empty.
        # EXPECTED: Trivially passes.
        result = assert_summary_faithfulness("", "", min_faithfulness=0.0)
        assert result.passed is True

    def test_empty_source_nonempty_summary(self) -> None:
        # SCENARIO: Source is empty but summary has content.
        # WHY: All summary tokens are novel -> faithfulness = 0.0.
        # EXPECTED: fails with min_faithfulness=0.5.
        with pytest.raises(MltkAssertionError) as exc_info:
            assert_summary_faithfulness(
                "", "hallucinated content", min_faithfulness=0.5,
            )
        assert exc_info.value.result.details["faithfulness"] == 0.0

    def test_single_shared_word(self) -> None:
        # SCENARIO: Summary is one word that exists in source.
        # WHY: 1/1 = 1.0 faithfulness.
        # EXPECTED: passes.
        result = assert_summary_faithfulness(
            "hello world", "hello", min_faithfulness=0.5,
        )
        assert result.passed is True
        assert result.details["faithfulness"] == 1.0
        assert result.details["novel_tokens"] == 0


# ── Cross-metric integration ────────────────────────────────────────


class TestCrossMetric:
    """Verify coverage and faithfulness are independent metrics."""

    def test_high_coverage_low_faithfulness(self) -> None:
        # SCENARIO: Summary covers source well but adds hallucinated
        #   content, inflating the summary with novel tokens.
        # WHY: Coverage looks at source perspective (good), but
        #   faithfulness looks at summary perspective (bad).
        source = "The sky is blue on clear days."
        summary = (
            "The sky is blue on clear days and also the "
            "quantum entanglement phenomenon is fascinating "
            "and revolutionary for modern physics."
        )
        cov = assert_summary_coverage(source, summary, min_coverage=0.5)
        assert cov.passed is True

        with pytest.raises(MltkAssertionError):
            assert_summary_faithfulness(
                source, summary, min_faithfulness=0.8,
            )

    def test_long_source_short_summary(self) -> None:
        # SCENARIO: A realistic long source with a short summary.
        # WHY: Tests all three metrics work on longer text.
        source = (
            "Artificial intelligence is transforming industries "
            "across the globe. Healthcare uses AI for diagnosis "
            "and drug discovery. Finance leverages machine learning "
            "for fraud detection and algorithmic trading. Education "
            "is being reshaped by adaptive learning platforms. "
            "Manufacturing benefits from predictive maintenance "
            "and quality control automation."
        )
        summary = "AI transforms healthcare, finance, and education."
        cov = assert_summary_coverage(source, summary, min_coverage=0.05)
        assert cov.passed is True

        comp = assert_summary_compression(
            source, summary, min_ratio=0.05, max_ratio=0.5,
        )
        assert comp.passed is True

        faith = assert_summary_faithfulness(
            source, summary, min_faithfulness=0.3,
        )
        assert faith.passed is True


# -- Hardened edge-case tests (S62 test hardening) ----------------


class TestCoverageParametrizedMinCoverage:
    """Parametrize min_coverage thresholds."""

    @pytest.mark.parametrize(
        "min_cov", [0.0, 0.3, 0.5, 0.8, 1.0],
    )
    def test_identical_text_all_thresholds(
        self, min_cov: float,
    ) -> None:
        # SCENARIO: Identical text means coverage = 1.0.
        # WHY: Should pass any min_coverage in [0, 1].
        text = "The quick brown fox jumps over the lazy dog."
        result = assert_summary_coverage(
            text, text, min_coverage=min_cov,
        )
        assert result.passed is True
        assert result.details["coverage"] == 1.0


class TestCompressionSingleChar:
    """Compression with source of length 1 char."""

    def test_source_one_char(self) -> None:
        # SCENARIO: Source is a single character.
        # WHY: Smallest non-empty source; ratio = 1/1 = 1.0.
        # EXPECTED: ratio = 1.0 => fails default max_ratio.
        with pytest.raises(MltkAssertionError) as exc:
            assert_summary_compression("x", "x")
        ratio = exc.value.result.details[
            "compression_ratio"
        ]
        assert abs(ratio - 1.0) < 1e-9

    def test_source_one_char_custom_range(self) -> None:
        # SCENARIO: Source 1 char with max_ratio=1.0.
        # EXPECTED: passes.
        result = assert_summary_compression(
            "x", "x", min_ratio=0.0, max_ratio=1.0,
        )
        assert result.passed is True


class TestFaithfulnessRepeatedWords:
    """Faithfulness with repeated words in summary."""

    def test_repeated_words_still_faithful(self) -> None:
        # SCENARIO: Summary repeats "the" many times.
        # WHY: Token sets ignore count; only unique tokens
        #   matter. All summary tokens exist in source.
        # EXPECTED: faithfulness = 1.0.
        source = "the cat sat on the mat by the door"
        summary = "the the the the"
        result = assert_summary_faithfulness(
            source, summary, min_faithfulness=0.5,
        )
        assert result.passed is True
        assert result.details["faithfulness"] == 1.0


class TestCoverageFaithfulnessCombined:
    """Coverage + faithfulness on same source/summary."""

    def test_same_pair_different_thresholds(self) -> None:
        # SCENARIO: Evaluate both metrics on the same data.
        # WHY: Verifies both can run independently on
        #   identical input without interference.
        source = (
            "Python is a popular high-level language "
            "used for web development and data science."
        )
        summary = "Python is a popular language."
        cov = assert_summary_coverage(
            source, summary, min_coverage=0.2,
        )
        assert cov.passed is True
        faith = assert_summary_faithfulness(
            source, summary, min_faithfulness=0.5,
        )
        assert faith.passed is True
        assert faith.details["faithfulness"] >= 0.5
        assert cov.details["coverage"] >= 0.2


class TestVeryLongSourcePerformance:
    """Very long source (10,000 chars) performance."""

    def test_10k_char_source_coverage(self) -> None:
        # SCENARIO: Source is 10,000 chars.
        # WHY: Must complete in reasonable time without
        #   quadratic blowup in token set operations.
        # EXPECTED: Completes; result is valid.
        source = ("word " * 2000).strip()  # 9999 chars
        summary = "word"
        result = assert_summary_coverage(
            source, summary, min_coverage=0.0,
        )
        assert result.passed is True
        assert result.details["source_tokens"] > 0

    def test_10k_char_source_faithfulness(self) -> None:
        # SCENARIO: 10K-char source, short summary.
        # EXPECTED: Completes without exception.
        source = ("alpha beta gamma " * 600).strip()
        summary = "alpha beta"
        result = assert_summary_faithfulness(
            source, summary, min_faithfulness=0.5,
        )
        assert result.passed is True


class TestSummaryLongerThanSource:
    """Summary longer than source -- compression > 1.0."""

    def test_summary_longer_than_source(self) -> None:
        # SCENARIO: Summary is 3x longer than source.
        # WHY: LLMs sometimes "expand" instead of summarize.
        # EXPECTED: ratio = 3.0 > max_ratio => fails.
        source = "short"
        summary = "this is a much longer summary text"
        with pytest.raises(MltkAssertionError) as exc:
            assert_summary_compression(source, summary)
        ratio = exc.value.result.details[
            "compression_ratio"
        ]
        assert ratio > 1.0

    def test_summary_longer_custom_range(self) -> None:
        # SCENARIO: Ratio > 1.0 with max_ratio=5.0.
        # EXPECTED: passes with permissive range.
        source = "hi"
        summary = "hello world foo bar"
        result = assert_summary_compression(
            source, summary, min_ratio=0.0, max_ratio=20.0,
        )
        assert result.passed is True
        assert result.details["compression_ratio"] > 1.0
