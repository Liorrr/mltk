"""Tests for mltk.domains.llm.text_quality — length, format, readability."""

import pytest

from mltk.core.assertion import MltkAssertionError
from mltk.domains.llm.text_quality import (
    assert_output_format,
    assert_readability,
    assert_text_length,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SHORT_TEXT = "Hello world today"  # 3 words
_MEDIUM_TEXT = " ".join(["word"] * 50)  # 50 words
_LONG_TEXT = " ".join(["word"] * 200)  # 200 words

# A deliberately simple sentence (low FK grade level)
_EASY_TEXT = (
    "The cat sat on the mat. "
    "A dog ran in the park. "
    "She saw the sun. "
    "He ate a big red apple. "
    "We like to play all day long."
)

# Dense academic prose (high FK grade level)
_COMPLEX_TEXT = (
    "The epistemological underpinnings of contemporary computational methodologies "
    "necessitate a rigorous reevaluation of foundational axiomatic frameworks. "
    "Phenomenological investigations into the ontological distinctions between "
    "deterministic and stochastic representational architectures illuminate "
    "the multifaceted interdependencies inherent within hierarchical abstraction layers. "
    "Consequentially, the parameterisation of probabilistic inference mechanisms "
    "transcends conventional algorithmic paradigms, engendering unprecedented "
    "epistemological sophistication within the broader theoretical discourse."
)


# ===========================================================================
# assert_text_length
# ===========================================================================


class TestTextLength:
    """Word-count boundary assertions."""

    def test_text_length_within_bounds(self) -> None:
        # SCENARIO: 50-word text, bounds 10-100
        # WHY: Typical valid output — word count sits comfortably inside range
        # EXPECTED: pass
        result = assert_text_length(_MEDIUM_TEXT, min_words=10, max_words=100)
        assert result.passed is True
        assert result.details["word_count"] == 50

    def test_text_length_too_short(self) -> None:
        # SCENARIO: 3-word text, min_words=10
        # WHY: Model output is too terse — should be flagged as under-length
        # EXPECTED: fail with "too short" message
        with pytest.raises(MltkAssertionError) as exc:
            assert_text_length(_SHORT_TEXT, min_words=10)
        assert "too short" in str(exc.value).lower()

    def test_text_length_too_long(self) -> None:
        # SCENARIO: 200-word text, max_words=100
        # WHY: Model output is excessively verbose — should be capped
        # EXPECTED: fail with "too long" message
        with pytest.raises(MltkAssertionError) as exc:
            assert_text_length(_LONG_TEXT, max_words=100)
        assert "too long" in str(exc.value).lower()

    def test_text_length_exact_min_boundary(self) -> None:
        # SCENARIO: exactly min_words=50
        # WHY: Boundary value — should pass (inclusive lower bound)
        # EXPECTED: pass
        result = assert_text_length(_MEDIUM_TEXT, min_words=50)
        assert result.passed is True

    def test_text_length_exact_max_boundary(self) -> None:
        # SCENARIO: exactly max_words=50
        # WHY: Boundary value — should pass (inclusive upper bound)
        # EXPECTED: pass
        result = assert_text_length(_MEDIUM_TEXT, max_words=50)
        assert result.passed is True

    def test_text_length_only_min(self) -> None:
        # SCENARIO: min_words only, long text
        # WHY: Common usage — enforce minimum length without cap
        # EXPECTED: pass
        result = assert_text_length(_LONG_TEXT, min_words=50)
        assert result.passed is True

    def test_text_length_only_max(self) -> None:
        # SCENARIO: max_words only, short text
        # WHY: Common usage — enforce maximum length without floor
        # EXPECTED: pass
        result = assert_text_length(_SHORT_TEXT, max_words=50)
        assert result.passed is True

    def test_text_length_no_bounds_raises(self) -> None:
        # SCENARIO: neither min_words nor max_words specified
        # WHY: Misconfigured call — neither bound makes the assertion meaningless
        # EXPECTED: fail immediately with config error
        with pytest.raises(MltkAssertionError):
            assert_text_length(_MEDIUM_TEXT)


# ===========================================================================
# assert_output_format
# ===========================================================================


class TestOutputFormat:
    """Regex pattern format assertions."""

    def test_output_format_match(self) -> None:
        # SCENARIO: text "Result: 42" matches r"^Result: \d+$"
        # WHY: Model should produce structured output matching expected format
        # EXPECTED: pass
        result = assert_output_format("Result: 42", pattern=r"^Result: \d+$")
        assert result.passed is True

    def test_output_format_no_match(self) -> None:
        # SCENARIO: plain sentence doesn't match JSON pattern
        # WHY: Model should produce JSON but returned prose — must be caught
        # EXPECTED: fail, description appears in error message
        with pytest.raises(MltkAssertionError) as exc:
            assert_output_format(
                "I cannot answer that.", pattern=r"^\{.*\}$",
                description="JSON object",
            )
        assert "json object" in str(exc.value).lower()

    def test_output_format_starts_with(self) -> None:
        # SCENARIO: text starts with "ANSWER:"
        # WHY: Instruction-tuned models often require a fixed prefix
        # EXPECTED: pass
        result = assert_output_format("ANSWER: Paris", pattern=r"^ANSWER:")
        assert result.passed is True

    def test_output_format_uuid(self) -> None:
        # SCENARIO: text is a UUID-4 string
        # WHY: Service response must be a valid UUID
        # EXPECTED: pass
        uuid_str = "550e8400-e29b-41d4-a716-446655440000"
        result = assert_output_format(
            uuid_str,
            pattern=r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
            description="UUID-4",
        )
        assert result.passed is True

    def test_output_format_description_in_failure(self) -> None:
        # SCENARIO: failure message must include the human-readable description
        # WHY: description kwarg exists to make errors actionable
        # EXPECTED: fail, description appears in error message
        with pytest.raises(MltkAssertionError) as exc:
            assert_output_format(
                "not a number", pattern=r"^\d+$", description="integer string"
            )
        assert "integer string" in str(exc.value).lower()

    def test_output_format_invalid_regex(self) -> None:
        # SCENARIO: caller passes a malformed regex
        # WHY: Should fail gracefully with a clear error, not an uncaught exception
        # EXPECTED: fail with regex error message
        with pytest.raises(MltkAssertionError):
            assert_output_format("hello", pattern=r"[invalid(")


# ===========================================================================
# assert_readability
# ===========================================================================


class TestReadability:
    """Flesch-Kincaid grade level assertions."""

    def test_readability_easy(self) -> None:
        # SCENARIO: simple short-sentence text, max_grade_level=10
        # WHY: Customer-facing copy must be readable at 10th-grade level or below
        # EXPECTED: pass
        result = assert_readability(_EASY_TEXT, max_grade_level=10.0)
        assert result.passed is True
        assert result.details["grade_level"] <= 10.0

    def test_readability_complex(self) -> None:
        # SCENARIO: dense academic prose, max_grade_level=12
        # WHY: Academic jargon far exceeds typical readability threshold
        # EXPECTED: fail — grade level too high
        with pytest.raises(MltkAssertionError) as exc:
            assert_readability(_COMPLEX_TEXT, max_grade_level=12.0)
        assert "complex" in str(exc.value).lower()

    def test_readability_metrics_present(self) -> None:
        # SCENARIO: check that all expected details keys are returned
        # WHY: Downstream dashboards depend on word/sentence/syllable counts
        # EXPECTED: pass and details keys are populated
        result = assert_readability(_EASY_TEXT, max_grade_level=20.0)
        assert result.passed is True
        for key in ("grade_level", "total_words", "total_sentences", "total_syllables"):
            assert key in result.details

    def test_readability_empty_text_fails(self) -> None:
        # SCENARIO: empty string passed
        # WHY: Guard against calling with empty LLM output
        # EXPECTED: fail with "no words" message
        with pytest.raises(MltkAssertionError):
            assert_readability("", max_grade_level=12.0)

    def test_readability_high_threshold_always_passes(self) -> None:
        # SCENARIO: absurdly high threshold (grade 50) for complex text
        # WHY: Verify direction — an unreachably lenient threshold should always pass;
        #      the complex fixture scores ~33 so 50 is safely above it
        # EXPECTED: pass
        result = assert_readability(_COMPLEX_TEXT, max_grade_level=50.0)
        assert result.passed is True
