"""Tests for detect_unicode_attacks and assert_no_unicode_attacks."""

from __future__ import annotations

import pytest

from mltk.core.assertion import MltkAssertionError
from mltk.core.result import Severity
from mltk.domains.llm.unicode_attacks import (
    assert_no_unicode_attacks,
    detect_unicode_attacks,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

# A token with mixed ASCII Latin + Cyrillic -- classic paypal phishing style.
# 'p', 'y', 'l' are ASCII; chr(0x0430) is Cyrillic SMALL LETTER A.
_HOMOGLYPH_TOKEN = "p" + chr(0x0430) + "yp" + chr(0x0430) + "l"

# Zero-width space (U+200B)
_ZWSP = chr(0x200B)

# Right-to-left override (U+202E)
_RLO = chr(0x202E)


# ---------------------------------------------------------------------------
# Tests for the pure detector
# ---------------------------------------------------------------------------


class TestDetectUnicodeAttacks:
    """Unit tests for detect_unicode_attacks (pure-detector, no TestResult)."""

    def test_plain_ascii_returns_no_findings(self) -> None:
        """PASS: Plain ASCII text produces zero findings in all categories."""
        result = detect_unicode_attacks("Hello, world! This is clean ASCII.")
        assert result["total"] == 0
        assert result["zero_width"] == []
        assert result["bidi"] == []
        assert result["homoglyph"] == []

    def test_empty_string_returns_no_findings(self) -> None:
        """PASS: Empty string is always clean."""
        result = detect_unicode_attacks("")
        assert result["total"] == 0

    def test_zero_width_space_detected(self) -> None:
        """DETECT: U+200B is reported with correct codepoint and index."""
        text = "hello" + _ZWSP + "world"
        result = detect_unicode_attacks(text)
        assert result["total"] == 1
        assert len(result["zero_width"]) == 1
        assert result["zero_width"][0]["codepoint"] == "U+200B"
        assert result["zero_width"][0]["index"] == 5

    def test_all_explicit_zero_width_chars_detected(self) -> None:
        """DETECT: All six explicit invisible chars are caught."""
        invisible = (
            chr(0x200B)  # ZERO WIDTH SPACE
            + chr(0x200C)  # ZERO WIDTH NON-JOINER
            + chr(0x200D)  # ZERO WIDTH JOINER
            + chr(0xFEFF)  # BOM
            + chr(0x2060)  # WORD JOINER
            + chr(0x00AD)  # SOFT HYPHEN
        )
        result = detect_unicode_attacks(invisible, checks=("zero_width",))
        assert result["total"] == 6
        assert len(result["zero_width"]) == 6

    def test_bidi_rlo_detected(self) -> None:
        """DETECT: U+202E right-to-left override is reported correctly."""
        text = "abc" + _RLO + "def"
        result = detect_unicode_attacks(text)
        assert result["total"] == 1
        assert len(result["bidi"]) == 1
        assert result["bidi"][0]["codepoint"] == "U+202E"
        assert result["bidi"][0]["index"] == 3

    def test_all_nine_bidi_controls_detected(self) -> None:
        """DETECT: All nine bidi control codepoints are caught."""
        bidi_chars = "".join(chr(cp) for cp in (
            0x202A, 0x202B, 0x202C, 0x202D, 0x202E,
            0x2066, 0x2067, 0x2068, 0x2069,
        ))
        result = detect_unicode_attacks(bidi_chars, checks=("bidi",))
        assert result["total"] == 9
        assert len(result["bidi"]) == 9

    def test_cyrillic_homoglyph_token_detected(self) -> None:
        """DETECT: Token mixing ASCII Latin and Cyrillic is flagged.

        The token is whitespace-split, so punctuation attached to the token
        is included. We assert on the start index and mixed-script presence.
        """
        text = "Visit " + _HOMOGLYPH_TOKEN + " now!"
        result = detect_unicode_attacks(text)
        assert result["total"] == 1
        assert len(result["homoglyph"]) == 1
        assert result["homoglyph"][0]["token"] == _HOMOGLYPH_TOKEN
        assert result["homoglyph"][0]["index"] == 6

    def test_emoji_not_flagged_as_zero_width(self) -> None:
        """PASS: Emoji (category So) must not be treated as zero-width."""
        text = "Hello \U0001F389 world \U0001F600"
        result = detect_unicode_attacks(text)
        assert result["zero_width"] == []
        assert result["total"] == 0

    def test_hebrew_letters_not_flagged_as_bidi(self) -> None:
        """PASS: Hebrew RTL letters are not bidi control characters."""
        text = "שלום"  # shalom in Hebrew
        result = detect_unicode_attacks(text)
        assert result["bidi"] == []
        assert result["total"] == 0

    def test_arabic_letters_not_flagged_as_bidi(self) -> None:
        """PASS: Arabic RTL letters are not bidi control characters."""
        text = "مرحبا"  # marhaba
        result = detect_unicode_attacks(text)
        assert result["bidi"] == []
        assert result["total"] == 0

    def test_pure_cyrillic_token_not_flagged_as_homoglyph(self) -> None:
        """PASS: Token with only Cyrillic (no ASCII Latin) is not a homoglyph."""
        text = "КИЕВ"  # KIEV in Cyrillic
        result = detect_unicode_attacks(text, checks=("homoglyph",))
        assert result["total"] == 0
        assert result["homoglyph"] == []

    def test_checks_subset_excludes_other_categories(self) -> None:
        """SCOPE: Keys for unchecked categories must not appear in the result.

        RLO (U+202E) is a bidi control — it must NOT bleed into zero_width
        even though it is technically a Cf-category character.
        """
        text = "abc" + _RLO + "def" + _ZWSP + "ghi"
        result = detect_unicode_attacks(text, checks=("zero_width",))
        assert "bidi" not in result
        assert "homoglyph" not in result
        assert "zero_width" in result
        # RLO is excluded from zero_width; only ZWSP counts
        assert result["total"] == 1
        assert result["zero_width"][0]["codepoint"] == "U+200B"

    def test_multiple_categories_sum_in_total(self) -> None:
        """DETECT: total reflects combined findings across all checked categories."""
        text = "abc" + _ZWSP + "def" + _RLO + "ghi"
        result = detect_unicode_attacks(text)
        # 1 zero_width + 1 bidi + 0 homoglyph
        assert result["total"] == 2
        assert len(result["zero_width"]) == 1
        assert len(result["bidi"]) == 1
        assert result["homoglyph"] == []

    def test_result_contains_total_key_always(self) -> None:
        """STRUCTURE: 'total' key is always present regardless of checks."""
        result = detect_unicode_attacks("clean", checks=())
        assert "total" in result
        assert result["total"] == 0


# ---------------------------------------------------------------------------
# Tests for the assertion wrapper
# ---------------------------------------------------------------------------


class TestAssertNoUnicodeAttacks:
    """Tests for assert_no_unicode_attacks (assertion that returns TestResult)."""

    def test_plain_ascii_passes(self) -> None:
        """PASS: Clean ASCII text passes with total_attacks == 0."""
        result = assert_no_unicode_attacks(
            "The quick brown fox jumps over the lazy dog."
        )
        assert result.passed is True
        assert result.details["total_attacks"] == 0

    def test_empty_string_passes(self) -> None:
        """PASS: Empty string always passes."""
        result = assert_no_unicode_attacks("")
        assert result.passed is True

    def test_zero_width_char_raises_on_critical(self) -> None:
        """FAIL/CRITICAL: Zero-width char raises AssertionError by default."""
        with pytest.raises(AssertionError):
            assert_no_unicode_attacks("hello" + _ZWSP + "world")

    def test_zero_width_warning_returns_failed_result(self) -> None:
        """FAIL/WARNING: Returns a failed result instead of raising."""
        result = assert_no_unicode_attacks(
            "hello" + _ZWSP + "world",
            severity=Severity.WARNING,
        )
        assert result.passed is False
        assert result.details["total_attacks"] == 1

    def test_bidi_rlo_raises(self) -> None:
        """FAIL/CRITICAL: Bidi right-to-left override raises AssertionError."""
        with pytest.raises(AssertionError):
            assert_no_unicode_attacks("normal" + _RLO + "text")

    def test_homoglyph_raises(self) -> None:
        """FAIL/CRITICAL: Mixed-script token raises AssertionError."""
        with pytest.raises(AssertionError):
            assert_no_unicode_attacks("Check " + _HOMOGLYPH_TOKEN + ".com!")

    def test_checks_subset_ignores_other_categories(self) -> None:
        """PASS: Bidi char present but excluded from checks — should pass."""
        result = assert_no_unicode_attacks(
            "abc" + _RLO + "def",
            checks=("zero_width",),
        )
        assert result.passed is True

    def test_details_contain_counts_and_sample_on_failure(self) -> None:
        """FAIL/WARNING: Details include per-category count and sample list."""
        text = "a" + chr(0x200B) + "b" + chr(0x200C) + "c"
        result = assert_no_unicode_attacks(text, severity=Severity.WARNING)
        assert "zero_width_count" in result.details
        assert result.details["zero_width_count"] == 2
        assert "zero_width_sample" in result.details
        assert isinstance(result.details["zero_width_sample"], list)

    def test_result_name_is_correct(self) -> None:
        """STRUCTURE: TestResult name must be 'llm.no_unicode_attacks'."""
        result = assert_no_unicode_attacks("safe text")
        assert result.name == "llm.no_unicode_attacks"

    def test_mltk_assertion_error_carries_result(self) -> None:
        """FAIL/CRITICAL: MltkAssertionError carries the failed TestResult."""
        with pytest.raises(MltkAssertionError) as exc_info:
            assert_no_unicode_attacks("bad" + _ZWSP + "text")
        assert exc_info.value.result.passed is False
        assert exc_info.value.result.details["total_attacks"] >= 1


# ---------------------------------------------------------------------------
# Review-fix regression: legitimate Cf format chars must NOT be flagged
# ---------------------------------------------------------------------------


def test_legit_arabic_number_sign_not_flagged() -> None:
    """U+0600 (Arabic number sign, category Cf) is legitimate, not zero-width."""
    text = chr(0x0600) + "تست"  # number sign + Arabic letters
    assert detect_unicode_attacks(text, checks=("zero_width",))["total"] == 0


def test_legit_syriac_abbreviation_mark_not_flagged() -> None:
    """U+070F (Syriac abbreviation mark, category Cf) is legitimate."""
    assert detect_unicode_attacks(chr(0x070F) + "word", checks=("zero_width",))["total"] == 0


def test_legit_arabic_text_passes_assertion() -> None:
    """A real Arabic string with a Cf number sign must not raise."""
    result = assert_no_unicode_attacks(chr(0x0600) + "١٢", checks=("zero_width",))
    assert result.passed is True


def test_real_zero_width_still_flagged_after_allowlist() -> None:
    """The legit-Cf allowlist must not weaken detection of genuine ZWSP attacks."""
    with pytest.raises(MltkAssertionError):
        assert_no_unicode_attacks("hi" + chr(0x200B) + "there", checks=("zero_width",))
