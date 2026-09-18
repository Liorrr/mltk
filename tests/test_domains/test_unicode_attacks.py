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

# Whole-word Cyrillic lookalikes of "paypa" + palochka-as-l (U+04CF).
_CYRILLIC_SPOOF = "раура" + chr(0x04CF)

# Fullwidth Latin "paypal" (U+FF50..).
_FULLWIDTH_SPOOF = "".join(chr(ord(c) + 0xFEE0) for c in "paypal")

# Mathematical bold "paypal" (U+1D41A MATHEMATICAL BOLD SMALL A ...).
_MATH_SPOOF = "".join(chr(0x1D41A + (ord(c) - ord("a"))) for c in "paypal")

# Greek spoof of Latin "ABO" -- every letter is a Latin twin.
_GREEK_SPOOF = "ΑΒΟ"

# Ordinary words built ENTIRELY from Latin twins. These are the false
# positives the context gate exists to prevent: nothing about them is
# adversarial, they are simply drawn from the lookalike subset.
_REAL_CYRILLIC_WORDS = (
    "МОСКВА",  # Moscow
    "СССР",  # USSR
    "хор",  # choir
    "оса",  # wasp
)
_REAL_GREEK_WORDS = (
    "και",  # "and" -- among the most frequent words in Greek
    "τον",  # "the" (accusative)
    "ναι",  # "yes"
)

# Prose in each script, long enough to fill a token's context window.
_RUSSIAN_SENTENCE = "Я живу в МОСКВА и пою в хор каждый день"
_GREEK_SENTENCE = "Το βιβλίο και τον κόσμο ειναι ωραια"


# "paypal" with only its tail fullwidth -- a genuine mixed-script spoof, and
# the shape the CJK false positive must not cost us.
_HALF_FULLWIDTH_SPOOF = "pay" + "".join(chr(ord(c) + 0xFEE0) for c in "pal")

# CJK prose carrying a fullwidth acronym beside an ASCII word. CJK writes no
# spaces, so each of these is a SINGLE \S+ token.
_CJK_WITH_ASCII = (
    # "This is a PDF. See https://example.com for details."
    "これは" + "".join(chr(ord(c) + 0xFEE0) for c in "PDF") + "です。"
    "詳細はhttps://example.comをご覧ください。",
    # "Set it to ON and enable WiFi."
    "設定は" + "".join(chr(ord(c) + 0xFEE0) for c in "ON") + "にしてWiFiを有効化する。",
    # "Please set the CPU temp parameter."
    "请把" + "".join(chr(ord(c) + 0xFEE0) for c in "CPU") + "的temp参数设置好。",
)

# Space-free scripts carrying a spoof with NO separator around it. This is the
# positive counterpart the run-2 suite lacked: the same property that makes CJK
# a false-positive risk (no spaces) also means no punctuation between words, so
# a spoof lands in the same run as the surrounding script.
_SPOOF_IN_SPACE_FREE_PROSE = (
    "ログインは{}へアクセスしてください。",  # Japanese
    "请访问{}网站登录",  # Chinese
    "로그인은{}에서",  # Korean
    "ไปที่{}เลย",  # Thai
)

# Unicode mathematical notation inside ordinary English prose.
_MATH_BOLD_ABC = "".join(chr(0x1D41A + i) for i in range(3))  # 𝐚𝐛𝐜
_MATH_IN_PROSE = (
    f"We define the set {_MATH_BOLD_ABC} to be the closure of S.",
    "Let the matrix \U0001d400 act on f(\U0001d431) for all x.",
    "the field \U0001d53d and ring \U0001d546\U0001d546\U0001d546 in algebra",
)

# Mathematics where a bold letter is juxtaposed with a plain one -- no
# separator, which is how a bold matrix applied to a variable renders.
# Notation, not a spoof.
_MATH_JUXTAPOSED = (
    "Let \U0001d400x = b be the system",
    "the product \U0001d400x is defined",
    "Consider \U0001d411n and its dual",
    "A vector x\U0001d422 in the basis",
    "the operator \U0001d400\U0001d401x applied",
)

# Math letters replacing a chunk of an ASCII word -- no notational reading.
def _math_bold(text: str) -> str:
    """Render lowercase ASCII as Mathematical Bold Small letters."""
    return "".join(chr(0x1D41A + ord(c) - ord("a")) for c in text)


_MATH_SPLICED_SPOOFS = (
    "pay" + _math_bold("pal"),
    _math_bold("pay") + "pal",
)

# Every separator str.splitlines() breaks on.
_LINE_SEPARATORS = (
    "\n",
    "\r\n",
    "\r",
    "\v",
    "\f",
    "\x1c",
    "\x1d",
    "\x1e",
    "\x85",
    "\u2028",
    "\u2029",
)


def _in_english(token: str) -> str:
    """Embed *token* in unambiguously English prose (Latin script context)."""
    return f"Please sign in to your {token} account again today"


# Zero-width space (U+200B)
_ZWSP = chr(0x200B)

# Right-to-left override (U+202E)
_RLO = chr(0x202E)

# Zero-width joiner (U+200D) and legitimate emoji ZWJ sequences
_ZWJ = chr(0x200D)
_EMOJI_MAN_TECH = "\U0001F468" + _ZWJ + "\U0001F4BB"  # 👨‍💻 man technologist
_EMOJI_FAMILY = "\U0001F468" + _ZWJ + "\U0001F469" + _ZWJ + "\U0001F467"  # 👨‍👩‍👧


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
        """PASS: Real Cyrillic (КИЕВ contains И, not a Latin twin) is not a spoof."""
        text = "КИЕВ"  # KIEV in Cyrillic
        result = detect_unicode_attacks(text, checks=("homoglyph",))
        assert result["total"] == 0
        assert result["homoglyph"] == []

    def test_whole_word_cyrillic_lookalike_is_flagged(self) -> None:
        """DETECT: All-lookalike Cyrillic spoof inside English prose is flagged."""
        result = detect_unicode_attacks(
            _in_english(_CYRILLIC_SPOOF), checks=("homoglyph",)
        )
        assert result["total"] == 1
        assert result["homoglyph"][0]["token"] == _CYRILLIC_SPOOF
        assert result["homoglyph"][0]["reason"] == "single_script"

    def test_real_cyrillic_sentence_not_flagged(self) -> None:
        """PASS: Ordinary Russian is not a lookalike spoof."""
        result = detect_unicode_attacks("привет мир", checks=("homoglyph",))
        assert result["homoglyph"] == []

    def test_greek_in_scientific_text_not_flagged(self) -> None:
        """PASS: Mixed Latin+Greek scientific text stays unflagged."""
        result = detect_unicode_attacks("α-helix 5μm kΩ", checks=("homoglyph",))
        assert result["homoglyph"] == []

    def test_whole_word_greek_lookalike_is_flagged(self) -> None:
        """DETECT: All-lookalike Greek spoof (ΑΒΟ ~ ABO) in English is flagged."""
        result = detect_unicode_attacks(
            _in_english(_GREEK_SPOOF), checks=("homoglyph",)
        )
        assert result["total"] == 1
        assert result["homoglyph"][0]["token"] == _GREEK_SPOOF
        assert result["homoglyph"][0]["reason"] == "single_script"

    def test_fullwidth_latin_is_flagged(self) -> None:
        """DETECT: Fullwidth Latin paypal lookalike in English is a homoglyph."""
        result = detect_unicode_attacks(
            _in_english(_FULLWIDTH_SPOOF), checks=("homoglyph",)
        )
        assert result["total"] == 1
        assert result["homoglyph"][0]["token"] == _FULLWIDTH_SPOOF

    def test_math_alphanumeric_needs_always_mode(self) -> None:
        """SCOPE: Math notation only ever sits in Latin prose, so "auto" cannot
        tell a spoof from a formula -- it is opt-in under "always".
        """
        text = _in_english(_MATH_SPOOF)
        assert detect_unicode_attacks(text, checks=("homoglyph",))["total"] == 0
        result = detect_unicode_attacks(
            text, checks=("homoglyph",), single_script_spoofs="always"
        )
        assert result["total"] == 1
        assert result["homoglyph"][0]["token"] == _MATH_SPOOF

    def test_homoglyph_finding_carries_reason(self) -> None:
        """STRUCTURE: Each homoglyph finding names the rule that produced it."""
        result = detect_unicode_attacks(
            "Visit " + _HOMOGLYPH_TOKEN, checks=("homoglyph",)
        )
        assert result["homoglyph"][0]["reason"] == "mixed_script"

    def test_mixed_latin_and_fullwidth_is_flagged(self) -> None:
        """DETECT: ASCII Latin mixed with fullwidth is mixed-script homoglyph."""
        token = "pay" + _FULLWIDTH_SPOOF
        result = detect_unicode_attacks(token, checks=("homoglyph",))
        assert result["total"] == 1

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

    def test_emoji_zwj_sequence_not_flagged(self) -> None:
        """PASS: ZWJ between two emoji (man technologist) is not flagged."""
        result = detect_unicode_attacks(_EMOJI_MAN_TECH, checks=("zero_width",))
        assert result["total"] == 0
        assert result["zero_width"] == []

    def test_emoji_family_multiple_zwj_not_flagged(self) -> None:
        """PASS: Multiple ZWJs joining a family emoji sequence are not flagged."""
        result = detect_unicode_attacks(_EMOJI_FAMILY, checks=("zero_width",))
        assert result["total"] == 0

    def test_zwj_smuggled_in_plain_text_flagged(self) -> None:
        """DETECT: ZWJ between plain ASCII letters (not emoji) is still flagged."""
        text = "ad" + _ZWJ + "min"
        result = detect_unicode_attacks(text, checks=("zero_width",))
        assert result["total"] == 1
        assert result["zero_width"][0]["codepoint"] == "U+200D"
        assert result["zero_width"][0]["index"] == 2

    def test_zwj_one_sided_emoji_flagged(self) -> None:
        """DETECT: ZWJ with an emoji on only one side is still flagged."""
        text = "hi\U0001F44D" + _ZWJ + "there"
        result = detect_unicode_attacks(text, checks=("zero_width",))
        assert result["total"] == 1

    def test_leading_zwj_flagged(self) -> None:
        """DETECT: A ZWJ at the start of the string (no left neighbour) is flagged."""
        result = detect_unicode_attacks(_ZWJ + "hello", checks=("zero_width",))
        assert result["total"] == 1

    def test_greek_latin_scientific_token_not_flagged(self) -> None:
        """PASS: Scientific tokens mixing Latin and Greek are not homoglyphs."""
        for token in ("α-helix", "5μm", "kΩ"):
            result = detect_unicode_attacks(token, checks=("homoglyph",))
            assert result["total"] == 0, token

    def test_cyrillic_latin_homoglyph_still_flagged(self) -> None:
        """DETECT: Cyrillic/Latin mixed token is still flagged after Greek carve-out."""
        text = "p" + chr(0x0430) + "yp" + chr(0x0430) + "l"  # pаypаl
        result = detect_unicode_attacks(text, checks=("homoglyph",))
        assert result["total"] == 1


# ---------------------------------------------------------------------------
# Single-script spoofs: the whole-token rule is context-gated (UTS #39)
# ---------------------------------------------------------------------------


class TestSingleScriptSpoofContext:
    """The whole-token rule must fire on spoofs, not on ordinary foreign words.

    "Every letter is a Latin twin" is a property of the alphabet subset, not
    of spoofing. МОСКВА and και satisfy it exactly as раура does, so the rule
    is resolved against the script surrounding the token.
    """

    # -- ordinary words in their own script must stay clean -----------------

    @pytest.mark.parametrize("word", _REAL_CYRILLIC_WORDS + _REAL_GREEK_WORDS)
    def test_real_word_alone_is_not_a_spoof(self, word: str) -> None:
        """PASS: A bare all-twin word has no context and is never flagged."""
        result = detect_unicode_attacks(word, checks=("homoglyph",))
        assert result["homoglyph"] == [], word

    @pytest.mark.parametrize("word", _REAL_CYRILLIC_WORDS)
    def test_real_cyrillic_word_in_russian_prose_is_clean(self, word: str) -> None:
        """PASS: An all-twin Russian word inside Russian prose is not an attack."""
        text = f"{_RUSSIAN_SENTENCE} {word} {_RUSSIAN_SENTENCE}"
        assert detect_unicode_attacks(text, checks=("homoglyph",))["total"] == 0

    @pytest.mark.parametrize("word", _REAL_GREEK_WORDS)
    def test_real_greek_word_in_greek_prose_is_clean(self, word: str) -> None:
        """PASS: An all-twin Greek word inside Greek prose is not an attack."""
        text = f"{_GREEK_SENTENCE} {word} {_GREEK_SENTENCE}"
        assert detect_unicode_attacks(text, checks=("homoglyph",))["total"] == 0

    def test_foreign_clause_inside_english_document_is_clean(self) -> None:
        """PASS: Document-level Latin majority must not convict a local clause.

        A long English paragraph followed by a short Russian one: the Russian
        words are judged by their own neighbourhood, not the document average.
        """
        text = (
            "The weather is fine today and everything works as expected "
            "in the office this week. " + _RUSSIAN_SENTENCE
        )
        assert detect_unicode_attacks(text, checks=("homoglyph",))["total"] == 0

    @pytest.mark.parametrize("sep", _LINE_SEPARATORS)
    def test_context_does_not_cross_a_line_break(self, sep: str) -> None:
        """PASS: An English prompt line must not lend Latin context to the next.

        The prompt/response pair is the common shape of an eval record, and
        the separator is whatever the extractor emitted -- U+2028 and U+0085
        arrive routinely out of PDF, DOCX and JSON-origin pipelines.
        """
        text = (
            "Translate the quick brown fox jumps over the lazy dog"
            + sep
            + _RUSSIAN_SENTENCE
        )
        assert detect_unicode_attacks(text, checks=("homoglyph",))["total"] == 0

    @pytest.mark.parametrize("sep", _LINE_SEPARATORS)
    def test_bare_spoof_on_its_own_line_is_clean(self, sep: str) -> None:
        """PASS: Every separator must isolate a line, not just the newline."""
        text = "Please sign in to your account" + sep + _CYRILLIC_SPOOF
        assert detect_unicode_attacks(text, checks=("homoglyph",))["total"] == 0

    def test_interleaved_scripts_on_one_line_are_not_convicted(self) -> None:
        """PASS: Text genuinely mixing two scripts is ambiguous, so it is clean."""
        text = "Hello world friend " + _RUSSIAN_SENTENCE
        assert detect_unicode_attacks(text, checks=("homoglyph",))["total"] == 0

    def test_fullwidth_latin_in_cjk_text_is_clean(self) -> None:
        """PASS: Fullwidth Latin is ordinary typography inside CJK text."""
        text = "これは" + _FULLWIDTH_SPOOF + "です。"
        assert detect_unicode_attacks(text, checks=("homoglyph",))["total"] == 0

    # -- the same tokens inside Latin context are spoofs --------------------

    @pytest.mark.parametrize(
        "spoof", [_CYRILLIC_SPOOF, _GREEK_SPOOF, _FULLWIDTH_SPOOF]
    )
    def test_spoof_in_english_prose_is_flagged(self, spoof: str) -> None:
        """DETECT: The same token inside English prose is a whole-script spoof."""
        result = detect_unicode_attacks(_in_english(spoof), checks=("homoglyph",))
        assert [f["token"] for f in result["homoglyph"]] == [spoof]
        assert result["homoglyph"][0]["reason"] == "single_script"

    def test_spoof_at_start_of_text_is_flagged(self) -> None:
        """DETECT: Context from one side alone is enough to convict."""
        text = _CYRILLIC_SPOOF + " is the official login page for members"
        assert detect_unicode_attacks(text, checks=("homoglyph",))["total"] == 1

    @pytest.mark.parametrize(
        "carrier",
        [
            "Đây là trang {} của bạn nhé",  # Vietnamese
            "Proszę zalogować się na {} zaraz",  # Polish
            "café naïve résumé {} account page",  # accented English
        ],
    )
    def test_accented_latin_counts_as_latin_context(self, carrier: str) -> None:
        """DETECT: Diacritics are Latin script, so they still convict a spoof."""
        text = carrier.format(_CYRILLIC_SPOOF)
        assert detect_unicode_attacks(text, checks=("homoglyph",))["total"] == 1

    def test_scientific_greek_does_not_shield_a_spoof(self) -> None:
        """DETECT: A few Greek symbols do not turn English into Greek context."""
        text = "The α-helix " + _CYRILLIC_SPOOF + " domain is 5μm wide overall"
        result = detect_unicode_attacks(text, checks=("homoglyph",))
        assert [f["token"] for f in result["homoglyph"]] == [_CYRILLIC_SPOOF]

    # -- explicit modes -----------------------------------------------------

    @pytest.mark.parametrize("word", _REAL_CYRILLIC_WORDS + _REAL_GREEK_WORDS)
    def test_always_mode_flags_every_candidate(self, word: str) -> None:
        """SCOPE: always drops the context gate -- for Latin-only corpora."""
        result = detect_unicode_attacks(
            word, checks=("homoglyph",), single_script_spoofs="always"
        )
        assert result["total"] == 1, word

    def test_always_mode_flags_a_context_free_spoof_list(self) -> None:
        """SCOPE: A bulk list of spoof tokens has no prose to judge it by."""
        text = "\n".join([_CYRILLIC_SPOOF, _FULLWIDTH_SPOOF, _MATH_SPOOF])
        assert detect_unicode_attacks(text, checks=("homoglyph",))["total"] == 0
        assert (
            detect_unicode_attacks(
                text, checks=("homoglyph",), single_script_spoofs="always"
            )["total"]
            == 3
        )

    def test_never_mode_disables_the_whole_token_rule(self) -> None:
        """SCOPE: never restores mixed-script-only detection."""
        result = detect_unicode_attacks(
            _in_english(_CYRILLIC_SPOOF),
            checks=("homoglyph",),
            single_script_spoofs="never",
        )
        assert result["homoglyph"] == []

    def test_never_mode_keeps_mixed_script_detection(self) -> None:
        """SCOPE: never must not weaken the unambiguous mixed-script rule."""
        result = detect_unicode_attacks(
            "Visit " + _HOMOGLYPH_TOKEN,
            checks=("homoglyph",),
            single_script_spoofs="never",
        )
        assert [f["reason"] for f in result["homoglyph"]] == ["mixed_script"]

    @pytest.mark.parametrize("mode", ["auto", "always", "never"])
    def test_mixed_script_rule_is_unaffected_by_mode(self, mode: str) -> None:
        """INVARIANT: The mixed-script rule needs no context in any mode."""
        result = detect_unicode_attacks(
            _HOMOGLYPH_TOKEN, checks=("homoglyph",), single_script_spoofs=mode
        )
        assert result["total"] == 1

    def test_unknown_mode_raises_value_error(self) -> None:
        """GUARD: A typo in the mode must fail loudly, not silently disable."""
        with pytest.raises(ValueError, match="single_script_spoofs"):
            detect_unicode_attacks("text", single_script_spoofs="sometimes")

    def test_unknown_mode_raises_even_without_homoglyph_check(self) -> None:
        """GUARD: The argument is validated regardless of the checks requested."""
        with pytest.raises(ValueError, match="single_script_spoofs"):
            detect_unicode_attacks(
                "text", checks=("zero_width",), single_script_spoofs="yes"
            )


# ---------------------------------------------------------------------------
# Mixed-script evidence: the partner must sit in the same word, in a word that
# is Latin throughout (run-2 review)
# ---------------------------------------------------------------------------


class TestMixedScriptEvidence:
    """A token is not a word. `\\S+` hands this rule URLs, formulae and — since
    CJK writes no spaces — whole Japanese and Chinese sentences.
    """

    @pytest.mark.parametrize("text", _CJK_WITH_ASCII)
    @pytest.mark.parametrize("mode", ["auto", "always", "never"])
    def test_cjk_with_fullwidth_and_ascii_is_clean(self, text: str, mode: str) -> None:
        """PASS: Fullwidth acronyms beside ASCII are ordinary CJK typography.

        The mixed-script rule is ungated by design, so this must hold in every
        mode -- there is no parameter a caller could reach for otherwise.
        """
        result = detect_unicode_attacks(
            text, checks=("homoglyph",), single_script_spoofs=mode
        )
        assert result["homoglyph"] == [], text

    @pytest.mark.parametrize("text", _MATH_IN_PROSE)
    @pytest.mark.parametrize("mode", ["auto", "never"])
    def test_math_notation_in_english_prose_is_clean(
        self, text: str, mode: str
    ) -> None:
        """PASS: `f(𝐱)` is a Latin function applied to a math variable.

        The two scripts are in one token but not in one word, which is the
        distinction the run boundary draws.
        """
        result = detect_unicode_attacks(
            text, checks=("homoglyph",), single_script_spoofs=mode
        )
        assert result["homoglyph"] == [], text

    def test_cyrillic_url_path_segment_is_clean(self) -> None:
        """PASS: A Cyrillic path segment is a separate word from the domain."""
        text = "See https://example.com/wiki/МОСКВА for details about it"
        assert detect_unicode_attacks(text, checks=("homoglyph",))["total"] == 0

    @pytest.mark.parametrize("mode", ["auto", "always", "never"])
    def test_partial_fullwidth_word_is_still_flagged(self, mode: str) -> None:
        """DETECT: Latin and fullwidth inside ONE word is the spoof shape."""
        result = detect_unicode_attacks(
            _in_english(_HALF_FULLWIDTH_SPOOF),
            checks=("homoglyph",),
            single_script_spoofs=mode,
        )
        assert [f["reason"] for f in result["homoglyph"]] == ["mixed_script"]

    @pytest.mark.parametrize("mode", ["auto", "always", "never"])
    def test_cyrillic_in_a_latin_word_is_still_flagged(self, mode: str) -> None:
        """DETECT: No language writes Latin and Cyrillic inside one word."""
        result = detect_unicode_attacks(
            "Visit " + _HOMOGLYPH_TOKEN + ".com now",
            checks=("homoglyph",),
            single_script_spoofs=mode,
        )
        assert [f["reason"] for f in result["homoglyph"]] == ["mixed_script"]

    @pytest.mark.parametrize("carrier", _SPOOF_IN_SPACE_FREE_PROSE)
    @pytest.mark.parametrize("mode", ["auto", "always", "never"])
    def test_spoof_inside_space_free_prose_is_flagged(
        self, carrier: str, mode: str
    ) -> None:
        """DETECT: The rule that clears CJK must not go blind inside it.

        This is the mirror of test_cjk_with_fullwidth_and_ascii_is_clean, and
        its absence is what let the purity test silently drop `pаypal` in
        Japanese, Chinese, Korean and Thai prose -- a false NEGATIVE on the
        assertion's headline vector, in every mode.
        """
        result = detect_unicode_attacks(
            carrier.format(_HOMOGLYPH_TOKEN),
            checks=("homoglyph",),
            single_script_spoofs=mode,
        )
        assert [f["token"] for f in result["homoglyph"]] != [], carrier
        assert result["homoglyph"][0]["reason"] == "mixed_script"

    @pytest.mark.parametrize("sep", [" ", "・", ":"])
    def test_spoof_in_cjk_with_a_separator_is_still_flagged(self, sep: str) -> None:
        """DETECT: Detection must not depend on the attacker leaving a space."""
        text = "ログインは" + sep + _HOMOGLYPH_TOKEN + sep + "へ"
        assert detect_unicode_attacks(text, checks=("homoglyph",))["total"] == 1

    @pytest.mark.parametrize("text", _MATH_JUXTAPOSED)
    @pytest.mark.parametrize("mode", ["auto", "never"])
    def test_juxtaposed_math_notation_is_clean(self, text: str, mode: str) -> None:
        """PASS: `𝐀x` is a bold matrix applied to a plain variable.

        The parenthesised form is cleared by the run boundary; juxtaposition
        is the more common shape and has no separator to split on, so the
        math-letter floor is what clears it.
        """
        result = detect_unicode_attacks(
            text, checks=("homoglyph",), single_script_spoofs=mode
        )
        assert result["homoglyph"] == [], text

    @pytest.mark.parametrize("text", _MATH_JUXTAPOSED)
    def test_juxtaposed_math_notation_convicts_under_always(self, text: str) -> None:
        """SCOPE: "always" asserts a Latin-only corpus free of mathematics."""
        result = detect_unicode_attacks(
            text, checks=("homoglyph",), single_script_spoofs="always"
        )
        assert result["homoglyph"] != [], text

    @pytest.mark.parametrize("spoof", _MATH_SPLICED_SPOOFS)
    @pytest.mark.parametrize("mode", ["auto", "always", "never"])
    def test_math_spliced_into_an_ascii_word_is_flagged(
        self, spoof: str, mode: str
    ) -> None:
        """DETECT: Three math letters replacing part of a word is a spoof.

        Keeps default coverage of the realistic math-homoglyph shape while the
        one-letter notation forms above stay clean.
        """
        result = detect_unicode_attacks(
            _in_english(spoof), checks=("homoglyph",), single_script_spoofs=mode
        )
        assert [f["token"] for f in result["homoglyph"]] == [spoof]
        assert result["homoglyph"][0]["reason"] == "mixed_script"

    def test_fullwidth_acronym_abutting_ascii_is_flagged(self) -> None:
        """DECISION: `ＵＲＬhttps` convicts -- it is the `payｐａｌ` shape.

        Fullwidth is Latin family, so no script boundary is drawn between the
        acronym and the ASCII word, and nothing distinguishes this from a
        spoof. Splitting here instead would drop `payｐａｌ`, which matters more.
        Japanese normally writes `ＵＲＬ：https`, which the punctuation splits --
        pinned below so the pair is not silently changed.
        """
        assert detect_unicode_attacks("ＵＲＬhttpsで接続", checks=("homoglyph",))[
            "total"
        ] == 1
        assert detect_unicode_attacks("ＵＲＬ：httpsで接続", checks=("homoglyph",))[
            "total"
        ] == 0

    def test_digits_do_not_split_a_spoofed_word(self) -> None:
        """DETECT: A digit inside the word must not hide the script mixture."""
        token = "p" + chr(0x0430) + "yp" + chr(0x0430) + "l1"
        result = detect_unicode_attacks("Go to " + token, checks=("homoglyph",))
        assert [f["reason"] for f in result["homoglyph"]] == ["mixed_script"]


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

    def test_emoji_zwj_sequence_passes_assertion(self) -> None:
        """PASS: Legitimate emoji ZWJ sequence does not raise."""
        result = assert_no_unicode_attacks(_EMOJI_MAN_TECH, checks=("zero_width",))
        assert result.passed is True

    def test_greek_scientific_token_passes_assertion(self) -> None:
        """PASS: Greek/Latin scientific token does not raise."""
        result = assert_no_unicode_attacks("α-helix", checks=("homoglyph",))
        assert result.passed is True

    def test_zwj_smuggle_raises(self) -> None:
        """FAIL/CRITICAL: ZWJ smuggled into plain text still raises."""
        with pytest.raises(MltkAssertionError):
            assert_no_unicode_attacks("ad" + _ZWJ + "min", checks=("zero_width",))

    def test_cyrillic_homoglyph_still_raises(self) -> None:
        """FAIL/CRITICAL: Cyrillic/Latin homoglyph still raises after Greek carve-out."""
        token = "p" + chr(0x0430) + "yp" + chr(0x0430) + "l"
        with pytest.raises(MltkAssertionError):
            assert_no_unicode_attacks("Visit " + token, checks=("homoglyph",))

    def test_russian_prose_passes_assertion(self) -> None:
        """PASS: A security gate must not fail a suite written in Russian."""
        result = assert_no_unicode_attacks(_RUSSIAN_SENTENCE, checks=("homoglyph",))
        assert result.passed is True

    def test_greek_prose_passes_assertion(self) -> None:
        """PASS: A security gate must not fail a suite written in Greek."""
        result = assert_no_unicode_attacks(_GREEK_SENTENCE, checks=("homoglyph",))
        assert result.passed is True

    def test_single_script_spoof_in_english_raises(self) -> None:
        """FAIL/CRITICAL: The same shape inside English prose is an attack."""
        with pytest.raises(MltkAssertionError):
            assert_no_unicode_attacks(
                _in_english(_CYRILLIC_SPOOF), checks=("homoglyph",)
            )

    def test_always_mode_raises_on_a_bare_candidate(self) -> None:
        """FAIL/CRITICAL: always opts a Latin-only corpus into the strict rule."""
        with pytest.raises(MltkAssertionError):
            assert_no_unicode_attacks(
                _REAL_CYRILLIC_WORDS[0],
                checks=("homoglyph",),
                single_script_spoofs="always",
            )

    def test_never_mode_passes_on_a_single_script_spoof(self) -> None:
        """PASS: never opts out of the whole-token rule entirely."""
        result = assert_no_unicode_attacks(
            _in_english(_CYRILLIC_SPOOF),
            checks=("homoglyph",),
            single_script_spoofs="never",
        )
        assert result.passed is True

    def test_details_record_the_active_spoof_mode(self) -> None:
        """STRUCTURE: A homoglyph result names the policy that produced it."""
        result = assert_no_unicode_attacks("clean text", checks=("homoglyph",))
        assert result.details["single_script_spoofs"] == "auto"

    def test_details_omit_spoof_mode_when_homoglyph_not_checked(self) -> None:
        """STRUCTURE: The mode is irrelevant when the category is not scanned."""
        result = assert_no_unicode_attacks("clean text", checks=("zero_width",))
        assert "single_script_spoofs" not in result.details

    def test_unknown_mode_raises_value_error(self) -> None:
        """GUARD: An unrecognised mode fails loudly rather than silently."""
        with pytest.raises(ValueError, match="single_script_spoofs"):
            assert_no_unicode_attacks("text", single_script_spoofs="maybe")


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
