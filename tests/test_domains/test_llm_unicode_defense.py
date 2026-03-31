"""Tests for unicode normalization defense in _utils.py.

These tests verify that the NFKC normalization and zero-width
character stripping in ``_tokenize`` and ``_normalize`` defend
against homoglyph, zero-width injection, and bidirectional
override attacks that could bypass token-overlap checks.
"""

from __future__ import annotations

from mltk.domains.llm._utils import _normalize, _tokenize

# ===================================================================
# _normalize defense tests
# ===================================================================


class TestNormalize:
    """Low-level normalization used before tokenization."""

    def test_nfkc_fullwidth_chars(self) -> None:
        """Fullwidth Latin letters normalize to ASCII."""
        # U+FF28=H, U+FF45=e, U+FF4C=l, U+FF4C=l, U+FF4F=o
        fullwidth = "\uff28\uff45\uff4c\uff4c\uff4f"
        assert _normalize(fullwidth) == "Hello"

    def test_zero_width_space_removed(self) -> None:
        """Zero-width space (U+200B) is stripped."""
        text = "hel\u200blo"
        assert _normalize(text) == "hello"

    def test_zero_width_joiner_removed(self) -> None:
        """Zero-width joiner (U+200D) is stripped."""
        text = "wor\u200dld"
        assert _normalize(text) == "world"

    def test_zero_width_nonjoiner_removed(self) -> None:
        """Zero-width non-joiner (U+200C) is stripped."""
        text = "te\u200cst"
        assert _normalize(text) == "test"

    def test_bidi_override_removed(self) -> None:
        """Right-to-left override (U+202E) is stripped."""
        text = "abc\u202edef"
        assert _normalize(text) == "abcdef"

    def test_bidi_embedding_removed(self) -> None:
        """Left-to-right embedding (U+202A) is stripped."""
        text = "abc\u202adef"
        assert _normalize(text) == "abcdef"

    def test_soft_hyphen_removed(self) -> None:
        """Soft hyphen (U+00AD) is stripped (Cf category)."""
        text = "pro\u00adgram"
        assert _normalize(text) == "program"

    def test_newline_preserved(self) -> None:
        """Newline chars survive the Cc filter."""
        text = "line1\nline2"
        assert _normalize(text) == "line1\nline2"

    def test_tab_preserved(self) -> None:
        """Tab chars survive the Cc filter."""
        text = "col1\tcol2"
        assert _normalize(text) == "col1\tcol2"

    def test_carriage_return_preserved(self) -> None:
        """Carriage return survives the Cc filter."""
        text = "line1\rline2"
        assert _normalize(text) == "line1\rline2"

    def test_combining_acute_composed(self) -> None:
        """Combining diacritical mark is composed by NFKC."""
        # e + combining acute accent = e-acute
        text = "caf\u0065\u0301"
        result = _normalize(text)
        assert "\u00e9" in result  # composed e-acute


# ===================================================================
# _tokenize unicode defense tests
# ===================================================================


class TestUnicodeDefense:
    """Tokenization resilience against unicode attacks."""

    def test_zero_width_no_phantom_tokens(self) -> None:
        """Zero-width chars do not create phantom tokens."""
        # Without normalization: "hel\u200blo" might split
        # into {"hel", "lo"} instead of {"hello"}
        tokens = _tokenize("hel\u200blo world")
        assert "hello" in tokens
        assert "world" in tokens
        assert len(tokens) == 2

    def test_homoglyph_normalized(self) -> None:
        """Fullwidth 'A' (U+FF21) normalizes to 'a' token."""
        # U+FF21 = fullwidth A -> NFKC -> A -> lower -> a
        tokens = _tokenize("\uff21pple")
        assert "apple" in tokens

    def test_cyrillic_a_survives(self) -> None:
        """Cyrillic 'a' (U+0430) is distinct from Latin 'a'.

        NFKC does NOT map Cyrillic to Latin — they are
        different scripts. This tests that NFKC preserves
        the distinction rather than collapsing them.
        """
        # Cyrillic small a = U+0430 (looks identical to 'a')
        tokens = _tokenize("\u0430pple")
        # Cyrillic a + "pple" = one token
        assert len(tokens) == 1
        # The token contains Cyrillic a, not Latin a
        token = list(tokens)[0]
        assert token[0] == "\u0430"

    def test_bidi_override_stripped(self) -> None:
        """Bidirectional override chars do not affect tokens."""
        text = "hello\u202e world\u202c test"
        tokens = _tokenize(text)
        assert "hello" in tokens
        assert "world" in tokens
        assert "test" in tokens

    def test_normal_text_tokenizes_correctly(self) -> None:
        """Regular text tokenizes the same as always."""
        tokens = _tokenize(
            "The quick brown fox jumps over the lazy dog"
        )
        expected = {
            "the", "quick", "brown", "fox", "jumps",
            "over", "lazy", "dog",
        }
        assert tokens == expected

    def test_mixed_attacks_in_claim(self) -> None:
        """Combined zero-width + fullwidth in one string."""
        # "P\u200Baris" with zero-width space + fullwidth I
        text = "P\u200baris is in \uff26rance"
        tokens = _tokenize(text)
        assert "paris" in tokens
        assert "france" in tokens

    def test_zero_width_between_words(self) -> None:
        """Zero-width between words does not merge them."""
        # Space is a real separator; zero-width inside a word
        text = "hello \u200b world"
        tokens = _tokenize(text)
        assert "hello" in tokens
        assert "world" in tokens

    def test_empty_after_stripping(self) -> None:
        """String of only invisible chars -> empty set."""
        text = "\u200b\u200c\u200d\u202e"
        tokens = _tokenize(text)
        assert tokens == set()

    def test_overlap_with_zero_width_attack(self) -> None:
        """Token overlap is not defeated by zero-width injection.

        An attacker might inject zero-width chars into claims to
        reduce overlap with source tokens. After normalization,
        the overlap should be the same as without the attack.
        """
        source = "Paris is the capital of France"
        # Attacker claim: zero-width space inside words
        attack = "Par\u200bis is the cap\u200bital"
        clean = "Paris is the capital"
        attack_tokens = _tokenize(attack)
        clean_tokens = _tokenize(clean)
        source_tokens = _tokenize(source)
        # Attack tokens should match clean tokens
        attack_overlap = len(attack_tokens & source_tokens)
        clean_overlap = len(clean_tokens & source_tokens)
        assert attack_overlap == clean_overlap

    def test_fullwidth_attack_on_overlap(self) -> None:
        """Fullwidth substitution does not defeat overlap.

        Replacing ASCII with fullwidth equivalents should
        produce the same tokens after NFKC normalization.
        """
        source = "Machine learning is powerful"
        # Fullwidth 'M' = U+FF2D
        attack = "\uff2dachine learning"
        attack_tokens = _tokenize(attack)
        source_tokens = _tokenize(source)
        assert "machine" in attack_tokens
        assert "machine" in source_tokens
        overlap = len(attack_tokens & source_tokens)
        assert overlap >= 2  # "machine" + "learning"


# ===================================================================
# Hardening: parametrized + edge-case tests (appended)
# ===================================================================


class TestNormalizeHardening:
    """Extra edge-case tests for _normalize."""

    def test_normalize_empty_string(self) -> None:
        """Empty string stays empty."""
        assert _normalize("") == ""

    def test_normalize_only_whitespace(self) -> None:
        """Whitespace-only string preserved as-is."""
        assert _normalize("   ") == "   "

    def test_normalize_mixed_scripts(self) -> None:
        """Latin + Cyrillic + Arabic coexist."""
        text = "Hello \u041f\u0440\u0438\u0432\u0435\u0442"
        text += " \u0645\u0631\u062d\u0628\u0627"
        result = _normalize(text)
        assert "Hello" in result
        # Cyrillic preserved
        assert "\u041f" in result or "\u043f" in result
        # Arabic preserved
        assert "\u0645" in result

    def test_normalize_combining_diacritics(self) -> None:
        """e + combining accent -> composed char."""
        import unicodedata
        raw = "caf\u0065\u0301"
        result = _normalize(raw)
        expected = unicodedata.normalize("NFKC", raw)
        assert result == expected
        assert "\u00e9" in result

    def test_normalize_fullwidth_ascii(self) -> None:
        """Fullwidth digits and letters normalize."""
        fw = "\uff11\uff12\uff13\uff21\uff22"
        result = _normalize(fw)
        assert result == "123AB"

    def test_normalize_hangul_jamo(self) -> None:
        """Korean compatibility jamo normalized."""
        import unicodedata
        compat = "\u3131\u3132\u3133"
        result = _normalize(compat)
        expected = unicodedata.normalize("NFKC", compat)
        assert result == expected

    def test_normalize_idempotent(self) -> None:
        """normalize(normalize(x)) == normalize(x)."""
        texts = [
            "Hello\u200bWorld",
            "\uff21\uff22\uff23",
            "caf\u0065\u0301",
            "\u202eReverse\u202c",
            "",
        ]
        for t in texts:
            once = _normalize(t)
            twice = _normalize(once)
            assert once == twice


class TestTokenizeHardening:
    """Extra edge-case tests for _tokenize."""

    def test_tokenize_zero_width_chars(self) -> None:
        """ZWJ, ZWNJ, ZWSP stripped before tokenizing."""
        text = "hel\u200dlo\u200cwor\u200bld"
        tokens = _tokenize(text)
        assert "helloworld" in tokens or (
            "hello" in tokens and "world" in tokens
        )

    def test_tokenize_preserves_numbers(self) -> None:
        """Alphanumeric tokens preserved."""
        tokens = _tokenize("test123 hello456")
        assert "test123" in tokens
        assert "hello456" in tokens

    def test_tokenize_empty_after_strip(self) -> None:
        """All zero-width chars -> empty token set."""
        text = (
            "\u200b\u200c\u200d\ufeff"
            "\u200b\u200c\u200d"
        )
        tokens = _tokenize(text)
        assert tokens == set()
