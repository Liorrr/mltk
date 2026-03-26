"""Tests for mltk.domains.llm._utils — shared LLM tokenization helper."""

from mltk.domains.llm._utils import _tokenize


class TestTokenize:
    """Unit tests for the canonical ``_tokenize`` helper.

    ``_tokenize`` is used by rag, ragas, agentic, coherence, conversation,
    and (after Run 2) safety to normalise text before token-overlap
    computations.  Any future change to the function's behaviour (e.g.
    switching from ``\\w`` to a Unicode-aware regex) will be caught here
    before it silently alters overlap scores across five callers.
    """

    def test_punctuation_stripped(self) -> None:
        """Punctuation is removed before splitting.

        WHY: Claims like "Paris, capital of France!" must match source tokens
        "Paris" and "France" without the comma and exclamation mark
        interfering with set membership.
        """
        tokens = _tokenize("Hello, world!")
        assert tokens == {"hello", "world"}

    def test_lowercase(self) -> None:
        """All tokens are lowercased.

        WHY: Token overlap checks must be case-insensitive so "Python" and
        "python" are treated as the same word.
        """
        tokens = _tokenize("Python IS great")
        assert "python" in tokens
        assert "is" in tokens
        assert "great" in tokens
        # no uppercase variants should survive
        assert "Python" not in tokens
        assert "IS" not in tokens

    def test_empty_string(self) -> None:
        """Empty string produces an empty set, not an error.

        WHY: ``assert_no_hallucination`` checks ``if not claim_tokens``
        to skip empty claims.  That guard only works correctly if
        ``_tokenize("")`` returns an empty set rather than, say, ``{""}``
        or raising an exception.
        """
        assert _tokenize("") == set()

    def test_unicode_text(self) -> None:
        """Non-ASCII characters are preserved in lowercased tokens.

        WHY: LLM outputs often contain accented characters, CJK text, or
        emoji-adjacent punctuation.  The ``\\w`` pattern in the regex is
        Unicode-aware in Python 3, so letters such as 'e' with an accent
        survive stripping while adjacent punctuation is removed.
        """
        tokens = _tokenize("Café au lait, s'il vous plaît!")
        # accented letters survive; punctuation (comma, !, ') is removed
        assert "café" in tokens
        assert "lait" in tokens
        # punctuation-only tokens must not appear
        assert "!" not in tokens
        assert "," not in tokens
