"""Unicode-aware ROUGE tokenization and the opt-in BLEU tokenizer.

The ROUGE parity tests are the mirror of the Unicode fix: on ASCII text the
new tokenizer must give exactly the tokens and scores of rouge-score's default
tokenizer, so English results do not move.
"""

from __future__ import annotations

import random
import string

import pytest

from mltk.core.assertion import MltkAssertionError
from mltk.domains.nlp.generation import (
    _unicode_words,
    _UnicodeRougeTokenizer,
    assert_bleu,
    assert_rouge,
)

# ASCII corpus with punctuation, digits, case, underscores, contractions,
# hyphens, newlines (rougeLsum) and words the Porter stemmer changes.
ASCII_CORPUS = [
    "The cats are running quickly!",
    "A cat ran quick.",
    "Running, runner, runs -- RAN; run?",
    "In 2024, revenue grew 12.5% to $3,400,000 (year-over-year).",
    "Revenue grew 12% in 2024 to $3.4M year over year",
    "snake_case and kebab-case and CamelCase identifiers",
    "It's the user's choice; they're connected... aren't they?",
    "Line one of the summary.\nLine two, generalization and connection.\nLine 3",
    "   leading and trailing whitespace\t\ttabs   ",
    "e-mail: someone@example.com, URL https://example.com/a_b?c=1",
    "studies studying studied student",
    "abc ab a abcd abcde",
    "",
    "!!! ??? ...",
]

PAIRS = [
    (ASCII_CORPUS[0], ASCII_CORPUS[1]),
    (ASCII_CORPUS[2], ASCII_CORPUS[10]),
    (ASCII_CORPUS[3], ASCII_CORPUS[4]),
    (ASCII_CORPUS[5], ASCII_CORPUS[9]),
    (ASCII_CORPUS[6], ASCII_CORPUS[6].upper()),
    (ASCII_CORPUS[7], "Line two, generalization.\nLine one of the summary"),
    (ASCII_CORPUS[8], ASCII_CORPUS[11]),
]


def _random_ascii(rng: random.Random, n: int) -> str:
    alphabet = string.printable  # letters, digits, punctuation, whitespace
    return "".join(rng.choice(alphabet) for _ in range(n))


class FakeStemmer:
    """Marks every stemmed token so tests can see which tokens were stemmed."""

    def stem(self, token: str) -> str:
        return f"<{token}>"


# ---------------------------------------------------------------------------
# Tokenizer behaviour (no optional dependencies)
# ---------------------------------------------------------------------------


class TestUnicodeWords:
    def test_hebrew_words_kept(self) -> None:
        assert _unicode_words("שלום, עולם!") == ["שלום", "עולם"]

    def test_combining_marks_stay_inside_the_word(self) -> None:
        # Hebrew niqqud and Devanagari vowel signs are combining marks (M*).
        assert _unicode_words("שָׁלוֹם עוֹלָם") == ["שָׁלוֹם", "עוֹלָם"]
        assert _unicode_words("नमस्ते दुनिया") == ["नमस्ते", "दुनिया"]

    def test_casefold_and_separators(self) -> None:
        assert _unicode_words("Straße_ÉTÉ--Ωμέγα") == ["strasse", "été", "ωμέγα"]

    def test_cjk_is_one_token_per_character(self) -> None:
        # Chinese and Japanese have no spaces between words.
        assert _unicode_words("Привет мир。你好") == ["привет", "мир", "你", "好"]
        assert _unicode_words("ひらがなとカタカナ") == list("ひらがなとカタカナ")
        assert _unicode_words("東京abc々") == ["東", "京", "abc", "々"]

    def test_hangul_stays_word_based(self) -> None:
        assert _unicode_words("안녕하세요 세계") == ["안녕하세요", "세계"]

    def test_digits_kept_punctuation_dropped(self) -> None:
        assert _unicode_words("v2.0 (beta) #1") == ["v2", "0", "beta", "1"]


class TestUnicodeRougeTokenizer:
    def test_stems_only_ascii_tokens_longer_than_three(self) -> None:
        tok = _UnicodeRougeTokenizer(FakeStemmer())
        assert tok.tokenize("abc abcd שלוםם café") == [
            "abc", "<abcd>", "שלוםם", "café",
        ]


# ---------------------------------------------------------------------------
# English parity with rouge-score's default tokenizer (mirror test)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def default_tokenizer():  # noqa: ANN201
    pytest.importorskip("rouge_score", reason="rouge-score required")
    from rouge_score.tokenizers import DefaultTokenizer

    return DefaultTokenizer(use_stemmer=True)


@pytest.fixture(scope="module")
def unicode_tokenizer() -> _UnicodeRougeTokenizer:
    porter = pytest.importorskip("nltk.stem.porter", reason="nltk required")
    return _UnicodeRougeTokenizer(porter.PorterStemmer())


class TestEnglishParity:
    @pytest.mark.parametrize("text", ASCII_CORPUS)
    def test_tokens_identical_on_ascii_corpus(
        self, text: str, default_tokenizer, unicode_tokenizer,  # noqa: ANN001
    ) -> None:
        assert unicode_tokenizer.tokenize(text) == default_tokenizer.tokenize(text)

    def test_tokens_identical_on_random_ascii(
        self, default_tokenizer, unicode_tokenizer,  # noqa: ANN001
    ) -> None:
        rng = random.Random(20261115)
        for _ in range(400):
            text = _random_ascii(rng, rng.randint(0, 60))
            assert unicode_tokenizer.tokenize(text) == default_tokenizer.tokenize(text), text

    @pytest.mark.parametrize("variant", ["rouge1", "rouge2", "rougeL", "rougeLsum"])
    def test_scores_identical_to_default(self, variant: str) -> None:
        pytest.importorskip("rouge_score", reason="rouge-score required")
        from rouge_score import rouge_scorer

        default = rouge_scorer.RougeScorer([variant], use_stemmer=True)
        refs = [r for r, _ in PAIRS]
        hyps = [h for _, h in PAIRS]
        expected = sum(
            default.score(r, h)[variant].fmeasure for r, h in PAIRS
        ) / len(PAIRS)
        result = assert_rouge(refs, hyps, variant=variant, min_score=0.0)
        assert result.details["score"] == expected

    def test_accented_latin_is_the_documented_difference(
        self, default_tokenizer, unicode_tokenizer,  # noqa: ANN001
    ) -> None:
        # rouge-score's default drops "é" and leaves "caf"; the new tokenizer
        # keeps the word. This is the one expected change for Latin text.
        assert default_tokenizer.tokenize("café") == ["caf"]
        assert unicode_tokenizer.tokenize("café") == ["café"]


# ---------------------------------------------------------------------------
# Non-ASCII ROUGE
# ---------------------------------------------------------------------------


class TestUnicodeRouge:
    def test_identical_hebrew_scores_one(self) -> None:
        pytest.importorskip("rouge_score", reason="rouge-score required")
        text = "המודל ענה על השאלה בצורה נכונה ומדויקת"
        result = assert_rouge([text], [text], variant="rougeL", min_score=0.99)
        assert result.details["score"] == 1.0

    def test_default_tokenizer_scored_identical_hebrew_zero(self) -> None:
        # Mirror: the bug this fixes. The default tokenizer sees no tokens.
        pytest.importorskip("rouge_score", reason="rouge-score required")
        from rouge_score import rouge_scorer

        text = "המודל ענה על השאלה"
        legacy = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        assert legacy.score(text, text)["rougeL"].fmeasure == 0.0

    def test_unrelated_hebrew_fails(self) -> None:
        pytest.importorskip("rouge_score", reason="rouge-score required")
        with pytest.raises(MltkAssertionError):
            assert_rouge(["שלום עולם"], ["חתול כלב"], min_score=0.3)

    def test_near_identical_chinese_scores_partial_overlap(self) -> None:
        # Mirror of the CJK fix: as whole clauses these shared nothing (0.0).
        pytest.importorskip("rouge_score", reason="rouge-score required")
        result = assert_rouge(
            ["我喜欢吃苹果和香蕉"], ["我喜欢吃苹果和橘子"],
            variant="rouge1", min_score=0.0,
        )
        assert result.details["score"] == pytest.approx(7 / 9)

    def test_partial_hebrew_overlap(self) -> None:
        pytest.importorskip("rouge_score", reason="rouge-score required")
        result = assert_rouge(
            ["אחד שניים שלושה ארבעה"], ["אחד שניים חמש שש"],
            variant="rouge1", min_score=0.0,
        )
        assert result.details["score"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# BLEU tokenizer option
# ---------------------------------------------------------------------------


class TestBleuTokenizer:
    def test_whitespace_is_default_and_unchanged(self) -> None:
        nltk_bleu = pytest.importorskip("nltk.translate.bleu_score")
        refs = ["The cat sat on the mat.", "A quick brown fox jumps"]
        hyps = ["The cat is on the mat.", "A quick brown dog jumps"]
        expected = float(nltk_bleu.corpus_bleu(
            [[r.split()] for r in refs], [h.split() for h in hyps],
            smoothing_function=nltk_bleu.SmoothingFunction().method1,
        ))
        result = assert_bleu(refs, hyps, min_score=0.0)
        assert result.details["score"] == expected
        assert result.details["tokenizer"] == "whitespace"
        explicit = assert_bleu(refs, hyps, min_score=0.0, tokenizer="whitespace")
        assert explicit.details["score"] == expected

    def test_unicode_ignores_case_and_punctuation(self) -> None:
        pytest.importorskip("nltk")
        refs = ["שלום, עולם! מה שלומך היום?"]
        hyps = ["שלום עולם מה שלומך היום"]
        unicode = assert_bleu(refs, hyps, min_score=0.99, tokenizer="unicode")
        assert unicode.details["score"] == pytest.approx(1.0)
        assert unicode.details["tokenizer"] == "unicode"
        whitespace = assert_bleu(refs, hyps, min_score=0.0)
        assert whitespace.details["score"] < unicode.details["score"]

    def test_unknown_tokenizer_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown tokenizer"):
            assert_bleu(["a"], ["a"], tokenizer="spacy")  # type: ignore[arg-type]

    def test_tokenizer_is_keyword_only(self) -> None:
        with pytest.raises(TypeError):
            assert_bleu(["a"], ["a"], 0.3, "unicode")  # type: ignore[misc]
