"""NLP generation quality testing -- BLEU, ROUGE, and chrF scores."""

from __future__ import annotations

import unicodedata
from typing import Any, Literal

from mltk.core.assertion import assert_true, timed_assertion
from mltk.core.result import Severity, TestResult
from mltk.core.validation import require_same_length

BleuTokenizer = Literal["whitespace", "unicode"]
_BLEU_TOKENIZERS: tuple[str, ...] = ("whitespace", "unicode")


# Chinese and Japanese are written without spaces between words, so a run of
# letters there is a whole clause. Han ideographs and kana are tokenized one
# character each instead (the usual choice for CJK ROUGE/BLEU). Hangul is
# left word-based: Korean separates words with spaces.
_CJK_RANGES: tuple[tuple[int, int], ...] = (
    (0x3005, 0x3007),    # 々 〆 〇
    (0x3040, 0x309F),    # Hiragana
    (0x30A0, 0x30FF),    # Katakana
    (0x31F0, 0x31FF),    # Katakana Phonetic Extensions
    (0x3400, 0x4DBF),    # CJK Unified Ideographs Extension A
    (0x4E00, 0x9FFF),    # CJK Unified Ideographs
    (0xF900, 0xFAFF),    # CJK Compatibility Ideographs
    (0xFF66, 0xFF9F),    # Halfwidth Katakana
    (0x20000, 0x3FFFF),  # Supplementary and Tertiary Ideographic Planes
)


def _is_cjk_char(ch: str) -> bool:
    cp = ord(ch)
    return cp >= 0x3005 and any(lo <= cp <= hi for lo, hi in _CJK_RANGES)


def _unicode_words(text: str) -> list[str]:
    """Casefold *text* and split it into runs of letters, digits and marks.

    Every character whose Unicode category is not a letter (L*), number (N*)
    or combining mark (M*) separates tokens, so punctuation, symbols,
    whitespace and ``_`` never appear in a token. Keeping combining marks
    keeps words in scripts such as Hebrew with niqqud or Devanagari whole.
    Han ideographs and kana become one token per character, because Chinese
    and Japanese do not separate words with spaces. Other scripts written
    without spaces (Thai, Lao, Khmer, Myanmar) stay one token per run; use
    chrF for them. On ASCII text this matches rouge-score's default split
    exactly.
    """
    chars: list[str] = []
    for ch in text.casefold():
        if unicodedata.category(ch)[0] not in "LNM":
            # Separators first: the kana blocks also hold punctuation and
            # symbols (・ ゠ ゛ ゜), which must not become tokens.
            chars.append(" ")
        elif _is_cjk_char(ch):
            chars.append(f" {ch} ")
        else:
            chars.append(ch)
    return "".join(chars).split()


class _UnicodeRougeTokenizer:
    """rouge-score tokenizer that keeps non-ASCII words.

    rouge-score's default tokenizer drops every character outside
    ``[a-z0-9]``, so Hebrew, Arabic, CJK or Cyrillic text tokenizes to
    nothing and identical texts score 0. This tokenizer splits with
    :func:`_unicode_words` (Chinese and Japanese per character) and, like
    the default, Porter-stems tokens longer than three characters, but only
    ASCII ones; other scripts are kept as written. On ASCII input the tokens
    equal the default tokenizer's.
    """

    def __init__(self, stemmer: Any) -> None:
        self._stemmer = stemmer

    def tokenize(self, text: str) -> list[str]:
        return [
            self._stemmer.stem(tok) if len(tok) > 3 and tok.isascii() else tok
            for tok in _unicode_words(text)
        ]


@timed_assertion
def assert_bleu(
    references: list[str],
    hypotheses: list[str],
    min_score: float = 0.3,
    *,
    tokenizer: BleuTokenizer = "whitespace",
) -> TestResult:
    """Assert BLEU score meets minimum threshold.

    BLEU matches whole word n-grams, so it under-scores morphologically rich
    languages such as Hebrew, Arabic or Finnish, where one word carries
    prefixes and suffixes. Prefer :func:`assert_chrf` for those languages.

    Args:
        references: Reference translations/texts. Must have the same length as
            hypotheses; mismatched lengths raise ``ValueError``. An empty
            corpus fails rather than scoring.
        hypotheses: Model-generated translations/texts.
        min_score: Minimum required BLEU score (0-1).
        tokenizer: ``"whitespace"`` (default) splits on whitespace and keeps
            case and punctuation. ``"unicode"`` casefolds and splits into
            runs of Unicode letters, digits and combining marks, dropping
            punctuation and symbols, with Chinese and Japanese characters
            as one token each.

    Returns:
        TestResult with BLEU score.

    Example:
        >>> refs = ["the cat sat on the mat"]
        >>> hyps = ["the cat is on the mat"]
        >>> assert_bleu(refs, hyps, min_score=0.2)
    """
    require_same_length(
        "assert_bleu", references=references, hypotheses=hypotheses,
    )
    if tokenizer not in _BLEU_TOKENIZERS:
        raise ValueError(
            f"Unknown tokenizer {tokenizer!r}; "
            f"supported: {', '.join(_BLEU_TOKENIZERS)}"
        )
    # Two empty lists have the same length, so they reach corpus_bleu
    # and divide by zero. Fail as an assertion, the way assert_chrf and
    # assert_rouge already do, rather than raising out of nltk.
    if len(references) == 0:
        return assert_true(
            False,
            name="nlp.bleu",
            message="BLEU: empty corpus",
            severity=Severity.CRITICAL,
            score=None,
            min_score=min_score,
            num_references=0,
            num_hypotheses=0,
            tokenizer=tokenizer,
        )

    try:
        from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
    except ImportError as err:
        raise ImportError(
            "nltk is required for BLEU scoring. Install: pip install mlspec[nlp]"
        ) from err

    split = _unicode_words if tokenizer == "unicode" else str.split
    ref_tokens = [[split(ref)] for ref in references]
    hyp_tokens = [split(hyp) for hyp in hypotheses]

    smoothie = SmoothingFunction().method1
    score = float(corpus_bleu(ref_tokens, hyp_tokens, smoothing_function=smoothie))

    passed = score >= min_score
    message = (
        f"BLEU: {score:.4f} >= {min_score}"
        if passed
        else f"BLEU: {score:.4f} < {min_score}"
    )

    return assert_true(
        passed, name="nlp.bleu", message=message,
        severity=Severity.CRITICAL,
        score=score, min_score=min_score,
        num_references=len(references), num_hypotheses=len(hypotheses),
        tokenizer=tokenizer,
    )


@timed_assertion
def assert_rouge(
    references: list[str],
    hypotheses: list[str],
    variant: str = "rougeL",
    min_score: float = 0.3,
) -> TestResult:
    """Assert ROUGE score meets minimum threshold.

    Texts are tokenized Unicode-aware: casefolded, split on every character
    that is not a letter, digit or combining mark, with Chinese and Japanese
    characters as one token each, and Porter-stemmed for ASCII tokens longer
    than three characters. On ASCII text the scores are
    identical to rouge-score's default tokenizer. Unlike that default, which
    drops every non-ASCII character, words in other scripts and accented
    Latin letters are kept, so identical Hebrew texts score 1.0, not 0.

    Args:
        references: Reference texts. Must have the same length as hypotheses.
        hypotheses: Model-generated texts.
        variant: ROUGE variant -- "rouge1", "rouge2", "rougeL", "rougeLsum".
        min_score: Minimum required F-measure (0-1).

    Returns:
        TestResult with ROUGE scores.

    Example:
        >>> refs = ["the cat sat on the mat"]
        >>> hyps = ["the cat is on the mat"]
        >>> assert_rouge(refs, hyps, variant="rougeL", min_score=0.3)
    """
    require_same_length("assert_rouge", references=references, hypotheses=hypotheses)

    try:
        from nltk.stem import porter
        from rouge_score import rouge_scorer
    except ImportError as err:
        raise ImportError(
            "rouge-score is required for ROUGE scoring. Install: pip install mlspec[nlp]"
        ) from err

    scorer = rouge_scorer.RougeScorer(
        [variant], tokenizer=_UnicodeRougeTokenizer(porter.PorterStemmer()),
    )

    scores = []
    for ref, hyp in zip(references, hypotheses, strict=True):
        result = scorer.score(ref, hyp)
        scores.append(result[variant].fmeasure)

    avg_score = sum(scores) / len(scores) if scores else 0.0

    passed = avg_score >= min_score
    message = (
        f"ROUGE-{variant}: {avg_score:.4f} >= {min_score}"
        if passed
        else f"ROUGE-{variant}: {avg_score:.4f} < {min_score}"
    )

    return assert_true(
        passed, name=f"nlp.rouge.{variant}", message=message,
        severity=Severity.CRITICAL,
        score=avg_score, variant=variant, min_score=min_score,
    )


@timed_assertion
def assert_chrf(
    references: list[str],
    hypotheses: list[str],
    min_score: float = 0.3,
    *,
    word_order: int = 0,
) -> TestResult:
    """Assert corpus chrF (Popović 2015) meets minimum threshold.

    Score is reported in ``[0, 1]``. ``word_order=2`` is chrF++.

    Args:
        references: Reference texts. Must have the same length as hypotheses.
        hypotheses: Model-generated texts.
        min_score: Minimum required chrF score (0-1).
        word_order: Character n-gram order extra word n-grams. ``0`` is
            chrF; ``2`` is chrF++.

    Returns:
        TestResult with chrF score.

    Example:
        >>> refs = ["the cat sat on the mat"]
        >>> hyps = ["the cat is on the mat"]
        >>> assert_chrf(refs, hyps, min_score=0.2)
    """
    require_same_length(
        "assert_chrf", references=references, hypotheses=hypotheses,
    )
    name = "nlp.chrfpp" if word_order == 2 else "nlp.chrf"
    if len(references) == 0:
        return assert_true(
            False,
            name=name,
            message="chrF: empty corpus",
            severity=Severity.CRITICAL,
            score=None,
            min_score=min_score,
            word_order=word_order,
        )

    try:
        from sacrebleu.metrics import CHRF
    except ImportError as err:
        raise ImportError(
            "sacrebleu is required for chrF scoring. "
            "Install: pip install mlspec[nlp]"
        ) from err

    raw = CHRF(word_order=word_order).corpus_score(
        hypotheses, [references],
    ).score
    score = float(raw) / 100.0
    passed = score >= min_score
    label = "chrF++" if word_order == 2 else "chrF"
    message = (
        f"{label}: {score:.4f} >= {min_score}"
        if passed
        else f"{label}: {score:.4f} < {min_score}"
    )
    return assert_true(
        passed,
        name=name,
        message=message,
        severity=Severity.CRITICAL,
        score=score,
        min_score=min_score,
        word_order=word_order,
        num_references=len(references),
        num_hypotheses=len(hypotheses),
    )
