"""NLP generation quality testing -- BLEU, ROUGE, and chrF scores."""

from __future__ import annotations

from mltk.core.assertion import assert_true, timed_assertion
from mltk.core.result import Severity, TestResult
from mltk.core.validation import require_same_length


@timed_assertion
def assert_bleu(
    references: list[str],
    hypotheses: list[str],
    min_score: float = 0.3,
) -> TestResult:
    """Assert BLEU score meets minimum threshold.

    Args:
        references: Reference translations/texts. Must have the same length as
            hypotheses; mismatched lengths raise ``ValueError``.
        hypotheses: Model-generated translations/texts.
        min_score: Minimum required BLEU score (0-1).

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

    try:
        from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
    except ImportError as err:
        raise ImportError(
            "nltk is required for BLEU scoring. Install: pip install mlspec[nlp]"
        ) from err

    # Tokenize
    ref_tokens = [[ref.split()] for ref in references]
    hyp_tokens = [hyp.split() for hyp in hypotheses]

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
    )


@timed_assertion
def assert_rouge(
    references: list[str],
    hypotheses: list[str],
    variant: str = "rougeL",
    min_score: float = 0.3,
) -> TestResult:
    """Assert ROUGE score meets minimum threshold.

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
        from rouge_score import rouge_scorer
    except ImportError as err:
        raise ImportError(
            "rouge-score is required for ROUGE scoring. Install: pip install mlspec[nlp]"
        ) from err

    scorer = rouge_scorer.RougeScorer([variant], use_stemmer=True)

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
