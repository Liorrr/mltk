"""Vision-Language Model (VLM) evaluation assertions.

VLMs (GPT-4o, Claude, Gemini, LLaVA) answer questions about images.
Evaluating VLM outputs requires checking two properties:

1. **Image helpfulness**: Does the image actually help answer the
   question?  An irrelevant image wastes context and can mislead
   the model.

2. **VQA accuracy**: Is the model's answer to a visual question
   correct?  VQA (Visual Question Answering) is the standard task
   for measuring VLM comprehension.

Both assertions use the LLM-as-Judge pattern established in
``mltk.domains.llm.judge``: the user provides a ``judge_fn``
callable that sends a prompt to any LLM and returns the response.
mltk builds the evaluation prompt; the user owns the LLM call.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable

from mltk.core.assertion import assert_true, timed_assertion
from mltk.core.result import Severity, TestResult
from mltk.domains.multimodal._image import (
    ImageInput,
    _build_image_prompt,
)

__all__ = [
    "assert_image_helpfulness",
    "assert_vqa_accuracy",
    "assert_ocr_accuracy",
]

# ---------------------------------------------------------------
# Score parsing (reused from judge.py pattern)
# ---------------------------------------------------------------

_FLOAT_PATTERN = re.compile(r"(\d+(?:\.\d+)?)")


def _parse_score(raw: str) -> float | None:
    """Extract the first numeric value from a judge response."""
    # Try JSON first
    try:
        data = json.loads(raw)
        if isinstance(data, dict) and "score" in data:
            return float(data["score"])
    except (json.JSONDecodeError, ValueError, TypeError):
        pass
    # Fallback to regex
    match = _FLOAT_PATTERN.search(raw.strip())
    if match:
        return float(match.group(1))
    return None


# ---------------------------------------------------------------
# Assertions
# ---------------------------------------------------------------


@timed_assertion
def assert_image_helpfulness(
    question: str,
    image: ImageInput | None,
    answer: str,
    judge_fn: Callable[[str], str],
    min_score: float = 0.7,
    image_description: str | None = None,
) -> TestResult:
    """Assert that an image helps understand or answer a question.

    An image is "helpful" when it provides visual context that
    improves comprehension of the question or validates the answer.
    An irrelevant or misleading image is a common failure mode in
    RAG systems that retrieve images alongside text.

    The assertion builds an evaluation prompt asking the judge to
    score how helpful the image is for understanding the question
    and answer, on a 0-1 scale.

    Args:
        question: The question being asked about or with the image.
        image: Image source (path, bytes) or None if using
            image_description.
        answer: The answer produced (by a VLM or human).
        judge_fn: Callable that takes a prompt string and returns
            the judge's response (should contain a score 0-1).
        min_score: Minimum helpfulness score to pass (default 0.7).
        image_description: Optional pre-computed text description
            of the image.  If provided, the image parameter is
            ignored and Pillow is not needed.

    Returns:
        TestResult with details: ``score``, ``min_score``,
        ``question``, ``answer``.

    Raises:
        MltkAssertionError: If score < min_score (CRITICAL).

    Example:
        >>> def mock_judge(prompt: str) -> str:
        ...     return '{"score": 0.9, "reasoning": "good"}'
        >>> result = assert_image_helpfulness(
        ...     question="What color is the car?",
        ...     image=None,
        ...     answer="The car is red.",
        ...     judge_fn=mock_judge,
        ...     image_description="A red sedan in a parking lot.",
        ... )
        >>> result.passed
        True
    """
    instruction = (
        "You are an impartial evaluation judge.\n\n"
        "## Task\n"
        "Rate how helpful the image is for understanding or "
        "answering the following question.\n\n"
        f"## Question\n{question}\n\n"
        f"## Answer\n{answer}\n\n"
        "## Scoring\n"
        "Return a JSON object with 'score' (0.0 to 1.0) and "
        "'reasoning' (brief explanation).\n"
        "A score of 1.0 means the image is essential for "
        "understanding the answer.\n"
        "A score of 0.0 means the image is completely "
        "irrelevant or misleading.\n"
    )

    prompt = _build_image_prompt(
        instruction,
        image=image,
        image_description=image_description,
    )

    score: float | None = None
    error: str | None = None

    try:
        raw = judge_fn(prompt)
        score = _parse_score(str(raw))
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    if score is None:
        score = 0.0
        if error is None:
            error = "Could not parse score from judge response"

    passed = score >= min_score

    message = (
        f"Image helpfulness: {score:.4f} >= "
        f"{min_score}"
        if passed
        else f"Image not helpful enough: {score:.4f} < "
        f"{min_score}"
    )

    return assert_true(
        passed,
        name="multimodal.vlm.image_helpfulness",
        message=message,
        severity=Severity.CRITICAL,
        score=round(score, 4),
        min_score=min_score,
        question=question,
        answer=answer,
        error=error,
    )


@timed_assertion
def assert_vqa_accuracy(
    question: str,
    image: ImageInput | None,
    expected_answer: str,
    actual_answer: str,
    judge_fn: Callable[[str], str] | None = None,
    min_score: float = 0.7,
    image_description: str | None = None,
) -> TestResult:
    """Assert that a VQA answer is correct.

    Visual Question Answering (VQA) is the standard task for
    evaluating VLM comprehension: given an image and a question,
    does the model produce the correct answer?

    Two evaluation modes:

    1. **With judge_fn**: An LLM judge scores semantic equivalence
       between expected and actual answers (handles paraphrasing,
       different wording for the same meaning).

    2. **Without judge_fn** (text matching): Normalized exact match.
       Case-insensitive, whitespace-stripped comparison. Fast but
       brittle -- "a red car" != "red car" even though both are
       correct. Use this only for short, unambiguous answers
       (colors, counts, yes/no).

    Args:
        question: The question asked about the image.
        image: Image source (path, bytes) or None if using
            image_description or text-only matching.
        expected_answer: The ground-truth correct answer.
        actual_answer: The model's answer to evaluate.
        judge_fn: Optional callable for semantic comparison.
            If None, uses normalized exact match.
        min_score: Minimum score to pass (default 0.7).
            For exact match mode: 1.0 if match, 0.0 if not.
        image_description: Optional pre-computed text description
            of the image.

    Returns:
        TestResult with details: ``score``, ``min_score``,
        ``question``, ``expected_answer``, ``actual_answer``,
        ``method``.

    Raises:
        MltkAssertionError: If score < min_score (CRITICAL).

    Example:
        >>> result = assert_vqa_accuracy(
        ...     question="How many dogs?",
        ...     image=None,
        ...     expected_answer="2",
        ...     actual_answer="2",
        ... )
        >>> result.passed
        True
    """
    score: float | None = None
    error: str | None = None
    method: str

    if judge_fn is not None:
        method = "llm_judge"
        instruction = (
            "You are an impartial evaluation judge.\n\n"
            "## Task\n"
            "Rate how well the actual answer matches the "
            "expected answer for the given visual question.\n\n"
            f"## Question\n{question}\n\n"
            f"## Expected Answer\n{expected_answer}\n\n"
            f"## Actual Answer\n{actual_answer}\n\n"
            "## Scoring\n"
            "Return a JSON object with 'score' (0.0 to 1.0) "
            "and 'reasoning'.\n"
            "1.0 = semantically identical answers.\n"
            "0.0 = completely wrong answer.\n"
        )

        if image is not None or image_description is not None:
            prompt = _build_image_prompt(
                instruction,
                image=image,
                image_description=image_description,
            )
        else:
            prompt = instruction

        try:
            raw = judge_fn(prompt)
            score = _parse_score(str(raw))
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
    else:
        method = "exact_match"
        norm_expected = expected_answer.strip().lower()
        norm_actual = actual_answer.strip().lower()
        score = 1.0 if norm_expected == norm_actual else 0.0

    if score is None:
        score = 0.0
        if error is None:
            error = "Could not parse score from judge response"

    passed = score >= min_score

    message = (
        f"VQA accuracy ({method}): {score:.4f} >= "
        f"{min_score}"
        if passed
        else f"VQA accuracy too low ({method}): "
        f"{score:.4f} < {min_score}"
    )

    return assert_true(
        passed,
        name="multimodal.vlm.vqa_accuracy",
        message=message,
        severity=Severity.CRITICAL,
        score=round(score, 4),
        min_score=min_score,
        question=question,
        expected_answer=expected_answer,
        actual_answer=actual_answer,
        method=method,
        error=error,
    )


# ---------------------------------------------------------------
# OCR accuracy (CER / WER)
# ---------------------------------------------------------------


def _levenshtein_distance(a: list, b: list) -> int:
    """Compute Levenshtein (edit) distance between two sequences.

    Uses the classic O(n*m) dynamic programming algorithm with
    O(n) space optimization (single-row DP array).

    Levenshtein distance counts the minimum number of single-element
    edits (insertions, deletions, substitutions) needed to transform
    sequence ``a`` into sequence ``b``.

    Works with any comparable elements -- characters (for CER) or
    words (for WER).

    Args:
        a: Source sequence (list of comparable elements).
        b: Target sequence (list of comparable elements).

    Returns:
        Integer edit distance (0 = identical sequences).

    Example:
        >>> _levenshtein_distance(list("kitten"), list("sitting"))
        3
    """
    m, n = len(a), len(b)
    dp = list(range(n + 1))

    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp

    return dp[n]


@timed_assertion
def assert_ocr_accuracy(
    expected_text: str,
    actual_text: str,
    method: str = "cer",
    threshold: float = 0.1,
) -> TestResult:
    """Assert that OCR output is accurate enough vs ground truth.

    Optical Character Recognition (OCR) extracts text from images.
    This assertion measures extraction accuracy using standard
    error rate metrics -- no external dependencies required.

    **Two methods:**

    1. **CER** (Character Error Rate): Levenshtein distance on
       individual characters, divided by the reference length.
       Sensitive to single-character errors -- ideal for OCR where
       "Iuvoice" vs "Invoice" is a meaningful failure.

    2. **WER** (Word Error Rate): Levenshtein distance on
       whitespace-split words, divided by the reference word count.
       More forgiving of minor character errors within words but
       strict about word-level differences.

    Both metrics can exceed 1.0 if the actual text is much longer
    than the expected text (more insertions than reference length).

    Args:
        expected_text: Ground-truth text (the correct text).
        actual_text: OCR-extracted text to evaluate.
        method: Error metric -- "cer" or "wer" (default "cer").
        threshold: Maximum error rate to pass (default 0.1 = 10%).
            Lower is stricter.

    Returns:
        TestResult with details: ``error_rate``, ``threshold``,
        ``method``, ``edit_distance``, ``reference_length``,
        ``expected_text``, ``actual_text``.

    Raises:
        MltkAssertionError: If error_rate > threshold (CRITICAL).
        ValueError: If method is not "cer" or "wer".

    Example:
        >>> result = assert_ocr_accuracy(
        ...     expected_text="Invoice #1234",
        ...     actual_text="Invoice #1234",
        ...     method="cer",
        ... )
        >>> result.passed
        True
    """
    if method not in ("cer", "wer"):
        raise ValueError(
            f"method must be 'cer' or 'wer', got '{method}'"
        )

    if method == "cer":
        ref_seq = list(expected_text)
        hyp_seq = list(actual_text)
    else:
        ref_seq = expected_text.split()
        hyp_seq = actual_text.split()

    ref_len = len(ref_seq)

    # Handle empty reference
    if ref_len == 0:
        hyp_len = len(hyp_seq)
        error_rate = 0.0 if hyp_len == 0 else 1.0
        edit_dist = hyp_len
    else:
        edit_dist = _levenshtein_distance(hyp_seq, ref_seq)
        error_rate = edit_dist / ref_len

    # For OCR, we pass when error_rate <= threshold
    passed = error_rate <= threshold

    # Score is inverted: 1.0 - error_rate (higher = better)
    score = max(0.0, 1.0 - error_rate)

    method_label = "CER" if method == "cer" else "WER"

    message = (
        f"OCR accuracy ({method_label}): error_rate "
        f"{error_rate:.4f} <= {threshold}"
        if passed
        else f"OCR accuracy too low ({method_label}): "
        f"error_rate {error_rate:.4f} > {threshold}"
    )

    return assert_true(
        passed,
        name="multimodal.vlm.ocr_accuracy",
        message=message,
        severity=Severity.CRITICAL,
        score=round(score, 4),
        error_rate=round(error_rate, 4),
        threshold=threshold,
        method=method,
        edit_distance=edit_dist,
        reference_length=ref_len,
        expected_text=expected_text,
        actual_text=actual_text,
    )
