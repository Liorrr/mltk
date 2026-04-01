"""Object hallucination detection -- POPE-style probing.

Vision-Language Models (VLMs) often "hallucinate" objects that are
not present in an image.  POPE (Li et al., NeurIPS 2023) is a
systematic protocol for measuring this:

1. Ask the VLM "Is there a [object] in the image?" for objects
   known to be **present** -- expect "yes".
2. Ask the same question for objects known to be **absent** --
   expect "no".
3. Compute hallucination_rate = false_positives / total_absent.

A high hallucination rate means the model claims to see things
that are not there -- a critical failure mode for safety-sensitive
applications (medical imaging, autonomous driving, security).

Design note: mltk does NOT bundle COCO object lists or implement
probe strategies (random/popular/adversarial).  The user provides
the object lists.  This follows mltk's policy of testing outputs,
not bundling datasets.
"""

from __future__ import annotations

import re
from collections.abc import Callable

from mltk.core.assertion import assert_true, timed_assertion
from mltk.core.result import Severity, TestResult
from mltk.domains.multimodal._image import ImageInput

__all__ = [
    "assert_object_hallucination",
]


# ---------------------------------------------------------------
# Answer parsing
# ---------------------------------------------------------------

_YES_PATTERN = re.compile(r"\byes\b", re.IGNORECASE)
_NO_PATTERN = re.compile(r"\bno\b", re.IGNORECASE)


def _parse_yes_no(answer: str) -> str | None:
    """Parse a VLM answer into "yes", "no", or None (ambiguous).

    Accepts case-insensitive "yes"/"no" as standalone words.
    If both appear, checks which appears first.  If neither
    appears, returns None (treated as hallucination -- conservative).
    """
    yes_match = _YES_PATTERN.search(answer)
    no_match = _NO_PATTERN.search(answer)

    if yes_match and no_match:
        # Both present -- use whichever appears first
        if yes_match.start() < no_match.start():
            return "yes"
        return "no"

    if yes_match:
        return "yes"
    if no_match:
        return "no"
    return None


# ---------------------------------------------------------------
# Assertion
# ---------------------------------------------------------------


@timed_assertion
def assert_object_hallucination(
    vqa_fn: Callable[
        [str, ImageInput | None, str | None], str
    ],
    image: ImageInput | None,
    objects_present: list[str],
    objects_absent: list[str],
    threshold: float = 0.8,
    image_description: str | None = None,
) -> TestResult:
    """Assert that a VLM does not hallucinate objects in an image.

    POPE-style probing: asks the VLM "Is there a [object] in the
    image?" for each object in both the present and absent lists.
    Objects present should get "yes"; objects absent should get "no".

    The ``vqa_fn`` signature is:
    ``(question: str, image: ImageInput | None, description: str | None) -> str``

    This lets users wrap any VLM backend -- GPT-4V, Claude, LLaVA,
    Gemini.  The function receives the binary yes/no question, the
    image (or None if using description), and an optional text
    description.

    Ambiguous answers (neither "yes" nor "no" detected) are treated
    as incorrect -- this is the conservative choice for safety.

    Args:
        vqa_fn: Callable that takes (question, image, description)
            and returns a text answer.
        image: Image source (path, bytes) or None if using
            image_description.
        objects_present: Objects known to be in the image.
            Expected answer: "yes".
        objects_absent: Objects known to NOT be in the image.
            Expected answer: "no".
        threshold: Minimum overall accuracy to pass (default 0.8).
            accuracy = correct_answers / total_questions.
        image_description: Optional text description of the image.
            Passed to vqa_fn as the third argument.

    Returns:
        TestResult with details: ``score`` (accuracy),
        ``threshold``, ``hallucination_rate``,
        ``false_positives``, ``false_negatives``,
        ``total_present``, ``total_absent``,
        ``per_object`` (list of per-object results).

    Raises:
        MltkAssertionError: If accuracy < threshold (CRITICAL).
        ValueError: If both objects_present and objects_absent
            are empty.

    Example:
        >>> def mock_vqa(q, img, desc):
        ...     return "yes" if "sofa" in q else "no"
        >>> result = assert_object_hallucination(
        ...     vqa_fn=mock_vqa,
        ...     image=None,
        ...     objects_present=["sofa"],
        ...     objects_absent=["elephant"],
        ...     image_description="A living room with a sofa.",
        ... )
        >>> result.passed
        True
    """
    if not objects_present and not objects_absent:
        raise ValueError(
            "At least one of objects_present or "
            "objects_absent must be non-empty."
        )

    per_object: list[dict] = []
    correct = 0
    false_positives = 0
    false_negatives = 0
    errors: list[str] = []

    # Probe present objects -- expect "yes"
    for obj in objects_present:
        question = f"Is there a {obj} in the image?"
        try:
            raw_answer = vqa_fn(
                question, image, image_description
            )
            parsed = _parse_yes_no(str(raw_answer))
            is_correct = parsed == "yes"
            if is_correct:
                correct += 1
            else:
                false_negatives += 1
            per_object.append({
                "object": obj,
                "expected": "yes",
                "answer": str(raw_answer),
                "parsed": parsed,
                "correct": is_correct,
            })
        except Exception as exc:
            false_negatives += 1
            error_msg = f"{type(exc).__name__}: {exc}"
            errors.append(error_msg)
            per_object.append({
                "object": obj,
                "expected": "yes",
                "answer": None,
                "parsed": None,
                "correct": False,
                "error": error_msg,
            })

    # Probe absent objects -- expect "no"
    for obj in objects_absent:
        question = f"Is there a {obj} in the image?"
        try:
            raw_answer = vqa_fn(
                question, image, image_description
            )
            parsed = _parse_yes_no(str(raw_answer))
            is_correct = parsed == "no"
            if is_correct:
                correct += 1
            else:
                false_positives += 1
            per_object.append({
                "object": obj,
                "expected": "no",
                "answer": str(raw_answer),
                "parsed": parsed,
                "correct": is_correct,
            })
        except Exception as exc:
            false_positives += 1
            error_msg = f"{type(exc).__name__}: {exc}"
            errors.append(error_msg)
            per_object.append({
                "object": obj,
                "expected": "no",
                "answer": None,
                "parsed": None,
                "correct": False,
                "error": error_msg,
            })

    total = len(objects_present) + len(objects_absent)
    accuracy = correct / total if total > 0 else 0.0

    total_absent = len(objects_absent)
    hallucination_rate = (
        false_positives / total_absent
        if total_absent > 0
        else 0.0
    )

    passed = accuracy >= threshold

    message = (
        f"Object hallucination check: accuracy "
        f"{accuracy:.4f} >= {threshold}"
        if passed
        else f"Object hallucination detected: accuracy "
        f"{accuracy:.4f} < {threshold}"
    )

    return assert_true(
        passed,
        name="multimodal.hallucination.object_hallucination",
        message=message,
        severity=Severity.CRITICAL,
        score=round(accuracy, 4),
        threshold=threshold,
        hallucination_rate=round(hallucination_rate, 4),
        false_positives=false_positives,
        false_negatives=false_negatives,
        total_present=len(objects_present),
        total_absent=total_absent,
        per_object=per_object,
        errors=errors if errors else None,
    )
