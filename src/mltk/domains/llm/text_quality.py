"""Text quality assertions — length, format, readability."""

from __future__ import annotations

import re

from mltk.core.assertion import assert_true, timed_assertion
from mltk.core.result import Severity, TestResult


def _count_syllables(word: str) -> int:
    """Count syllables in a word using a vowel-group heuristic."""
    word = word.lower().strip(".,!?;:\"'")
    if not word:
        return 0
    # Count contiguous vowel groups
    vowel_groups = re.findall(r"[aeiouy]+", word)
    count = len(vowel_groups)
    # Silent 'e' at end: subtract if word ends in 'e' and has >1 vowel group
    if word.endswith("e") and count > 1:
        count -= 1
    return max(count, 1)


@timed_assertion
def assert_text_length(
    text: str,
    min_words: int | None = None,
    max_words: int | None = None,
) -> TestResult:
    """Assert text word count is within bounds.

    At least one of min_words or max_words must be specified.

    Args:
        text: The text string to evaluate.
        min_words: Minimum word count (inclusive). None = no lower bound.
        max_words: Maximum word count (inclusive). None = no upper bound.

    Returns:
        TestResult with word count details.

    Example:
        >>> assert_text_length("The quick brown fox", min_words=3, max_words=10)
    """
    if min_words is None and max_words is None:
        return assert_true(
            False, name="llm.text_length",
            message="At least one of min_words or max_words must be specified",
            severity=Severity.CRITICAL,
        )

    words = text.split()
    word_count = len(words)

    too_short = min_words is not None and word_count < min_words
    too_long = max_words is not None and word_count > max_words
    passed = not too_short and not too_long

    if passed:
        bounds = []
        if min_words is not None:
            bounds.append(f">= {min_words}")
        if max_words is not None:
            bounds.append(f"<= {max_words}")
        message = f"Word count {word_count} within bounds ({', '.join(bounds)})"
    elif too_short:
        message = f"Text too short: {word_count} words < min_words={min_words}"
    else:
        message = f"Text too long: {word_count} words > max_words={max_words}"

    return assert_true(
        passed, name="llm.text_length", message=message,
        severity=Severity.CRITICAL,
        word_count=word_count,
        min_words=min_words,
        max_words=max_words,
    )


@timed_assertion
def assert_output_format(
    text: str,
    pattern: str,
    description: str | None = None,
) -> TestResult:
    """Assert text matches a regex pattern.

    Useful for: starts-with checks, contains checks, ends-with checks,
    JSON format, UUID format, date format, etc.

    Args:
        text: The text string to evaluate.
        pattern: Regex string to match against the full text (re.search).
        description: Human-readable description of what the pattern expects.
            Used in the failure message. Defaults to the pattern itself.

    Returns:
        TestResult with match details.

    Example:
        >>> assert_output_format("Result: 42", pattern=r"^Result: \\d+$")
        >>> assert_output_format(json_str, pattern=r"^\\{.*\\}$", description="JSON object")
    """
    label = description or f"pattern={pattern!r}"
    try:
        matched = bool(re.search(pattern, text, re.DOTALL))
    except re.error as exc:
        return assert_true(
            False, name="llm.output_format",
            message=f"Invalid regex pattern {pattern!r}: {exc}",
            severity=Severity.CRITICAL,
        )

    message = (
        f"Text matches {label}"
        if matched
        else f"Text does not match {label}"
    )

    return assert_true(
        matched, name="llm.output_format", message=message,
        severity=Severity.CRITICAL,
        pattern=pattern,
        description=label,
        text_preview=text[:100] + ("..." if len(text) > 100 else ""),
    )


@timed_assertion
def assert_readability(
    text: str,
    max_grade_level: float = 12.0,
) -> TestResult:
    """Assert text readability using Flesch-Kincaid grade level.

    Grade level formula:
    0.39 * (total_words / total_sentences) + 11.8 * (total_syllables / total_words) - 15.59

    Lower grade levels = easier to read. 6 = middle school, 8 = 8th grade,
    12 = high school senior, 16+ = college+.

    Args:
        text: The text string to evaluate.
        max_grade_level: Maximum allowed FK grade level. Defaults to 12.0.

    Returns:
        TestResult with grade level and component counts.

    Example:
        >>> assert_readability("The cat sat on the mat.", max_grade_level=8.0)
    """
    # Count sentences — split on terminal punctuation
    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    total_sentences = max(len(sentences), 1)

    # Count words
    words = [w for w in text.split() if w.strip()]
    total_words = len(words)

    if total_words == 0:
        return assert_true(
            False, name="llm.readability",
            message="Text has no words — cannot compute readability",
            severity=Severity.CRITICAL,
        )

    # Count syllables using vowel-group heuristic
    total_syllables = sum(_count_syllables(w) for w in words)

    # Flesch-Kincaid Grade Level
    grade_level = (
        0.39 * (total_words / total_sentences)
        + 11.8 * (total_syllables / total_words)
        - 15.59
    )
    grade_level = round(grade_level, 2)

    passed = grade_level <= max_grade_level

    message = (
        f"Readability OK: grade level {grade_level:.2f} <= {max_grade_level}"
        if passed
        else f"Text too complex: grade level {grade_level:.2f} > {max_grade_level}"
    )

    return assert_true(
        passed, name="llm.readability", message=message,
        severity=Severity.CRITICAL,
        grade_level=grade_level,
        max_grade_level=max_grade_level,
        total_words=total_words,
        total_sentences=total_sentences,
        total_syllables=total_syllables,
    )
