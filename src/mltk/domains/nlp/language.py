"""Language identification via the langdetect library."""

from __future__ import annotations

from mltk.core.assertion import assert_true, timed_assertion
from mltk.core.result import Severity, TestResult

_INSTALL = "pip install mlspec[langdetect]"


@timed_assertion
def assert_language(
    text: str,
    expected: str,
    *,
    min_prob: float | None = None,
    severity: Severity = Severity.CRITICAL,
) -> TestResult:
    """Assert that *text* is detected as *expected* (ISO 639-1).

    Uses ``langdetect``. Detection is seeded for determinism. Empty or
    whitespace-only input fails closed.

    Args:
        text: Input text to classify.
        expected: ISO 639-1 language code (e.g. ``"en"``, ``"fr"``).
        min_prob: Optional minimum probability for *expected*. ``None``
            checks identity only.
        severity: CRITICAL raises on failure.

    Returns:
        TestResult named ``nlp.language``. Details carry ``detected``,
        ``expected``, and two separate confidences:
        ``detected_probability`` (langdetect's confidence in the
        top-ranked language) and ``expected_probability`` (the
        confidence it assigned to *expected*, ``0.0`` when *expected* is
        not among the candidates at all). ``min_prob`` reads against
        ``expected_probability``. Both are ``None`` when detection never
        ran -- empty input, or langdetect could not classify.
    """
    stripped = text.strip() if isinstance(text, str) else ""
    if not stripped:
        return assert_true(
            False,
            name="nlp.language",
            message="language detection: empty text",
            severity=severity,
            detected=None,
            expected=expected,
            detected_probability=None,
            expected_probability=None,
        )

    try:
        from langdetect import DetectorFactory, detect_langs
        from langdetect.lang_detect_exception import LangDetectException
    except ImportError as err:
        raise ImportError(
            "langdetect is required for language identification. "
            f"Install: {_INSTALL}"
        ) from err

    DetectorFactory.seed = 0
    try:
        candidates = detect_langs(stripped)
    except LangDetectException:
        return assert_true(
            False,
            name="nlp.language",
            message="language detection: could not determine language",
            severity=severity,
            detected=None,
            expected=expected,
            detected_probability=None,
            expected_probability=None,
        )

    ranked = [(c.lang, float(c.prob)) for c in candidates]
    detected = ranked[0][0] if ranked else None
    expected_prob = next((p for lang, p in ranked if lang == expected), 0.0)
    identity_ok = detected == expected
    prob_ok = True if min_prob is None else expected_prob >= min_prob
    passed = bool(identity_ok and prob_ok)

    if passed:
        message = f"language {detected} == {expected}"
    elif not identity_ok:
        message = f"language {detected} != {expected}"
    else:
        message = (
            f"language {detected} probability {expected_prob:.4f} "
            f"< {min_prob}"
        )

    return assert_true(
        passed,
        name="nlp.language",
        message=message,
        severity=severity,
        detected=detected,
        expected=expected,
        detected_probability=ranked[0][1] if ranked else None,
        expected_probability=expected_prob,
    )
