"""Speech recognition testing -- WER and CER metrics."""

from __future__ import annotations

from mltk.core.assertion import assert_true, timed_assertion
from mltk.core.result import Severity, TestResult


@timed_assertion
def assert_wer(
    references: list[str],
    hypotheses: list[str],
    max_wer: float = 0.1,
) -> TestResult:
    """Assert Word Error Rate is below threshold.

    Args:
        references: Ground truth transcriptions. Length must match hypotheses;
            jiwer evaluates all pairs together as a corpus-level metric.
        hypotheses: Model transcriptions.
        max_wer: Maximum allowed WER (0.1 = 10% error rate).

    Returns:
        TestResult with WER value.

    Example:
        >>> refs = ["hello world"]
        >>> hyps = ["hello word"]
        >>> assert_wer(refs, hyps, max_wer=0.5)
    """
    try:
        import jiwer
    except ImportError as err:
        raise ImportError(
            "jiwer is required for WER computation. Install: pip install mltk[speech]"
        ) from err

    wer_value = float(jiwer.wer(references, hypotheses))

    passed = wer_value <= max_wer
    message = (
        f"WER: {wer_value:.4f} <= {max_wer}"
        if passed
        else f"WER: {wer_value:.4f} > {max_wer}"
    )

    return assert_true(
        passed, name="speech.wer", message=message,
        severity=Severity.CRITICAL,
        wer=wer_value, max_wer=max_wer,
        num_samples=len(references),
    )


@timed_assertion
def assert_cer(
    references: list[str],
    hypotheses: list[str],
    max_cer: float = 0.05,
) -> TestResult:
    """Assert Character Error Rate is below threshold.

    Args:
        references: Ground truth transcriptions. Length must match hypotheses;
            jiwer evaluates all pairs together as a corpus-level metric.
        hypotheses: Model transcriptions.
        max_cer: Maximum allowed CER (0.05 = 5% error rate).

    Returns:
        TestResult with CER value.

    Example:
        >>> refs = ["hello world"]
        >>> hyps = ["helo world"]
        >>> assert_cer(refs, hyps, max_cer=0.1)
    """
    try:
        import jiwer
    except ImportError as err:
        raise ImportError(
            "jiwer is required for CER computation. Install: pip install mltk[speech]"
        ) from err

    cer_value = float(jiwer.cer(references, hypotheses))

    passed = cer_value <= max_cer
    message = (
        f"CER: {cer_value:.4f} <= {max_cer}"
        if passed
        else f"CER: {cer_value:.4f} > {max_cer}"
    )

    return assert_true(
        passed, name="speech.cer", message=message,
        severity=Severity.CRITICAL,
        cer=cer_value, max_cer=max_cer,
        num_samples=len(references),
    )
