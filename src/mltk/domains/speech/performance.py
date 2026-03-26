"""Speech performance testing -- RTF and accent coverage."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from mltk.core.assertion import assert_true, timed_assertion
from mltk.core.result import Severity, TestResult


@timed_assertion
def assert_rtf(
    process_fn: Callable[..., Any],
    audio_durations: list[float],
    max_rtf: float = 1.0,
) -> TestResult:
    """Assert Real-Time Factor is below threshold.

    RTF = processing_time / audio_duration. RTF < 1.0 = real-time capable.

    Args:
        process_fn: Function that processes audio (called once per duration).
        audio_durations: Audio durations in seconds.
        max_rtf: Maximum allowed RTF.

    Returns:
        TestResult with RTF value.

    Example:
        >>> def fast_asr(duration): pass  # simulate fast processing
        >>> assert_rtf(fast_asr, audio_durations=[5.0, 10.0], max_rtf=1.0)
    """
    if not audio_durations:
        return assert_true(
            False,
            name="speech.rtf",
            message="No audio durations provided — cannot compute RTF",
            severity=Severity.CRITICAL,
            rtf=float("nan"),
            max_rtf=max_rtf,
            total_processing_sec=0.0,
            total_audio_sec=0.0,
        )

    total_processing = 0.0
    total_audio = sum(audio_durations)

    for duration in audio_durations:
        start = time.perf_counter()
        process_fn(duration)
        total_processing += time.perf_counter() - start

    rtf = total_processing / total_audio if total_audio > 0 else 0.0

    passed = rtf <= max_rtf
    message = (
        f"RTF: {rtf:.4f} <= {max_rtf} (real-time capable)"
        if passed
        else f"RTF: {rtf:.4f} > {max_rtf} (too slow for real-time)"
    )

    return assert_true(
        passed, name="speech.rtf", message=message,
        severity=Severity.CRITICAL,
        rtf=rtf, max_rtf=max_rtf,
        total_processing_sec=total_processing,
        total_audio_sec=total_audio,
    )


@timed_assertion
def assert_accent_coverage(
    wer_by_accent: dict[str, float],
    max_gap: float = 0.05,
) -> TestResult:
    """Assert WER difference across accents is within bounds.

    Args:
        wer_by_accent: Dict mapping accent name to WER value.
        max_gap: Maximum allowed WER gap between best and worst accent.

    Returns:
        TestResult with per-accent WER and gap.

    Example:
        >>> wers = {"US_English": 0.08, "UK_English": 0.10, "Indian_English": 0.12}
        >>> assert_accent_coverage(wers, max_gap=0.05)
    """
    if len(wer_by_accent) < 2:
        return assert_true(
            True, name="speech.accent_coverage",
            message="Need >= 2 accents for coverage check",
            severity=Severity.INFO,
        )

    wers = list(wer_by_accent.values())
    best = min(wers)
    worst = max(wers)
    gap = worst - best

    best_accent = [k for k, v in wer_by_accent.items() if v == best][0]
    worst_accent = [k for k, v in wer_by_accent.items() if v == worst][0]

    passed = gap <= max_gap
    message = (
        f"Accent gap: {gap:.4f} <= {max_gap}"
        if passed
        else f"Accent bias: gap={gap:.4f} > {max_gap} "
        f"(best='{best_accent}' {best:.4f}, worst='{worst_accent}' {worst:.4f})"
    )

    return assert_true(
        passed, name="speech.accent_coverage", message=message,
        severity=Severity.CRITICAL,
        gap=gap, max_gap=max_gap,
        best_accent=best_accent, best_wer=best,
        worst_accent=worst_accent, worst_wer=worst,
        wer_by_accent=wer_by_accent,
    )
