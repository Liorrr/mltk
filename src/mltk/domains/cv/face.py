"""Face recognition testing — FAR/FRR for biometric systems."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from mltk.core.assertion import assert_true, timed_assertion
from mltk.core.empty import ON_EMPTY_OPTIONS, empty_input_result, unknown_on_empty_result
from mltk.core.result import Severity, TestResult

_NAME = "cv.face_far"


def _validate_inputs(
    similarities: Any, labels: Any, threshold: float, max_far: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(similarities, labels)`` as 1-D arrays, or raise ValueError."""
    if not np.isfinite(threshold):
        raise ValueError(f"threshold must be a finite number, got {threshold!r}")
    if not (np.isfinite(max_far) and 0.0 <= max_far <= 1.0):
        raise ValueError(f"max_far must be in [0, 1], got {max_far!r}")
    sims = np.asarray(similarities, dtype=np.float64)
    raw_labels = np.asarray(labels)
    if sims.ndim != 1 or raw_labels.ndim != 1:
        raise ValueError("similarities and labels must be 1-D")
    if len(sims) != len(raw_labels):
        raise ValueError(
            f"similarities and labels must have equal lengths, "
            f"got {len(sims)} and {len(raw_labels)}"
        )
    if not np.isfinite(sims).all():
        raise ValueError("similarities must be finite (no NaN or inf)")
    if raw_labels.size and not np.isin(raw_labels, (0, 1)).all():
        raise ValueError("labels must be 0 (non-mate) or 1 (mate)")
    return sims, raw_labels.astype(int)


@timed_assertion
def assert_face_far(
    similarities: Any,
    labels: Any,
    *,
    threshold: float,
    max_far: float = 0.001,
    on_empty: Literal["fail", "skip", "pass"] = "fail",
) -> TestResult:
    """Assert the False Accept Rate at the operating threshold is low enough.

    FAR is the fraction of non-mate pairs accepted at the deployed decision
    threshold: a pair is accepted when its similarity is ``>= threshold``.
    Standard operating points are FAR = 1e-3 to 1e-6. The True Accept Rate
    (TAR) of the mate pairs at the same threshold is reported in the details.

    Args:
        similarities: Pairwise similarity scores (1-D, finite).
        labels: Binary labels (1 = mate, 0 = non-mate), same length.
        threshold: The operating threshold the system accepts at. Required:
            FAR is only meaningful at the threshold that is deployed.
        max_far: Maximum allowed FAR, in ``[0, 1]``.
        on_empty: Policy when there are no non-mate pairs (FAR undefined) or
            no mate pairs (TAR undefined): ``"fail"`` (default), ``"skip"``,
            or ``"pass"``. An empty non-mate set is skipped or passed. An
            empty mate set is waived under ``"skip"`` and ``"pass"``
            (``tar`` is ``None``), but FAR is still checked.

    Returns:
        TestResult with ``far``, ``max_far``, ``threshold``,
        ``false_accepts``, ``total_non_mates``, ``tar``, ``true_accepts``
        and ``total_mates`` details. ``tar`` is ``None`` when there are no
        mate pairs.

    Raises:
        ValueError: If the inputs differ in length, are not 1-D, contain a
            non-finite similarity or a label outside {0, 1}, or if
            ``threshold`` is not finite or ``max_far`` is outside [0, 1].

    Example:
        >>> sims = [0.9, 0.8, 0.2, 0.1]  # 2 mates, 2 non-mates
        >>> labels = [1, 1, 0, 0]
        >>> assert_face_far(sims, labels, threshold=0.5, max_far=0.1)
    """
    if on_empty not in ON_EMPTY_OPTIONS:
        return unknown_on_empty_result(_NAME, on_empty)
    sims, labs = _validate_inputs(similarities, labels, threshold, max_far)
    threshold = float(threshold)

    non_mate_sims = sims[labs == 0]
    mate_sims = sims[labs == 1]

    if len(non_mate_sims) == 0:
        return empty_input_result(
            name=_NAME,
            reason="No non-mate pairs: FAR is undefined",
            on_empty=on_empty,
            legacy_message="No non-mate pairs to evaluate",
            threshold=threshold,
            max_far=max_far,
            total_non_mates=0,
            total_mates=len(mate_sims),
        )
    if len(mate_sims) == 0 and on_empty == "fail":
        # With "skip" or "pass" only TAR is waived: FAR is still measurable
        # here, so it is still checked rather than skipped or passed.
        return empty_input_result(
            name=_NAME,
            reason="No mate pairs: TAR at the threshold is undefined",
            on_empty=on_empty,
            legacy_message="",
        )

    false_accepts = int((non_mate_sims >= threshold).sum())
    far = false_accepts / len(non_mate_sims)
    true_accepts = int((mate_sims >= threshold).sum())
    tar = true_accepts / len(mate_sims) if len(mate_sims) else None
    tar_text = f"{tar:.4f}" if tar is not None else "n/a (no mate pairs)"

    passed = far <= max_far
    message = (
        f"FAR: {far:.6f} <= {max_far} at threshold={threshold:.4f} "
        f"(TAR {tar_text})"
        if passed
        else (
            f"FAR too high: {far:.6f} > {max_far} at threshold={threshold:.4f} "
            f"({false_accepts}/{len(non_mate_sims)} false accepts, TAR {tar_text})"
        )
    )

    return assert_true(
        passed, name=_NAME, message=message,
        severity=Severity.CRITICAL,
        far=far, max_far=max_far, threshold=threshold,
        false_accepts=false_accepts, total_non_mates=len(non_mate_sims),
        tar=tar, true_accepts=true_accepts, total_mates=len(mate_sims),
    )
