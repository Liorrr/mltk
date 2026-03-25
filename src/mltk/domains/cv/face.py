"""Face recognition testing — FAR/FRR for biometric systems."""

from __future__ import annotations

from typing import Any

import numpy as np

from mltk.core.assertion import assert_true, timed_assertion
from mltk.core.result import Severity, TestResult


@timed_assertion
def assert_face_far(
    similarities: Any,
    labels: Any,
    max_far: float = 0.001,
) -> TestResult:
    """Assert False Accept Rate is below threshold.

    FAR = fraction of non-mate pairs incorrectly accepted.
    Standard operating points: FAR = 1e-3 to 1e-6.

    Args:
        similarities: Pairwise similarity scores (1D array).
        labels: Binary labels (1=mate, 0=non-mate).
        max_far: Maximum allowed FAR.

    Returns:
        TestResult with FAR and threshold details.

    Example:
        >>> sims = [0.9, 0.8, 0.2, 0.1]  # 2 mates, 2 non-mates
        >>> labels = [1, 1, 0, 0]
        >>> assert_face_far(sims, labels, max_far=0.1)
    """
    sims = np.asarray(similarities, dtype=np.float64)
    labs = np.asarray(labels, dtype=int)

    non_mate_mask = labs == 0
    non_mate_sims = sims[non_mate_mask]

    if len(non_mate_sims) == 0:
        return assert_true(
            True, name="cv.face_far",
            message="No non-mate pairs to evaluate",
            severity=Severity.INFO,
        )

    # Find threshold that gives best TAR while meeting FAR constraint
    # Use median of mate similarities as threshold
    mate_sims = sims[labs == 1]
    if len(mate_sims) > 0:
        threshold = float(np.median(mate_sims))
    else:
        threshold = 0.5

    # Compute FAR at this threshold
    false_accepts = int((non_mate_sims >= threshold).sum())
    far = false_accepts / len(non_mate_sims)

    passed = far <= max_far
    message = (
        f"FAR: {far:.6f} <= {max_far} at threshold={threshold:.4f}"
        if passed
        else f"FAR too high: {far:.6f} > {max_far} ({false_accepts} false accepts)"
    )

    return assert_true(
        passed, name="cv.face_far", message=message,
        severity=Severity.CRITICAL,
        far=far, max_far=max_far, threshold=threshold,
        false_accepts=false_accepts, total_non_mates=len(non_mate_sims),
    )
