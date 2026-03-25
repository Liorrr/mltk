"""Video analytics testing -- frame accuracy and temporal consistency.

Frame accuracy validates per-frame detection quality.
Temporal consistency checks tracking stability across consecutive frames.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from mltk.core.assertion import assert_true, timed_assertion
from mltk.core.result import Severity, TestResult


@timed_assertion
def assert_frame_accuracy(
    frame_preds: Any,
    frame_labels: Any,
    threshold: float = 0.8,
) -> TestResult:
    """Assert per-frame classification/detection accuracy.

    Args:
        frame_preds: Predictions per frame (1D array of labels).
        frame_labels: Ground truth per frame (1D array of labels).
        threshold: Minimum required accuracy.

    Returns:
        TestResult with accuracy and frame count.
    """
    preds = np.asarray(frame_preds)
    labels = np.asarray(frame_labels)

    if len(preds) == 0:
        return assert_true(
            False, name="cv.frame_accuracy",
            message="No frames to evaluate", severity=Severity.CRITICAL,
        )

    correct = int((preds == labels).sum())
    total = len(preds)
    accuracy = correct / total

    passed = accuracy >= threshold
    message = (
        f"Frame accuracy: {accuracy:.4f} >= {threshold} ({correct}/{total})"
        if passed
        else f"Frame accuracy: {accuracy:.4f} < {threshold} ({correct}/{total})"
    )

    return assert_true(
        passed, name="cv.frame_accuracy", message=message,
        severity=Severity.CRITICAL,
        accuracy=accuracy, correct=correct, total=total, threshold=threshold,
    )


@timed_assertion
def assert_temporal_consistency(
    tracked_boxes: list[Any],
    min_smoothness: float = 0.7,
) -> TestResult:
    """Assert frame-to-frame IoU stability for tracked objects.

    Args:
        tracked_boxes: List of (4,) boxes for consecutive frames of one tracked object.
        min_smoothness: Minimum mean frame-to-frame IoU.

    Returns:
        TestResult with smoothness score.
    """
    if len(tracked_boxes) < 2:
        return assert_true(
            True, name="cv.temporal_consistency",
            message="Need >= 2 frames for consistency check",
            severity=Severity.INFO,
        )

    from mltk.domains.cv.detection import compute_iou

    ious = []
    for i in range(len(tracked_boxes) - 1):
        box_a = np.asarray(tracked_boxes[i]).reshape(1, 4)
        box_b = np.asarray(tracked_boxes[i + 1]).reshape(1, 4)
        iou = float(compute_iou(box_a, box_b)[0, 0])
        ious.append(iou)

    smoothness = float(np.mean(ious))
    min_iou = float(np.min(ious))

    passed = smoothness >= min_smoothness
    message = (
        f"Temporal consistency: {smoothness:.4f} >= {min_smoothness}"
        if passed
        else f"Jittery tracking: smoothness={smoothness:.4f} < {min_smoothness} "
        f"(min frame IoU={min_iou:.4f})"
    )

    return assert_true(
        passed, name="cv.temporal_consistency", message=message,
        severity=Severity.CRITICAL,
        smoothness=smoothness, min_iou=min_iou,
        num_frames=len(tracked_boxes), min_smoothness=min_smoothness,
    )
