"""Tests for mltk.domains.cv -- computer vision assertions.

Tests cover object detection (IoU, mAP), video (frame accuracy, tracking),
and image classification (top-K accuracy). Each test validates a specific
CV evaluation scenario that gates model deployment.
"""

import numpy as np
import pytest

from mltk.core.assertion import MltkAssertionError
from mltk.domains.cv.classification import assert_topk_accuracy
from mltk.domains.cv.detection import assert_iou, assert_map, compute_iou
from mltk.domains.cv.video import assert_frame_accuracy, assert_temporal_consistency


class TestComputeIoU:
    """IoU computation tests.

    Intersection over Union (IoU) is the standard metric for object detection
    localization quality. These tests verify the geometric computation is
    correct for edge cases: perfect overlap, zero overlap, and partial overlap.
    """

    def test_perfect_overlap(self) -> None:
        """PASS: Identical boxes produce IoU = 1.0.

        WHY: This is the calibration test for IoU computation. If a box
        compared to itself does not yield exactly 1.0, the intersection
        or union calculation has a bug.
        Expected: IoU == 1.0 (within float precision).
        """
        boxes = np.array([[10, 10, 50, 50]])
        iou = compute_iou(boxes, boxes)
        assert abs(iou[0, 0] - 1.0) < 1e-6

    def test_no_overlap(self) -> None:
        """PASS: Non-overlapping boxes produce IoU = 0.0.

        WHY: Two boxes that do not touch have zero intersection. If IoU
        is non-zero here, the coordinate math is wrong (e.g., using min
        instead of max for intersection bounds).
        Expected: IoU == 0.0.
        """
        a = np.array([[0, 0, 10, 10]])
        b = np.array([[20, 20, 30, 30]])
        iou = compute_iou(a, b)
        assert iou[0, 0] == 0.0

    def test_partial_overlap(self) -> None:
        """PASS: Overlapping boxes produce 0 < IoU < 1.

        WHY: Most real detections partially overlap ground truth. This
        verifies the computation handles the general case correctly.
        Expected: IoU strictly between 0 and 1.
        """
        a = np.array([[0, 0, 20, 20]])
        b = np.array([[10, 10, 30, 30]])
        iou = compute_iou(a, b)
        assert 0 < iou[0, 0] < 1.0


class TestAssertIoU:
    """IoU assertion tests.

    Validates that assert_iou correctly gates on localization quality
    thresholds. Used to verify object detection models place bounding
    boxes accurately.
    """

    def test_high_iou_passes(self) -> None:
        """PASS: Predicted box is 1 pixel off from ground truth (IoU ~0.96).

        WHY: A 1-pixel offset on a 40x40 box gives very high IoU. This
        verifies the assertion passes for well-localized detections.
        Expected: result.passed is True.
        """
        pred = np.array([[10, 10, 50, 50]])
        gt = np.array([[11, 11, 51, 51]])
        result = assert_iou(pred, gt, threshold=0.8)
        assert result.passed is True

    def test_low_iou_fails(self) -> None:
        """FAIL: Predicted box is nowhere near ground truth (IoU = 0.0).

        WHY: The predicted box is in the upper-left corner while the ground
        truth is in the lower-right. This is a complete localization failure
        that must be caught before deployment.
        Expected: MltkAssertionError raised.
        """
        pred = np.array([[0, 0, 10, 10]])
        gt = np.array([[50, 50, 100, 100]])
        with pytest.raises(MltkAssertionError):
            assert_iou(pred, gt, threshold=0.5)

    def test_empty_boxes(self) -> None:
        """FAIL: Empty prediction boxes handled gracefully.

        WHY: A detection model that produces zero boxes (failed to detect
        anything) should fail cleanly rather than crash with an index error.
        Expected: MltkAssertionError raised.
        """
        with pytest.raises(MltkAssertionError):
            assert_iou(np.empty((0, 4)), np.array([[10, 10, 50, 50]]))


class TestAssertMAP:
    """mAP assertion tests.

    Mean Average Precision (mAP) is the standard end-to-end metric for
    object detection. It combines precision, recall, and localization
    quality across all classes and confidence thresholds.
    """

    def test_perfect_detection(self) -> None:
        """PASS: Perfect detection with exact box match and high confidence.

        WHY: A single correct detection at the exact ground truth location
        with 99% confidence should give mAP near 1.0. This is the baseline
        test for the mAP computation.
        Expected: result.passed is True.
        """
        predictions = [
            {"boxes": [[10, 10, 50, 50]], "labels": [1], "scores": [0.99]},
        ]
        ground_truth = [
            {"boxes": [[10, 10, 50, 50]], "labels": [1]},
        ]
        result = assert_map(predictions, ground_truth, min_map=0.5)
        assert result.passed is True

    def test_no_detections_fails(self) -> None:
        """FAIL: Zero detections against one ground truth gives mAP = 0.

        WHY: A model that detects nothing misses all objects. This is a
        complete failure, often caused by a broken model checkpoint or
        wrong input preprocessing.
        Expected: MltkAssertionError raised.
        """
        predictions = [{"boxes": [], "labels": [], "scores": []}]
        ground_truth = [{"boxes": [[10, 10, 50, 50]], "labels": [1]}]
        with pytest.raises(MltkAssertionError):
            assert_map(predictions, ground_truth, min_map=0.5)

    def test_multiclass(self) -> None:
        """PASS: mAP computed across multiple object classes.

        WHY: Real detection tasks have multiple classes (car, pedestrian,
        cyclist). mAP must average per-class APs. The result details should
        include per-class breakdown for debugging.
        Expected: result.passed is True, class_aps in details.
        """
        predictions = [
            {"boxes": [[10, 10, 50, 50], [60, 60, 90, 90]],
             "labels": [1, 2], "scores": [0.95, 0.90]},
        ]
        ground_truth = [
            {"boxes": [[10, 10, 50, 50], [60, 60, 90, 90]], "labels": [1, 2]},
        ]
        result = assert_map(predictions, ground_truth, min_map=0.3)
        assert result.passed is True
        assert "class_aps" in result.details


class TestFrameAccuracy:
    """Video frame accuracy tests.

    Frame accuracy measures per-frame classification correctness in video
    tasks (action recognition, scene classification). Unlike image tasks,
    video has temporal context that can be exploited.
    """

    def test_high_accuracy(self) -> None:
        """PASS: All 10 frames correctly classified (100% accuracy).

        WHY: Perfect frame-level predictions. This is the happy path for
        video classification models. Verifies the accuracy computation
        handles the perfect-score case.
        Expected: result.passed is True.
        """
        preds = np.array([0, 1, 1, 0, 1, 1, 0, 1, 1, 0])
        labels = np.array([0, 1, 1, 0, 1, 1, 0, 1, 1, 0])
        result = assert_frame_accuracy(preds, labels, threshold=0.9)
        assert result.passed is True

    def test_low_accuracy_fails(self) -> None:
        """FAIL: All frames misclassified (0% accuracy).

        WHY: Predicting all-zero when ground truth is all-one means the
        model learned the wrong class. This catches inverted label mappings
        or completely broken models.
        Expected: MltkAssertionError raised.
        """
        preds = np.array([0, 0, 0, 0, 0])
        labels = np.array([1, 1, 1, 1, 1])
        with pytest.raises(MltkAssertionError):
            assert_frame_accuracy(preds, labels, threshold=0.5)


class TestTemporalConsistency:
    """Tracking smoothness tests.

    Temporal consistency measures how smoothly an object tracker follows
    targets across frames. Jittery tracking (box jumping around) indicates
    tracker failures or ID switches.
    """

    def test_smooth_tracking(self) -> None:
        """PASS: Object moves 1 pixel per frame in each direction.

        WHY: A smoothly moving object (constant velocity) should have
        high temporal consistency. This verifies the smoothness metric
        correctly identifies stable tracking.
        Expected: result.passed is True.
        """
        boxes = [
            [10, 10, 50, 50],
            [11, 11, 51, 51],
            [12, 12, 52, 52],
            [13, 13, 53, 53],
        ]
        result = assert_temporal_consistency(boxes, min_smoothness=0.9)
        assert result.passed is True

    def test_jittery_tracking(self) -> None:
        """FAIL: Object teleports 190 pixels between frames.

        WHY: A box jumping from (10,10) to (200,200) and back indicates
        the tracker lost the object and re-acquired it, or an ID switch
        occurred. This level of jitter makes tracking unusable.
        Expected: MltkAssertionError raised.
        """
        boxes = [
            [10, 10, 50, 50],
            [200, 200, 250, 250],
            [10, 10, 50, 50],
        ]
        with pytest.raises(MltkAssertionError):
            assert_temporal_consistency(boxes, min_smoothness=0.5)


class TestTopKAccuracy:
    """Image classification top-K accuracy tests.

    Top-K accuracy measures whether the correct label is among the K
    highest-confidence predictions. Top-5 is more lenient than top-1
    and is standard for ImageNet evaluation.
    """

    def test_top1_accuracy(self) -> None:
        """PASS: Top-1 prediction is correct for all 3 samples.

        WHY: Each sample's highest-probability class matches the true label.
        This is the strictest classification accuracy measure.
        Expected: result.passed is True.
        """
        y_true = np.array([0, 1, 2])
        y_probs = np.array([
            [0.9, 0.05, 0.05],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
        ])
        result = assert_topk_accuracy(y_true, y_probs, k=1, threshold=0.9)
        assert result.passed is True

    def test_top5_more_lenient(self) -> None:
        """PASS: True label is in top-5 even though it is not top-1.

        WHY: Top-5 accuracy is forgiving for fine-grained classification
        (e.g., 1000 dog breeds). The correct class at rank 3 still counts.
        This verifies the K parameter actually expands the acceptance window.
        Expected: result.passed is True.
        """
        y_true = np.array([2])
        y_probs = np.array([[0.3, 0.25, 0.2, 0.15, 0.1]])
        result = assert_topk_accuracy(y_true, y_probs, k=5, threshold=0.5)
        assert result.passed is True

    def test_below_threshold(self) -> None:
        """FAIL: Top-1 accuracy is 0% (true class always gets lowest prob).

        WHY: The model consistently assigns the highest probability to the
        wrong class. This catches inverted softmax outputs or label mapping
        errors.
        Expected: MltkAssertionError raised.
        """
        y_true = np.array([0, 0, 0])
        y_probs = np.array([
            [0.1, 0.9],
            [0.1, 0.9],
            [0.1, 0.9],
        ])
        with pytest.raises(MltkAssertionError):
            assert_topk_accuracy(y_true, y_probs, k=1, threshold=0.5)
