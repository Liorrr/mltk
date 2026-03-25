"""Tests for mltk.domains.cv — computer vision assertions.

Tests cover object detection (IoU, mAP), video (frame accuracy, tracking),
and image classification (top-K accuracy).
"""

import numpy as np
import pytest

from mltk.core.assertion import MltkAssertionError
from mltk.domains.cv.classification import assert_topk_accuracy
from mltk.domains.cv.detection import assert_iou, assert_map, compute_iou
from mltk.domains.cv.video import assert_frame_accuracy, assert_temporal_consistency


class TestComputeIoU:
    """IoU computation tests."""

    def test_perfect_overlap(self) -> None:
        """Identical boxes have IoU = 1.0."""
        boxes = np.array([[10, 10, 50, 50]])
        iou = compute_iou(boxes, boxes)
        assert abs(iou[0, 0] - 1.0) < 1e-6

    def test_no_overlap(self) -> None:
        """Non-overlapping boxes have IoU = 0.0."""
        a = np.array([[0, 0, 10, 10]])
        b = np.array([[20, 20, 30, 30]])
        iou = compute_iou(a, b)
        assert iou[0, 0] == 0.0

    def test_partial_overlap(self) -> None:
        """Partially overlapping boxes have 0 < IoU < 1."""
        a = np.array([[0, 0, 20, 20]])
        b = np.array([[10, 10, 30, 30]])
        iou = compute_iou(a, b)
        assert 0 < iou[0, 0] < 1.0


class TestAssertIoU:
    """IoU assertion tests."""

    def test_high_iou_passes(self) -> None:
        """PASS: Near-identical boxes meet IoU threshold."""
        pred = np.array([[10, 10, 50, 50]])
        gt = np.array([[11, 11, 51, 51]])
        result = assert_iou(pred, gt, threshold=0.8)
        assert result.passed is True

    def test_low_iou_fails(self) -> None:
        """FAIL: Poor localization detected."""
        pred = np.array([[0, 0, 10, 10]])
        gt = np.array([[50, 50, 100, 100]])
        with pytest.raises(MltkAssertionError):
            assert_iou(pred, gt, threshold=0.5)

    def test_empty_boxes(self) -> None:
        """FAIL: Empty boxes handled gracefully."""
        with pytest.raises(MltkAssertionError):
            assert_iou(np.empty((0, 4)), np.array([[10, 10, 50, 50]]))


class TestAssertMAP:
    """mAP assertion tests."""

    def test_perfect_detection(self) -> None:
        """PASS: Perfect detections give high mAP."""
        predictions = [
            {"boxes": [[10, 10, 50, 50]], "labels": [1], "scores": [0.99]},
        ]
        ground_truth = [
            {"boxes": [[10, 10, 50, 50]], "labels": [1]},
        ]
        result = assert_map(predictions, ground_truth, min_map=0.5)
        assert result.passed is True

    def test_no_detections_fails(self) -> None:
        """FAIL: No predictions give mAP = 0."""
        predictions = [{"boxes": [], "labels": [], "scores": []}]
        ground_truth = [{"boxes": [[10, 10, 50, 50]], "labels": [1]}]
        with pytest.raises(MltkAssertionError):
            assert_map(predictions, ground_truth, min_map=0.5)

    def test_multiclass(self) -> None:
        """mAP computed across multiple classes."""
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
    """Video frame accuracy tests."""

    def test_high_accuracy(self) -> None:
        """PASS: Most frames correctly classified."""
        preds = np.array([0, 1, 1, 0, 1, 1, 0, 1, 1, 0])
        labels = np.array([0, 1, 1, 0, 1, 1, 0, 1, 1, 0])
        result = assert_frame_accuracy(preds, labels, threshold=0.9)
        assert result.passed is True

    def test_low_accuracy_fails(self) -> None:
        """FAIL: Too many misclassified frames."""
        preds = np.array([0, 0, 0, 0, 0])
        labels = np.array([1, 1, 1, 1, 1])
        with pytest.raises(MltkAssertionError):
            assert_frame_accuracy(preds, labels, threshold=0.5)


class TestTemporalConsistency:
    """Tracking smoothness tests."""

    def test_smooth_tracking(self) -> None:
        """PASS: Object moves smoothly across frames."""
        boxes = [
            [10, 10, 50, 50],
            [11, 11, 51, 51],
            [12, 12, 52, 52],
            [13, 13, 53, 53],
        ]
        result = assert_temporal_consistency(boxes, min_smoothness=0.9)
        assert result.passed is True

    def test_jittery_tracking(self) -> None:
        """FAIL: Object jumps between frames — tracking failure."""
        boxes = [
            [10, 10, 50, 50],
            [200, 200, 250, 250],
            [10, 10, 50, 50],
        ]
        with pytest.raises(MltkAssertionError):
            assert_temporal_consistency(boxes, min_smoothness=0.5)


class TestTopKAccuracy:
    """Image classification top-K tests."""

    def test_top1_accuracy(self) -> None:
        """PASS: Top-1 predictions are correct."""
        y_true = np.array([0, 1, 2])
        y_probs = np.array([
            [0.9, 0.05, 0.05],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
        ])
        result = assert_topk_accuracy(y_true, y_probs, k=1, threshold=0.9)
        assert result.passed is True

    def test_top5_more_lenient(self) -> None:
        """PASS: True label in top-5 even if not top-1."""
        y_true = np.array([2])
        y_probs = np.array([[0.3, 0.25, 0.2, 0.15, 0.1]])
        result = assert_topk_accuracy(y_true, y_probs, k=5, threshold=0.5)
        assert result.passed is True

    def test_below_threshold(self) -> None:
        """FAIL: Top-K accuracy below threshold."""
        y_true = np.array([0, 0, 0])
        y_probs = np.array([
            [0.1, 0.9],
            [0.1, 0.9],
            [0.1, 0.9],
        ])
        with pytest.raises(MltkAssertionError):
            assert_topk_accuracy(y_true, y_probs, k=1, threshold=0.5)
