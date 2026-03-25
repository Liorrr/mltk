"""Example: mycompany-style CV testing with mltk.

This file demonstrates how to test video analytics and object detection
models at mycompany using mltk's CV domain kit.
"""

import numpy as np
import pytest

from mltk.domains.cv import (
    assert_frame_accuracy,
    assert_iou,
    assert_map,
    assert_temporal_consistency,
    assert_topk_accuracy,
)


@pytest.mark.ml_model
def test_object_detection_quality():
    """Validate object detection model meets mAP threshold."""
    predictions = [
        {"boxes": [[10, 10, 50, 50]], "labels": [1], "scores": [0.95]},
        {"boxes": [[20, 20, 80, 80]], "labels": [2], "scores": [0.88]},
    ]
    ground_truth = [
        {"boxes": [[10, 10, 50, 50]], "labels": [1]},
        {"boxes": [[20, 20, 80, 80]], "labels": [2]},
    ]
    assert_map(predictions, ground_truth, min_map=0.5)


@pytest.mark.ml_model
def test_localization_accuracy():
    """Validate bounding box localization quality."""
    pred_boxes = np.array([[11, 11, 49, 49], [21, 21, 79, 79]])
    gt_boxes = np.array([[10, 10, 50, 50], [20, 20, 80, 80]])
    assert_iou(pred_boxes, gt_boxes, threshold=0.9)


@pytest.mark.ml_model
def test_video_frame_accuracy():
    """Validate per-frame detection accuracy for video pipeline."""
    rng = np.random.default_rng(42)
    n_frames = 1000
    labels = rng.integers(0, 5, n_frames)
    preds = labels.copy()
    # 5% error rate
    errors = rng.choice(n_frames, size=50, replace=False)
    preds[errors] = rng.integers(0, 5, 50)
    assert_frame_accuracy(preds, labels, threshold=0.90)


@pytest.mark.ml_model
def test_tracking_stability():
    """Validate object tracking doesn't jitter across frames."""
    # Simulate smooth object movement
    boxes = [[10 + i, 10 + i, 50 + i, 50 + i] for i in range(30)]
    assert_temporal_consistency(boxes, min_smoothness=0.95)


@pytest.mark.ml_model
def test_image_classification():
    """Validate image classifier top-5 accuracy."""
    rng = np.random.default_rng(42)
    n_images = 100
    n_classes = 10
    y_true = rng.integers(0, n_classes, n_images)
    # Generate probs where true class has high probability
    y_probs = rng.random((n_images, n_classes)) * 0.1
    for i in range(n_images):
        y_probs[i, y_true[i]] += 0.8
    assert_topk_accuracy(y_true, y_probs, k=5, threshold=0.95)
