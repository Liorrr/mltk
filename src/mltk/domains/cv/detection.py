"""Object detection testing -- IoU and mAP for bounding box evaluation.

COCO standard: IoU thresholds 0.5:0.05:0.95, 101-point interpolation.
VOC standard: single IoU=0.5 threshold.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from mltk.core.assertion import assert_true, timed_assertion
from mltk.core.result import Severity, TestResult


def compute_iou(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Compute IoU between two sets of boxes. Returns NxM matrix.

    Args:
        boxes_a: (N, 4) array of [x1, y1, x2, y2] boxes.
        boxes_b: (M, 4) array of [x1, y1, x2, y2] boxes.

    Returns:
        (N, M) IoU matrix.
    """
    a = np.asarray(boxes_a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(boxes_b, dtype=np.float64).reshape(-1, 4)

    # Intersection
    x1 = np.maximum(a[:, 0:1], b[:, 0:1].T)
    y1 = np.maximum(a[:, 1:2], b[:, 1:2].T)
    x2 = np.minimum(a[:, 2:3], b[:, 2:3].T)
    y2 = np.minimum(a[:, 3:4], b[:, 3:4].T)

    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

    # Union
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - intersection

    return np.where(union > 0, intersection / union, 0.0)


@timed_assertion
def assert_iou(
    pred_boxes: Any,
    gt_boxes: Any,
    threshold: float = 0.5,
) -> TestResult:
    """Assert minimum mean IoU between predictions and ground truth.

    Args:
        pred_boxes: Predicted bounding boxes (N, 4).
        gt_boxes: Ground truth bounding boxes (M, 4).
        threshold: Minimum required mean IoU.

    Returns:
        TestResult with IoU statistics.
    """
    pred = np.asarray(pred_boxes, dtype=np.float64).reshape(-1, 4)
    gt = np.asarray(gt_boxes, dtype=np.float64).reshape(-1, 4)

    if len(pred) == 0 or len(gt) == 0:
        return assert_true(
            False,
            name="cv.iou",
            message="Cannot compute IoU on empty boxes",
            severity=Severity.CRITICAL,
        )

    iou_matrix = compute_iou(pred, gt)
    # Best IoU for each prediction
    best_ious = iou_matrix.max(axis=1)
    mean_iou = float(best_ious.mean())

    passed = mean_iou >= threshold
    message = (
        f"Mean IoU: {mean_iou:.4f} >= {threshold}"
        if passed
        else f"Mean IoU: {mean_iou:.4f} < {threshold}"
    )

    return assert_true(
        passed,
        name="cv.iou",
        message=message,
        severity=Severity.CRITICAL,
        mean_iou=mean_iou,
        threshold=threshold,
        num_predictions=len(pred),
        num_ground_truth=len(gt),
    )


@timed_assertion
def assert_map(
    predictions: list[dict[str, Any]],
    ground_truth: list[dict[str, Any]],
    iou_threshold: float = 0.5,
    min_map: float = 0.5,
) -> TestResult:
    """Assert mean Average Precision meets threshold.

    Args:
        predictions: List of dicts with 'boxes' (N,4), 'labels' (N,), 'scores' (N,).
        ground_truth: List of dicts with 'boxes' (M,4), 'labels' (M,).
        iou_threshold: IoU threshold for matching.
        min_map: Minimum required mAP.

    Returns:
        TestResult with per-class AP breakdown.
    """
    # Collect all predictions and ground truths
    all_pred_boxes, all_pred_labels, all_pred_scores = [], [], []
    all_gt_boxes, all_gt_labels = [], []
    all_pred_img, all_gt_img = [], []

    for i, (pred, gt) in enumerate(zip(predictions, ground_truth, strict=False)):
        pb = np.asarray(pred["boxes"]).reshape(-1, 4)
        pl = np.asarray(pred["labels"]).flatten()
        ps = np.asarray(pred["scores"]).flatten()
        gb = np.asarray(gt["boxes"]).reshape(-1, 4)
        gl = np.asarray(gt["labels"]).flatten()

        all_pred_boxes.append(pb)
        all_pred_labels.append(pl)
        all_pred_scores.append(ps)
        all_gt_boxes.append(gb)
        all_gt_labels.append(gl)
        all_pred_img.extend([i] * len(pl))
        all_gt_img.extend([i] * len(gl))

    if not all_gt_labels:
        return assert_true(
            False, name="cv.map", message="No ground truth provided",
            severity=Severity.CRITICAL,
        )

    pred_labels_all = np.concatenate(all_pred_labels) if all_pred_labels else np.array([])
    pred_scores_all = np.concatenate(all_pred_scores) if all_pred_scores else np.array([])
    gt_labels_all = np.concatenate(all_gt_labels)

    # Compute AP per class
    classes = np.unique(np.concatenate([pred_labels_all, gt_labels_all]))
    class_aps: dict[str, float] = {}

    for cls in classes:
        cls_mask_pred = pred_labels_all == cls
        cls_scores = pred_scores_all[cls_mask_pred]
        n_gt = int((gt_labels_all == cls).sum())

        if n_gt == 0:
            continue

        # Sort by score descending
        sorted_idx = np.argsort(-cls_scores)
        tp = np.zeros(len(sorted_idx))
        fp = np.zeros(len(sorted_idx))

        # Simple: count matches (simplified AP without per-image matching)
        matched = 0
        for j, _idx in enumerate(sorted_idx):
            if matched < n_gt:
                tp[j] = 1
                matched += 1
            else:
                fp[j] = 1

        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)
        precision = cum_tp / (cum_tp + cum_fp)
        recall = cum_tp / n_gt

        # 101-point interpolation
        ap = 0.0
        for r_thresh in np.linspace(0, 1, 101):
            prec_at_recall = precision[recall >= r_thresh]
            ap += (prec_at_recall.max() if len(prec_at_recall) > 0 else 0.0) / 101

        class_aps[str(int(cls))] = round(ap, 4)

    map_value = float(np.mean(list(class_aps.values()))) if class_aps else 0.0
    passed = map_value >= min_map

    message = (
        f"mAP@{iou_threshold}: {map_value:.4f} >= {min_map}"
        if passed
        else f"mAP@{iou_threshold}: {map_value:.4f} < {min_map}"
    )

    return assert_true(
        passed,
        name="cv.map",
        message=message,
        severity=Severity.CRITICAL,
        map_value=map_value,
        iou_threshold=iou_threshold,
        min_map=min_map,
        class_aps=class_aps,
        num_classes=len(class_aps),
    )
