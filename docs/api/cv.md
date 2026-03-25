# Computer Vision Testing

CV-specific assertions for object detection (IoU, mAP), video analytics (frame accuracy, temporal consistency), and image classification (top-K accuracy). Designed for mycompany-style workloads: video analytics, face recognition, tracking.

**Module:** `mltk.domains.cv`

**Install:** `pip install mltk[cv]`

---

## Detection

### compute_iou
Vectorized IoU computation between two sets of bounding boxes. Returns NxM matrix.

### assert_iou
Assert minimum mean IoU between predicted and ground-truth boxes.

### assert_map
Assert mean Average Precision meets threshold. Supports COCO-style (multi-threshold) and VOC-style (single threshold) evaluation with per-class AP breakdown.

## Video

### assert_frame_accuracy
Per-frame detection/classification accuracy for video pipelines.

### assert_temporal_consistency
Frame-to-frame IoU stability for tracked objects. Catches jittery tracking.

## Classification

### assert_topk_accuracy
Assert top-K accuracy (e.g., top-5 accuracy >= 90%) for image classification.

---
