# Computer Vision Testing

CV-specific assertions for object detection (IoU, mAP), video analytics (frame accuracy, temporal consistency), and image classification (top-K accuracy). Designed for Kaleidoo-style workloads: video analytics, face recognition, tracking.

**Module:** `mltk.domains.cv`

**Install:** `pip install mltk[cv]`

---

## Detection

### compute_iou

Vectorized IoU computation between two sets of bounding boxes. Returns NxM matrix.

```python
from mltk.domains.cv import compute_iou

iou_matrix = compute_iou(pred_boxes, gt_boxes)
```

#### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `boxes_a` | `np.ndarray` | *(required)* | (N, 4) array of `[x1, y1, x2, y2]` boxes |
| `boxes_b` | `np.ndarray` | *(required)* | (M, 4) array of `[x1, y1, x2, y2]` boxes |

#### Returns

`np.ndarray` of shape (N, M) with IoU values.

---

### assert_iou

Assert minimum mean IoU between predicted and ground-truth boxes.

```python
from mltk.domains.cv import assert_iou

assert_iou(pred_boxes, gt_boxes, threshold=0.5)
```

#### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `pred_boxes` | `array-like` | *(required)* | Predicted bounding boxes (N, 4) |
| `gt_boxes` | `array-like` | *(required)* | Ground truth bounding boxes (M, 4) |
| `threshold` | `float` | `0.5` | Minimum required mean IoU |

#### Returns

`TestResult` with details:
- `mean_iou` -- actual mean IoU (best match per prediction)
- `threshold` -- configured threshold
- `num_predictions` -- number of predicted boxes
- `num_ground_truth` -- number of ground truth boxes

---

### assert_map

Assert mean Average Precision meets threshold. Supports COCO-style (101-point interpolation) evaluation with per-class AP breakdown.

```python
from mltk.domains.cv import assert_map

assert_map(predictions, ground_truth, iou_threshold=0.5, min_map=0.5)
```

#### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `predictions` | `list[dict]` | *(required)* | List of dicts with `boxes` (N,4), `labels` (N,), `scores` (N,) |
| `ground_truth` | `list[dict]` | *(required)* | List of dicts with `boxes` (M,4), `labels` (M,) |
| `iou_threshold` | `float` | `0.5` | IoU threshold for matching |
| `min_map` | `float` | `0.5` | Minimum required mAP |

#### Returns

`TestResult` with details:
- `map_value` -- computed mAP
- `iou_threshold` -- configured IoU threshold
- `min_map` -- configured minimum mAP
- `class_aps` -- dict of per-class AP values
- `num_classes` -- number of classes evaluated

---

## Video

### assert_frame_accuracy

Per-frame detection/classification accuracy for video pipelines.

```python
from mltk.domains.cv import assert_frame_accuracy

assert_frame_accuracy(frame_preds, frame_labels, threshold=0.8)
```

#### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `frame_preds` | `array-like` | *(required)* | Predictions per frame (1D array of labels) |
| `frame_labels` | `array-like` | *(required)* | Ground truth per frame (1D array of labels) |
| `threshold` | `float` | `0.8` | Minimum required accuracy |

#### Returns

`TestResult` with details:
- `accuracy` -- actual frame-level accuracy
- `correct` -- number of correct frames
- `total` -- total number of frames
- `threshold` -- configured threshold

---

### assert_temporal_consistency

Frame-to-frame IoU stability for tracked objects. Catches jittery tracking.

```python
from mltk.domains.cv import assert_temporal_consistency

assert_temporal_consistency(tracked_boxes, min_smoothness=0.7)
```

#### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `tracked_boxes` | `list[array-like]` | *(required)* | List of (4,) boxes for consecutive frames of one tracked object |
| `min_smoothness` | `float` | `0.7` | Minimum mean frame-to-frame IoU |

#### Returns

`TestResult` with details:
- `smoothness` -- mean frame-to-frame IoU
- `min_iou` -- lowest frame-to-frame IoU observed
- `num_frames` -- number of frames
- `min_smoothness` -- configured threshold

#### Edge Cases

- **Fewer than 2 frames**: Returns a passing result with `INFO` severity (need at least 2 frames for comparison).

---

## Classification

### assert_topk_accuracy

Assert top-K accuracy (e.g., top-5 accuracy >= 90%) for image classification.

```python
from mltk.domains.cv import assert_topk_accuracy

assert_topk_accuracy(y_true, y_probs, k=5, threshold=0.9)
```

#### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `y_true` | `array-like` | *(required)* | Ground truth class indices (1D array) |
| `y_probs` | `array-like` | *(required)* | Predicted probabilities (N x num_classes matrix) |
| `k` | `int` | `5` | Number of top predictions to consider |
| `threshold` | `float` | `0.9` | Minimum required top-K accuracy |

#### Returns

`TestResult` with details:
- `accuracy` -- actual top-K accuracy
- `k` -- number of top predictions considered
- `threshold` -- configured threshold
- `correct` -- number of correct samples
- `total` -- total number of samples

---
