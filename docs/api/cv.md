# Computer Vision Testing

CV-specific assertions for object detection (IoU, mAP), video analytics (frame accuracy, temporal consistency), and image classification (top-K accuracy). Designed for production CV workloads: video analytics, face recognition, tracking.

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

## Multi-Object Tracking

Standard MOT metrics for evaluating object trackers. All functions take per-frame ground truth and prediction tracks.

**Input format**: Lists of dicts per frame, each with `ids` (1D array of track IDs) and `boxes` (Nx4 array of `[x1, y1, x2, y2]`).

### assert_mota

Multi-Object Tracking Accuracy. MOTA = 1 - (FN + FP + IDSW) / total_gt. The standard metric for tracking quality.

```python
from mltk.domains.cv import assert_mota

gt = [{"ids": [1, 2], "boxes": [[0,0,10,10], [20,20,30,30]]}]
pred = [{"ids": [1, 2], "boxes": [[1,1,11,11], [21,21,31,31]]}]
assert_mota(gt, pred, min_mota=0.5)
```

#### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `gt_tracks` | `list[dict]` | *(required)* | Per-frame GT with `ids` and `boxes` |
| `pred_tracks` | `list[dict]` | *(required)* | Per-frame predictions with `ids` and `boxes` |
| `min_mota` | `float` | `0.5` | Minimum required MOTA |
| `iou_threshold` | `float` | `0.5` | IoU threshold for matching |

#### Returns

`TestResult` with details:
- `mota` -- computed MOTA score
- `fn` -- total false negatives (missed GT)
- `fp` -- total false positives (extra predictions)
- `idsw` -- total ID switches
- `total_gt` -- total ground truth detections

---

### assert_motp

Multi-Object Tracking Precision. MOTP = mean IoU of all matched pairs. Measures localization quality.

```python
from mltk.domains.cv import assert_motp

assert_motp(gt, pred, min_motp=0.5)
```

#### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `gt_tracks` | `list[dict]` | *(required)* | Per-frame GT with `ids` and `boxes` |
| `pred_tracks` | `list[dict]` | *(required)* | Per-frame predictions with `ids` and `boxes` |
| `min_motp` | `float` | `0.5` | Minimum required MOTP |
| `iou_threshold` | `float` | `0.5` | IoU threshold for matching |

#### Returns

`TestResult` with details:
- `motp` -- mean IoU of matched pairs
- `num_matches` -- total matched pairs

---

### assert_idf1

ID F1 score for identity-aware tracking. Measures how well the tracker maintains consistent IDs. IDF1 = 2 * IDTP / (2 * IDTP + IDFP + IDFN).

```python
from mltk.domains.cv import assert_idf1

assert_idf1(gt, pred, min_idf1=0.5)
```

#### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `gt_tracks` | `list[dict]` | *(required)* | Per-frame GT with `ids` and `boxes` |
| `pred_tracks` | `list[dict]` | *(required)* | Per-frame predictions with `ids` and `boxes` |
| `min_idf1` | `float` | `0.5` | Minimum required IDF1 |
| `iou_threshold` | `float` | `0.5` | IoU threshold for matching |

#### Returns

`TestResult` with details:
- `idf1` -- computed IDF1 score
- `idtp` -- identity true positives
- `idfp` -- identity false positives
- `idfn` -- identity false negatives

---

## Face Recognition

### assert_face_far

Assert False Accept Rate is below threshold. FAR = fraction of non-mate pairs incorrectly accepted. Standard for biometric systems (NIST FRVT).

```python
from mltk.domains.cv import assert_face_far

assert_face_far(similarities, labels, max_far=0.001)
```

#### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `similarities` | `array-like` | *(required)* | Pairwise similarity scores |
| `labels` | `array-like` | *(required)* | Binary labels (1=mate, 0=non-mate) |
| `max_far` | `float` | `0.001` | Maximum allowed FAR |

#### Returns

`TestResult` with details:
- `far` -- computed FAR
- `max_far` -- configured threshold
- `threshold` -- similarity threshold used
- `false_accepts` -- number of false accepts
- `total_non_mates` -- total non-mate pairs

---
