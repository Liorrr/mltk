# Model Metrics Testing

Model metrics assertions are the core of ML quality testing. They catch the most dangerous bugs: wrong metric selection on imbalanced data, models that barely beat random, and silent performance drops.

**Module:** `mltk.model.metrics`

**ML Lifecycle Stage:** Post-training evaluation / Pre-deployment gate

**ML Bug caught:** Using accuracy on 99% negative data (model predicts "negative" always, gets 99% accuracy, is useless)

---

## assert_metric

Unified metric assertion supporting classification and regression metrics.

```python
from mltk.model import assert_metric

# Classification
assert_metric(y_true, y_pred, metric="f1", threshold=0.85)
assert_metric(y_true, y_pred, metric="accuracy", threshold=0.90)
assert_metric(y_true, y_prob, metric="auc", threshold=0.95)

# Regression (error metrics auto-detect: lower is better)
assert_metric(y_true, y_pred, metric="mse", threshold=0.1)
assert_metric(y_true, y_pred, metric="r2", threshold=0.8)
```

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `y_true` | `array-like` | *(required)* | Ground truth labels/values |
| `y_pred` | `array-like` | *(required)* | Model predictions (class labels, probabilities, or continuous) |
| `metric` | `str` | `"accuracy"` | Metric name (see supported metrics below) |
| `threshold` | `float` | `0.8` | Minimum required value (or maximum for error metrics) |
| `average` | `str` | `"weighted"` | Averaging for multiclass: `"weighted"`, `"macro"`, `"micro"` |

### Supported Metrics

| Metric | Type | Higher is better? | Use when |
|--------|------|-------------------|----------|
| `accuracy` | Classification | Yes | Balanced classes only |
| `f1` | Classification | Yes | Imbalanced data (default weighted) |
| `precision` | Classification | Yes | False positives costly (spam filter) |
| `recall` | Classification | Yes | False negatives costly (cancer detection) |
| `auc` | Classification | Yes | Overall classifier quality (binary only) |
| `mse` | Regression | No (lower=better) | Mean squared error |
| `rmse` | Regression | No (lower=better) | Root mean squared error |
| `mae` | Regression | No (lower=better) | Mean absolute error |
| `r2` | Regression | Yes | Explained variance (0-1) |

### Example

```python
import pytest
from mltk.model import assert_metric

@pytest.mark.ml_model
def test_classifier_quality(y_true, y_pred):
    # For imbalanced data, use F1 not accuracy
    assert_metric(y_true, y_pred, metric="f1", threshold=0.85)

@pytest.mark.ml_model
def test_regression_model(y_true, y_pred):
    assert_metric(y_true, y_pred, metric="r2", threshold=0.8)
    assert_metric(y_true, y_pred, metric="rmse", threshold=5.0)
```

### Related Tests

| Test | What it validates |
|------|-------------------|
| `test_accuracy_above_threshold` | Good predictions pass accuracy check |
| `test_accuracy_below_threshold` | Bad predictions fail with clear message |
| `test_f1_weighted` | Weighted F1 for multiclass |
| `test_auc_binary` | AUC on binary classification probabilities |
| `test_mse_regression` | MSE below threshold passes |
| `test_r2_regression` | R2 above threshold passes |
| `test_unknown_metric` | Invalid metric name raises error |
| `test_perfect_predictions` | Perfect predictions give metric=1.0 |

---

## assert_no_overfitting

Assert the gap between training and test metrics is bounded. Overfitting is the most common silent ML failure: the model memorizes training data but generalizes poorly.

**Module:** `mltk.model.overfitting`

```python
from mltk.model import assert_no_overfitting

assert_no_overfitting(train_score=0.95, test_score=0.88, max_gap=0.1)
```

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `train_score` | `float` | *(required)* | Metric value on the training set |
| `test_score` | `float` | *(required)* | Metric value on the held-out test set |
| `max_gap` | `float` | `0.1` | Maximum allowed gap (train - test). Default 0.1 (10 pp). |
| `metric_name` | `str` | `"accuracy"` | Human-readable metric label used in messages |

### Returns

`TestResult` with details:
- `train_score` -- training metric value
- `test_score` -- test metric value
- `gap` -- actual gap (train_score - test_score)
- `max_gap` -- configured threshold
- `metric_name` -- metric label

### How it works

```
gap = train_score - test_score
PASS if gap <= max_gap
FAIL if gap > max_gap
```

### Example

```python
import pytest
from mltk.model import assert_no_overfitting

@pytest.mark.ml_model
def test_model_not_overfitting():
    """Training/test accuracy gap is acceptable."""
    assert_no_overfitting(
        train_score=0.97,
        test_score=0.91,
        max_gap=0.1,
        metric_name="f1",
    )
```

---

## assert_label_drift

Assert label distribution hasn't shifted between train and test splits. Computes total variation distance (TV) between label distributions. A label shift can silently invalidate evaluation metrics.

**Module:** `mltk.model.overfitting`

```python
from mltk.model import assert_label_drift

assert_label_drift([0, 0, 1, 1], [0, 1, 1, 1], max_drift=0.2)
```

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `train_labels` | `list \| np.ndarray` | *(required)* | Label array from the training split |
| `test_labels` | `list \| np.ndarray` | *(required)* | Label array from the test split |
| `max_drift` | `float` | `0.1` | Maximum allowed total variation distance |

### Returns

`TestResult` with details:
- `tv_distance` -- computed total variation distance
- `max_drift` -- configured threshold
- `train_distribution` -- dict mapping label to fraction in train set
- `test_distribution` -- dict mapping label to fraction in test set

### How it works

```
TV = 0.5 * sum(|P(y) - Q(y)|)  for all unique labels
PASS if TV <= max_drift
FAIL if TV > max_drift
```

### Example

```python
import pytest
from mltk.model import assert_label_drift

@pytest.mark.ml_model
def test_label_distribution_stable():
    """Train/test label distributions are close."""
    assert_label_drift(
        train_labels=y_train,
        test_labels=y_test,
        max_drift=0.1,
    )
```

---
