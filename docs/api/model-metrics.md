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
