# Model Regression Testing

Model regression testing detects when a new model performs worse than a previous version. 67% of organizations detect silent model degradation more than 6 months late. These assertions catch it immediately.

**Module:** `mltk.model.regression`

**ML Lifecycle Stage:** Model evaluation / CI/CD gate

**ML Bug caught:** New model version silently degrades — metrics look "fine" but are worse than baseline

---

## save_baseline

Compute and save model metrics as a JSON baseline for future comparisons.

```python
from mltk.model import save_baseline

save_baseline(y_true, y_pred, metrics=["accuracy", "f1"], path="baselines/v1.json")
```

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `y_true` | `array-like` | *(required)* | Ground truth labels |
| `y_pred` | `array-like` | *(required)* | Model predictions |
| `metrics` | `list[str]` | *(required)* | Metrics to compute and save |
| `path` | `str` | *(required)* | File path for JSON output |
| `average` | `str` | `"weighted"` | Averaging for multiclass metrics |

### Baseline JSON Format

```json
{
  "metrics": {"accuracy": 0.95, "f1": 0.93},
  "sample_count": 10000,
  "timestamp": "2026-03-25T12:00:00"
}
```

---

## assert_no_regression

Assert current model metrics have not regressed from a saved baseline.

```python
from mltk.model import assert_no_regression

assert_no_regression(y_true, y_pred, baseline=0.95, metric="accuracy", tolerance=0.02)
assert_no_regression(y_true, y_pred, baseline="baselines/v1.json", metric="f1")
```

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `y_true` | `array-like` | *(required)* | Ground truth labels |
| `y_pred` | `array-like` | *(required)* | Current model predictions |
| `baseline` | `float \| dict \| str` | *(required)* | Baseline value, dict of metrics, or path to JSON |
| `metric` | `str` | `"accuracy"` | Which metric to compare |
| `tolerance` | `float` | `0.02` | Max allowed regression (0.02 = 2% drop allowed) |
| `average` | `str` | `"weighted"` | Averaging for multiclass |

### How tolerance works

```
PASS if: current_metric >= baseline_metric - tolerance
FAIL if: current_metric <  baseline_metric - tolerance

Example: baseline=0.95, tolerance=0.02
  current=0.94 → PASS (0.94 >= 0.93)
  current=0.92 → FAIL (0.92 < 0.93)
```

### Example

```python
@pytest.mark.ml_model
def test_no_accuracy_regression(y_true, y_pred):
    # Allow max 2% drop from last known good model
    assert_no_regression(y_true, y_pred, baseline=0.95, metric="accuracy", tolerance=0.02)

@pytest.mark.ml_model
def test_no_regression_from_file(y_true, y_pred):
    assert_no_regression(y_true, y_pred, baseline="baselines/prod_v2.json", metric="f1")
```

### Related Tests

| Test | What it validates |
|------|-------------------|
| `test_no_regression` | Current meets baseline — passes |
| `test_regression_detected` | Current below baseline-tolerance — fails |
| `test_baseline_from_float` | Direct float baseline value |
| `test_baseline_from_dict` | Dict with metric keys |
| `test_baseline_from_file` | Load from JSON file |
| `test_save_load_roundtrip` | save_baseline then assert_no_regression |
| `test_tolerance_boundary` | Exactly at tolerance passes |
| `test_empty_predictions` | Empty input fails gracefully |

---
