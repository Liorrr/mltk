# Model Slicing & Calibration

Slicing tests catch the most insidious ML bug: a model that works great on average but fails for specific subgroups. Overall accuracy 92%, but accuracy for users under 18 is 52%. Calibration tests verify that prediction confidence scores actually mean what they say.

**Module:** `mltk.model.slicing`

**ML Lifecycle Stage:** Post-training evaluation / Fairness gate

**ML Bugs caught:**
- Model performs well overall but fails for demographic subgroups
- Model says 90% confidence but is correct only 60% of the time

---

## assert_slice_performance

Assert model meets minimum performance on EVERY data slice.

```python
from mltk.model import assert_slice_performance

slices = {
    "age_18_25": age_mask_young,
    "age_65_plus": age_mask_senior,
    "female": gender_mask_female,
}
assert_slice_performance(y_true, y_pred, slices=slices, metric="accuracy", min_threshold=0.75)
```

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `y_true` | `array-like` | *(required)* | Ground truth labels |
| `y_pred` | `array-like` | *(required)* | Model predictions |
| `slices` | `dict[str, array-like]` | *(required)* | Slice name to boolean mask mapping |
| `metric` | `str` | `"accuracy"` | Metric to compute per slice |
| `min_threshold` | `float` | `0.7` | Minimum metric value required for EVERY slice |
| `average` | `str` | `"weighted"` | Averaging for multiclass metrics |

### Example

```python
@pytest.mark.ml_model
def test_model_fair_across_age_groups(y_true, y_pred, age_groups):
    slices = {
        "young": age_groups == "18-25",
        "middle": age_groups == "26-50",
        "senior": age_groups == "50+",
    }
    assert_slice_performance(y_true, y_pred, slices=slices, metric="f1", min_threshold=0.75)
```

### Related Tests

| Test | What it validates |
|------|-------------------|
| `test_all_slices_pass` | Every subgroup meets threshold |
| `test_one_slice_fails` | Minority group underperforms — detected |
| `test_empty_slice` | Slice with 0 samples handled gracefully |

---

## assert_calibration

Assert prediction probabilities match actual outcomes (Expected Calibration Error).

```python
from mltk.model import assert_calibration

assert_calibration(y_true, y_prob, max_error=0.05, n_bins=10)
```

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `y_true` | `array-like` | *(required)* | Binary ground truth (0/1) |
| `y_prob` | `array-like` | *(required)* | Predicted probabilities (0.0-1.0) |
| `max_error` | `float` | `0.05` | Maximum allowed Expected Calibration Error |
| `n_bins` | `int` | `10` | Number of bins for calibration curve |

### How ECE works

```
ECE = weighted average of |predicted_prob - actual_prob| per bin

Example:
  Bin [0.8, 0.9]: model predicts ~0.85 prob, actual positive rate = 0.60
  → This bin contributes |0.85 - 0.60| = 0.25 to ECE

ECE < 0.05 = well-calibrated
ECE > 0.10 = poorly calibrated (confidence scores are misleading)
```

### Example

```python
@pytest.mark.ml_model
def test_model_well_calibrated(y_true, y_prob):
    assert_calibration(y_true, y_prob, max_error=0.05)
```

### Related Tests

| Test | What it validates |
|------|-------------------|
| `test_well_calibrated` | Perfect calibration passes (ECE near 0) |
| `test_poorly_calibrated` | Overconfident model fails |
| `test_calibration_details` | ECE value and per-bin data in result |

---
