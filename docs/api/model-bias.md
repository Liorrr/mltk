# Model Bias & Fairness Testing

Bias testing ensures ML models treat demographic groups fairly. The EU AI Act (effective August 2, 2026) mandates bias detection for high-risk AI systems. The US four-fifths rule requires selection rates within 80% across groups. mltk implements 5 fairness metrics with zero dependencies (pure numpy).

**Module:** `mltk.model.bias`

**ML Lifecycle Stage:** Post-training evaluation / Compliance gate

**Impossibility theorem:** You cannot simultaneously satisfy demographic parity, equalized odds, AND predictive parity when group base rates differ (Chouldechova-Kleinberg). Choose the metric that matches your use case.

---

## assert_no_bias

Unified fairness assertion supporting 5 methods.

```python
from mltk.model import assert_no_bias

# Demographic parity (default) -- equal selection rates
assert_no_bias(y_true, y_pred, sensitive_feature=gender, method="demographic_parity")

# Equalized odds -- equal TPR and FPR across groups
assert_no_bias(y_true, y_pred, sensitive_feature=race, method="equalized_odds")

# Four-fifths rule (legal standard for hiring/lending)
assert_no_bias(y_true, y_pred, sensitive_feature=race, method="disparate_impact", threshold=0.80)
```

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `y_true` | `array-like` | *(required)* | Ground truth labels (binary: 0/1) |
| `y_pred` | `array-like` | *(required)* | Model predictions (binary: 0/1) |
| `sensitive_feature` | `array-like` | *(required)* | Protected attribute (e.g., gender, race, age group) |
| `method` | `str` | `"demographic_parity"` | Fairness metric (see below) |
| `threshold` | `float \| None` | `None` | Custom threshold. None = method-specific default |

### Methods & Default Thresholds

| Method | What it measures | Default | Pass condition |
|--------|-----------------|---------|----------------|
| `demographic_parity` | Max selection rate difference across groups | 0.10 | diff <= threshold |
| `equalized_odds` | Max of (TPR diff, FPR diff) across groups | 0.10 | diff <= threshold |
| `predictive_parity` | Max PPV (precision) difference across groups | 0.10 | diff <= threshold |
| `disparate_impact` | Min/max selection rate ratio | 0.80 | ratio >= threshold |
| `equal_opportunity` | Max TPR difference across groups | 0.10 | diff <= threshold |

### Example

```python
@pytest.mark.ml_model
def test_hiring_model_fairness(y_true, y_pred, gender):
    # Must pass four-fifths rule for legal compliance
    assert_no_bias(y_true, y_pred, sensitive_feature=gender,
                   method="disparate_impact", threshold=0.80)

    # Also check equalized odds
    assert_no_bias(y_true, y_pred, sensitive_feature=gender,
                   method="equalized_odds", threshold=0.10)
```

### Related Tests

| Test | What it validates |
|------|-------------------|
| `test_demographic_parity_fair` | Equal selection rates pass |
| `test_demographic_parity_biased` | Unequal selection rates fail |
| `test_equalized_odds_fair` | Equal TPR/FPR across groups |
| `test_equalized_odds_biased` | Different TPR/FPR detected |
| `test_disparate_impact_passes` | Ratio >= 0.80 passes |
| `test_disparate_impact_fails` | Ratio < 0.80 fails (four-fifths rule) |
| `test_predictive_parity` | PPV difference across groups |
| `test_equal_opportunity` | TPR-only fairness check |
| `test_unknown_method` | Invalid method raises error |
| `test_single_group` | Only one group -- passes trivially |

---
