# Concept Drift Detection

Detect when the relationship between inputs and outputs changes (P(Y|X) drift).

**Module:** `mltk.monitor.concept_drift`

**ML Lifecycle Stage:** Production monitoring / Retraining decisions

**When to use:**
- Deciding whether to retrain a model based on degraded input-output alignment
- Monitoring A/B tests for unexpected behavioral changes
- Regulatory compliance where model performance must be continuously validated
- Any scenario where ground truth labels become available after prediction

---

## Overview

Most drift detection focuses on inputs or outputs in isolation. But the most dangerous kind of drift is invisible to both: **concept drift** -- when the relationship between inputs and outputs changes, even if their individual distributions stay the same.

Consider a loan default model. The income distribution of applicants (P(X)) might stay identical. The overall default rate (P(Y)) might stay identical. But if the *meaning* of income changes -- for example, during an economic shift where the same income level now carries higher default risk -- the model's learned mapping from income to default probability is wrong. This is concept drift.

### The Four Types of Drift

| Type | Distribution | What Changed | Detectable Without Labels? |
|------|-------------|-------------|---------------------------|
| **Input drift** | P(X) | Feature distributions shifted | Yes |
| **Output drift** | P(Y-hat) | Model predictions shifted | Yes |
| **Label drift** | P(Y) | True outcome distribution shifted | Needs labels |
| **Concept drift** | P(Y\|X) | Relationship between inputs and outputs changed | Needs labels |

Concept drift is the hardest to detect because it requires ground truth labels. You cannot see it by looking at features alone or predictions alone -- you need to compare how well predictions match reality across two time periods.

---

## assert_no_concept_drift

Compares model error patterns between a reference period and a current period. If the relationship between predictions and true labels has changed significantly, the assertion fails.

```python
from mltk.monitor.concept_drift import assert_no_concept_drift

result = assert_no_concept_drift(
    y_true_ref=ref_labels,
    y_pred_ref=ref_predictions,
    y_true_cur=cur_labels,
    y_pred_cur=cur_predictions,
    method="chi2",
    alpha=0.05,
)
```

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `y_true_ref` | `list \| np.ndarray` | *(required)* | True labels from the reference period. |
| `y_pred_ref` | `list \| np.ndarray` | *(required)* | Model predictions from the reference period. |
| `y_true_cur` | `list \| np.ndarray` | *(required)* | True labels from the current period. |
| `y_pred_cur` | `list \| np.ndarray` | *(required)* | Model predictions from the current period. |
| `method` | `str` | `"chi2"` | Statistical test: `"chi2"`, `"fisher"`, or `"proportion"`. |
| `alpha` | `float` | `0.05` | Significance level. The assertion fails if p-value < alpha. |

### Returns

`TestResult` with:

- `name`: `"monitor.concept_drift"`
- `passed`: `True` if no concept drift detected, `False` otherwise
- `severity`: `CRITICAL`
- `details.drift_detected`: boolean flag
- `details.method`: statistical method used
- `details.statistic`: test statistic value
- `details.p_value`: p-value of the test
- `details.alpha`: significance level used
- `details.ref_error_rate`: error rate in the reference period
- `details.cur_error_rate`: error rate in the current period
- `details.error_rate_change`: absolute difference in error rates

### Example

```python
import numpy as np
import pytest
from mltk.monitor.concept_drift import assert_no_concept_drift

@pytest.mark.ml_drift
def test_model_concept_stable():
    """The relationship between features and outcomes has not changed."""
    # Reference period: last month's predictions vs. ground truth
    ref_labels = load_labels(period="2025-01")
    ref_preds = load_predictions(period="2025-01")

    # Current period: this month
    cur_labels = load_labels(period="2025-02")
    cur_preds = load_predictions(period="2025-02")

    assert_no_concept_drift(
        y_true_ref=ref_labels,
        y_pred_ref=ref_preds,
        y_true_cur=cur_labels,
        y_pred_cur=cur_preds,
        method="chi2",
        alpha=0.05,
    )
```

---

## Methods

### Chi-Squared (`method="chi2"`)

Constructs a contingency table of correct/incorrect predictions for the reference and current periods, then applies the chi-squared test of independence.

```
                  Reference    Current
Correct              a            b
Incorrect            c            d
```

If the error pattern differs significantly between periods, P(Y|X) has likely changed.

**Best for:** Large samples (n > 30 per cell). General-purpose default.

```python
assert_no_concept_drift(
    y_true_ref, y_pred_ref,
    y_true_cur, y_pred_cur,
    method="chi2",
)
```

### Fisher's Exact Test (`method="fisher"`)

Same contingency table as chi-squared, but uses Fisher's exact test instead of the chi-squared approximation. Computes the exact p-value rather than relying on the chi-squared distribution.

**Best for:** Small samples where the chi-squared approximation may be unreliable (any cell count < 5). Computationally more expensive on large datasets.

```python
assert_no_concept_drift(
    y_true_ref, y_pred_ref,
    y_true_cur, y_pred_cur,
    method="fisher",
)
```

### Proportion Test (`method="proportion"`)

Compares the error rates directly using a two-proportion z-test. Tests whether the proportion of incorrect predictions differs between the reference and current periods.

**Best for:** When you care specifically about the error rate difference and want an interpretable z-statistic. Works well for binary classification. Requires moderately large samples (n > 30).

```python
assert_no_concept_drift(
    y_true_ref, y_pred_ref,
    y_true_cur, y_pred_cur,
    method="proportion",
)
```

### Choosing a Method

| Method | Sample Size | Speed | Interpretability | Use When |
|--------|-------------|-------|-------------------|----------|
| `chi2` | Large (n > 30/cell) | Fast | Moderate | Default choice, general purpose |
| `fisher` | Any (exact test) | Slower | High | Small samples, regulatory audits |
| `proportion` | Moderate (n > 30) | Fast | High (z-score, error rates) | Binary classification, error rate monitoring |

---

## When to Use Concept Drift Detection

### Model Retraining Triggers

The most common use case. Run concept drift detection on a schedule (daily, weekly) as labels become available. When drift is detected, trigger a retraining pipeline.

```python
@pytest.mark.ml_drift
def test_retrain_needed():
    """Check if the model needs retraining based on concept drift."""
    result = assert_no_concept_drift(
        y_true_ref=baseline_labels,
        y_pred_ref=baseline_preds,
        y_true_cur=recent_labels,
        y_pred_cur=recent_preds,
    )
    # If this test fails, the CI/CD pipeline triggers retraining
```

### A/B Test Monitoring

When running A/B tests on model versions, concept drift detection can reveal whether one model variant's error pattern has diverged from the control.

```python
@pytest.mark.ml_drift
def test_ab_variant_stable():
    """Model variant B has not diverged from control in error patterns."""
    assert_no_concept_drift(
        y_true_ref=control_labels,
        y_pred_ref=control_preds,
        y_true_cur=variant_b_labels,
        y_pred_cur=variant_b_preds,
        method="fisher",
        alpha=0.01,  # stricter threshold for A/B tests
    )
```

### Regulatory Compliance

Regulatory frameworks increasingly require continuous model monitoring. Concept drift detection provides evidence that a model's decision-making relationship has remained stable.

- **EU AI Act (Article 72):** High-risk AI systems must be monitored for performance degradation. Concept drift detection directly addresses this requirement by testing whether the learned input-output relationship has changed.
- **FDA (21 CFR Part 11):** Medical device ML models require documented evidence of continued validity. Concept drift tests produce auditable `TestResult` records with timestamps, statistics, and pass/fail outcomes.
- **Fair lending (ECOA/Reg B):** Financial models must demonstrate consistent behavior. Concept drift detection across demographic segments can reveal disparate performance changes.

```python
@pytest.mark.ml_compliance
def test_eu_ai_act_model_monitoring():
    """Article 72 compliance: model performance has not degraded."""
    result = assert_no_concept_drift(
        y_true_ref=quarterly_baseline_labels,
        y_pred_ref=quarterly_baseline_preds,
        y_true_cur=current_quarter_labels,
        y_pred_cur=current_quarter_preds,
        method="chi2",
        alpha=0.01,
    )
    # TestResult is stored in mltk server for audit trail
```

---

## Edge Cases

- **Empty arrays**: Returns a passing result if any input array is empty (no data to compare).
- **Perfect predictions in both periods**: Returns a passing result (0% error rate in both, no divergence).
- **All predictions wrong in both periods**: Returns a passing result (100% error rate in both, no divergence -- the relationship is consistently bad, not changed).
- **Mismatched lengths**: `y_true_ref` and `y_pred_ref` must have the same length. Same for the current pair. Raises `ValueError` if mismatched.
- **Unknown method**: Fails with a descriptive error listing supported methods.
- **Single observation**: Fisher's exact test handles this; chi-squared and proportion tests may produce unreliable results.

---

## Drift Story: Complete Coverage

mltk provides four layers of drift detection, each answering a different question:

| Type | What Changed | Question Answered | mltk Assertion |
|------|-------------|-------------------|----------------|
| Input drift P(X) | Feature distributions | Has the data the model sees changed? | `assert_no_drift` |
| Output drift P(Y-hat) | Prediction distributions | Have the model's outputs shifted? | `assert_no_output_drift` |
| Streaming drift | Real-time distribution shift | Is the data changing right now? | `assert_no_streaming_drift` |
| Concept drift P(Y\|X) | Input-output relationship | Has the meaning of the data changed? | `assert_no_concept_drift` |

Concept drift is the final piece. Without it, you can detect *that* something changed but not *what* changed at the semantic level. A model that passes input drift, output drift, and streaming drift checks can still be silently wrong if the concept has shifted.

---

## References

- Gama, J., et al. (2014). "A Survey on Concept Drift Adaptation." *ACM Computing Surveys*, 46(4), 1-37.
- Lu, J., et al. (2019). "Learning under Concept Drift: A Review." *IEEE Transactions on Knowledge and Data Engineering*, 31(12), 2346-2363.

---
