# Healthcare ML Evaluation

mltk provides **clinical-grade diagnostic quality metrics** as pytest-native assertions. These go beyond generic accuracy to catch the failure modes that matter in healthcare: missed diagnoses, unnecessary procedures, and the base-rate illusion on imbalanced datasets.

## Why Generic Accuracy Fails in Healthcare

A cancer screening model tested on a population where 99% of patients are healthy can achieve **99% accuracy by always predicting "healthy"** -- and miss every single cancer case. Generic accuracy metrics are dangerous in healthcare because they hide catastrophic failures behind impressive numbers.

| Failure Mode | Clinical Impact | Which Metric Catches It |
|---|---|---|
| **Missed diagnosis** | Patient walks away with undetected cancer | `assert_sensitivity` |
| **False alarm** | Unnecessary biopsy, surgery, anxiety | `assert_specificity` |
| **Low positive trust** | "Positive" result is wrong half the time | `assert_ppv` |
| **Low negative trust** | "Clear" result misses actual disease | `assert_npv` |
| **Chance-level agreement** | Model adds nothing beyond guessing majority class | `assert_clinical_agreement` |

## The 5-Metric Clinical Evaluation

Use all five together for a complete diagnostic evaluation:

```
Sensitivity        --> Does the model catch sick patients?
Specificity        --> Does the model avoid false alarms?
PPV                --> When it says "positive," is it right?
NPV                --> When it says "negative," is it right?
Clinical Agreement --> Does the model agree beyond random chance?
```

All five take `y_true` and `y_pred` as numpy arrays of binary labels (0=negative, 1=positive).

---

## assert_sensitivity

True positive rate (recall). Of all patients who ARE sick, how many does the model detect?

**Why it matters:** A screening model with 60% sensitivity misses 40% of cancers. Those patients leave with a false sense of security.

**Formula:**

```
sensitivity = TP / (TP + FN)
```

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `y_true` | `np.ndarray` | required | Ground truth binary labels (0/1) |
| `y_pred` | `np.ndarray` | required | Predicted binary labels (0/1) |
| `min_sensitivity` | `float` | `0.9` | Minimum acceptable sensitivity |
| `severity` | `Severity` | `CRITICAL` | Assertion severity level |

**Returns:** TestResult with `sensitivity`, `min_sensitivity`, `tp`, `fn`, `n_positive`

**Example:**

```python
import numpy as np
from mltk.domains.healthcare import assert_sensitivity

# Cancer screening model: 5 actual cancers in the dataset
y_true = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
y_pred = np.array([1, 1, 1, 0, 1, 0, 0, 1, 0, 0])

# 4 out of 5 cancers detected = 80% sensitivity
result = assert_sensitivity(y_true, y_pred, min_sensitivity=0.8)
assert result.details["sensitivity"] == 0.8
assert result.details["fn"] == 1  # 1 missed cancer
```

**When to use:** For screening tests (mammography, COVID rapid tests) where missing a positive case is the primary danger. Regulatory thresholds are typically 90-95%.

---

## assert_specificity

True negative rate. Of all patients who are HEALTHY, how many does the model correctly clear?

**Why it matters:** A model with 80% specificity sends 20% of healthy patients for unnecessary biopsies, imaging, and anxiety.

**Formula:**

```
specificity = TN / (TN + FP)
```

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `y_true` | `np.ndarray` | required | Ground truth binary labels (0/1) |
| `y_pred` | `np.ndarray` | required | Predicted binary labels (0/1) |
| `min_specificity` | `float` | `0.9` | Minimum acceptable specificity |
| `severity` | `Severity` | `CRITICAL` | Assertion severity level |

**Returns:** TestResult with `specificity`, `min_specificity`, `tn`, `fp`, `n_negative`

**Example:**

```python
import numpy as np
from mltk.domains.healthcare import assert_specificity

# 6 healthy patients, model incorrectly flags 1
y_true = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1])
y_pred = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

# 5/6 negatives correctly cleared = 83.3% specificity
result = assert_specificity(y_true, y_pred, min_specificity=0.8)
assert result.details["fp"] == 1  # 1 false alarm
```

**When to use:** For confirmatory tests (biopsy decisions, surgical recommendations) where a false positive leads to invasive procedures.

---

## assert_ppv

Positive predictive value (precision). When the model says "positive," how often is it actually correct?

**Why it matters:** The base-rate trap. For a rare disease with 1% prevalence, even a test with 99% sensitivity and 99% specificity has a PPV of only ~50%. Half of all "positive" results are false alarms.

**Formula:**

```
ppv = TP / (TP + FP)
```

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `y_true` | `np.ndarray` | required | Ground truth binary labels (0/1) |
| `y_pred` | `np.ndarray` | required | Predicted binary labels (0/1) |
| `min_ppv` | `float` | `0.8` | Minimum acceptable PPV |
| `severity` | `Severity` | `CRITICAL` | Assertion severity level |

**Returns:** TestResult with `ppv`, `min_ppv`, `tp`, `fp`, `n_predicted_positive`

**Example:**

```python
import numpy as np
from mltk.domains.healthcare import assert_ppv

# Model predicts 5 positives, but only 3 are correct
y_true = np.array([1, 1, 1, 0, 0, 0, 0, 0])
y_pred = np.array([1, 1, 1, 1, 1, 0, 0, 0])

# PPV = 3/5 = 0.6
result = assert_ppv(y_true, y_pred, min_ppv=0.5)
assert result.details["ppv"] == 0.6
assert result.details["fp"] == 2  # 2 false positives
```

**When to use:** Always check PPV alongside sensitivity, especially for rare conditions. Patients and clinicians care about this metric most -- it answers "given I tested positive, what is the chance I actually have it?"

---

## assert_npv

Negative predictive value. When the model says "negative," how often is it actually correct?

**Why it matters:** Critical for ruling-out tests. In an emergency room, sending a patient home with a missed heart attack because the model said "negative" is catastrophic.

**Formula:**

```
npv = TN / (TN + FN)
```

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `y_true` | `np.ndarray` | required | Ground truth binary labels (0/1) |
| `y_pred` | `np.ndarray` | required | Predicted binary labels (0/1) |
| `min_npv` | `float` | `0.9` | Minimum acceptable NPV |
| `severity` | `Severity` | `CRITICAL` | Assertion severity level |

**Returns:** TestResult with `npv`, `min_npv`, `tn`, `fn`, `n_predicted_negative`

**Example:**

```python
import numpy as np
from mltk.domains.healthcare import assert_npv

# Model predicts 6 negatives, but 1 is a missed positive
y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1])
y_pred = np.array([0, 0, 0, 0, 0, 0, 1, 1])

# NPV = 5/6 = 0.833 (1 missed positive among negative predictions)
result = assert_npv(y_true, y_pred, min_npv=0.8)
assert result.details["fn"] == 1  # 1 missed case
```

**When to use:** For rule-out tests in emergency settings (troponin for heart attack, D-dimer for pulmonary embolism). NPV must be extremely high (>99%) when the consequence of a false negative is death.

---

## assert_clinical_agreement

Cohen's Kappa -- agreement between model and ground truth beyond random chance.

**Why it matters:** Raw accuracy is an illusion on imbalanced datasets. A model and a doctor agreeing 95% of the time might only yield Kappa = 0.2 if the base rate is 95% -- the model adds nothing beyond always guessing the majority class.

**Interpretation (Landis & Koch, 1977):**

| Kappa | Interpretation |
|---|---|
| < 0.00 | Less than chance agreement |
| 0.01 - 0.20 | Slight agreement |
| 0.21 - 0.40 | Fair agreement |
| 0.41 - 0.60 | Moderate agreement |
| 0.61 - 0.80 | Substantial agreement |
| 0.81 - 1.00 | Almost perfect agreement |

**Formula:**

```
p_observed = (TP + TN) / N
p_expected = (TP+FN)(TP+FP)/N^2 + (TN+FP)(TN+FN)/N^2
kappa = (p_observed - p_expected) / (1 - p_expected)
```

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `y_true` | `np.ndarray` | required | Ground truth binary labels (0/1) |
| `y_pred` | `np.ndarray` | required | Predicted binary labels (0/1) |
| `min_kappa` | `float` | `0.6` | Minimum acceptable Kappa |
| `severity` | `Severity` | `CRITICAL` | Assertion severity level |

**Returns:** TestResult with `kappa`, `min_kappa`, `p_observed`, `p_expected`, `n_samples`

**Example:**

```python
import numpy as np
from mltk.domains.healthcare import assert_clinical_agreement

# Imbalanced: 95% healthy, model always predicts healthy
y_true = np.zeros(100, dtype=int)
y_true[:5] = 1  # 5% disease prevalence
y_pred = np.zeros(100, dtype=int)  # always predict healthy

# Accuracy = 95% but Kappa ~ 0 (no better than chance)
# This FAILS -- exposing the accuracy illusion
from mltk.core.assertion import MltkAssertionError
try:
    assert_clinical_agreement(y_true, y_pred, min_kappa=0.1)
except MltkAssertionError as e:
    print(f"Kappa: {e.result.details['kappa']:.4f}")
    # Kappa is near 0 despite 95% accuracy
```

**When to use:** As a reality check on any accuracy claim. If Kappa is low despite high accuracy, the model is exploiting class imbalance, not learning diagnostic patterns.

---

## Complete Clinical Evaluation Pipeline

Run all five assertions together for a thorough diagnostic model evaluation:

```python
import numpy as np
from mltk.domains.healthcare import (
    assert_sensitivity,
    assert_specificity,
    assert_ppv,
    assert_npv,
    assert_clinical_agreement,
)

# Your model's predictions
y_true = np.array([...])  # ground truth diagnoses
y_pred = np.array([...])  # model predictions

# Gate 1: Does the model catch disease? (screening)
assert_sensitivity(y_true, y_pred, min_sensitivity=0.90)

# Gate 2: Does the model avoid false alarms? (confirmatory)
assert_specificity(y_true, y_pred, min_specificity=0.85)

# Gate 3: Are positive results trustworthy?
assert_ppv(y_true, y_pred, min_ppv=0.70)

# Gate 4: Are negative results trustworthy?
assert_npv(y_true, y_pred, min_npv=0.95)

# Gate 5: Does the model add value beyond chance?
assert_clinical_agreement(y_true, y_pred, min_kappa=0.60)
```

## Edge Cases Handled

All assertions handle these edge cases gracefully:

- **Empty arrays**: Fails with "Cannot compute metric on empty arrays"
- **Length mismatch**: Fails with "Array length mismatch" and both lengths
- **Non-binary values**: Fails with "Non-binary values detected" and actual values
- **No positive cases** (sensitivity): Fails with "undefined" message
- **No negative cases** (specificity): Fails with "undefined" message
- **Model never predicts positive** (PPV): Fails with "undefined" message
- **Model never predicts negative** (NPV): Fails with "undefined" message
- **Degenerate kappa** (all same class): Fails with "undefined" message
