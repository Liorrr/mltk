# Data Drift Detection

Drift detection answers the critical question: **has the data changed since the model was trained?** A model that performed well on training data may silently degrade as real-world distributions shift. Drift is the #2 QA pain point in ML systems (after non-deterministic outputs).

**Module:** `mltk.data.drift`

**ML Lifecycle Stage:** Pre-training validation / Production monitoring

**When to use:**
- Before retraining: compare current data against training data baseline
- In production: detect when incoming data no longer matches what the model expects
- In CI/CD: gate deployments on drift thresholds

---

## assert_no_drift

Unified drift detection supporting 4 statistical methods.

```python
from mltk.data import assert_no_drift

# KS test (default) — best for continuous numeric features
assert_no_drift(reference, current, method="ks")

# PSI — industry standard for credit scoring / financial models
assert_no_drift(reference, current, method="psi", threshold=0.2)

# KL divergence — information-theoretic measure
assert_no_drift(reference, current, method="kl")

# Chi-squared — for categorical features
assert_no_drift(reference, current, method="chi2")
```

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `reference` | `pd.Series` | *(required)* | Baseline distribution (e.g., training data) |
| `current` | `pd.Series` | *(required)* | Current distribution to compare against baseline |
| `method` | `str` | `"ks"` | Detection method: `"ks"`, `"psi"`, `"kl"`, `"chi2"` |
| `threshold` | `float \| None` | `None` | Custom threshold. If None, uses method-specific default |

### Default Thresholds

| Method | Default | Interpretation |
|--------|---------|----------------|
| `ks` | p-value > 0.05 | No statistically significant difference |
| `psi` | < 0.1 | Stable distribution (0.1-0.2 = moderate, >0.2 = significant) |
| `kl` | < 0.1 | Low information divergence |
| `chi2` | p-value > 0.05 | No significant categorical distribution change |

### Returns

`TestResult` with details:
- `method` -- which detection method was used
- `statistic` -- the computed test statistic
- `p_value` -- p-value (for KS and chi2)
- `threshold` -- threshold used for pass/fail
- `drift_detected` -- boolean

### Why it matters for ML

**Drift is silent.** A model doesn't crash when data drifts -- it just produces increasingly wrong predictions. A fraud detection model trained on pre-COVID transaction patterns will miss new fraud patterns post-COVID. A recommendation model trained on summer behavior will fail in winter. Drift detection is the bridge between "model works on test set" and "model works in production."

### Choosing a method

| Method | Best for | Strengths | Weaknesses |
|--------|----------|-----------|------------|
| **KS** | Continuous numeric | Non-parametric, no binning needed | Only univariate |
| **PSI** | Production monitoring | Industry standard, interpretable buckets | Sensitive to binning |
| **KL** | Information theory | Captures distributional shape | Asymmetric, undefined if P(x)=0 |
| **Chi2** | Categorical features | Handles discrete values natively | Needs sufficient samples per category |

### Example

```python
import pandas as pd
import pytest
from mltk.data import assert_no_drift

@pytest.mark.ml_drift
def test_feature_distribution_stable():
    """Training data distribution hasn't changed since last month."""
    reference = pd.read_csv("data/train_features.csv")["income"]
    current = pd.read_csv("data/current_features.csv")["income"]
    assert_no_drift(reference, current, method="psi", threshold=0.2)

@pytest.mark.ml_drift
def test_categorical_feature_stable():
    """Category distribution hasn't shifted."""
    reference = pd.Series(["A", "A", "B", "B", "C"] * 100)
    current = pd.Series(["A", "A", "A", "B", "C"] * 100)  # A grew
    assert_no_drift(reference, current, method="chi2")
```

### Performance

KS test and PSI have optional **Rust acceleration** via `mltk._rust`. When the Rust extension is installed, these computations run 10-100x faster on large arrays. Falls back to scipy/numpy automatically.

### Edge Cases

- **Identical distributions** always pass (statistic = 0, p-value = 1.0)
- **Empty arrays** raise an error
- **Single-value arrays** may produce unreliable results (need sufficient samples)
- **KL divergence** clips bins to 1e-6 to avoid log(0)
- **PSI** uses 10 equal-width bins by default

### Related Tests

| Test | What it validates |
|------|-------------------|
| `test_identical_distributions_ks` | Same data passes KS test |
| `test_shifted_distribution_ks` | Mean-shifted data fails KS test |
| `test_identical_distributions_psi` | Same data gives PSI near 0 |
| `test_shifted_distribution_psi` | Shifted data gives PSI > threshold |
| `test_kl_divergence_known` | KL on known distributions matches expected |
| `test_chi2_categorical_no_drift` | Same category proportions pass |
| `test_chi2_categorical_drift` | Changed proportions fail |
| `test_custom_threshold` | User-provided threshold overrides default |
| `test_unknown_method` | Invalid method raises error |

---
