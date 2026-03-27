# Counterfactual Fairness & Causal Inference

Individual-level fairness testing and causal validation for ML experiments.

**Modules:** `mltk.model.counterfactual`, `mltk.model.causal`

---

## Why Counterfactual Fairness

Standard fairness metrics (demographic parity, equalized odds) measure **aggregate** disparities: "Does group A get approved more often than group B?" But aggregate metrics can hide individual-level discrimination. A model might achieve perfect demographic parity while still using gender to decide *specific* cases -- as long as the errors balance out across the population.

Counterfactual fairness asks a sharper question for **each individual**:

> "If THIS person had a different gender/race/age -- but everything else stayed the same -- would the model's prediction change?"

### The Loan Example

A loan model approves John (male, 30, $80K salary). We create a counterfactual twin: Jane (female, 30, $80K salary) -- identical except for the protected attribute. If the model now denies Jane, it is **causally using gender** in its decision, even if aggregate metrics look clean.

The **flip rate** is the fraction of individuals whose prediction changes under the counterfactual intervention. A fair model should have a flip rate near zero.

### When to Use Counterfactual vs. Group Fairness

| Approach | Question It Answers | Catches | Misses |
|----------|-------------------|---------|--------|
| Group fairness | "Are outcomes equal across groups?" | Systematic group-level bias | Individual discrimination hidden by balanced errors |
| Counterfactual fairness | "Would THIS person's outcome change?" | Causal use of protected attributes | Group-level disparities that arise from legitimate features |

**Use both.** Group fairness catches population-level issues. Counterfactual fairness catches individual-level causal reliance on protected attributes.

Reference: Kusner et al., "Counterfactual Fairness" (NeurIPS 2017).

---

## `assert_counterfactual_fairness`

Assert that a model's predictions do not change when the protected attribute is perturbed.

**Module:** `mltk.model.counterfactual`

### How It Works

1. Get original predictions: `y_orig = model_fn(X)`
2. Perturb ONLY the sensitive column (flip gender, change race, etc.)
3. Get counterfactual predictions: `y_cf = model_fn(X_perturbed)`
4. Count how many predictions flipped: `flip_rate = mean(y_orig != y_cf)`
5. Pass if `flip_rate <= max_flip_rate`

### Basic Example

```python
import numpy as np
from mltk.model.counterfactual import assert_counterfactual_fairness

# Feature matrix: [gender, income, credit_score]
X = np.array([
    [0, 50000, 720],
    [1, 50000, 720],
    [0, 80000, 680],
    [1, 80000, 680],
    [0, 30000, 750],
    [1, 30000, 750],
], dtype=float)

# A fair model that ignores gender (column 0)
def fair_model(X):
    return (X[:, 2] > 700).astype(int)  # uses only credit score

result = assert_counterfactual_fairness(
    model_fn=fair_model,
    X=X,
    sensitive_col=0,       # gender is column 0
    max_flip_rate=0.05,    # allow at most 5% of predictions to flip
)
assert result.passed  # flip_rate = 0.0
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_fn` | `Callable` | *(required)* | Takes 2-D ndarray, returns 1-D predictions |
| `X` | `np.ndarray` | *(required)* | Feature matrix, shape `(n_samples, n_features)` |
| `sensitive_col` | `int` | *(required)* | Column index of the protected attribute |
| `perturbation_fn` | `Callable \| None` | `None` | Custom perturbation function (see below) |
| `max_flip_rate` | `float` | `0.05` | Maximum allowed fraction of predictions that change |
| `severity` | `Severity` | `CRITICAL` | Severity level for the assertion |

Returns `TestResult` (name: `model.counterfactual_fairness`) with details:

| Detail Key | Type | Description |
|------------|------|-------------|
| `flip_rate` | `float` | Fraction of predictions that changed |
| `max_flip_rate` | `float` | The threshold that was required |
| `n_flipped` | `int` | Number of predictions that changed |
| `n_total` | `int` | Total number of samples |
| `sensitive_col` | `int` | Which column was perturbed |

### Default Perturbation

When `perturbation_fn` is `None`, the default perturbation logic handles:

- **Binary features** (exactly 2 unique values): flips each value to the other (e.g. 0 becomes 1, 1 becomes 0)
- **Categorical features** (3+ unique values): cycles each value to the next in sorted order (wrapping around), guaranteeing every sample receives a different value
- **Constant columns** (1 unique value): returns unchanged (cannot perturb)

### Custom Perturbation Functions

For complex perturbations (e.g. changing race while also adjusting correlated features like zip code), provide a custom function:

```python
def perturb_race_and_zip(X):
    """Perturb race (col 0) and associated zip code (col 3)."""
    X_new = X.copy()
    # Flip race
    X_new[:, 0] = 1 - X_new[:, 0]
    # Reassign zip codes to match new race demographics
    X_new[:, 3] = np.where(X_new[:, 0] == 1, 90210, 10001)
    return X_new

result = assert_counterfactual_fairness(
    model_fn=model.predict,
    X=X,
    sensitive_col=0,
    perturbation_fn=perturb_race_and_zip,
    max_flip_rate=0.05,
)
```

The custom function signature: `(X: np.ndarray) -> np.ndarray`. It receives the full feature matrix and must return a perturbed copy of the same shape.

---

## Why Causal Inference for ML

In ML, we constantly make causal claims: "Model B is better than Model A," "this feature improves conversion," "the new ranking algorithm increases engagement." But correlation is not causation. Without proper causal analysis, teams deploy "improvements" that are just noise or confounded by other factors.

### The A/B Testing Problem

You A/B test two recommendation models. Model B shows +2% click rate. Is that real, or would you see a similar difference by chance?

- **Small sample size**: +2% on 50 users is noise. +2% on 50,000 users is real.
- **Confounded assignment**: If power users are routed to Model B, the lift may reflect user quality, not model quality.

mltk provides two causal checks to validate ML experiments:

1. **ATE significance** -- Is the treatment effect statistically significant?
2. **No confounding** -- Is treatment assignment independent of features?

References: Rubin (1974), Imbens & Rubin (2015).

---

## `assert_ate_significant`

Assert that the Average Treatment Effect is statistically significant using Welch's two-sample t-test.

**Module:** `mltk.model.causal`

### How It Works

The ATE is the difference in mean outcomes between the treatment group (treatment=1) and the control group (treatment=0). Statistical significance is assessed via Welch's t-test, which does not assume equal variance between groups.

- **PASS** (p < alpha): The treatment effect is statistically significant. The observed difference is unlikely to arise from chance alone.
- **FAIL** (p >= alpha): The effect is not distinguishable from noise. Do not ship based on this evidence.

### Example

```python
import numpy as np
from mltk.model.causal import assert_ate_significant

rng = np.random.default_rng(42)

# A/B test: 500 users in control, 500 in treatment
treatment = np.array([0] * 500 + [1] * 500)

# Treatment group has genuinely higher conversion
outcome = np.concatenate([
    rng.normal(0.10, 0.05, 500),   # control: 10% avg
    rng.normal(0.13, 0.05, 500),   # treatment: 13% avg
])

result = assert_ate_significant(
    treatment=treatment,
    outcome=outcome,
    alpha=0.05,
)
assert result.passed
print(f"ATE = {result.details['ate']:.4f}")
print(f"p-value = {result.details['p_value']:.4f}")
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `treatment` | `np.ndarray` | *(required)* | Binary array (0=control, 1=treatment) |
| `outcome` | `np.ndarray` | *(required)* | Numeric outcome array (e.g. conversion, revenue) |
| `alpha` | `float` | `0.05` | Significance level. Lower = stricter. |
| `severity` | `Severity` | `CRITICAL` | Severity level for the assertion |

Returns `TestResult` (name: `model.causal.ate_significant`) with details:

| Detail Key | Type | Description |
|------------|------|-------------|
| `ate` | `float` | Average Treatment Effect (mean_treatment - mean_control) |
| `p_value` | `float` | Two-sided p-value from Welch's t-test |
| `alpha` | `float` | The significance threshold |
| `n_treatment` | `int` | Number of treated observations |
| `n_control` | `int` | Number of control observations |
| `mean_treatment` | `float` | Mean outcome in the treatment group |
| `mean_control` | `float` | Mean outcome in the control group |

### Why Welch's Instead of Student's t-test

Student's t-test assumes equal variance in both groups, which rarely holds in ML experiments (treatment and control groups often have different variance). Welch's t-test relaxes this assumption and is more robust.

mltk implements Welch's t-test using the Welch-Satterthwaite approximation for degrees of freedom and a continued-fraction expansion for the p-value -- no scipy dependency required.

---

## `assert_no_confounding`

Assert that no features are correlated with treatment assignment. In a well-randomized experiment, treatment assignment should be independent of all observed covariates.

**Module:** `mltk.model.causal`

### Why Confounding Breaks A/B Tests

Imagine an e-commerce A/B test where premium users (high spend, many sessions) are more likely to be routed to the new model. The new model "wins" -- but is it the model or the user quality? If `user_activity` correlates with treatment assignment, the estimated ATE is biased.

This check computes the Pearson correlation between each feature column and the treatment indicator. If ANY correlation exceeds `max_correlation`, the check fails and reports which features are confounded.

### Example

```python
import numpy as np
from mltk.model.causal import assert_no_confounding

rng = np.random.default_rng(42)

# Well-randomized experiment: features uncorrelated with treatment
X = rng.standard_normal((1000, 3))  # 3 features
treatment = rng.integers(0, 2, 1000)  # random assignment

result = assert_no_confounding(
    X=X,
    treatment=treatment,
    max_correlation=0.1,
)
assert result.passed
print(f"Max correlation: {result.details['max_observed_correlation']:.4f}")
```

### Detecting a Confounded Experiment

```python
# BAD: treatment assignment correlates with feature 0
X = rng.standard_normal((1000, 3))
treatment = (X[:, 0] > 0).astype(int)  # assigned based on feature!

result = assert_no_confounding(X, treatment, max_correlation=0.1)
assert not result.passed
print(f"Confounded features: {result.details['confounded_features']}")
# Confounded features: [0]
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `X` | `np.ndarray` | *(required)* | Feature matrix, shape `(n_samples, n_features)` |
| `treatment` | `np.ndarray` | *(required)* | Binary treatment indicator (0/1) |
| `max_correlation` | `float` | `0.1` | Maximum allowed absolute Pearson correlation |
| `severity` | `Severity` | `CRITICAL` | Severity level for the assertion |

Returns `TestResult` (name: `model.causal.no_confounding`) with details:

| Detail Key | Type | Description |
|------------|------|-------------|
| `max_observed_correlation` | `float` | Highest absolute correlation found |
| `confounded_features` | `list[int]` | Column indices exceeding the threshold |
| `correlations` | `dict[int, float]` | Per-feature absolute correlation values |

---

## pytest Integration

### Counterfactual Fairness in CI

```python
import pytest
import numpy as np
from mltk.model.counterfactual import assert_counterfactual_fairness

def test_loan_model_counterfactual_fairness(trained_model, test_data):
    """Loan model must not causally use gender."""
    result = assert_counterfactual_fairness(
        model_fn=trained_model.predict,
        X=test_data.values,
        sensitive_col=0,       # gender column
        max_flip_rate=0.02,    # strict: 2% max
    )
    assert result.passed, (
        f"Model causally uses gender: "
        f"{result.details['n_flipped']}/{result.details['n_total']} "
        f"predictions changed ({result.details['flip_rate']:.2%})"
    )
```

### A/B Test Validation in CI

```python
from mltk.model.causal import assert_ate_significant, assert_no_confounding

def test_ab_experiment_validity(experiment_data):
    """Validate A/B test before shipping model change."""
    X = experiment_data[["feature_1", "feature_2", "feature_3"]].values
    treatment = experiment_data["treatment_group"].values
    outcome = experiment_data["conversion"].values

    # Step 1: Verify no confounding
    conf_result = assert_no_confounding(X, treatment, max_correlation=0.1)
    assert conf_result.passed, (
        f"Confounded features: {conf_result.details['confounded_features']}"
    )

    # Step 2: Verify effect is significant
    ate_result = assert_ate_significant(treatment, outcome, alpha=0.05)
    assert ate_result.passed, (
        f"ATE not significant: p={ate_result.details['p_value']:.4f}"
    )
```
