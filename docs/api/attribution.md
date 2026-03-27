# Attribution Stability

Verify that feature attribution methods (SHAP, LIME, Integrated Gradients) produce reproducible, trustworthy explanations.

**Module:** `mltk.model.attribution`

**ML Lifecycle Stage:** Post-training evaluation / Explainability audit

**When to use:**

- Validating that model explanations are stable enough for regulatory audit trails
- CI/CD gates before deploying models that must provide consistent feature importance rankings
- Comparing explanation methods (KernelSHAP vs. TreeSHAP vs. LIME) for reliability
- Any scenario where stakeholders will act on feature attribution results

---

## The Reproducibility Problem

Feature attribution methods like SHAP, LIME, and Integrated Gradients are the standard way to answer "why did the model make this prediction?" But many of these methods rely on randomized sampling internally:

- **KernelSHAP** approximates Shapley values by sampling random feature coalitions. Fewer samples = noisier estimates.
- **LIME** generates perturbed copies of the input by randomly toggling features on/off, then fits a local linear model to the perturbations.
- **Permutation Importance** shuffles feature columns randomly to measure prediction changes.
- **Integrated Gradients** can vary with different baseline choices and numerical integration steps.

The consequence: run the same explanation method twice on the same input and you can get different results. Sometimes slightly different. Sometimes dramatically different.

```
Run 1 (seed=42):
  #1  age            0.38
  #2  income         0.25
  #3  credit_score   0.18
  #4  employment     0.12
  #5  zip_code       0.07

Run 2 (seed=99):
  #1  income         0.31    <-- was #2
  #2  age            0.29    <-- was #1
  #3  employment     0.19    <-- was #4
  #4  credit_score   0.14    <-- was #3
  #5  debt_ratio     0.07    <-- was NOT in top-5
```

In this example, the top feature flipped between runs, and a completely different feature entered the top-5. If you showed "Run 1" to a loan officer as the explanation for a denial, they would see "age is the primary factor." If you showed "Run 2," they would see "income is the primary factor." These are fundamentally different explanations for the same prediction.

This is not a theoretical concern. In regulated industries -- healthcare, finance, insurance, criminal justice -- explanations must be:

1. **Reproducible**: the same input should produce the same explanation
2. **Auditable**: regulators or internal compliance teams must be able to verify explanations
3. **Defensible**: if challenged in court or by a regulator, the explanation must hold up

Unstable attributions fail all three criteria. They are noise dressed up as insight.

---

## Two Complementary Checks

Attribution stability has two dimensions that must be tested independently:

| Check | What it measures | What it misses |
|-------|-----------------|----------------|
| **Top-K stability** | Whether the same features appear in the top-K ranking | Magnitude differences -- features could rank identically but with wildly different importance scores |
| **Cosine stability** | Whether the full attribution vectors point in the same direction | Ranking swaps -- two vectors can have high cosine similarity even if their sorted order differs slightly |

A model explanation is only trustworthy when **both** checks pass. Top-K alone can mask magnitude instability. Cosine alone can mask ranking instability. Together, they provide a complete picture.

```
Scenario A: Top-K PASSES, Cosine FAILS
  Run 1: [age=0.50, income=0.30, score=0.20]
  Run 2: [age=0.10, income=0.80, score=0.10]
  Same top-3 features, but the relative magnitudes changed drastically.
  The explanation "narrative" is completely different.

Scenario B: Top-K FAILS, Cosine PASSES
  Run 1: [age=0.33, income=0.32, score=0.31, debt=0.04]
  Run 2: [income=0.33, score=0.32, age=0.31, debt=0.04]
  Features shuffled in ranking, but the overall "shape" of attributions is nearly identical.
  Cosine similarity is ~0.99. Top-3 Jaccard is 3/3 = 1.0... but the ORDER changed.
  With k=1 (top feature only), Jaccard drops to 0/1 = 0.0.

Scenario C: BOTH PASS
  Run 1: [age=0.40, income=0.30, score=0.20, debt=0.10]
  Run 2: [age=0.38, income=0.31, score=0.19, debt=0.12]
  Same ranking, similar magnitudes. This explanation is stable.
```

---

### assert_top_k_stable

Assert that the top-K most important features are consistent across two attribution runs. Uses Jaccard similarity (set overlap) to measure how many of the top features appear in both runs.

```python
from mltk.model.attribution import assert_top_k_stable

# Feature names for interpretability
feature_names = ["age", "income", "credit_score", "employment", "debt_ratio", "zip_code"]

# SHAP values from two runs (1D array: one value per feature)
shap_run_1 = [0.38, 0.25, 0.18, 0.12, 0.07, 0.00]
shap_run_2 = [0.29, 0.31, 0.14, 0.19, 0.07, 0.00]

result = assert_top_k_stable(
    attributions_a=shap_run_1,
    attributions_b=shap_run_2,
    k=3,
    min_jaccard=0.8,
    feature_names=feature_names,
)
```

#### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `attributions_a` | `list \| np.ndarray` | *(required)* | Attribution values from the first run. Shape: `(n_features,)` for single-sample, or `(n_samples, n_features)` for multi-sample (averaged internally). |
| `attributions_b` | `list \| np.ndarray` | *(required)* | Attribution values from the second run. Same shape as `attributions_a`. |
| `k` | `int` | `5` | Number of top features to compare. |
| `min_jaccard` | `float` | `0.8` | Minimum required Jaccard similarity between the two top-K sets (0-1). |
| `feature_names` | `list[str] \| None` | `None` | Human-readable feature names. If provided, included in error messages and result details. |

#### How It Works: Jaccard Similarity

The Jaccard similarity coefficient measures the overlap between two sets:

```
Jaccard(A, B) = |A intersection B| / |A union B|
```

Applied to top-K feature sets:

```
Run 1 top-3: {age, income, credit_score}
Run 2 top-3: {income, age, employment}

Intersection: {age, income}           -> size = 2
Union:        {age, income, credit_score, employment}  -> size = 4

Jaccard = 2/4 = 0.5
```

With `min_jaccard=0.8`, this would **fail** -- only 50% overlap in the top-3 features.

Note that Jaccard treats the top-K as **unordered sets**. It does not penalize rank swaps within the top-K. If you need strict rank ordering, use a lower `k` (e.g., k=1 checks only the top feature) or combine with cosine stability.

#### Return Value

Returns a result object with:

| Field | Type | Description |
|-------|------|-------------|
| `passed` | `bool` | Whether the Jaccard similarity met the threshold |
| `jaccard` | `float` | The computed Jaccard similarity |
| `top_k_a` | `list[str \| int]` | Top-K features from run A (names if provided, else indices) |
| `top_k_b` | `list[str \| int]` | Top-K features from run B |
| `overlap` | `list[str \| int]` | Features present in both top-K sets |
| `only_in_a` | `list[str \| int]` | Features in top-K of A but not B |
| `only_in_b` | `list[str \| int]` | Features in top-K of B but not A |

#### Choosing K

The right value of `k` depends on the total number of features:

| Total Features | Recommended K | Rationale |
|---------------|--------------|-----------|
| < 20 | 3-5 | With few features, top-5 is already 25%+ of the feature space. Too large a K makes Jaccard trivially high. |
| 20-100 | 5-10 | Standard range. Top-10 is 10-50% of features. |
| 100+ | 10-20 | With many features, you need a larger window to detect meaningful instability. |
| 1000+ (e.g., NLP, genomics) | 20-50 | Very high-dimensional spaces. Most features are irrelevant; focus on whether the "signal" features are consistent. |

A useful heuristic: `k` should cover the features that explain approximately 80% of the total attribution mass. If 3 features explain 80% of importance, use `k=3`.

---

### assert_attribution_cosine_stability

Assert that the full attribution vectors from two runs are geometrically similar, measured by cosine similarity. Unlike top-K which checks set membership, this checks whether the entire shape of the attribution landscape is consistent.

```python
from mltk.model.attribution import assert_attribution_cosine_stability

# Per-sample attributions from two runs (2D: n_samples x n_features)
# Each row is the attribution vector for one sample
attrs_run_1 = [
    [0.38, 0.25, 0.18, 0.12, 0.07],  # sample 0
    [0.10, 0.45, 0.20, 0.15, 0.10],  # sample 1
    [0.30, 0.30, 0.25, 0.10, 0.05],  # sample 2
]

attrs_run_2 = [
    [0.36, 0.27, 0.17, 0.13, 0.07],  # sample 0 -- very similar
    [0.12, 0.42, 0.22, 0.14, 0.10],  # sample 1 -- very similar
    [0.28, 0.32, 0.23, 0.11, 0.06],  # sample 2 -- very similar
]

result = assert_attribution_cosine_stability(
    attributions_a=attrs_run_1,
    attributions_b=attrs_run_2,
    min_cosine=0.95,
)
```

#### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `attributions_a` | `array-like` | *(required)* | Attribution values from the first run. Shape: `(n_features,)` for single-sample or `(n_samples, n_features)` for per-sample attributions. |
| `attributions_b` | `array-like` | *(required)* | Attribution values from the second run. Same shape as `attributions_a`. |
| `min_cosine` | `float` | `0.95` | Minimum required cosine similarity (0-1). |
| `aggregate` | `str` | `"mean"` | How to aggregate per-sample cosine similarities: `"mean"` (average across samples) or `"min"` (worst-case sample). |

#### How It Works: Cosine Similarity

Cosine similarity measures the angle between two vectors, ignoring their magnitude:

```
cosine(A, B) = (A . B) / (||A|| * ||B||)
```

Where `A . B` is the dot product, and `||A||` is the L2 norm.

```
A = [0.40, 0.30, 0.20, 0.10]
B = [0.80, 0.60, 0.40, 0.20]    <-- same direction, 2x magnitude

cosine(A, B) = 1.0    (perfectly aligned, magnitude doesn't matter)
```

This is intentional. Attribution methods can produce different scales depending on normalization settings, random seed, or number of samples. What matters is whether they agree on the *relative* importance of features -- the direction of the vector, not its length.

For 2D inputs (per-sample attributions), cosine similarity is computed row-by-row and then aggregated:

```
For each sample i:
    cos_i = cosine(attributions_a[i], attributions_b[i])

If aggregate="mean":  result = mean(cos_0, cos_1, ..., cos_n)
If aggregate="min":   result = min(cos_0, cos_1, ..., cos_n)
```

Use `aggregate="min"` when you need **every** sample's explanation to be stable (e.g., regulatory contexts where any single unstable explanation is a compliance violation). Use `aggregate="mean"` for overall quality assessment.

#### Return Value

Returns a result object with:

| Field | Type | Description |
|-------|------|-------------|
| `passed` | `bool` | Whether the aggregated cosine similarity met the threshold |
| `cosine_similarity` | `float` | The aggregated cosine similarity score |
| `per_sample_cosines` | `list[float]` | Cosine similarity for each sample (if 2D input) |
| `worst_sample_idx` | `int \| None` | Index of the sample with lowest cosine similarity |
| `worst_sample_cosine` | `float \| None` | Cosine similarity of the worst sample |

#### Choosing min_cosine

| Context | Recommended min_cosine | Rationale |
|---------|----------------------|-----------|
| Regulated (healthcare, finance, insurance) | 0.95 - 0.99 | Explanations may face regulatory scrutiny. Small differences can change the narrative. |
| Standard production ML | 0.90 - 0.95 | Good balance of stability vs. practical tolerance for sampling noise. |
| Exploratory / research | 0.80 - 0.90 | Useful for comparing explanation methods or early-stage models where some instability is expected. |
| High-dimensional (1000+ features) | 0.85 - 0.90 | Cosine in high dimensions is naturally noisier due to the curse of dimensionality. |

---

## Integration with SHAP

A complete workflow: compute SHAP values twice with different random seeds, then verify stability.

```python
import shap
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from mltk.model.attribution import assert_top_k_stable, assert_attribution_cosine_stability

# Train a model
model = GradientBoostingClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

feature_names = X_train.columns.tolist()

# --- TreeSHAP (deterministic -- should always pass) ---
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test[:50])

# TreeSHAP is exact, so running twice gives identical results.
# This is a sanity check, not a real stability test.
shap_values_again = explainer.shap_values(X_test[:50])

assert_top_k_stable(
    attributions_a=np.mean(np.abs(shap_values), axis=0),
    attributions_b=np.mean(np.abs(shap_values_again), axis=0),
    k=5,
    min_jaccard=1.0,  # exact match expected
    feature_names=feature_names,
)

# --- KernelSHAP (stochastic -- this is the real test) ---
background = shap.sample(X_train, 100)
explainer_kernel = shap.KernelExplainer(model.predict_proba, background)

# Run 1: low sample count (noisier)
shap_run_1 = explainer_kernel.shap_values(X_test[:20], nsamples=100)

# Run 2: same data, different internal randomness
shap_run_2 = explainer_kernel.shap_values(X_test[:20], nsamples=100)

# Check top-K stability (global: average absolute SHAP across samples)
assert_top_k_stable(
    attributions_a=np.mean(np.abs(shap_run_1[1]), axis=0),  # class 1
    attributions_b=np.mean(np.abs(shap_run_2[1]), axis=0),
    k=5,
    min_jaccard=0.8,
    feature_names=feature_names,
)

# Check cosine stability (per-sample)
assert_attribution_cosine_stability(
    attributions_a=shap_run_1[1],  # shape: (20, n_features)
    attributions_b=shap_run_2[1],
    min_cosine=0.90,
    aggregate="mean",
)
```

**Key insight:** TreeSHAP computes exact Shapley values for tree-based models and is deterministic. KernelSHAP approximates via sampling and is stochastic. If your model supports TreeSHAP, prefer it -- stability is guaranteed by construction. Use these assertions to validate KernelSHAP, LIME, or any sampling-based method.

---

## Integration with LIME

LIME generates random perturbations around each input, fits a local linear model, and uses the coefficients as feature attributions. This is inherently stochastic.

```python
import lime.lime_tabular
import numpy as np
from mltk.model.attribution import assert_top_k_stable, assert_attribution_cosine_stability

# Create LIME explainer
lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=feature_names,
    class_names=["reject", "approve"],
    mode="classification",
)

# Extract LIME attributions for a batch of samples, twice
def get_lime_attributions(explainer, model_fn, X, num_features, num_samples):
    """Run LIME on multiple samples and return attribution matrix."""
    all_attrs = []
    for i in range(len(X)):
        exp = explainer.explain_instance(
            X[i],
            model_fn,
            num_features=num_features,
            num_samples=num_samples,
        )
        # Build full attribution vector (LIME only returns requested features)
        attr_dict = dict(exp.as_map()[1])  # class 1 attributions
        attr_vector = [attr_dict.get(j, 0.0) for j in range(X.shape[1])]
        all_attrs.append(attr_vector)
    return np.array(all_attrs)

# Run LIME twice with different internal randomness
lime_run_1 = get_lime_attributions(lime_explainer, model.predict_proba, X_test[:10].values,
                                    num_features=len(feature_names), num_samples=500)
lime_run_2 = get_lime_attributions(lime_explainer, model.predict_proba, X_test[:10].values,
                                    num_features=len(feature_names), num_samples=500)

# Global top-K stability
assert_top_k_stable(
    attributions_a=np.mean(np.abs(lime_run_1), axis=0),
    attributions_b=np.mean(np.abs(lime_run_2), axis=0),
    k=5,
    min_jaccard=0.8,
    feature_names=feature_names,
)

# Per-sample cosine stability
assert_attribution_cosine_stability(
    attributions_a=lime_run_1,
    attributions_b=lime_run_2,
    min_cosine=0.85,       # LIME is noisier than SHAP; lower threshold is realistic
    aggregate="mean",
)
```

**Note:** LIME's `num_samples` parameter controls how many perturbations are generated. Lower values (e.g., 100) produce faster but noisier explanations. If stability tests fail, increasing `num_samples` to 1000-5000 usually helps significantly.

---

## When Explanations Are Unstable

If `assert_top_k_stable` or `assert_attribution_cosine_stability` fails, the attributions are not reliable enough for production use. Here is a systematic approach to improving stability, ordered from least to most effort:

### 1. Increase Sampling Budget

The most common fix. More samples = lower variance in the approximation.

| Method | Parameter to Increase | Typical Range | Effect |
|--------|----------------------|---------------|--------|
| KernelSHAP | `nsamples` | 100 -> 1000 -> 5000 | Linear reduction in variance, linear increase in compute time |
| LIME | `num_samples` | 500 -> 2000 -> 5000 | Same tradeoff |
| Permutation Importance | `n_repeats` | 5 -> 20 -> 50 | More repeats = more stable mean importance |

### 2. Switch to an Exact Method

If your model supports it, use a deterministic attribution method:

| Model Type | Exact Method | Why |
|------------|-------------|-----|
| Tree-based (XGBoost, LightGBM, Random Forest, GBM) | **TreeSHAP** | Computes exact Shapley values in polynomial time using the tree structure. No sampling. |
| Linear models | **Exact linear SHAP** | Shapley values have a closed-form solution for linear models. |
| Deep learning | **Integrated Gradients** (with fixed baseline and sufficient steps) | Deterministic given a fixed baseline; increase `n_steps` for numerical precision. |

### 3. Fix Random Seeds

For reproducibility in CI/CD, fix the random seed at the explainer level:

```python
# KernelSHAP -- no built-in seed, but you can set numpy's global seed
np.random.seed(42)
shap_values = explainer.shap_values(X_test, nsamples=500)

# LIME -- supports random_state
lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=feature_names,
    random_state=42,  # <-- fixes the perturbation sampling
)
```

**Caveat:** Fixed seeds give you reproducibility (same result every time) but not stability (robust to any seed). For audit purposes, you often need both: run with a fixed seed for reproducibility, AND run with multiple seeds to verify stability.

### 4. Aggregate Across Multiple Runs

If single-run stability is insufficient, compute attributions N times and average:

```python
runs = []
for seed in range(10):
    np.random.seed(seed)
    vals = explainer.shap_values(X_test, nsamples=500)
    runs.append(vals)

# Averaged attributions are much more stable
stable_attributions = np.mean(runs, axis=0)
```

This is the Monte Carlo approach: variance decreases proportionally to 1/N. Ten runs reduces standard deviation by ~3x.

### 5. Use Permutation Importance as a Fallback

When model-agnostic explanation stability is critical and sampling budgets are constrained, permutation importance is simpler and often more stable than SHAP or LIME:

```python
from sklearn.inspection import permutation_importance

result = permutation_importance(model, X_test, y_test, n_repeats=30, random_state=42)
# result.importances_mean gives stable feature importances
```

Permutation importance measures how much the model's score drops when a feature is shuffled. It does not explain individual predictions (no per-sample attributions), but for global feature importance rankings, it is typically more stable than KernelSHAP with equivalent compute budget.

---

## Decision Flowchart

```
Is your model tree-based?
  YES -> Use TreeSHAP (exact, deterministic, always stable)
  NO  -> Is it a linear model?
           YES -> Use exact linear SHAP
           NO  -> Use KernelSHAP or LIME
                    |
                    v
                  Run assert_top_k_stable + assert_attribution_cosine_stability
                    |
                  PASS? -> Ship it
                  FAIL? -> Increase nsamples/num_samples
                             |
                           Still FAIL? -> Average across N runs
                             |
                           Still FAIL? -> Fall back to permutation importance
                             |
                           Still FAIL? -> The model itself may be unstable
                                          (check assert_robust for adversarial fragility)
```

---

## Related Assertions

| Assertion | Module | Relationship |
|-----------|--------|-------------|
| `assert_robust` | `mltk.model.adversarial` | If predictions are unstable under noise, attributions will also be unstable |
| `assert_no_bias` | `mltk.model.bias` | Attribution stability is a prerequisite for trustworthy bias explanations |
| `assert_no_concept_drift` | `mltk.monitor.concept_drift` | If concepts drift, historical attributions become invalid |
| `assert_model_regression` | `mltk.model.regression` | Model changes that degrade accuracy often destabilize attributions too |
