# S72 Enhancements: Research-Driven Testing

Sprint 72 adds three capabilities grounded in peer-reviewed
research. Each one addresses a blind spot in traditional ML
testing -- scenarios where standard tools report "all clear"
while real problems go undetected.

This document teaches the **why** behind each feature, not
just the API surface. If you only read the docstrings, you
miss the story.

---

## 1. Multivariate Drift Detection (MMD)

### The Problem: Per-Feature Tests Are Blind to Correlation

Imagine a credit-scoring model trained on two features:
`income` and `debt_ratio`. In training, they are negatively
correlated -- higher income, lower debt ratio. Six months
later, both features still have the same marginal
distributions (mean, variance, shape all unchanged). A KS
test on each column passes. PSI passes. Every per-feature
drift test says "no drift."

But the *correlation flipped*. Now high-income applicants
also carry high debt ratios. The joint distribution shifted
dramatically, and the model's learned decision boundary is
wrong.

```
Training:           Production:
  income high         income high
  debt   low          debt   HIGH  <-- correlation flip
  KS: pass            KS: pass
  PSI: pass            PSI: pass
  MMD: ---             MMD: DRIFT DETECTED
```

Per-feature tests decompose a multivariate problem into
independent univariate checks. They literally cannot see
changes in the relationships *between* features. This is
not a theoretical concern -- covariate shift in production
is overwhelmingly a joint-distribution phenomenon.

### What MMD Is

Maximum Mean Discrepancy is a **kernel two-sample test**.
Given samples from two distributions P and Q, it asks:
"are these samples drawn from the same distribution?"

The core idea (Gretton et al., JMLR 2012): map both
sample sets into a Reproducing Kernel Hilbert Space (RKHS)
using a kernel function, then measure the distance between
their mean embeddings in that space.

```
MMD^2(P, Q) = E[k(x,x')] + E[k(y,y')] - 2*E[k(x,y)]

where x,x' ~ P and y,y' ~ Q
```

If MMD^2 = 0, the distributions are identical. If MMD^2 > 0,
they differ. The beauty is that this works in *any*
dimensionality without needing to bin, bucket, or histogram
anything.

### Why RBF Kernel

The Radial Basis Function (Gaussian) kernel is a
**characteristic kernel** -- meaning MMD=0 if and only if
P=Q. Not all kernels have this property. A linear kernel,
for example, only compares means. A polynomial kernel of
degree 2 compares means and covariances but misses higher
moments.

The RBF kernel compares *all* moments simultaneously:

```
k(x, y) = exp(-||x - y||^2 / (2 * sigma^2))
```

This is the mathematical guarantee that if two distributions
differ in *any* way -- mean, variance, skew, kurtosis,
correlation structure, tail behavior -- MMD with an RBF
kernel will eventually detect it given enough samples.

### Why Multi-Bandwidth

A single bandwidth sigma controls the kernel's "attention
span." Small sigma focuses on local structure (nearby
points). Large sigma captures global patterns (overall
shape). Neither alone is optimal for all types of drift.

The implementation averages MMD^2 across three bandwidths:

```python
sigmas = [0.5 * sigma, sigma, 2.0 * sigma]
```

This multi-bandwidth approach ensures sensitivity to both
fine-grained local changes (a new cluster appearing) and
broad distributional shifts (the entire cloud moving). It
follows the kernel-selection strategy from Gretton et al.
where mixing bandwidths improves power across diverse
alternatives.

### The Median Heuristic: No Tuning Needed

How to pick sigma? The **median heuristic** sets sigma to
the median of all pairwise Euclidean distances in the pooled
data. This is data-adaptive -- it automatically scales with
feature dimensionality and data spread.

```python
def _median_heuristic(ref, cur, rng, subsample=500):
    pooled = np.vstack([ref, cur])
    pooled = _subsample(pooled, subsample, rng)
    # Pairwise squared distances via expansion trick
    sq_norms = np.sum(pooled ** 2, axis=1)
    d2 = (
        sq_norms[:, None]
        + sq_norms[None, :]
        - 2.0 * pooled @ pooled.T
    )
    np.maximum(d2, 0.0, out=d2)
    triu_idx = np.triu_indices(len(pooled), k=1)
    dists = np.sqrt(d2[triu_idx])
    return max(float(np.median(dists)), 1e-8)
```

Why median specifically? If sigma is too small, the kernel
matrix becomes nearly diagonal (every point is "far" from
every other). If too large, everything collapses to 1.0.
The median sits at the natural scale of the data where
roughly half the pairwise distances produce meaningful
kernel values.

### Why Permutation Test

The asymptotic distribution of MMD^2 under the null
hypothesis is a complex infinite mixture of chi-squared
distributions. Approximating it requires large samples and
is unreliable for the sample sizes typical in ML testing
(hundreds to low thousands).

The permutation test sidesteps this entirely:

1. Compute MMD^2 on the real data (observed statistic)
2. Pool both datasets, randomly shuffle, re-split
3. Compute MMD^2 on the shuffled split
4. Repeat 200 times
5. P-value = fraction of shuffled MMD^2 >= observed MMD^2

This gives an **exact** test for any sample size. No
distributional assumptions, no approximation errors.

```python
pooled = np.vstack([ref, cur])
count_ge = 0
for _ in range(n_permutations):
    perm = rng.permutation(m + n)
    perm_ref = pooled[perm[:m]]
    perm_cur = pooled[perm[m:]]
    perm_mmd2 = _multi_bandwidth_mmd2(
        perm_ref, perm_cur, sigmas
    )
    if perm_mmd2 >= observed_mmd2:
        count_ge += 1
p_value = (count_ge + 1) / (n_permutations + 1)
```

The `+1` in the numerator and denominator is the standard
correction that ensures the p-value is never exactly zero
(which would be a false certainty claim).

### Working Example

```python
import numpy as np
from mltk.data import assert_no_multivariate_drift

# Simulate the correlation-flip scenario
rng = np.random.default_rng(42)

# Training: income and debt are negatively correlated
cov_train = [[1.0, -0.8], [-0.8, 1.0]]
ref = rng.multivariate_normal([0, 0], cov_train, 300)

# Production: correlation flipped to positive
cov_prod = [[1.0, 0.8], [0.8, 1.0]]
cur = rng.multivariate_normal([0, 0], cov_prod, 300)

# Per-feature KS would pass -- marginals are identical!
# MMD catches the joint distribution change:
result = assert_no_multivariate_drift(ref, cur)
# => FAIL: Multivariate drift detected
```

### API Reference

```python
from mltk.data import assert_no_multivariate_drift

assert_no_multivariate_drift(
    reference,           # np.ndarray or pd.DataFrame
    current,             # np.ndarray or pd.DataFrame
    threshold=0.05,      # p-value threshold
    n_permutations=200,  # permutation count
    max_samples=500,     # subsample cap per dataset
    kernel="rbf",        # only "rbf" supported
    sigma=None,          # None = median heuristic
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `reference` | `ndarray \| DataFrame` | *(required)* | Baseline data, shape (m, d) |
| `current` | `ndarray \| DataFrame` | *(required)* | Current data, shape (n, d) |
| `threshold` | `float` | `0.05` | P-value cutoff (pass if p > threshold) |
| `n_permutations` | `int` | `200` | Permutation count for p-value |
| `max_samples` | `int` | `500` | Max rows per dataset (random subsample) |
| `kernel` | `str` | `"rbf"` | Kernel type (only "rbf" supported) |
| `sigma` | `float \| None` | `None` | Bandwidth override (None = auto) |

**Returns:** `TestResult` with `statistic` (MMD^2),
`p_value`, `sigma`, `bandwidths`, `n_features`,
`ref_samples`, `cur_samples`, `drift_detected`.

### When to Use MMD vs Per-Feature Tests

| Scenario | Use |
|----------|-----|
| Single feature monitoring | KS or PSI |
| Categorical feature | Chi-squared or JS |
| Financial model (regulatory) | PSI (industry standard) |
| Multi-feature correlation check | **MMD** |
| Embedding drift (NLP/CV) | **MMD** |
| Quick CI gate (< 100ms) | KS (faster) |
| Production monitoring (many features) | **MMD** |

### Performance

- ~2-4 seconds for 500 samples, 200 permutations
- O(n^2) in sample count (kernel matrix computation)
- Subsampling keeps runtime bounded regardless of dataset
- Fixed seed (`rng = default_rng(42)`) for CI determinism

### Design Decisions

**Pure numpy, no scipy.** The entire MMD pipeline (kernel
matrix, distance computation, permutation test) uses only
numpy operations. This keeps the dependency footprint
minimal and avoids version-compatibility issues.

**Random subsampling, not sequential.** When datasets exceed
`max_samples`, rows are randomly selected rather than
taking the first N. Sequential slicing would introduce
temporal bias if data has time-ordered patterns -- the
subsample would not represent the full distribution.

**Fixed seed for CI.** The random generator is seeded with
`np.random.default_rng(42)` so identical inputs always
produce identical p-values. Non-deterministic test results
break CI pipelines.

---

## 2. SmoothECE Calibration

### The Problem: Standard ECE Lies

Expected Calibration Error (ECE) is the most widely used
calibration metric. It divides predictions into bins,
compares the average predicted probability to the actual
frequency of positives in each bin, and reports the
weighted average gap.

ECE has three fundamental problems:

**1. Bin boundary artifacts.** Move a bin edge by 0.01 and
the ECE value changes. A prediction of 0.799 lands in
bin [0.7, 0.8) while 0.801 lands in [0.8, 0.9). Same
model, same predictions, different ECE depending on where
you draw the lines.

**2. Empty bins.** If no predictions fall in a bin, that
bin contributes nothing to ECE. A model that avoids certain
confidence ranges gets a free pass on those ranges.

**3. Theoretical inconsistency.** This is the killer: **a
perfectly calibrated model can have non-zero ECE.** The
binned estimator has irreducible bias from discretization.
Conversely, a miscalibrated model can get ECE=0 if its
errors happen to cancel within bins. ECE is not a
*consistent* estimator of calibration error.

```
Example: the bin-boundary problem

Predictions: [0.79, 0.79, 0.81, 0.81]
True labels:  [1,    1,    0,    0]

With bins [0.7, 0.8), [0.8, 0.9):
  Bin [0.7, 0.8): avg_pred=0.79, avg_true=1.0
                   error=0.21
  Bin [0.8, 0.9): avg_pred=0.81, avg_true=0.0
                   error=0.81
  ECE = 0.5*0.21 + 0.5*0.81 = 0.51

With bins [0.75, 0.85):  (shift boundary by 0.05)
  All 4 predictions in one bin
  avg_pred=0.80, avg_true=0.50
  ECE = 0.30

Same data, different ECE. Which one is "correct"?
Neither -- the metric itself is unreliable.
```

### How SmoothECE Fixes Every Problem

SmoothECE (Blasiok et al., ICLR 2024) replaces binning
with kernel smoothing. Instead of assigning each prediction
to a single bin, it uses a continuous Gaussian kernel to
estimate the calibration function at every prediction point.

**No bins = no boundary artifacts.** The kernel gives every
prediction a smooth, continuous neighborhood. No arbitrary
boundaries to game.

**No empty regions.** The kernel density is defined
everywhere in [0, 1]. There are no "empty bins" to skip.

**Provably consistent.** SmoothECE converges to zero if and
only if the model is truly calibrated. This is the crucial
theoretical property that binned ECE lacks.

### The Reflected Gaussian Kernel

Probabilities live on [0, 1]. A standard Gaussian kernel
would "leak" probability mass outside this interval, causing
bias near the boundaries. The reflected kernel adds mirror
images at 0 and 1:

```python
def _reflected_gaussian_kernel(p, q, sigma):
    inv = 1.0 / (sigma * np.sqrt(2.0 * np.pi))
    half_inv_sq = -0.5 / (sigma * sigma)
    return inv * (
        np.exp(half_inv_sq * (p - q) ** 2)      # original
        + np.exp(half_inv_sq * (p + q) ** 2)     # reflect at 0
        + np.exp(half_inv_sq * (p - q - 2) ** 2) # reflect at 1
        + np.exp(half_inv_sq * (p - q + 2) ** 2) # reflect at 1
    )
```

This is a standard technique from boundary-corrected kernel
density estimation. The mirror images ensure the density
integrates correctly near 0 and 1 without artificial
dropoff.

### Nadaraya-Watson Estimator

Instead of binning, SmoothECE uses kernel regression to
estimate the true positive rate at each predicted
probability:

```
mu_hat(f) = sum_i K(f, f_i) * y_i / sum_i K(f, f_i)
```

This is the Nadaraya-Watson estimator -- a weighted average
of outcomes where the weights come from the kernel. Points
with similar predicted probabilities contribute more to the
estimate. The calibration error at each point is
`|mu_hat(f_i) - f_i|`, and SmoothECE is the average of
these errors across all predictions.

```python
def _smooth_ece_sigma(f, y, sigma):
    mu_hat = _nadaraya_watson(f, f, y, sigma)
    return float(np.mean(np.abs(mu_hat - f)))
```

### Auto-Bandwidth via Fixed-Point Iteration

The bandwidth sigma controls the smoothing level. Too small
and the estimate is noisy. Too large and it over-smooths
real miscalibration. The Blasiok et al. solution is
elegant: find the **self-consistent** sigma where:

```
smECE(sigma) >= sigma
```

The smallest such sigma is the optimal bandwidth. The
implementation finds it via binary search:

```python
def _smooth_ece_auto(f, y):
    lo, hi = 1e-4, 1.0
    for _ in range(50):   # 50 iterations = ~15 decimal places
        mid = (lo + hi) / 2.0
        val = _smooth_ece_sigma(f, y, mid)
        if val >= mid:
            lo = mid
        else:
            hi = mid
    sigma_star = (lo + hi) / 2.0
    return _smooth_ece_sigma(f, y, sigma_star), sigma_star
```

The intuition: if the true calibration error is large,
even a large sigma (heavy smoothing) still detects it.
If the error is small, you need a small sigma (fine
resolution) to see it. The fixed-point condition naturally
balances resolution against noise.

### Working Example

```python
import numpy as np
from mltk.model import assert_calibration

rng = np.random.default_rng(42)

# A reasonably calibrated model
y_true = rng.binomial(1, 0.6, size=500)
y_prob = np.clip(
    y_true * 0.7 + (1 - y_true) * 0.3
    + rng.normal(0, 0.1, 500),
    0, 1,
)

# Binned ECE -- sensitive to bin count
r1 = assert_calibration(
    y_true, y_prob,
    max_error=0.1,
    n_bins=10,
    method="ece",
)
print(f"ECE (10 bins): {r1.details['ece']:.4f}")

# SmoothECE -- no bins, consistent estimator
r2 = assert_calibration(
    y_true, y_prob,
    max_error=0.1,
    method="smooth_ece",
)
print(f"smECE: {r2.details['smooth_ece']:.4f}")
print(f"auto-sigma: {r2.details['sigma']:.4f}")
```

### API Reference

```python
from mltk.model import assert_calibration

# Classic binned ECE (default, backward compatible)
assert_calibration(
    y_true, y_prob,
    max_error=0.05,
    n_bins=10,
    method="ece",
)

# SmoothECE (kernel-smoothed, provably consistent)
assert_calibration(
    y_true, y_prob,
    max_error=0.05,
    method="smooth_ece",
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `y_true` | `array-like` | *(required)* | Binary ground truth (0/1) |
| `y_prob` | `array-like` | *(required)* | Predicted probabilities [0, 1] |
| `max_error` | `float` | `0.05` | Max allowed calibration error |
| `n_bins` | `int` | `10` | Bin count (ECE only, ignored for smooth_ece) |
| `method` | `str` | `"ece"` | `"ece"` or `"smooth_ece"` |

**Returns (method="smooth_ece"):** `TestResult` with
`smooth_ece`, `sigma` (auto-selected bandwidth),
`max_error`, `method`.

**Returns (method="ece"):** `TestResult` with `ece`,
`n_bins`, `max_error`, `method`, `bin_data`.

### When to Keep Binned ECE

SmoothECE is the better metric. But keep ECE as your
*default* in two cases:

1. **Regulatory compliance.** SR 11-7 (banking) and FDA
   guidance (medical devices) expect binned reliability
   diagrams. Regulators understand bins. Showing them a
   kernel-smoothed curve may invite questions you do not
   want to answer.

2. **Backward compatibility.** Existing test suites and
   dashboards expect ECE values. Switching the default
   would break comparisons against historical baselines.

ECE remains the default (`method="ece"`). Use SmoothECE
when you want the scientifically correct answer.

### Design Decisions

**~45 lines of numpy, not a dependency.** The reference
implementation is in the `relplot` package. We implement
the core algorithm ourselves because adding a dependency
for 45 lines of numpy carries maintenance risk. If
`relplot` changes its API or goes unmaintained, our tests
should not break.

**ECE stays as default.** Despite SmoothECE being
theoretically superior, changing the default would be a
breaking change for existing users. The `method` parameter
lets users opt in.

---

## 3. Intersectional Fairness

### The Problem: Simpson's Paradox of Fairness

In 1989, legal scholar Kimberle Crenshaw introduced the
concept of **intersectionality**: the insight that
overlapping identities create unique experiences of
discrimination that cannot be understood by examining each
identity in isolation.

Here is the concrete failure mode for ML:

```
Hiring model results:
  Gender fairness:  pass (men 60%, women 58% hired)
  Race fairness:    pass (White 59%, Black 57% hired)

Intersectional breakdown:
  White men:    62% hired
  White women:  56% hired
  Black men:    58% hired
  Black women:  31% hired   <-- BIAS HIDDEN BY AGGREGATION
```

The model is "fair" when you check gender alone. "Fair"
when you check race alone. But Black women are hired at
half the rate. This is Simpson's paradox applied to
fairness -- aggregate statistics conceal subgroup harm.

`assert_no_bias` catches the first two. Only
`assert_intersectional_fairness` catches the third.

### How It Works: Full Cartesian Enumeration

Given protected attributes, the function generates every
possible intersection:

```
gender: [M, F]
race: [White, Black, Asian, Hispanic, Other]
age: [young, middle, senior]

Total subgroups: 2 x 5 x 3 = 30
```

For each subgroup, it computes the selected fairness
metric and reports the **worst-case disparity** across
all subgroups. This is minimax fairness -- the system is
only as fair as its treatment of the most disadvantaged
intersection.

```python
# Enumerate all combinations
combo_values = [attr_uniques[k] for k in attr_names]
all_combos = list(itertools.product(*combo_values))

for combo in all_combos:
    attrs = dict(zip(attr_names, combo))
    mask = np.ones(len(y_t), dtype=bool)
    for k, v in attrs.items():
        mask &= attr_arrays[k] == v

    n = int(mask.sum())
    if n < min_subgroup_size:
        skipped[label] = n
        continue

    metrics = _subgroup_metric(y_t[mask], y_p[mask], method)
    evaluated[label] = metrics
```

### Why min_subgroup_size=30

The Central Limit Theorem guarantees that sample means
approximate a normal distribution when n >= 30 (for most
practical distributions). Below this threshold, metrics
like selection rate and TPR have such wide confidence
intervals that apparent bias could be pure sampling noise.

With 30 subgroups and a 0.05 significance level, you
*expect* 1.5 false positives from random data (the
multiple testing problem). Setting a floor of 30 samples
per subgroup is the minimum defense: it ensures each
reported metric is at least statistically meaningful,
even if the overall test does not correct for multiplicity.

Subgroups below the threshold are **skipped and reported
transparently** in the result:

```python
skipped_subgroups = {
    "gender=F & race=Other & age=senior": 12,
    "gender=M & race=Asian & age=senior": 8,
}
```

This lets auditors see exactly which intersections were
too small to test, rather than silently ignoring them.

### Three Metrics

| Metric | What It Measures | Threshold | Direction |
|--------|-----------------|-----------|-----------|
| `demographic_parity` | Max selection-rate gap across subgroups | 0.10 | diff <= threshold |
| `equalized_odds` | Max of (TPR gap, FPR gap) across subgroups | 0.10 | diff <= threshold |
| `disparate_impact` | Min/max selection-rate ratio | 0.80 | ratio >= threshold |

- **Demographic parity** asks: does every subgroup get
  approved/selected at the same rate?
- **Equalized odds** asks: does every subgroup get
  the same true positive and false positive rates?
- **Disparate impact** implements the US four-fifths
  rule: the lowest-rate group must have at least 80%
  of the highest-rate group's selection rate.

### Worst-Case Aggregation

The function reports the single worst disparity across
all evaluated subgroups. For demographic parity, this is
the difference between the highest and lowest selection
rates. For disparate impact, the ratio between the lowest
and highest.

This is **minimax fairness** -- the system passes only if
every intersection passes. A model cannot compensate for
harming one subgroup by being extra-fair to another.

### Working Example

```python
import numpy as np
from mltk.model import assert_intersectional_fairness

rng = np.random.default_rng(42)
n = 600

# Protected attributes
gender = rng.choice(["M", "F"], n)
race = rng.choice(["White", "Black", "Asian"], n)

# Ground truth
y_true = rng.binomial(1, 0.5, n)

# Model that is biased against one intersection
# Black women get worse predictions
y_pred = y_true.copy()
bias_mask = (gender == "F") & (race == "Black")
flip_idx = np.where(
    bias_mask & (y_true == 1)
)[0][:20]
y_pred[flip_idx] = 0  # flip 20 true positives to FN

result = assert_intersectional_fairness(
    y_true,
    y_pred,
    sensitive_features={
        "gender": gender,
        "race": race,
    },
    method="demographic_parity",
    min_subgroup_size=30,
)
# => Reports worst-case disparity across all
#    gender x race intersections
```

### API Reference

```python
from mltk.model import assert_intersectional_fairness

assert_intersectional_fairness(
    y_true,
    y_pred,
    sensitive_features={
        "gender": gender_array,
        "race": race_array,
        "age_group": age_array,
    },
    method="demographic_parity",
    threshold=None,         # None = method default
    min_subgroup_size=30,   # CLT floor
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `y_true` | `array-like` | *(required)* | Binary ground truth (0/1) |
| `y_pred` | `array-like` | *(required)* | Binary predictions (0/1) |
| `sensitive_features` | `dict[str, array-like]` | *(required)* | Attribute name to value array mapping |
| `method` | `str` | `"demographic_parity"` | `"demographic_parity"`, `"equalized_odds"`, or `"disparate_impact"` |
| `threshold` | `float \| None` | `None` | Custom threshold (None = method default) |
| `min_subgroup_size` | `int` | `30` | Minimum samples per subgroup |

**Returns:** `TestResult` with `worst_case_subgroup`,
`worst_case_statistic`, `evaluated_subgroups` (per-subgroup
metrics), `skipped_subgroups` (below-threshold subgroups
with their counts), `n_total_combos`, `n_evaluated`,
`n_skipped`.

### Design Decisions

**Full Cartesian enumeration, not heuristic search.** Some
fairness toolkits use sampling or heuristic methods to
find problematic subgroups (e.g., FairSquare, AIF360
subgroup discovery). We enumerate exhaustively because
correctness matters more than speed for fairness testing.
Missing a biased subgroup is worse than taking an extra
second to find it.

**Skipped subgroups reported transparently.** When a
subgroup is too small to evaluate, it appears in the
`skipped_subgroups` dict with its sample count. This
prevents the silent data censorship that plagues many
fairness tools -- an auditor can see exactly which
intersections were not tested and why.

**Three metrics, not five.** The single-attribute
`assert_no_bias` supports five metrics (including
predictive parity and equal opportunity). The
intersectional version supports three. Predictive
parity requires sufficient positive predictions per
subgroup, which is rarely achievable at fine-grained
intersections. Equal opportunity (TPR only) is a strict
subset of equalized odds and does not add information
in the intersectional setting.

---

## 4. Research Background

### Papers That Informed These Decisions

**Multivariate Drift (MMD):**

- Gretton, A., Borgwardt, K.M., Rasch, M.J., Scholkopf,
  B., & Smola, A. (2012). "A Kernel Two-Sample Test."
  *Journal of Machine Learning Research*, 13, 723-773.
  -- The foundational MMD paper. Establishes the theory
  of characteristic kernels and the permutation test.
  Our implementation follows their unbiased estimator
  (diagonal-excluded) and multi-bandwidth strategy.

- Rabanser, S., Gunnemann, S., & Lipton, Z.C. (2019).
  "Failing Loudly: An Empirical Study of Methods for
  Detecting Dataset Shift." *NeurIPS 2019*.
  -- Empirically demonstrates that per-feature tests
  miss joint distributional shifts. Motivates MMD as
  the default multivariate test. Our correlation-flip
  example directly mirrors their experimental setup.

**Calibration (SmoothECE):**

- Blasiok, J., Nakkiran, P., & Shetty, A. (2024).
  "Smooth ECE: Principled Reliability Diagrams via
  Kernel Smoothing." *ICLR 2024*.
  -- Proves binned ECE is inconsistent and proposes the
  kernel-smoothed alternative with the self-consistent
  bandwidth rule. Our auto-bandwidth binary search
  implements their Algorithm 1.

- Naeini, M.P., Cooper, G.F., & Hauskrecht, M. (2015).
  "Obtaining Well-Calibrated Probabilities Using Bayesian
  Binning into Quantiles." *AAAI 2015*.
  -- Documents the bin-boundary sensitivity problem.
  Proposes BBQ as a fix (averaging over bin schemes).
  SmoothECE supersedes this by eliminating bins entirely.

**Intersectional Fairness:**

- Crenshaw, K. (1989). "Demarginalizing the Intersection
  of Race and Sex." *University of Chicago Legal Forum*.
  -- The original intersectionality paper from legal
  scholarship. Demonstrates how single-axis frameworks
  fail to capture compound discrimination. Our test
  operationalizes this insight computationally.

- Kearns, M., Neel, S., Roth, A., & Wu, Z.S. (2018).
  "Preventing Fairness Gerrymandering in Dynamic
  Classifiers." *ICML 2018*.
  -- Formalizes the problem of fairness across rich
  subgroup collections. Proves that group-level fairness
  can be "gerrymandered" to hide subgroup harm. Our
  Cartesian enumeration is the brute-force solution to
  their theoretical concern.

- Bird, S., Dudik, M., Edgar, R., et al. (2020).
  "Fairlearn: A toolkit for assessing and improving
  fairness in AI." *Microsoft Research Technical Report*.
  -- The MetricFrame concept (computing metrics across
  intersectional groups) directly inspired our subgroup
  enumeration approach.

### What We Rejected and Why

| Alternative | Why Rejected |
|-------------|-------------|
| scipy for MMD | Added dependency for ~80 lines of numpy. Not worth the dependency chain. |
| Asymptotic MMD p-value | Unreliable for n < 1000. Permutation test is exact at any sample size. |
| Apple `relplot` for smECE | External dependency for ~45 lines. Maintenance risk outweighs convenience. |
| Adaptive binning (BBQ) | Still bin-based, still has artifacts. SmoothECE is theoretically cleaner. |
| Heuristic subgroup search (AIF360) | Missing a biased subgroup is worse than being thorough. Correctness > speed. |
| 5 metrics for intersectional | Predictive parity needs more data than intersections typically have. |
| Bonferroni correction | Overly conservative for correlated subgroups. Transparent reporting preferred. |

---

*Added in S72. Module paths: `mltk.data.drift`,
`mltk.model.slicing`, `mltk.model.bias`.*
