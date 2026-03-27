# Conformal Prediction

Validate prediction intervals and prediction sets from any interval-producing method (conformal, Bayesian, bootstrap, quantile regression).

**Module:** `mltk.model.conformal`

---

## assert_interval_coverage

Check that prediction intervals achieve target empirical coverage. For each sample, the true value must fall within `[y_lower, y_upper]`. The assertion passes when the observed coverage rate is at least `target_coverage - tolerance`.

Works with any interval source: conformal prediction, Bayesian credible intervals, bootstrap confidence intervals, or quantile regression.

```python
import numpy as np
from mltk.model.conformal import assert_interval_coverage

y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
y_lower = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
y_upper = np.array([1.5, 2.5, 3.5, 4.5, 5.5])

result = assert_interval_coverage(
    y_true, y_lower, y_upper,
    target_coverage=0.9,
    tolerance=0.05,
)
# coverage = 1.0 >= 0.9 - 0.05 => PASS
```

#### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `y_true` | `np.ndarray` | *(required)* | Ground truth values |
| `y_lower` | `np.ndarray` | *(required)* | Lower bounds of prediction intervals |
| `y_upper` | `np.ndarray` | *(required)* | Upper bounds of prediction intervals |
| `target_coverage` | `float` | `0.9` | Desired coverage probability (e.g. 0.9 for 90%) |
| `tolerance` | `float` | `0.05` | Allowed shortfall below `target_coverage` |
| `severity` | `Severity` | `CRITICAL` | Severity level for the assertion |

#### Returns

`TestResult` with details:

- `empirical_coverage` -- fraction of true values inside their interval
- `target_coverage` -- the requested coverage target
- `tolerance` -- the tolerance used
- `n_covered` -- number of samples where `y_lower <= y_true <= y_upper`
- `n_total` -- total number of samples
- `avg_width` -- mean interval width (`y_upper - y_lower`)
- `median_width` -- median interval width

The assertion passes when `empirical_coverage >= target_coverage - tolerance`.

---

## assert_prediction_set_size

Check that prediction sets are informatively sized. Prediction sets that are too large provide no useful information; empty sets indicate a calibration failure.

For **classification**, each prediction set is a list or set of predicted class labels and the size is the cardinality. For **regression**, each entry is a float representing the interval width.

The assertion checks two conditions:

1. Average set size <= `max_avg_size`
2. Fraction of empty sets <= `max_empty_frac`

### Classification example

```python
from mltk.model.conformal import assert_prediction_set_size

# Each set contains the classes the model considers plausible
prediction_sets = [
    {"cat", "dog"},      # size 2
    {"cat"},             # size 1
    {"dog", "bird"},     # size 2
]

result = assert_prediction_set_size(
    prediction_sets,
    max_avg_size=3.0,
)
# avg_size = 1.6667 <= 3.0, empty_frac = 0.0 <= 0.1 => PASS
```

### Regression example

```python
import numpy as np
from mltk.model.conformal import assert_prediction_set_size

# Interval widths from a conformal regression model
widths = np.array([0.8, 1.2, 0.9, 1.1, 1.0])

result = assert_prediction_set_size(
    widths,
    max_avg_size=2.0,
    max_empty_frac=0.05,
)
# avg_size = 1.0 <= 2.0, empty_frac = 0.0 <= 0.05 => PASS
```

#### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `prediction_sets` | `list[list] \| list[set] \| np.ndarray` | *(required)* | Prediction sets (lists/sets for classification) or ndarray of floats (interval widths for regression) |
| `max_avg_size` | `float` | *(required)* | Maximum allowed average set size / width |
| `max_empty_frac` | `float` | `0.1` | Maximum allowed fraction of empty sets |
| `severity` | `Severity` | `CRITICAL` | Severity level for the assertion |

#### Returns

`TestResult` with details:

- `avg_size` -- mean set cardinality (classification) or mean width (regression)
- `max_size` -- largest set size / width
- `min_size` -- smallest set size / width
- `empty_count` -- number of empty sets (cardinality 0 or width 0.0)
- `empty_frac` -- `empty_count / n_sets`
- `n_sets` -- total number of prediction sets

---

## Use Cases

- **Conformal prediction calibration checking** -- verify that a split-conformal or full-conformal predictor achieves the promised marginal coverage (e.g. 90%) on a held-out calibration set.
- **Bayesian credible interval validation** -- check that posterior credible intervals from MCMC or variational inference contain the true parameter at the stated credibility level.
- **Quantile regression coverage verification** -- confirm that quantile crossing is absent and that the interval between predicted quantiles (e.g. q=0.05 and q=0.95) covers the expected fraction of test points.
- **Bootstrap confidence interval testing** -- validate that bootstrap percentile or BCa intervals achieve nominal coverage on out-of-sample data.

---

## Advanced: Calibration and Fairness

### assert_conformal_calibration

#### The Calibration Promise

Conformal prediction makes a mathematical guarantee: if you ask for 90% coverage, then 90% of future predictions will contain the true value. This is not a hope or a heuristic -- it is a finite-sample validity result that holds under one key condition: exchangeability (which includes the common i.i.d. assumption as a special case).

But the guarantee has prerequisites:

1. **Exchangeability** -- the calibration and test data must be exchangeable (i.i.d. is sufficient but not necessary). If the data distribution shifts between calibration and deployment, the guarantee breaks.
2. **Sufficient calibration set size** -- conformal coverage holds in expectation for any calibration set size, but with small calibration sets (say, fewer than 100 samples), the *realized* coverage on a finite test set can deviate substantially from the nominal level due to randomness alone.
3. **Correct nonconformity score** -- the choice of score function does not affect the validity guarantee, but it affects the *efficiency* (width) of the intervals. A poorly chosen score produces valid but uselessly wide intervals.

When any of these assumptions are violated, the **actual** coverage deviates from the **promised** coverage. `assert_conformal_calibration` checks whether the promise is being kept.

#### How It Differs from assert_interval_coverage

These two assertions look similar but serve different purposes:

| | `assert_interval_coverage` | `assert_conformal_calibration` |
|---|---|---|
| **Question** | "Is coverage good enough?" | "Is coverage what was promised?" |
| **Check direction** | One-sided (fails only if coverage is too *low*) | Two-sided (fails if coverage is too *high* or too *low*) |
| **Tolerance meaning** | Allowed shortfall below target | Allowed deviation in *either* direction from nominal |
| **Over-coverage** | Acceptable (even desirable) | A problem -- indicates intervals are wider than necessary, wasting prediction precision |

Why is over-coverage a problem? If you request 90% coverage and get 99%, your intervals are far wider than they need to be. Every extra percent of coverage comes at the cost of wider intervals, which reduces the practical usefulness of the predictions. A well-calibrated conformal predictor should hit close to the nominal level -- not wildly above it.

```python
import numpy as np
from mltk.model.conformal import assert_conformal_calibration

# Well-calibrated model: asked for 90%, got 91%
y_true = np.random.randn(500)
noise = np.random.rand(500) * 0.5
y_lower = y_true - 1.65 - noise
y_upper = y_true + 1.65 + noise

result = assert_conformal_calibration(
    y_true, y_lower, y_upper,
    nominal_coverage=0.9,
    tolerance=0.03,
)
# empirical_coverage ≈ 0.91, deviation ≈ +0.01
# |0.01| <= 0.03 => PASS (calibrated)


# Miscalibrated model: asked for 90%, got 78%
y_lower_bad = y_true - 1.0
y_upper_bad = y_true + 1.0

result = assert_conformal_calibration(
    y_true, y_lower_bad, y_upper_bad,
    nominal_coverage=0.9,
    tolerance=0.03,
)
# empirical_coverage ≈ 0.78, deviation ≈ -0.12
# |0.12| > 0.03 => FAIL (under-covering)


# Over-calibrated model: asked for 90%, got 99%
y_lower_wide = y_true - 5.0
y_upper_wide = y_true + 5.0

result = assert_conformal_calibration(
    y_true, y_lower_wide, y_upper_wide,
    nominal_coverage=0.9,
    tolerance=0.03,
)
# empirical_coverage ≈ 0.99, deviation ≈ +0.09
# |0.09| > 0.03 => FAIL (over-covering -- intervals are wastefully wide)
```

#### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `y_true` | `np.ndarray` | *(required)* | Ground truth values |
| `y_lower` | `np.ndarray` | *(required)* | Lower bounds of prediction intervals |
| `y_upper` | `np.ndarray` | *(required)* | Upper bounds of prediction intervals |
| `nominal_coverage` | `float` | `0.9` | The coverage level that was promised (e.g. 0.9 for 90%) |
| `tolerance` | `float` | `0.03` | Maximum allowed absolute deviation from `nominal_coverage` in either direction |
| `severity` | `Severity` | `CRITICAL` | Severity level for the assertion |

#### Returns

`TestResult` with details:

- `empirical_coverage` -- fraction of true values inside their interval
- `nominal_coverage` -- the promised coverage level
- `deviation` -- `empirical_coverage - nominal_coverage` (positive = over-coverage, negative = under-coverage)
- `abs_deviation` -- absolute value of `deviation`
- `direction` -- `"over"` if empirical > nominal, `"under"` if empirical < nominal, `"exact"` if equal
- `tolerance` -- the tolerance used
- `n_covered` -- number of samples where `y_lower <= y_true <= y_upper`
- `n_total` -- total number of samples

The assertion passes when `abs(empirical_coverage - nominal_coverage) <= tolerance`.

#### When Calibration Fails

If `assert_conformal_calibration` fails, investigate these common causes:

1. **Calibration set too small (< 100 samples)** -- with a small calibration set, the conformal quantile is estimated imprecisely. The theoretical coverage guarantee holds in expectation, but finite-sample variance is high. Solution: use a larger calibration set, or accept a wider tolerance.

2. **Non-exchangeable data (time series, spatial data)** -- if the calibration data and the test data are drawn from different distributions (e.g., calibrating on January data and testing on July data), the exchangeability assumption is violated. Solution: use time-series-aware conformal methods (e.g., ACI, EnbPI) or recalibrate periodically.

3. **Label noise in the calibration set** -- noisy labels inflate the nonconformity scores during calibration, producing intervals that are too wide for clean test data (over-coverage) or too narrow for equally noisy test data (under-coverage depending on noise asymmetry). Solution: clean the calibration labels, or use robust nonconformity scores.

4. **Wrong nonconformity score for the data type** -- using absolute residuals for heavy-tailed data, or symmetric scores for asymmetric distributions. This does not break the validity guarantee in theory, but in finite samples it can cause miscalibration. Solution: use normalized residuals or quantile-based scores matched to the data distribution.

---

### assert_conditional_coverage

#### The Fairness-Coverage Gap

Marginal coverage -- the kind checked by `assert_interval_coverage` -- averages over *all* data points. A model can achieve 95% coverage on the majority group and 60% on a minority group, averaging to 90% overall. The aggregate number looks fine; the minority group is badly served.

This is the conformal prediction analog of the model fairness problem. Your intervals are discriminatory even if the overall coverage is correct. The 90% guarantee is meaningless to the subgroup that only gets 60%.

The real-world impact is concrete. Consider a medical diagnosis system with prediction intervals for blood test results. If it achieves 95% coverage for common conditions but only 50% for rare conditions, patients with rare conditions receive unreliable predictions -- and the overall 90% coverage statistic hides this entirely.

#### Mondrian Conformal Prediction

The formal framework for per-group coverage guarantees is called **Mondrian conformal prediction** (named after Piet Mondrian's grid paintings, where each rectangle gets its own treatment). The idea is simple: instead of computing one conformal quantile for all data, compute a *separate* quantile for each group. Each group then gets its own coverage guarantee, independent of the others.

`assert_conditional_coverage` does not *implement* Mondrian conformal prediction -- it *validates* whether your model (however it was built) achieves adequate coverage within each group. You can use it to test:

- A standard conformal predictor (to see if marginal coverage hides per-group failures)
- A Mondrian conformal predictor (to verify it delivers on its per-group promises)
- Any interval-producing method (Bayesian, bootstrap, quantile regression) applied to grouped data

```python
import numpy as np
from mltk.model.conformal import assert_conditional_coverage

# Simulate a model that covers Group A well but Group B poorly
np.random.seed(42)

y_true = np.concatenate([
    np.random.randn(400),       # Group A: 400 samples
    np.random.randn(100) + 3.0, # Group B: 100 samples, shifted distribution
])

# Intervals tuned for Group A's distribution
y_lower = y_true - 1.8
y_upper = y_true + 1.8

# Group B has higher variance that the model doesn't account for
y_lower[400:] = y_true[400:] - 0.8  # too narrow for Group B
y_upper[400:] = y_true[400:] + 0.8

groups = np.array(["A"] * 400 + ["B"] * 100)

result = assert_conditional_coverage(
    y_true, y_lower, y_upper,
    groups=groups,
    nominal_coverage=0.9,
    min_group_coverage=0.85,
    min_group_size=20,
)
# Group A: ~95% coverage => PASS
# Group B: ~60% coverage => FAIL (below min_group_coverage of 0.85)
# Overall: ~88% coverage looks acceptable, but Group B is badly served
```

#### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `y_true` | `np.ndarray` | *(required)* | Ground truth values |
| `y_lower` | `np.ndarray` | *(required)* | Lower bounds of prediction intervals |
| `y_upper` | `np.ndarray` | *(required)* | Upper bounds of prediction intervals |
| `groups` | `np.ndarray` | *(required)* | Group labels for each sample (same length as `y_true`). Can be strings, ints, or any hashable type. |
| `nominal_coverage` | `float` | `0.9` | The expected coverage level |
| `min_group_coverage` | `float` | `0.85` | Minimum acceptable coverage for any single group. Typically set slightly below `nominal_coverage` to allow for finite-sample variance within groups. |
| `min_group_size` | `int` | `10` | Groups with fewer samples than this are skipped (reported in `groups_skipped` but not used for pass/fail). See "Choosing min_group_size" below. |
| `severity` | `Severity` | `CRITICAL` | Severity level for the assertion |

#### Returns

`TestResult` with details:

- `per_group` -- dict mapping each group label to its coverage, count, and pass/fail status
- `worst_group` -- the group label with the lowest coverage (among groups with at least `min_group_size` samples)
- `worst_coverage` -- the coverage of the worst group
- `groups_below_threshold` -- list of group labels with coverage below `min_group_coverage`
- `groups_skipped` -- list of group labels skipped due to having fewer than `min_group_size` samples
- `overall_coverage` -- marginal coverage across all samples (for comparison)
- `n_groups` -- total number of distinct groups
- `n_groups_evaluated` -- number of groups with at least `min_group_size` samples

The assertion passes when *every* evaluated group has coverage >= `min_group_coverage`.

#### Choosing min_group_size

Groups with very few samples cannot give reliable coverage estimates. If a group has 3 samples and all 3 are covered, the empirical coverage is 100% -- but this tells you almost nothing about the true coverage rate. With 3 samples, even a model with 50% true coverage has a 12.5% chance of covering all 3 by luck alone.

The `min_group_size` parameter prevents false confidence by excluding groups that are too small to evaluate meaningfully. Groups below this threshold are still *reported* (in `groups_skipped`) so you know they exist, but they do not affect the pass/fail decision.

Rules of thumb:

- **min_group_size=10** (default) -- minimum for any rough estimate. At n=10, a 90% coverage model will show 100% empirical coverage about 35% of the time, so take results with a grain of salt.
- **min_group_size=30** -- reasonable for moderate confidence. The standard error of a coverage estimate at p=0.9, n=30 is about 5.5%.
- **min_group_size=100** -- good for rigorous validation. Standard error drops to about 3%.
- **min_group_size=500+** -- for high-stakes applications (medical, financial) where per-group guarantees matter.

If a critical group has too few samples, the solution is not to lower `min_group_size` -- it is to collect more data for that group.

---

## Recommended Validation Pipeline (Updated)

The four assertions in this module form a natural validation pipeline, from basic to rigorous:

```python
import numpy as np
from mltk.model.conformal import (
    assert_interval_coverage,
    assert_conformal_calibration,
    assert_conditional_coverage,
    assert_prediction_set_size,
)

# Step 1: Basic coverage check
# "Do we cover at least 90% of true values?"
assert_interval_coverage(y_true, y_lower, y_upper, target_coverage=0.9)

# Step 2: Is coverage what we promised? (not too high, not too low)
# "Are we calibrated, or are our intervals wastefully wide / dangerously narrow?"
assert_conformal_calibration(y_true, y_lower, y_upper, nominal_coverage=0.9, tolerance=0.02)

# Step 3: Does coverage hold for ALL subgroups?
# "Is any demographic or data slice getting worse coverage than the aggregate suggests?"
assert_conditional_coverage(
    y_true, y_lower, y_upper,
    groups=demographics,
    min_group_coverage=0.85,
)

# Step 4: Are intervals informatively sized? (not too wide)
# "Even if coverage is correct, are the intervals narrow enough to be useful?"
widths = y_upper - y_lower
assert_prediction_set_size(widths, max_avg_size=2.0)
```

Each step catches a different failure mode:

- **Step 1** catches under-coverage (the model is unreliable).
- **Step 2** catches miscalibration in either direction (the model is wasteful or the guarantee is broken).
- **Step 3** catches fairness gaps (the model is reliable *on average* but unreliable for specific groups).
- **Step 4** catches uninformative predictions (the model is reliable but useless -- covering everything by making intervals enormous).

A model that passes all four steps provides intervals that are calibrated, fair across groups, and practically useful.
