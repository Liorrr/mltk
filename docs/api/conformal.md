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
