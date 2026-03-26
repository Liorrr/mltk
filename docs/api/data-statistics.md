# Column Statistics

Column-level statistical assertions catch distribution shape problems that range checks miss. A column can have all values in [0, 100] but still have a wrong mean, skewed median, or a stdev so small the feature carries no signal. These assertions are especially useful after feature engineering steps where transformations can silently shift the distribution.

**Module:** `mltk.data.statistics`

**ML Lifecycle Stage:** Data Validation / Feature Engineering QA

---

## assert_column_mean

Assert column mean is within specified bounds. At least one bound (min or max) is required.

```python
from mltk.data import assert_column_mean

assert_column_mean(df, "age", min_val=18.0, max_val=65.0)
assert_column_mean(df, "score", min_val=0.4)  # at least 0.4 average
```

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `df` | `pd.DataFrame` | *(required)* | DataFrame to validate |
| `column` | `str` | *(required)* | Name of the column to check |
| `min_val` | `float \| None` | `None` | Lower bound for the mean (inclusive) |
| `max_val` | `float \| None` | `None` | Upper bound for the mean (inclusive) |

### Returns

`TestResult` with details:
- `column` -- column name checked
- `actual_mean` -- computed mean value
- `min_val` -- lower bound (or None)
- `max_val` -- upper bound (or None)

### Raises

`ValueError` if neither `min_val` nor `max_val` is provided.

---

## assert_column_median

Assert column median is within specified bounds. The median is more robust to outliers than the mean. Use this when you care about the typical value rather than the average.

```python
from mltk.data import assert_column_median

assert_column_median(df, "income", min_val=30000.0, max_val=80000.0)
```

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `df` | `pd.DataFrame` | *(required)* | DataFrame to validate |
| `column` | `str` | *(required)* | Name of the column to check |
| `min_val` | `float \| None` | `None` | Lower bound for the median (inclusive) |
| `max_val` | `float \| None` | `None` | Upper bound for the median (inclusive) |

### Returns

`TestResult` with details:
- `column` -- column name checked
- `actual_median` -- computed median value
- `min_val` -- lower bound (or None)
- `max_val` -- upper bound (or None)

### Raises

`ValueError` if neither `min_val` nor `max_val` is provided.

---

## assert_column_stdev

Assert column standard deviation is within bounds. Catches two failure modes: stdev too low (feature is nearly constant, carries no signal) and stdev too high (distribution unusually wide, may contain outliers or mixed populations).

```python
from mltk.data import assert_column_stdev

assert_column_stdev(df, "response_time_ms", min_val=1.0, max_val=500.0)
assert_column_stdev(df, "normalized_score", max_val=0.5)
```

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `df` | `pd.DataFrame` | *(required)* | DataFrame to validate |
| `column` | `str` | *(required)* | Name of the column to check |
| `min_val` | `float \| None` | `None` | Minimum allowed stdev |
| `max_val` | `float \| None` | `None` | Maximum allowed stdev |

### Returns

`TestResult` with details:
- `column` -- column name checked
- `actual_stdev` -- computed standard deviation
- `min_val` -- lower bound (or None)
- `max_val` -- upper bound (or None)

### Raises

`ValueError` if neither `min_val` nor `max_val` is provided.

---

## assert_quantiles

Assert column quantile values are within specified bounds. The most expressive statistical check: specify the expected shape of the distribution at multiple points simultaneously. A distribution that passes mean and stdev checks can still be wrong at the tails.

```python
from mltk.data import assert_quantiles

assert_quantiles(df, "age", {
    0.25: (20, 35),
    0.50: (30, 50),
    0.75: (45, 65),
    0.95: (60, 80),
})
```

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `df` | `pd.DataFrame` | *(required)* | DataFrame to validate |
| `column` | `str` | *(required)* | Name of the column to check |
| `quantiles` | `dict[float, tuple[float, float]]` | *(required)* | Mapping of quantile level to (min_bound, max_bound). Each quantile value must fall within [min_bound, max_bound]. |

### Returns

`TestResult` with details:
- `column` -- column name checked
- `quantiles_checked` -- number of quantiles validated
- `failures` -- number of quantiles that failed bounds check
- `actual_values` -- dict mapping quantile labels (e.g., "Q0.25") to actual values

### Example

```python
import pytest
import pandas as pd
from mltk.data import assert_column_mean, assert_column_stdev, assert_quantiles

@pytest.mark.ml_data
def test_feature_distribution():
    """Feature statistics are within expected bounds after transformation."""
    df = pd.read_parquet("data/features.parquet")

    assert_column_mean(df, "score", min_val=0.3, max_val=0.7)
    assert_column_stdev(df, "score", min_val=0.05, max_val=0.3)
    assert_quantiles(df, "score", {
        0.05: (0.0, 0.2),
        0.50: (0.3, 0.7),
        0.95: (0.8, 1.0),
    })
```

### Edge Cases

- **Empty DataFrame**: `assert_quantiles` fails with a descriptive message on empty DataFrames.
- **NaN values**: NaN values are dropped before computing quantiles.
- **Single bound**: `assert_column_mean`, `assert_column_median`, and `assert_column_stdev` require at least one of `min_val` or `max_val`.

---
