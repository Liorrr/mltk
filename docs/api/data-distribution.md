# Data Distribution

Distribution tests catch subtle data issues that schema tests miss: a column can have the right name and type but contain wildly wrong values. These assertions ensure numeric ranges, uniqueness, and outlier bounds before data enters the training pipeline.

**Module:** `mltk.data.distribution`

---

## assert_range

```python
@timed_assertion
def assert_range(
    series: pd.Series,
    min_val: float,
    max_val: float,
) -> TestResult
```

### What it tests

Validates that every value in a pandas Series falls within the inclusive range `[min_val, max_val]`.

### Why it matters for ML

Out-of-range values are a top cause of garbage model outputs. A probability column with values above 1.0 breaks calibration. A negative age is physically impossible. A temperature of 999.9 is a sensor glitch. Models trained on out-of-range data learn incorrect relationships, and the failure mode is silent -- the model still produces predictions, they are just wrong.

### When to use it

- **Feature validation** -- after feature engineering, verify that derived features have valid ranges
- **Model output validation** -- ensure predicted probabilities are in `[0, 1]` and regression outputs are in expected bounds
- **Data ingestion** -- catch sensor malfunctions, unit mismatches, or data entry errors at the boundary
- **Post-normalization** -- verify that scaling (min-max, z-score) produced values in the expected range

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `series` | `pd.Series` | *(required)* | The numeric Series to validate. |
| `min_val` | `float` | *(required)* | Minimum allowed value (inclusive). |
| `max_val` | `float` | *(required)* | Maximum allowed value (inclusive). |

### Returns

`TestResult` with:

- `name`: `"data.range[{series.name}]"` if the Series has a name, otherwise `"data.range"`
- `details.actual_min`: the actual minimum value in the Series
- `details.actual_max`: the actual maximum value in the Series
- `details.below_count`: number of values below `min_val`
- `details.above_count`: number of values above `max_val`

### Example

```python
import pandas as pd
import pytest
from mltk.data import assert_range
from mltk.core.assertion import MltkAssertionError

def test_probability_scores_valid():
    """Model output probabilities must be in [0, 1]."""
    scores = pd.Series([0.1, 0.5, 0.9, 0.3], name="probability")
    result = assert_range(scores, min_val=0.0, max_val=1.0)
    assert result.passed
    assert result.details["actual_min"] == 0.1
    assert result.details["actual_max"] == 0.9

def test_catches_negative_age():
    """Age cannot be negative -- indicates data corruption."""
    ages = pd.Series([-5, 10, 25, 50], name="age")
    with pytest.raises(MltkAssertionError) as exc:
        assert_range(ages, min_val=0, max_val=150)
    assert "outside" in str(exc.value)
```

### Edge Cases

- **NaN values** are not counted as violations by the comparison operators (`<` and `>`), but they still exist in the Series. Consider running `assert_no_nulls` first if NaN values should not be present.
- **Boundary values** are inclusive. A value exactly equal to `min_val` or `max_val` passes.
- **Empty Series** will not raise but `actual_min` and `actual_max` will be NaN.

---

## assert_unique

```python
@timed_assertion
def assert_unique(
    df: pd.DataFrame,
    columns: list[str],
) -> TestResult
```

### What it tests

Validates that there are no duplicate rows based on the specified column(s). For a single column, checks that all values are unique. For multiple columns, checks that the combination (composite key) is unique.

### Why it matters for ML

Duplicate records directly bias model training. If a data pipeline runs twice and appends duplicate rows, the model sees those examples twice as often, learning to overweight them. In recommendation systems, duplicates inflate engagement metrics. In classification, they skew class distributions. Deduplication checks are essential whenever data flows through multiple pipeline stages.

### When to use it

- **Primary key validation** -- verify that every record has a unique identifier
- **Post-merge validation** -- after joining datasets, check that the join did not create unintended duplicates
- **Time-series data** -- ensure one observation per (timestamp, entity) combination
- **Training data assembly** -- verify that sampling or concatenation did not introduce duplicates

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `df` | `pd.DataFrame` | *(required)* | The DataFrame to validate. |
| `columns` | `list[str]` | *(required)* | Column(s) to check for uniqueness. Multiple columns are treated as a composite key. |

### Returns

`TestResult` with:

- `name`: `"data.unique"`
- `details.columns`: the columns checked
- `details.duplicate_count`: total number of rows involved in duplicates (counts all copies, not just extras)
- `details.total_rows`: total row count

### Example

```python
import pandas as pd
import pytest
from mltk.data import assert_unique
from mltk.core.assertion import MltkAssertionError

def test_user_ids_are_unique():
    df = pd.DataFrame({"user_id": [1, 2, 3, 4]})
    result = assert_unique(df, columns=["user_id"])
    assert result.passed

def test_composite_key_unique():
    """Daily sales: one row per store per day."""
    df = pd.DataFrame({
        "date": ["2026-01-01", "2026-01-01", "2026-01-02"],
        "store_id": [1, 2, 1],
    })
    result = assert_unique(df, columns=["date", "store_id"])
    assert result.passed

def test_detects_pipeline_duplication():
    df = pd.DataFrame({"user_id": [1, 2, 2, 3]})
    with pytest.raises(MltkAssertionError):
        assert_unique(df, columns=["user_id"])
```

### Edge Cases

- **`duplicate_count` counts all copies**, not just extras. If user_id `2` appears 3 times, all 3 rows are counted as duplicates (using `keep=False`). This matches pandas `duplicated(keep=False)` behavior.
- **Null values**: Two NaN values in the same column are considered duplicates by pandas.
- **Column order** in the `columns` list does not affect the result. `["date", "store"]` and `["store", "date"]` produce the same uniqueness check.

---

## assert_no_outliers

```python
@timed_assertion
def assert_no_outliers(
    series: pd.Series,
    method: str = "iqr",
    threshold: float = 1.5,
) -> TestResult
```

### What it tests

Detects statistical outliers in a numeric Series using the IQR (Interquartile Range) method. A value is an outlier if it falls below `Q1 - threshold * IQR` or above `Q3 + threshold * IQR`.

### Why it matters for ML

Outliers have an outsized impact on ML models. A single extreme value can shift the mean dramatically, distort standardization/normalization, and cause gradient explosion during training. While some outliers are real (a CEO's salary in a salary dataset), many are data errors (unit mismatches, sensor glitches, data entry mistakes). Detecting outliers before training lets you decide whether to keep, clip, or remove them -- rather than having the model silently learn from corrupted data.

### When to use it

- **Pre-training validation** -- flag extreme values that could distort model weights
- **Feature engineering output** -- verify that transformations did not introduce extreme values
- **Monitoring** -- detect sudden distribution shifts in production data
- **Data cleaning verification** -- confirm that outlier removal or clipping was applied correctly

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `series` | `pd.Series` | *(required)* | The numeric Series to validate. |
| `method` | `str` | `"iqr"` | Outlier detection method. Currently only `"iqr"` is supported. |
| `threshold` | `float` | `1.5` | IQR multiplier for outlier bounds. `1.5` is the standard Tukey fence; `3.0` flags only extreme outliers. |

### Returns

`TestResult` with severity `WARNING` (not `CRITICAL`):

- `name`: `"data.no_outliers[{series.name}]"` if the Series has a name, otherwise `"data.no_outliers"`
- `details.method`: the detection method used
- `details.threshold`: the IQR multiplier used
- `details.q1`: first quartile
- `details.q3`: third quartile
- `details.iqr`: interquartile range (Q3 - Q1)
- `details.lower_bound`: `Q1 - threshold * IQR`
- `details.upper_bound`: `Q3 + threshold * IQR`
- `details.outlier_count`: number of outliers detected

!!! note "WARNING severity"
    Unlike most assertions, `assert_no_outliers` uses `Severity.WARNING` by default. This means it does **not** raise an exception on failure -- it returns a `TestResult` with `passed=False`. This is intentional: outliers are often worth investigating but should not block a pipeline automatically.

### Example

```python
import numpy as np
import pandas as pd
from mltk.data import assert_no_outliers

def test_salary_distribution_clean():
    """Normal salary data -- no extreme outliers."""
    rng = np.random.default_rng(42)
    salaries = pd.Series(rng.normal(50000, 10000, 100), name="salary")

    result = assert_no_outliers(salaries, threshold=3.0)
    assert result.passed

def test_detects_data_entry_error():
    """One salary is 10M -- likely a data entry error."""
    salaries = pd.Series(
        [40000, 45000, 50000, 55000, 60000, 10_000_000],
        name="salary",
    )
    result = assert_no_outliers(salaries, threshold=1.5)

    # WARNING severity: no exception raised, but result shows failure
    assert result.passed is False
    assert result.details["outlier_count"] > 0
    print(f"Bounds: [{result.details['lower_bound']:.0f}, {result.details['upper_bound']:.0f}]")
```

### Edge Cases

- **Unsupported method** raises `MltkAssertionError` (CRITICAL). Only `"iqr"` is currently supported. Passing any other string causes an immediate failure with a clear error message.
- **Constant data** (all values identical) produces `IQR = 0`, so `lower_bound == upper_bound == the value`. No outliers will be detected.
- **NaN values** are dropped before computing quartiles (via `series.dropna()`), but the outlier count check includes the full series. Run `assert_no_nulls` first if NaN handling matters.
- **Threshold tuning**: `1.5` is the standard Tukey fence (catches moderate outliers). `3.0` catches only extreme outliers. For ML feature validation, `3.0` is often more practical to avoid excessive false positives.

---

## Related Tests

Tests are located in `tests/test_data/test_distribution.py`.

### TestAssertRange

- **`test_all_in_range`** -- Verifies that a Series with all values within `[0, 1]` produces a passing result. The baseline case for probability score validation.
- **`test_below_minimum`** -- Verifies that a negative value in an age column is detected and raises `MltkAssertionError`. Simulates physically impossible data that indicates corruption.
- **`test_above_maximum`** -- Verifies that a single extreme value (999.9 in a temperature column) is detected with the correct violation count in the error message. Simulates a sensor malfunction.

### TestAssertUnique

- **`test_all_unique`** -- Verifies that a DataFrame with distinct user IDs produces a passing result. The baseline case for primary key validation.
- **`test_duplicates_found`** -- Verifies that duplicate user IDs are detected and raise `MltkAssertionError`. Simulates a data pipeline that ran twice and inserted duplicate records.
- **`test_composite_key_unique`** -- Verifies that composite key uniqueness (date + store) works correctly, where individual columns may repeat but the combination is unique. Validates multi-column deduplication.

### TestAssertNoOutliers

- **`test_no_outliers`** -- Verifies that normally distributed salary data passes outlier detection with `threshold=3.0`. Uses a fixed random seed for reproducibility.
- **`test_outliers_detected`** -- Verifies that a single extreme value (10 million in a salary column) is flagged as an outlier. Confirms that the result has `passed=False` and `outlier_count > 0`, and that no exception is raised because the severity is `WARNING`.
- **`test_unsupported_method`** -- Verifies that passing an invalid method name (e.g., `"invalid"`) raises `MltkAssertionError` with `CRITICAL` severity, ensuring clear failure on misconfiguration.
