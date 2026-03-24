# Data Freshness

Stale data is a silent killer for ML models. A model trained on data from six months ago may perform well on the test set but fail in production because the real world has changed (concept drift). Freshness assertions ensure your training data and feature pipelines are up-to-date. Row count validation catches data pipeline failures where a table that should have millions of rows suddenly has zero or unexpectedly few.

**Module:** `mltk.data.freshness`

---

## assert_freshness

```python
@timed_assertion
def assert_freshness(
    df: pd.DataFrame,
    date_column: str,
    max_age_days: int,
    reference_date: datetime | None = None,
) -> TestResult
```

### What it tests

Validates that the most recent date in a specified column is within `max_age_days` of a reference date (defaults to now). The assertion finds the maximum datetime value in the column and computes the age in days.

### Why it matters for ML

ML models are only as good as the data they were trained on. When a data pipeline silently fails -- an ETL job stops running, an API rate-limits your ingestion, a cron job gets disabled -- the data looks structurally perfect (right schema, right types, right distributions) but it is dangerously outdated. A fraud detection model trained on 3-week-old data will miss new fraud patterns. A recommendation model with stale user behavior data will suggest irrelevant items. Freshness checks are the only way to catch these invisible failures.

### When to use it

- **Training data validation** -- before starting a training run, verify the dataset was refreshed recently
- **Feature store monitoring** -- ensure feature tables are being updated on schedule
- **Daily pipeline checks** -- CI/CD job that runs every morning to verify last night's ETL completed
- **Backfill validation** -- with `reference_date`, verify that historical data was fresh at the time of a past training run

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `df` | `pd.DataFrame` | *(required)* | The DataFrame to validate. |
| `date_column` | `str` | *(required)* | Name of the datetime column to check. Values are parsed with `pd.to_datetime`. |
| `max_age_days` | `int` | *(required)* | Maximum allowed age in days. The assertion fails if the most recent date is older than this. |
| `reference_date` | `datetime \| None` | `None` | Reference point for age calculation. If `None`, uses `datetime.now()`. Pass a specific datetime for reproducible tests or backfill validation. |

### Returns

`TestResult` with:

- `name`: `"data.freshness"`
- `details.age_days`: actual age of the most recent record in days
- `details.max_age_days`: the configured threshold
- `details.most_recent`: string representation of the most recent date
- `details.reference_date`: string representation of the reference date used

### Example

```python
from datetime import datetime, timedelta
import pandas as pd
import pytest
from mltk.data import assert_freshness
from mltk.core.assertion import MltkAssertionError

def test_training_data_is_fresh():
    """Verify the ETL pipeline ran within the last 24 hours."""
    now = datetime.now()
    df = pd.DataFrame({
        "created_at": [
            now - timedelta(hours=6),
            now - timedelta(hours=3),
            now - timedelta(hours=1),
        ],
        "value": [1.0, 2.0, 3.0],
    })
    result = assert_freshness(df, date_column="created_at", max_age_days=1)
    assert result.passed

def test_catches_stale_pipeline():
    """ETL stopped 2 weeks ago -- data is dangerously outdated."""
    old = datetime.now() - timedelta(days=30)
    df = pd.DataFrame({
        "created_at": [old, old - timedelta(days=5)],
        "value": [1.0, 2.0],
    })
    with pytest.raises(MltkAssertionError) as exc:
        assert_freshness(df, date_column="created_at", max_age_days=7)
    assert "exceeds limit" in str(exc.value)

def test_backfill_was_fresh_at_training_time():
    """Historical validation: data was fresh when we trained on Jan 15."""
    ref = datetime(2026, 1, 15)
    df = pd.DataFrame({
        "created_at": [datetime(2026, 1, 14), datetime(2026, 1, 13)],
    })
    result = assert_freshness(
        df, date_column="created_at", max_age_days=7, reference_date=ref
    )
    assert result.passed
```

### Edge Cases

- **Missing date column**: If `date_column` does not exist in the DataFrame, the assertion fails immediately with a clear error message.
- **Unparseable dates**: Values that cannot be converted to datetime via `pd.to_datetime` are coerced to `NaT`. If all values become `NaT`, the assertion fails with "No valid dates" in the message.
- **Timezone-aware dates**: If the date column contains timezone-aware timestamps but `reference_date` is naive (or `None`), the timezone is stripped from the most recent date before comparison.
- **Age is measured in whole days**: The age is computed as `(reference_date - most_recent).days`, which truncates to whole days. Data that is 23 hours old has an age of 0 days.

---

## assert_row_count

```python
@timed_assertion
def assert_row_count(
    df: pd.DataFrame,
    min_rows: int | None = None,
    max_rows: int | None = None,
) -> TestResult
```

### What it tests

Validates that the DataFrame row count falls within the specified bounds. Either or both bounds can be `None` to skip that direction of the check.

### Why it matters for ML

Row count anomalies signal data pipeline failures. An empty dataset means the extract failed entirely. A dataset with 100 rows instead of the expected 100,000 means a partial load (wrong date filter, permission issue, timeout). A dataset with 10x more rows than expected means data was appended multiple times (pipeline re-run without idempotency). All of these produce models that look fine during training but fail in production.

### When to use it

- **Pipeline output validation** -- verify that the ETL produced a reasonable number of rows
- **Minimum training data** -- ensure enough examples for the model to learn from (e.g., at least 1,000 rows per class)
- **Data growth monitoring** -- set upper bounds to catch unexpected explosions (pipeline bugs, duplication)
- **Split validation** -- verify that train/test/validation splits have expected proportions

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `df` | `pd.DataFrame` | *(required)* | The DataFrame to validate. |
| `min_rows` | `int \| None` | `None` | Minimum expected row count (inclusive). `None` means no lower bound. |
| `max_rows` | `int \| None` | `None` | Maximum expected row count (inclusive). `None` means no upper bound. |

### Returns

`TestResult` with:

- `name`: `"data.row_count"`
- `details.row_count`: actual number of rows
- `details.min_rows`: the configured minimum (or `None`)
- `details.max_rows`: the configured maximum (or `None`)

### Example

```python
import pandas as pd
import pytest
from mltk.data import assert_row_count
from mltk.core.assertion import MltkAssertionError

def test_dataset_has_enough_rows():
    """Training needs at least 1000 rows."""
    df = pd.DataFrame({"x": range(5000)})
    result = assert_row_count(df, min_rows=1000)
    assert result.passed
    assert result.details["row_count"] == 5000

def test_catches_empty_extract():
    """Pipeline returned 5 rows instead of expected 100+."""
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
    with pytest.raises(MltkAssertionError) as exc:
        assert_row_count(df, min_rows=100)
    assert "below minimum" in str(exc.value)

def test_catches_data_duplication():
    """More rows than expected -- pipeline may have run twice."""
    df = pd.DataFrame({"a": range(1000)})
    with pytest.raises(MltkAssertionError) as exc:
        assert_row_count(df, max_rows=500)
    assert "exceeds maximum" in str(exc.value)

def test_informational_count():
    """No bounds -- just report the count."""
    df = pd.DataFrame({"x": range(42)})
    result = assert_row_count(df)
    assert result.passed
    assert result.details["row_count"] == 42
```

### Edge Cases

- **No bounds specified**: When both `min_rows` and `max_rows` are `None`, the assertion always passes. It acts as an informational check, recording the row count in the result details.
- **Zero rows**: An empty DataFrame has `row_count=0`. This passes if `min_rows` is `None` or `0`, and fails if `min_rows > 0`.
- **Both bounds equal**: Setting `min_rows=1000, max_rows=1000` requires exactly 1,000 rows. This is useful for fixed-size datasets like image classification benchmarks.

---

## Related Tests

Tests are located in `tests/test_data/test_freshness.py`.

### TestAssertFreshness

- **`test_fresh_data`** -- Verifies that a DataFrame with recent dates (within the last few hours) passes a 1-day freshness check. The happy path for a pipeline that ran recently.
- **`test_stale_data`** -- Verifies that a DataFrame where the newest record is 30 days old fails a 7-day freshness check, and that the error message contains "exceeds limit". Simulates an ETL pipeline that silently stopped running.
- **`test_missing_date_column`** -- Verifies that specifying a date column that does not exist in the DataFrame raises `MltkAssertionError`. Catches cases where a column was renamed upstream but the test configuration was not updated.
- **`test_custom_reference_date`** -- Verifies that passing a `reference_date` overrides `datetime.now()` for the age calculation. Enables reproducible tests and historical validation of backfilled data.

### TestAssertRowCount

- **`test_within_bounds`** -- Verifies that a DataFrame with 100 rows passes when bounds are set to `min_rows=50, max_rows=10000`. The normal case for pipeline output validation.
- **`test_below_minimum`** -- Verifies that a DataFrame with only 5 rows fails when `min_rows=100`, and the error message says "below minimum". Simulates a partial data load or failed extraction.
- **`test_above_maximum`** -- Verifies that a DataFrame with 1,000 rows fails when `max_rows=500`, and the error message says "exceeds maximum". Simulates data duplication from a non-idempotent pipeline.
- **`test_no_bounds`** -- Verifies that calling `assert_row_count` with no bounds always passes and correctly records the actual row count in `details["row_count"]`. Supports informational-only usage.
