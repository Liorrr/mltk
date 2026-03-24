# Data Schema

Schema validation is the first line of defense in ML data quality. If your data does not have the right columns, types, or completeness, everything downstream -- feature engineering, training, inference -- will silently produce garbage.

**Module:** `mltk.data.schema`

---

## assert_schema

```python
@timed_assertion
def assert_schema(
    df: pd.DataFrame,
    expected: dict[str, str],
    allow_extra_columns: bool = True,
) -> TestResult
```

### What it tests

Validates that a DataFrame has the expected columns with the correct dtypes. Optionally rejects unexpected extra columns.

### Why it matters for ML

Schema drift is one of the most common causes of silent ML failures. An upstream ETL job changes a column name or type, and suddenly your feature engineering pipeline produces zeros, NaNs, or crashes -- but the training loop might still complete successfully with garbage data. Catching schema changes at the data boundary prevents these failures from propagating.

### When to use it

- **Data ingestion** -- validate raw data immediately after loading from CSV, Parquet, or a database
- **Feature engineering** -- verify the output DataFrame of your feature pipeline before it enters training
- **Inference** -- validate incoming request payloads match the model's expected input schema
- **CI/CD** -- run as part of data validation tests before training begins

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `df` | `pd.DataFrame` | *(required)* | The DataFrame to validate. |
| `expected` | `dict[str, str]` | *(required)* | Mapping of column names to expected dtype strings (e.g., `{"id": "int64", "score": "float64"}`). |
| `allow_extra_columns` | `bool` | `True` | If `False`, the assertion fails when the DataFrame has columns not listed in `expected`. |

### Returns

`TestResult` with:

- `name`: `"data.schema"`
- `details.missing_columns`: sorted list of columns in `expected` but not in `df`
- `details.dtype_mismatches`: dict of columns with wrong dtypes, each containing `expected` and `actual`
- `details.extra_columns`: sorted list of unexpected columns (only populated when `allow_extra_columns=False`)

### Example

```python
import pandas as pd
import pytest
from mltk.data import assert_schema
from mltk.core.assertion import MltkAssertionError

def test_training_data_schema():
    df = pd.DataFrame({
        "user_id": [1, 2, 3],
        "age": [25, 30, 35],
        "score": [0.8, 0.9, 0.7],
        "label": [1, 0, 1],
    })

    result = assert_schema(df, {
        "user_id": "int64",
        "age": "int64",
        "score": "float64",
        "label": "int64",
    })
    assert result.passed

def test_strict_schema_rejects_extra_columns():
    df = pd.DataFrame({
        "id": [1, 2],
        "name": ["a", "b"],
        "debug_notes": ["x", "y"],  # unexpected column
    })

    with pytest.raises(MltkAssertionError):
        assert_schema(df, {"id": "int64", "name": "object"}, allow_extra_columns=False)
```

### Edge Cases

- **Empty DataFrames** pass if the column names and dtypes match the expected schema. Row count is validated separately by `assert_row_count`.
- **Dtype strings must match exactly.** For example, `"int64"` and `"Int64"` (nullable integer) are different dtypes. Use `str(df[col].dtype)` to check the exact string.
- **Pandas category dtype** shows as `"category"`, not the underlying type. If your column is categorical, use `"category"` in the expected dict.

---

## assert_no_nulls

```python
@timed_assertion
def assert_no_nulls(
    df: pd.DataFrame,
    columns: list[str] | None = None,
) -> TestResult
```

### What it tests

Validates that the specified columns (or all columns) contain no null or NaN values.

### Why it matters for ML

Null values silently corrupt ML pipelines. A null label means the model trains on an unlabeled example (producing random gradients). A null feature may get imputed to zero, which could be a valid value, masking the missing data. Explicit null detection at the boundary ensures you know exactly what data is missing before it affects model quality.

### When to use it

- **Label validation** -- ensure every training example has a label
- **Required feature columns** -- verify that critical features have no gaps
- **Post-join validation** -- after SQL joins, null values indicate failed lookups (missing foreign keys)
- **Preprocessing output** -- verify that imputation or cleaning steps removed all nulls

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `df` | `pd.DataFrame` | *(required)* | The DataFrame to validate. |
| `columns` | `list[str] \| None` | `None` | Columns to check. If `None`, checks all columns in the DataFrame. |

### Returns

`TestResult` with:

- `name`: `"data.no_nulls"`
- `details.null_counts`: dict mapping column names to their null count (only columns with nulls are included)
- `details.columns_checked`: list of columns that were validated

### Example

```python
import pandas as pd
import pytest
from mltk.data import assert_no_nulls
from mltk.core.assertion import MltkAssertionError

def test_labels_are_complete():
    df = pd.DataFrame({
        "feature": [1.0, 2.0, 3.0],
        "label": [0, 1, 1],
    })
    result = assert_no_nulls(df, columns=["label"])
    assert result.passed

def test_detects_missing_labels():
    df = pd.DataFrame({
        "feature": [1.0, 2.0, 3.0],
        "label": [0, None, 1],
    })
    with pytest.raises(MltkAssertionError) as exc:
        assert_no_nulls(df, columns=["label"])
    assert "null" in str(exc.value).lower()
```

### Edge Cases

- **NaN vs None**: Both `None` and `float('nan')` are detected. Pandas treats both as null.
- **Empty strings** are NOT null. A column with `""` values will pass `assert_no_nulls`. Use a separate assertion if empty strings are invalid.
- **Checking a subset**: When `columns` is specified, nulls in other columns are ignored. This lets you validate only the columns that matter for your pipeline stage.

---

## assert_dtypes

```python
@timed_assertion
def assert_dtypes(
    df: pd.DataFrame,
    expected: dict[str, str],
) -> TestResult
```

### What it tests

Validates that specific columns have the exact expected dtypes. Unlike `assert_schema`, this does not require all DataFrame columns to appear in the expected dict -- it only checks the columns you specify.

### Why it matters for ML

Type mismatches are a common source of silent ML bugs. Loading a CSV can turn numeric columns into `object` dtype if any value contains a comma, space, or special character. A model receiving string features where it expects floats will either crash or silently produce meaningless embeddings. `assert_dtypes` catches these issues before they reach the model.

### When to use it

- **Post-CSV-load validation** -- CSV files are schema-less; dtypes depend on the data content
- **After type casting** -- verify that `astype()` or `pd.to_numeric()` calls produced the expected types
- **Selective validation** -- when you only care about certain columns' types (e.g., just the features, not metadata)

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `df` | `pd.DataFrame` | *(required)* | The DataFrame to validate. |
| `expected` | `dict[str, str]` | *(required)* | Mapping of column names to expected dtype strings. Only the listed columns are checked. |

### Returns

`TestResult` with:

- `name`: `"data.dtypes"`
- `details.mismatches`: dict of columns with wrong dtypes, each containing `expected` and `actual`
- `details.missing_columns`: list of columns in `expected` that do not exist in the DataFrame

### Example

```python
import pandas as pd
import pytest
from mltk.data import assert_dtypes
from mltk.core.assertion import MltkAssertionError

def test_numeric_features_are_numeric():
    df = pd.DataFrame({
        "name": ["alice", "bob"],
        "score": [0.95, 0.87],
        "count": [10, 20],
    })
    result = assert_dtypes(df, {"score": "float64", "count": "int64"})
    assert result.passed

def test_catches_string_in_numeric_column():
    df = pd.DataFrame({"score": ["high", "low", "medium"]})
    with pytest.raises(MltkAssertionError):
        assert_dtypes(df, {"score": "float64"})
```

### Edge Cases

- **Missing columns fail the assertion.** If a column in `expected` does not exist in the DataFrame, it is reported in `details.missing_columns` and the assertion fails. This catches upstream renames.
- **Partial validation is intentional.** If your DataFrame has 50 columns but you only care about 3, pass only those 3 in `expected`. Extra columns are ignored.

---

## Related Tests

Tests are located in `tests/test_data/test_schema.py`.

### TestAssertSchema

- **`test_valid_schema`** -- Verifies that a DataFrame matching the expected columns and dtypes produces a passing result. The happy path for schema validation.
- **`test_missing_column`** -- Verifies that requesting a column not present in the DataFrame raises `MltkAssertionError` with "Missing columns" in the error message. Simulates an upstream ETL job dropping a field.
- **`test_wrong_dtype`** -- Verifies that a column with the right name but wrong dtype (e.g., `float64` instead of `int64`) raises an error with the expected/actual types in the message. Catches silent type casting issues.
- **`test_extra_columns_allowed_by_default`** -- Verifies that extra columns in the DataFrame are accepted when `allow_extra_columns=True` (the default). Supports partial schema validation where only specific columns matter.
- **`test_extra_columns_rejected`** -- Verifies that extra columns cause a failure when `allow_extra_columns=False`. Useful for strict security-sensitive pipelines where unexpected columns might contain PII.
- **`test_empty_dataframe`** -- Verifies that an empty DataFrame (zero rows) with correct column names and dtypes passes schema validation. Confirms that schema checks are structural, not data-dependent.

### TestAssertNoNulls

- **`test_no_nulls`** -- Verifies that a DataFrame with no null values in any column produces a passing result.
- **`test_nulls_detected`** -- Verifies that null values in a column are detected and raise `MltkAssertionError`. Simulates a labeling pipeline that failed mid-batch.
- **`test_subset_columns`** -- Verifies that specifying a `columns` list only checks those columns, ignoring nulls in unlisted columns. Supports cases where optional metadata columns may legitimately have nulls.

### TestAssertDtypes

- **`test_correct_dtypes`** -- Verifies that columns matching their expected dtypes produce a passing result. Validates the happy path after CSV loading.
- **`test_dtype_mismatch`** -- Verifies that a string column expected to be `float64` raises an error. Simulates CSV files with formatting issues (commas in numbers, mixed types).
- **`test_missing_column_in_dtypes`** -- Verifies that requesting a dtype check on a non-existent column raises an error. Catches configuration drift when columns are renamed upstream.
