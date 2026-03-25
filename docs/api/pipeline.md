# Pipeline Testing

Validate ML pipeline reproducibility and end-to-end execution. Catches non-deterministic training, broken pipelines, and artifact corruption.

**Module:** `mltk.pipeline`

---

## assert_reproducible

Assert a function produces identical output across multiple runs with the same seed.

```python
from mltk.pipeline import assert_reproducible

assert_reproducible(train_model, X, y, seed=42, runs=3, tolerance=0.001)
```

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `func` | `Callable` | *(required)* | Function to test (e.g., `train_model`) |
| `*args` | `Any` | | Positional arguments passed through to `func` on each run |
| `seed` | `int` | `42` | Random seed set before each run (both `random` and `numpy.random`) |
| `runs` | `int` | `3` | Number of runs to compare |
| `tolerance` | `float` | `0.001` | Max allowed difference between outputs (for numeric comparisons) |

### Returns

`TestResult` with details:
- `seed` -- the seed used
- `runs` -- number of runs performed
- `tolerance` -- configured tolerance
- `max_diff` -- maximum difference observed between runs

### How it works

For each run, sets `random.seed(seed)` and `numpy.random.seed(seed)`, then calls `func(*args)`. Compares all outputs:
- **numpy arrays**: max absolute element-wise difference
- **numeric scalars**: absolute difference
- **other types**: equality check (diff = 0.0 if equal, 1.0 if not)

### Example

```python
import pytest
from mltk.pipeline import assert_reproducible

@pytest.mark.ml_smoke
def test_training_deterministic(X_train, y_train):
    """Training produces the same model weights across runs."""
    def train(X, y):
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(random_state=42).fit(X, y).coef_

    assert_reproducible(train, X_train, y_train, seed=42, runs=3, tolerance=0.001)
```

---

## assert_checksum

Assert a model artifact matches an expected SHA-256 hash.

```python
from mltk.pipeline import assert_checksum

assert_checksum("model.pkl", expected_hash="sha256:abc123...")
```

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `path` | `str \| Path` | *(required)* | Path to file to verify |
| `expected_hash` | `str` | *(required)* | Expected SHA-256 hex digest (with or without `sha256:` prefix) |

### Returns

`TestResult` with details:
- `actual_hash` -- computed SHA-256 hex digest
- `expected_hash` -- expected hash (prefix stripped)

### Edge Cases

- **File not found**: Returns a failing `TestResult` with `CRITICAL` severity.
- **`sha256:` prefix**: Automatically stripped from `expected_hash` before comparison.

---

## assert_pipeline

Assert an end-to-end pipeline runs without errors.

```python
from mltk.pipeline import assert_pipeline

assert_pipeline([load_data, preprocess, train, evaluate], input_data=config)
```

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `steps` | `list[Callable]` | *(required)* | List of callables to run in sequence. Each receives the output of the previous step. |
| `input_data` | `Any` | *(required)* | Initial input passed to the first step |
| `expected_output_type` | `type \| None` | `None` | Expected type of final output. `None` = skip type check. |

### Returns

`TestResult` with details:
- `completed_steps` -- number of steps that completed successfully
- `total_steps` -- total number of steps in the pipeline

If a step raises an exception, additional details are included:
- `failed_step` -- index of the failed step
- `step_name` -- name of the failed function

### Example

```python
import pytest
from mltk.pipeline import assert_pipeline

@pytest.mark.ml_smoke
def test_pipeline_runs():
    """Full data->train->predict pipeline completes without errors."""
    def load(cfg):
        return pd.read_csv(cfg["path"])

    def preprocess(df):
        return df.dropna()

    def train(df):
        model = LogisticRegression().fit(df[["x"]], df["y"])
        return model

    assert_pipeline([load, preprocess, train], input_data={"path": "data.csv"})

@pytest.mark.ml_smoke
def test_pipeline_output_type():
    """Pipeline output must be a numpy array."""
    assert_pipeline([load, preprocess, predict], input_data=cfg, expected_output_type=np.ndarray)
```

### Edge Cases

- **Step failure**: If any step raises an exception, the assertion fails with `CRITICAL` severity and reports which step failed and the exception type.
- **Type checking**: If `expected_output_type` is provided and the final output does not match, the assertion fails with a type mismatch message.

---
