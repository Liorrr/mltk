# Testing Patterns

Utilities for writing robust ML tests: flaky-test detection, versioned golden baselines, confidence-interval retry, and smart test selection.

**Module:** `mltk.testing`

---

## detect_flaky

Run a test function N times and decide whether it is flaky (non-deterministic).

```python
from mltk.testing import detect_flaky

summary = detect_flaky(my_test_fn, runs=10, threshold=0.8)
if summary.is_flaky:
    print(f"Flaky! Pass rate: {summary.pass_rate:.0%}")
```

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `func` | `Callable[[], None]` | *(required)* | Zero-argument callable; raises on failure |
| `runs` | `int` | `5` | Number of executions |
| `threshold` | `float` | `0.8` | Pass-rate below which a partial pass is labelled flaky |
| `test_name` | `str \| None` | `None` | Label for the summary; defaults to `func.__name__` |

### Returns

`FlakySummary` with fields:

| Field | Type | Description |
|-------|------|-------------|
| `test_name` | `str` | Label for the test |
| `pass_count` | `int` | Number of runs that passed |
| `fail_count` | `int` | Number of runs that failed |
| `pass_rate` | `float` | `pass_count / runs` |
| `is_flaky` | `bool` | `True` when `0 < pass_rate < threshold` |

### How it works

A test that **always passes** (`pass_rate == 1.0`) or **always fails** (`pass_rate == 0.0`) is considered stable or broken — not flaky. Only tests that sometimes pass and sometimes fail within the threshold window are labelled flaky.

### Example

```python
import pytest
from mltk.testing import detect_flaky

@pytest.mark.ml_smoke
def test_no_random_seed_flakiness():
    """Model accuracy should not vary across runs without seed."""
    summary = detect_flaky(lambda: train_and_evaluate(), runs=5, threshold=0.8)
    assert not summary.is_flaky, f"Test is flaky: {summary.pass_rate:.0%} pass rate"
```

---

## save_golden / load_golden

Persist and retrieve versioned baseline data for regression testing.

```python
from mltk.testing import save_golden, load_golden

# Save at pipeline acceptance time
save_golden({"accuracy": 0.923, "f1": 0.901}, "baselines/v1.json", version="1.0.0")

# Load in tests
envelope = load_golden("baselines/v1.json")
baseline = envelope["data"]  # {"accuracy": 0.923, "f1": 0.901}
```

### save_golden Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `data` | `dict \| list \| np.ndarray` | *(required)* | Baseline data to persist |
| `path` | `str \| Path` | *(required)* | Destination JSON file path |
| `version` | `str \| None` | `"1.0.0"` | Version tag stored in metadata |

### save_golden Returns

`Path` — the resolved path of the written file.

### load_golden Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `path` | `str \| Path` | *(required)* | Path to a previously saved golden JSON file |

### load_golden Returns

`dict` with keys:

| Key | Type | Description |
|-----|------|-------------|
| `version` | `str` | Version tag set at save time |
| `timestamp` | `str` | ISO-8601 timestamp of when the file was saved |
| `data` | `dict \| list` | The original data (numpy arrays serialised as nested lists) |

### Edge Cases

- **Parent directory**: Created automatically if it does not exist.
- **Numpy arrays**: Converted to nested Python lists via `.tolist()` before serialisation. Reloaded as plain lists.
- **Missing file**: `load_golden` raises `FileNotFoundError`.

---

## assert_matches_golden

Assert that current data matches a saved golden baseline within a numeric tolerance.

```python
from mltk.testing import assert_matches_golden

assert_matches_golden(current_metrics, "baselines/v1.json", tolerance=0.02)
```

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `current` | `dict \| list \| np.ndarray` | *(required)* | Data produced by the system under test |
| `golden_path` | `str \| Path` | *(required)* | Path to the golden file |
| `tolerance` | `float` | `0.01` | Maximum allowed numeric deviation |

### Returns

`TestResult` with details:

| Detail key | Description |
|------------|-------------|
| `max_diff` | Maximum absolute difference observed |
| `tolerance` | Configured tolerance |
| `golden_version` | Version tag from the golden file |

### How it works

Comparison is recursive:

- **numpy arrays** — converted to lists, then compared element-wise
- **dicts** — all keys present in the golden are compared recursively
- **lists** — element-wise comparison
- **scalars** — `abs(current - golden) <= tolerance`

### Example

```python
import pytest
from mltk.testing import save_golden, assert_matches_golden

@pytest.mark.ml_regression
def test_model_metrics_stable(trained_model, test_data):
    """Key metrics must not regress beyond 1% from the accepted baseline."""
    metrics = evaluate(trained_model, test_data)
    assert_matches_golden(metrics, "baselines/model_v2.json", tolerance=0.01)
```

---

## retry_until_confident

Run a non-deterministic test repeatedly and evaluate its pass verdict using a Wilson score confidence interval.

```python
from mltk.testing import retry_until_confident

result = retry_until_confident(my_stochastic_test, max_runs=20, confidence=0.95)
assert result.is_passing
```

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `func` | `Callable[[], None]` | *(required)* | Zero-argument callable; raises on failure |
| `min_runs` | `int` | `3` | Minimum executions before early termination |
| `max_runs` | `int` | `10` | Hard cap on total executions |
| `confidence` | `float` | `0.95` | Confidence level for the Wilson interval |
| `failure_threshold` | `float` | `0.5` | Lower CI bound must exceed this to declare passing |

### Returns

`RetryResult` with fields:

| Field | Type | Description |
|-------|------|-------------|
| `pass_count` | `int` | Number of successful runs |
| `fail_count` | `int` | Number of failed runs |
| `pass_rate` | `float` | `pass_count / total` |
| `confidence_lower` | `float` | Lower bound of the Wilson CI |
| `confidence_upper` | `float` | Upper bound of the Wilson CI |
| `is_passing` | `bool` | `True` if `confidence_lower > failure_threshold` |

### How it works

Uses the **Wilson score interval**, which gives better coverage than the naive Wald interval at small sample sizes. After each run (starting from `min_runs`), the CI is evaluated for early exit: if the verdict is already clear (lower bound above or upper bound below `failure_threshold`) the loop terminates early.

### Example

```python
import pytest
from mltk.testing import retry_until_confident

@pytest.mark.ml_smoke
def test_sampling_quality():
    """Generated samples must pass quality check 90%+ of the time."""
    def check():
        sample = model.generate()
        assert quality_score(sample) > 0.7

    result = retry_until_confident(check, max_runs=30, failure_threshold=0.8)
    assert result.is_passing, f"Only {result.pass_rate:.0%} samples passed quality check"
```

---

## build_test_map

Build a dependency map from source files to the test files that import them, using Python AST parsing.

```python
from mltk.testing import build_test_map

mapping = build_test_map("tests/", "src/")
# {"src/mltk/data/drift.py": ["tests/test_data/test_drift.py"], ...}
```

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `test_dir` | `str \| Path` | *(required)* | Root directory containing test files |
| `src_dir` | `str \| Path` | *(required)* | Root directory of the source tree |

### Returns

`dict[str, list[str]]` — maps source-file path strings to lists of test-file path strings.

### How it works

1. Walks every `test_*.py` file under `test_dir`.
2. Parses each file with `ast` to extract `import` and `from ... import` statements.
3. Converts source file paths to dotted module names relative to `src_dir`.
4. Matches imported module names against the source module index (exact and prefix matches).

---

## select_affected_tests

Given a list of changed source files, return the test files that need re-running.

```python
from mltk.testing import build_test_map, select_affected_tests

mapping = build_test_map("tests/", "src/")
affected = select_affected_tests(["src/mltk/data/drift.py"], mapping)
# ["tests/test_data/test_drift.py"]
```

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `changed_files` | `list[str]` | *(required)* | Source file paths that were modified |
| `test_map` | `dict[str, list[str]]` | *(required)* | Map produced by `build_test_map` |

### Returns

`list[str]` — deduplicated list of test file paths that cover the changed sources. Empty list if no tests are affected.

### Example

```python
import subprocess
from mltk.testing import build_test_map, select_affected_tests

# Get files changed since last commit
changed = subprocess.check_output(
    ["git", "diff", "--name-only", "HEAD~1"]
).decode().splitlines()

mapping = build_test_map("tests/", "src/")
affected = select_affected_tests(changed, mapping)

if affected:
    subprocess.run(["pytest"] + affected)
else:
    print("No affected tests — skipping run")
```

---
