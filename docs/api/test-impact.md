# Test Impact Analysis & Anomaly Detection

Run only affected tests, and catch relative metric changes that absolute thresholds miss.

**Modules:** `mltk.testing.impact`, `mltk.monitor.anomaly`

---

## Why Test Impact Analysis

Running an entire test suite on every commit is wasteful. If you modified `data/drift.py`, you do not need to re-run `test_fairness.py` or `test_golden.py`. Test impact analysis builds a dependency graph from Python imports and returns only the test files whose execution paths include the changed source code.

### The Problem

| Approach | Tests Run | CI Time | Coverage Guarantee |
|----------|:---------:|:-------:|:------------------:|
| Run everything | 1500+ | 45 min | Full |
| Manual selection | 5-10 | 2 min | None (error-prone) |
| **Impact analysis** | 10-50 | 3-5 min | Verified by assertion |

Impact analysis gives you the speed of manual selection with the coverage guarantee of running everything.

---

## `analyze_impact`

Determine which test files should run based on changed source files.

**Module:** `mltk.testing.impact`

### How the Dependency Graph Works

1. **Module index**: Scans the `src/` tree to build a module-name to file-path lookup table.
2. **Import parsing**: For each `test_*.py` file, parses its imports using `ast.parse` and maps them to source files.
3. **Transitive dependencies**: Builds a transitive dependency graph between source files. If `drift.py` imports from `schema.py`, and `monitor.py` imports from `drift.py`, then changing `schema.py` impacts tests of both `drift.py` and `monitor.py`.
4. **Impact collection**: For each changed file, collects all test files that depend on it directly or transitively.

### Example

```python
from mltk.testing.impact import analyze_impact

# Which tests should run after changing drift.py?
impacted = analyze_impact(
    changed_files=["src/mltk/data/drift.py"],
    project_root=".",
    test_dir="tests",
)
# Returns:
# [
#   "tests/test_data/test_drift.py",
#   "tests/test_monitor/test_concept_drift.py",  # transitively depends on drift
# ]
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `changed_files` | `list[str]` | *(required)* | Changed source file paths (relative to project root or absolute) |
| `project_root` | `str` | `"."` | Root of the project |
| `test_dir` | `str` | `"tests"` | Subdirectory containing tests |

Returns a sorted, deduplicated list of test file paths that should be executed. Empty list if no tests are impacted.

### CI Integration

```bash
# Get changed files from git
CHANGED=$(git diff --name-only HEAD~1)

# Use mltk to find impacted tests
python -c "
from mltk.testing.impact import analyze_impact
changed = '''$CHANGED'''.strip().split('\n')
impacted = analyze_impact(changed)
print('\n'.join(impacted))
" > impacted_tests.txt

# Run only impacted tests
pytest $(cat impacted_tests.txt)
```

---

## `assert_impact_coverage`

Assert that all impacted tests were actually executed. Catches the situation where your CI pipeline skipped tests due to misconfigured path filters, parallelism bugs, or shard imbalances.

**Module:** `mltk.testing.impact`

### Why This Assertion Exists

CI pipelines sometimes skip tests silently. You change `drift.py` but `test_drift.py` never runs -- a coverage gap that defeats the purpose of having tests. This assertion closes that gap by verifying every impacted test was executed.

### Example

```python
from mltk.testing.impact import assert_impact_coverage

# After your CI run, verify coverage
result = assert_impact_coverage(
    changed_files=["src/mltk/data/drift.py"],
    executed_tests=[
        "tests/test_data/test_drift.py",
        "tests/test_monitor/test_concept_drift.py",
    ],
    project_root=".",
    test_dir="tests",
)
assert result.passed, f"Missing tests: {result.details['missing_tests']}"
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `changed_files` | `list[str]` | *(required)* | Source files that were modified |
| `executed_tests` | `list[str]` | *(required)* | Test files that were actually run |
| `project_root` | `str` | `"."` | Project root directory |
| `test_dir` | `str` | `"tests"` | Test subdirectory name |

Returns `TestResult` (name: `impact.coverage`) with details:

| Detail Key | Type | Description |
|------------|------|-------------|
| `required_tests` | `list[str]` | Tests that should have been executed |
| `executed_tests` | `list[str]` | Tests that were actually executed |
| `missing_tests` | `list[str]` | Tests that were required but not executed |
| `changed_files` | `list[str]` | The source files that triggered the analysis |

---

## Why Anomaly Detection on Test Metrics

A standard assertion like `assert accuracy >= 0.80` catches **absolute** failures: the model accuracy dropped below a hard threshold. But what if your model usually scores 0.95 and today it scores 0.88? It still passes the 0.80 threshold, but something is clearly wrong -- a 7-point drop from baseline is a significant regression that warrants investigation.

Anomaly detection catches **relative** changes by comparing the current metric value against its own history.

### Absolute vs. Relative Failures

| Scenario | Absolute Threshold (0.80) | Anomaly Detection |
|----------|:-------------------------:|:-----------------:|
| Usual: 0.95, Today: 0.93 | PASS | PASS (within normal range) |
| Usual: 0.95, Today: 0.88 | PASS | **FAIL** (7-point drop) |
| Usual: 0.95, Today: 0.78 | FAIL | FAIL |
| Usual: 0.82, Today: 0.81 | PASS | PASS (within normal range) |

Both types of checks are needed for robust ML testing. Absolute thresholds define the floor. Anomaly detection catches silent degradation above the floor.

---

## `assert_no_test_anomaly`

Assert that the current test metric is not anomalous compared to its history.

**Module:** `mltk.monitor.anomaly`

### Three Detection Methods

#### 1. Z-score (default)

How many standard deviations is the current value from the historical mean?

```
z = (current - mean) / std
```

Anomalous if `|z| > threshold` (default: 3.0, corresponding to ~0.3% probability under normality).

Best for: roughly normal distributions (most ML metrics).

```python
from mltk.monitor.anomaly import assert_no_test_anomaly

latency_history = [12.1, 11.8, 12.3, 12.0, 11.9, 12.2, 12.1, 11.7, 12.4]

# Normal value -- passes
result = assert_no_test_anomaly(
    history=latency_history,
    current=12.0,
    method="zscore",
    threshold=3.0,
)
assert result.passed

# Extreme value -- fails
result = assert_no_test_anomaly(
    history=latency_history,
    current=45.2,
    method="zscore",
    threshold=3.0,
)
assert not result.passed
print(f"Z-score: {result.details['z_score']}")  # very high
```

#### 2. IQR (Interquartile Range)

The "box" in a box plot. Values outside `[Q1 - multiplier*IQR, Q3 + multiplier*IQR]` are outliers.

More robust than Z-score when the history is skewed or has heavy tails.

```python
result = assert_no_test_anomaly(
    history=latency_history,
    current=45.2,
    method="iqr",
    threshold=1.5,  # standard Tukey fence
)
assert not result.passed
print(f"Bounds: [{result.details['lower_bound']:.2f}, {result.details['upper_bound']:.2f}]")
```

#### 3. Percentile

Flags values below the Nth or above the (100-N)th percentile of the history. Intuitive and non-parametric (no normality assumption).

```python
result = assert_no_test_anomaly(
    history=latency_history,
    current=45.2,
    method="percentile",
    threshold=5.0,  # flag bottom/top 5%
)
assert not result.passed
print(f"Range: [{result.details['lower_percentile']:.2f}, {result.details['upper_percentile']:.2f}]")
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `history` | `list[float]` | *(required)* | Previous metric values (at least 3 recommended) |
| `current` | `float` | *(required)* | The current observation to test |
| `method` | `str` | `"zscore"` | Detection method: `"zscore"`, `"iqr"`, or `"percentile"` |
| `threshold` | `float` | `3.0` | Sensitivity threshold (interpretation depends on method) |

### Threshold Reference

| Method | Parameter | Default | Meaning |
|--------|-----------|:-------:|---------|
| `zscore` | Max absolute Z-score | `3.0` | ~0.3% false positive rate under normality |
| `iqr` | IQR multiplier | `1.5` | Standard Tukey fence; use 3.0 for extreme outliers only |
| `percentile` | Percentile cutoff | `5.0` | Flag values below 5th or above 95th percentile |

Returns `TestResult` (name: `anomaly.test_metric`) with method-specific details:

#### Z-score Details

| Detail Key | Type | Description |
|------------|------|-------------|
| `method` | `str` | `"zscore"` |
| `z_score` | `float` | Computed Z-score |
| `mean` | `float` | Historical mean |
| `std` | `float` | Historical standard deviation |
| `threshold` | `float` | The Z-score threshold |
| `current` | `float` | The tested value |
| `is_anomaly` | `bool` | Whether an anomaly was detected |

#### IQR Details

| Detail Key | Type | Description |
|------------|------|-------------|
| `method` | `str` | `"iqr"` |
| `q1` | `float` | 25th percentile |
| `q3` | `float` | 75th percentile |
| `iqr` | `float` | Q3 - Q1 |
| `lower_bound` | `float` | Q1 - threshold * IQR |
| `upper_bound` | `float` | Q3 + threshold * IQR |
| `current` | `float` | The tested value |
| `is_anomaly` | `bool` | Whether an anomaly was detected |

#### Percentile Details

| Detail Key | Type | Description |
|------------|------|-------------|
| `method` | `str` | `"percentile"` |
| `lower_percentile` | `float` | Value at the threshold-th percentile |
| `upper_percentile` | `float` | Value at the (100-threshold)-th percentile |
| `threshold_pct` | `float` | The percentile cutoff |
| `current` | `float` | The tested value |
| `is_anomaly` | `bool` | Whether an anomaly was detected |

### Edge Cases

- **Fewer than 3 history values**: Returns a passing result with an `INFO` severity message. Not enough data for meaningful statistics.
- **Constant history** (Z-score): Any deviation from the constant value is flagged as anomalous.
- **Unknown method**: Returns a failing result with `CRITICAL` severity.

---

## pytest Integration

### Impact-Aware Testing

```python
import pytest
from mltk.testing.impact import analyze_impact, assert_impact_coverage

def test_impact_coverage_gate(changed_files, executed_tests):
    """Verify CI ran all impacted tests."""
    result = assert_impact_coverage(
        changed_files=changed_files,
        executed_tests=executed_tests,
        project_root="/path/to/project",
        test_dir="tests",
    )
    assert result.passed, (
        f"Skipped {len(result.details['missing_tests'])} impacted tests: "
        f"{result.details['missing_tests']}"
    )
```

### Anomaly Detection on CI Metrics

```python
from mltk.monitor.anomaly import assert_no_test_anomaly

def test_inference_latency_stability(latency_history_db):
    """Catch latency regressions that are still above the absolute threshold."""
    history = latency_history_db.get_last_n(30)  # last 30 runs
    current = measure_current_latency()

    # Z-score detects sudden spikes
    result = assert_no_test_anomaly(
        history=history,
        current=current,
        method="zscore",
        threshold=3.0,
    )
    assert result.passed, (
        f"Latency anomaly: {current}ms "
        f"(Z-score={result.details['z_score']:.2f}, "
        f"mean={result.details['mean']:.2f}ms)"
    )

def test_accuracy_stability(accuracy_history_db):
    """Catch gradual accuracy degradation."""
    history = accuracy_history_db.get_last_n(30)
    current = evaluate_current_accuracy()

    # IQR is robust to occasional outliers in history
    result = assert_no_test_anomaly(
        history=history,
        current=current,
        method="iqr",
        threshold=1.5,
    )
    assert result.passed, (
        f"Accuracy anomaly: {current:.4f} "
        f"outside [{result.details['lower_bound']:.4f}, "
        f"{result.details['upper_bound']:.4f}]"
    )
```

### Combining Absolute and Relative Checks

```python
def test_model_accuracy_comprehensive(model, test_data, history_db):
    """Both absolute floor AND relative stability must hold."""
    accuracy = evaluate(model, test_data)
    history = history_db.get_last_n(30)

    # Absolute floor: never below 0.80
    assert accuracy >= 0.80, f"Accuracy {accuracy:.4f} below absolute floor 0.80"

    # Relative check: no anomalous drops from baseline
    result = assert_no_test_anomaly(
        history=history,
        current=accuracy,
        method="zscore",
        threshold=2.5,
    )
    assert result.passed, (
        f"Accuracy dropped from baseline: "
        f"{accuracy:.4f} vs mean {result.details['mean']:.4f}"
    )
```
