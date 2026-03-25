# Core API

The core module provides the foundational types and assertion framework that all mltk assertions are built on.

**Module:** `mltk.core`

---

## MltkConfig

```python
@dataclass
class MltkConfig:
    drift_method: str = "ks"
    drift_threshold: float = 0.05
    report_dir: str = "./mltk-reports"
    report_format: str = "html"
    baseline_dir: str = "./mltk-baselines"
    seed: int = 42
    pii_patterns: list[str] = field(
        default_factory=lambda: ["email", "phone", "ssn", "credit_card"]
    )
```

### What it is

Global configuration dataclass for mltk test runs. Holds settings for drift detection methods, thresholds, report output, and reproducibility seeds.

### Why it matters for ML

ML test pipelines need consistent, reproducible configuration. A drift threshold of 0.05 in development but 0.01 in CI will cause confusing pass/fail differences. `MltkConfig` centralizes these settings so the entire team runs tests with identical parameters.

### When to use it

- **Project setup** -- define defaults in `pyproject.toml` or `mltk.yaml` once
- **CI/CD pipelines** -- load config at the start of the test run to ensure consistency
- **Report generation** -- serialize config into reports for reproducibility audits

### Methods

#### `MltkConfig.load`

```python
@classmethod
def load(cls, path: str | Path | None = None) -> MltkConfig
```

Load configuration with cascade priority: explicit path > `mltk.yaml` > `pyproject.toml [tool.mltk]` > defaults.

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `path` | `str \| Path \| None` | `None` | Explicit path to a YAML config file. If `None`, auto-detects from the current directory. |

**Returns:** `MltkConfig` instance.

#### `MltkConfig.to_dict`

```python
def to_dict(self) -> dict[str, Any]
```

Serialize the config to a plain dictionary. Useful for logging, report metadata, or debugging.

**Returns:** `dict[str, Any]` with all configuration values.

### Example

```python
from mltk.core import MltkConfig

# Auto-detect config file
config = MltkConfig.load()
print(config.drift_method)   # "ks"
print(config.seed)           # 42

# Override with explicit values
config = MltkConfig(drift_method="psi", drift_threshold=0.1)
print(config.to_dict())
```

---

## Severity

```python
class Severity(Enum):
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"
```

### What it is

Enum controlling how assertion failures are handled.

### Why it matters for ML

Not all test failures are equal in ML. A missing label column is `CRITICAL` -- training cannot proceed. A statistical outlier is `WARNING` -- worth investigating but not necessarily a blocker. An informational metric is `INFO` -- logged for monitoring dashboards but never fails the build.

### Severity behavior

| Level | On failure |
|-------|------------|
| `CRITICAL` | Raises `MltkAssertionError` (test fails) |
| `WARNING` | Returns `TestResult` with `passed=False` (test continues) |
| `INFO` | Returns `TestResult` (never fails) |

---

## TestResult

```python
@dataclass
class TestResult:
    name: str
    passed: bool
    severity: Severity
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
```

### What it is

The structured result object returned by every mltk assertion. Contains the pass/fail status, a human-readable message, and a details dictionary with assertion-specific data.

### Why it matters for ML

ML tests produce rich diagnostic information beyond pass/fail. When `assert_range` fails, you need to know which values were out of bounds, how many, and the actual min/max. `TestResult.details` carries this structured data for reports, dashboards, and debugging.

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Assertion identifier (e.g., `"data.schema"`, `"data.range[age]"`) |
| `passed` | `bool` | Whether the assertion passed |
| `severity` | `Severity` | `CRITICAL`, `WARNING`, or `INFO` |
| `message` | `str` | Human-readable result description |
| `details` | `dict[str, Any]` | Assertion-specific metadata (counts, bounds, mismatches, etc.) |
| `duration_ms` | `float` | Execution time in milliseconds |
| `timestamp` | `datetime` | When the assertion was executed |

### Example

```python
from mltk.data import assert_range
import pandas as pd

s = pd.Series([0.1, 0.5, 0.9], name="score")
result = assert_range(s, min_val=0.0, max_val=1.0)

print(result.passed)       # True
print(result.name)         # "data.range[score]"
print(result.duration_ms)  # 0.23 (varies)
print(result.details)      # {"actual_min": 0.1, "actual_max": 0.9, ...}
```

---

## TestSuite

```python
@dataclass
class TestSuite:
    results: list[TestResult] = field(default_factory=list)
```

### What it is

A collection of `TestResult` objects from a test run. Provides aggregate metrics like total count, pass rate, and overall pass/fail status.

### Why it matters for ML

ML validation often involves running dozens of assertions against a dataset. `TestSuite` aggregates the results and determines overall status: the suite passes only if all `CRITICAL` assertions pass. Warnings alone do not fail the suite.

### Methods and Properties

#### `TestSuite.add`

```python
def add(self, result: TestResult) -> None
```

Add a test result to the suite.

#### `TestSuite.passed`

```python
@property
def passed(self) -> bool
```

`True` if all `CRITICAL` severity results passed. Warning failures do not affect this.

#### `TestSuite.total` / `passed_count` / `failed_count`

```python
@property
def total(self) -> int

@property
def passed_count(self) -> int

@property
def failed_count(self) -> int
```

#### `TestSuite.score`

```python
@property
def score(self) -> float
```

Pass rate as a percentage (0.0 -- 100.0). Empty suite returns 0.0.

### Example

```python
from mltk.core import TestSuite, TestResult, Severity

suite = TestSuite()
suite.add(TestResult(name="schema", passed=True, severity=Severity.CRITICAL, message="ok"))
suite.add(TestResult(name="outliers", passed=False, severity=Severity.WARNING, message="3 outliers"))

print(suite.passed)       # True (only CRITICAL matters)
print(suite.total)        # 2
print(suite.passed_count) # 1
print(suite.failed_count) # 1
print(suite.score)        # 50.0
```

---

## assert_true

```python
def assert_true(
    condition: bool,
    name: str,
    message: str,
    severity: Severity = Severity.CRITICAL,
    **details: Any,
) -> TestResult
```

### What it tests

The base assertion function. Evaluates a boolean condition and produces a `TestResult`. All higher-level assertions (like `assert_schema`, `assert_range`, etc.) are built on top of this.

### Why it matters for ML

`assert_true` lets you write custom ML assertions for domain-specific checks that mltk does not cover yet. Need to assert that a confusion matrix is not degenerate? That a tokenizer vocabulary size matches your model config? Use `assert_true` with a custom condition.

### When to use it

- **Custom assertions** -- when no built-in assertion fits your check
- **Building new assertion functions** -- as the foundation for your own `assert_*` helpers
- **One-off checks** -- quick validation during exploratory testing

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `condition` | `bool` | *(required)* | The condition to assert. `True` = pass, `False` = fail. |
| `name` | `str` | *(required)* | Identifier for this assertion (appears in reports). |
| `message` | `str` | *(required)* | Human-readable description of the result. |
| `severity` | `Severity` | `Severity.CRITICAL` | Controls failure behavior. `CRITICAL` raises on failure; `WARNING` and `INFO` do not. |
| `**details` | `Any` | `{}` | Arbitrary keyword arguments stored in `TestResult.details`. |

### Returns

`TestResult` with the assertion outcome. If `condition` is `False` and `severity` is `CRITICAL`, raises `MltkAssertionError` instead of returning.

### Example

```python
import pytest
from mltk.core.assertion import assert_true
from mltk.core.result import Severity

def test_vocabulary_size():
    """Custom assertion: tokenizer vocab matches model config."""
    vocab_size = 32000
    model_vocab = 32000

    result = assert_true(
        vocab_size == model_vocab,
        name="model.vocab_match",
        message=f"Vocab size {vocab_size} matches model config {model_vocab}",
        actual=vocab_size,
        expected=model_vocab,
    )
    assert result.passed

def test_warning_does_not_block():
    """WARNING severity logs the failure but does not raise."""
    result = assert_true(
        False,
        name="data.optional_check",
        message="Non-critical issue detected",
        severity=Severity.WARNING,
    )
    # No exception raised -- test continues
    assert result.passed is False
    assert result.severity == Severity.WARNING
```

### Edge Cases

- Passing `severity=Severity.WARNING` with a `False` condition will **not** raise an exception. The result is returned with `passed=False`. This is intentional for non-blocking checks.
- The `**details` kwargs are stored as-is in `TestResult.details`. Make sure values are JSON-serializable if you plan to write reports.

---

## timed_assertion

```python
def timed_assertion(func: Callable) -> Callable
```

### What it is

A decorator that adds timing to assertion functions. It measures execution time using `time.perf_counter()` and stores the result in `TestResult.duration_ms`.

### Why it matters for ML

ML assertions can be expensive -- computing drift statistics on large datasets, running inference for robustness checks, or scanning for PII across millions of rows. `timed_assertion` automatically records how long each assertion takes, which feeds into reports and helps identify slow tests that need optimization.

### When to use it

- **Building custom assertions** -- wrap your `assert_*` function with `@timed_assertion` to get automatic timing
- **All built-in assertions** use this decorator already

### Example

```python
from mltk.core.assertion import timed_assertion, assert_true
from mltk.core.result import Severity, TestResult

@timed_assertion
def assert_custom_check(data, threshold=0.5) -> TestResult:
    score = compute_expensive_metric(data)
    return assert_true(
        score >= threshold,
        name="custom.check",
        message=f"Score {score:.4f} vs {threshold}",
        severity=Severity.CRITICAL,
        score=score,
    )

result = assert_custom_check(my_data)
print(result.duration_ms)  # automatically populated
```

---

## MltkAssertionError

```python
class MltkAssertionError(AssertionError):
    def __init__(self, result: TestResult) -> None
```

### What it is

Exception raised when a `CRITICAL` severity assertion fails. It is a subclass of Python's built-in `AssertionError`, so pytest catches it naturally without any special configuration.

### Why it matters for ML

The error carries the full `TestResult` object, so error handlers, report generators, and CI/CD integrations can extract structured diagnostic data (not just a string message).

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `result` | `TestResult` | The full test result with name, message, details, timing |

### Example

```python
import pytest
from mltk.core.assertion import MltkAssertionError, assert_true

def test_catches_structured_error():
    with pytest.raises(MltkAssertionError) as exc_info:
        assert_true(False, name="check", message="something failed")

    error = exc_info.value
    assert error.result.passed is False
    assert error.result.name == "check"
    assert error.result.message == "something failed"
```

---

## Related Tests

Tests are located in `tests/test_core/`.

### tests/test_core/test_assertion.py

- **`test_assert_true_passes`** -- Verifies that `assert_true` with a `True` condition returns a `TestResult` with `passed=True`. The happy path.
- **`test_assert_true_fails_critical`** -- Verifies that `assert_true` with `False` and `CRITICAL` severity raises `MltkAssertionError` with the correct result attached.
- **`test_assert_true_warning_does_not_raise`** -- Verifies that `assert_true` with `False` and `WARNING` severity returns a result without raising, confirming non-blocking behavior.
- **`test_assert_true_carries_details`** -- Verifies that keyword arguments passed to `assert_true` are stored in `TestResult.details`, enabling structured diagnostics.

### tests/test_core/test_config.py

- **`test_default_config`** -- Verifies that `MltkConfig()` with no arguments uses sensible defaults (KS test, 0.05 threshold, seed=42).
- **`test_config_to_dict`** -- Verifies that `to_dict()` serializes all config fields to a plain dictionary for report metadata.
- **`test_config_load_defaults`** -- Verifies that `MltkConfig.load()` returns defaults when no config files exist in the current directory.
- **`test_config_from_pyproject`** -- Verifies loading `[tool.mltk]` from a `pyproject.toml` file, including that unset values keep their defaults.
- **`test_config_from_yaml`** -- Verifies loading flat key-value pairs from an `mltk.yaml` file.
- **`test_config_from_yaml_nested`** -- Verifies loading the nested `mltk: { ... }` format from YAML, supporting namespaced config files.
- **`test_config_load_missing_path`** -- Verifies that `load()` with a non-existent file path returns defaults instead of crashing.
- **`test_config_from_dict_ignores_unknown_keys`** -- Verifies that unknown keys in config dictionaries are silently ignored for forward compatibility.
- **`test_config_from_pyproject_no_mltk_section`** -- Verifies that a `pyproject.toml` without a `[tool.mltk]` section returns defaults.

### tests/test_core/test_result.py

- **`test_test_result_creation`** -- Verifies that a `TestResult` can be created with all required fields and that attributes are stored correctly.
- **`test_test_suite_passed`** -- Verifies that a suite with all passing results reports `passed=True`, correct total count, and 100% score.
- **`test_test_suite_failed_critical`** -- Verifies that a suite with one failed `CRITICAL` result reports `passed=False` and correct `failed_count`.
- **`test_test_suite_warning_only_still_passes`** -- Verifies that a suite where only `WARNING` results fail still reports `passed=True`, confirming that only `CRITICAL` failures affect suite status.
- **`test_empty_suite_score`** -- Verifies that an empty `TestSuite` returns a score of 0.0 instead of raising a division-by-zero error.
