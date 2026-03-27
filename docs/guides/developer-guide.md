# Developer Guide

The definitive guide for developers extending, integrating, and contributing to mltk.

Whether you need to write custom assertions for your ML domain, wire mltk into an existing pipeline, extend the plugin system, or contribute upstream -- this guide covers it all.

---

## 1. Architecture Overview

### Package Structure

```text
src/mltk/
  __init__.py            Convenience re-exports (assert_schema, assert_metric, ...)
  _rust.py               Rust acceleration bridge (fallback to scipy/numpy)
  _mltk_rust.*.pyd/.so   Compiled Rust extension (PyO3)

  core/                  Foundation layer
    result.py              TestResult, TestSuite, Severity enum
    assertion.py           assert_true(), @timed_assertion, MltkAssertionError
    config.py              MltkConfig -- cascade loader (env > yaml > pyproject > defaults)
    plugin.py              @register_assertion, discover_plugins()

  data/                  Data quality (25+ assertions)
    schema.py              assert_schema, assert_no_nulls, assert_dtypes
    distribution.py        assert_range, assert_unique, assert_no_outliers
    drift.py               assert_no_drift (KS, PSI, chi-squared, JS, Wasserstein)
    freshness.py           assert_freshness, assert_row_count
    pii.py                 assert_no_pii, scan_pii, PiiMatch
    labels.py              assert_label_balance, assert_label_coverage
    statistics.py          assert_column_mean, assert_column_median, assert_column_stdev
    validation.py          assert_datetime_format, assert_values_in_set, ...
    embedding_drift.py     assert_no_embedding_drift
    lineage.py             LineageGraph, @track_lineage, assert_lineage_complete
    preset.py              assert_data_quality (one-call quality sweep)

  model/                 Model quality (10+ assertions)
    metrics.py             assert_metric (accuracy, F1, AUC, ...)
    regression.py          assert_no_regression, save_baseline
    slicing.py             assert_slice_performance, assert_calibration
    bias.py                assert_no_bias (demographic parity, equalized odds)
    adversarial.py         assert_robust
    overfitting.py         assert_no_overfitting, assert_label_drift

  training/              Training bug detection (19 assertions)
    gradient.py            assert_gradient_flow, assert_no_vanishing_gradient, ...
    numerical.py           assert_no_nan_inf, assert_loss_decreasing, ...
    leakage.py             assert_no_train_test_overlap, assert_temporal_split, ...
    checkpoint.py          assert_checkpoint_complete, assert_resume_loss_continuous
    augmentation.py        assert_no_augmentation_on_test, ...
    distributed.py         assert_effective_batch_size, assert_gradient_sync
    memory.py              assert_no_memory_leak, assert_loss_is_detached
    skew.py                assert_no_training_serving_skew

  inference/             Serving performance
    latency.py             assert_latency, assert_cold_start
    throughput.py          assert_throughput
    contract.py            assert_api_contract

  pipeline/              Pipeline validation
    e2e.py                 assert_pipeline
    reproducibility.py     assert_reproducible, assert_checksum

  monitor/               Production monitoring
    drift_monitor.py       assert_no_degradation, assert_sla, assert_no_output_drift
    prometheus.py          assert_prometheus_metric, assert_gpu_utilization, ...
    aws.py / gcp.py / azure.py   Cloud-provider endpoint health

  domains/               Domain-specific kits
    cv/                    IoU, mAP, tracking (MOTA/MOTP/IDF1), face FAR, top-K
    nlp/                   BLEU, ROUGE, NER F1, prompt injection, sentiment drift
    speech/                WER, CER, RTF, accent coverage
    tabular/               Feature drift, importance stability, class balance
    llm/                   RAG eval, BERTScore, toxicity, hallucination, TTFT/ITL,
                           agentic eval, conversation, coherence, RAGAS composite

  testing/               Meta-testing utilities
    flaky.py               detect_flaky -- run N times, compute pass rate
    golden.py              save_golden, load_golden, assert_matches_golden
    retry.py               retry_until_confident -- Wilson CI early-exit
    selection.py           build_test_map, select_affected_tests

  pytest_plugin/         Automatic pytest integration (entry-point registered)
  cli/                   Typer CLI (mltk doctor, mltk test, mltk report, ...)
  report/                HTML report generation (Plotly + Jinja2)
  server/                Self-hosted results platform (FastAPI + SQLite)
  integrations/          Jira, GitHub, Slack, Linear, Asana, MLflow
  registry/              Test resource registry (save/load collections)
  contracts/             YAML-based data contract definitions
  compliance/            EU AI Act, FDA audit trail, PDF export

rust/                    PyO3 Rust crate (_mltk_rust)
  Cargo.toml               pyo3 0.28, regex
  src/lib.rs               KS, PSI, KL, chi-squared, JS, Wasserstein,
                           cosine similarity, BERTScore P/R/F1, PII scan
```

### The Assertion Pattern

Every assertion in mltk follows the same contract:

```text
@timed_assertion       Wraps the function, measures wall-clock time (ms)
def assert_*(...)      Performs validation logic
    return assert_true(    Builds TestResult, raises on CRITICAL failure
        condition,
        name="module.check_name",
        message="human-readable outcome",
        severity=Severity.CRITICAL,
        **details,
    )
```

This uniformity means:

- Every assertion returns a `TestResult` (or raises `MltkAssertionError`).
- Every assertion is timed automatically.
- Every assertion stores structured details for reporting.
- Assertions compose into `TestSuite` objects for batch analysis.

### Config Cascade

`MltkConfig` resolves configuration with this priority (highest wins):

```text
MLTK_* environment variables
       |
function arguments
       |
mltk.yaml (project root)
       |
pyproject.toml [tool.mltk]
       |
built-in defaults
```

Supported environment variables:

| Variable | Type | Default |
|----------|------|---------|
| `MLTK_DRIFT_METHOD` | `str` | `"ks"` |
| `MLTK_DRIFT_THRESHOLD` | `float` | `0.05` |
| `MLTK_REPORT_DIR` | `str` | `"./mltk-reports"` |
| `MLTK_REPORT_FORMAT` | `str` | `"html"` |
| `MLTK_BASELINE_DIR` | `str` | `"./mltk-baselines"` |
| `MLTK_SEED` | `int` | `42` |
| `MLTK_PII_PATTERNS` | `str` (comma-separated) | `"email,phone,ssn,credit_card"` |

### Rust Acceleration Bridge

The `_rust.py` module wraps 10 compute-heavy functions in a transparent fallback pattern:

```python
if RUST_AVAILABLE:
    return _ks_test_rust(reference, current)      # fastest
try:
    from scipy.stats import ks_2samp              # scipy fallback
    ...
except ImportError:
    # pure numpy fallback                         # always works
```

Users never need to think about which backend runs. The bridge auto-selects the fastest available implementation.

---

## 2. Writing Custom Assertions

### Step-by-Step Guide

#### Step 1: Create the function skeleton

```python
# src/mltk/data/my_module.py  (or your own package)
from __future__ import annotations

import numpy as np

from mltk.core.assertion import assert_true, timed_assertion
from mltk.core.result import Severity, TestResult
```

#### Step 2: Write the assertion function

```python
@timed_assertion
def assert_feature_variance(
    data: np.ndarray,
    min_variance: float = 1e-4,
    severity: Severity = Severity.CRITICAL,
) -> TestResult:
    """Assert that all features have non-trivial variance.

    A feature column with near-zero variance carries no discriminative
    signal and may indicate a data pipeline bug (e.g., a constant column
    that should have been populated by an upstream source).

    Args:
        data: 2-D array of shape (n_samples, n_features).
        min_variance: Minimum acceptable per-column variance.
        severity: Severity level for the assertion.

    Returns:
        TestResult with per-column variance details.

    Example:
        >>> import numpy as np
        >>> data = np.random.randn(100, 5)
        >>> result = assert_feature_variance(data, min_variance=1e-4)
        >>> assert result.passed
    """
    variances = np.var(data, axis=0)
    dead_cols = [i for i, v in enumerate(variances) if v < min_variance]
    passed = len(dead_cols) == 0

    return assert_true(
        passed,
        name="data.feature_variance",
        message=(
            f"All {data.shape[1]} features have variance >= {min_variance}"
            if passed
            else f"{len(dead_cols)} feature(s) have near-zero variance: columns {dead_cols}"
        ),
        severity=severity,
        dead_columns=dead_cols,
        min_variance=min_variance,
        variances=variances.tolist(),
    )
```

#### Step 3: Understand the decorator

`@timed_assertion` wraps your function to:

1. Record `time.perf_counter()` before the call.
2. Call your function normally.
3. Set `result.duration_ms` on the returned `TestResult`.
4. Return the result (or let `MltkAssertionError` propagate).

The decorator is transparent -- the function signature, docstring, and return type are preserved via `@functools.wraps`.

#### Step 4: Understand `assert_true()`

```python
def assert_true(
    condition: bool,        # True = test passes
    name: str,              # Dotted identifier: "module.check_name"
    message: str,           # Human-readable outcome description
    severity: Severity,     # CRITICAL | WARNING | INFO
    **details: Any,         # Arbitrary key-value pairs for reporting
) -> TestResult:
```

Behavior by severity:

| Severity | On Failure | On Pass |
|----------|-----------|---------|
| `CRITICAL` | Raises `MltkAssertionError` | Returns `TestResult(passed=True)` |
| `WARNING` | Returns `TestResult(passed=False)` | Returns `TestResult(passed=True)` |
| `INFO` | Returns `TestResult(passed=False)` | Returns `TestResult(passed=True)` |

This means:

- **CRITICAL** assertions halt execution on failure (like `assert` in Python).
- **WARNING** assertions flag issues without stopping the test suite.
- **INFO** assertions are purely informational metrics.

#### Step 5: The Details Dict

The `**details` kwargs are stored in `TestResult.details` and appear in:

- HTML reports (as a table of key-value pairs)
- JSON exports (serialized alongside the result)
- Jupyter notebook rich display (`_repr_html_`)
- MLflow metrics (logged per-test)
- Server API responses

Include any data that helps diagnose failures:

```python
return assert_true(
    passed,
    name="model.accuracy",
    message=f"Accuracy {value:.4f} >= {threshold}",
    severity=Severity.CRITICAL,
    # All of these end up in TestResult.details:
    value=value,
    threshold=threshold,
    num_samples=len(y_true),
    class_distribution=dict(zip(*np.unique(y_true, return_counts=True))),
)
```

### Complete Example: Custom Assertion from Scratch

Here is a complete, self-contained assertion for validating that a time-series dataset has no large gaps:

```python
"""Time-series gap detection for sensor/IoT data pipelines."""
from __future__ import annotations

import numpy as np
import pandas as pd

from mltk.core.assertion import assert_true, timed_assertion
from mltk.core.result import Severity, TestResult


@timed_assertion
def assert_no_time_gaps(
    timestamps: pd.Series,
    max_gap_seconds: float = 3600.0,
    severity: Severity = Severity.CRITICAL,
) -> TestResult:
    """Assert that a time-series has no gaps exceeding max_gap_seconds.

    Useful for sensor data pipelines where missing readings indicate
    hardware failure or ingestion bugs.

    Args:
        timestamps: Pandas Series of datetime values (sorted or unsorted).
        max_gap_seconds: Maximum allowed gap between consecutive timestamps.
        severity: Severity level for the assertion.

    Returns:
        TestResult with gap statistics.

    Example:
        >>> import pandas as pd
        >>> ts = pd.Series(pd.date_range("2024-01-01", periods=100, freq="1min"))
        >>> result = assert_no_time_gaps(ts, max_gap_seconds=120)
        >>> assert result.passed
    """
    sorted_ts = timestamps.sort_values().reset_index(drop=True)
    diffs = sorted_ts.diff().dt.total_seconds().dropna()

    if len(diffs) == 0:
        return assert_true(
            True,
            name="data.time_gaps",
            message="No gaps to check (single timestamp)",
            severity=severity,
        )

    max_gap = float(diffs.max())
    num_violations = int((diffs > max_gap_seconds).sum())
    passed = num_violations == 0

    return assert_true(
        passed,
        name="data.time_gaps",
        message=(
            f"No gaps exceed {max_gap_seconds}s (max observed: {max_gap:.1f}s)"
            if passed
            else f"{num_violations} gap(s) exceed {max_gap_seconds}s (max: {max_gap:.1f}s)"
        ),
        severity=severity,
        max_gap_seconds=round(max_gap, 2),
        threshold_seconds=max_gap_seconds,
        num_violations=num_violations,
        total_intervals=len(diffs),
    )
```

Use it in a test:

```python
import pandas as pd
import pytest

from my_assertions import assert_no_time_gaps
from mltk.core.assertion import MltkAssertionError


class TestTimeGaps:
    def test_continuous_data_passes(self) -> None:
        """PASS: 1-minute interval data with no gaps.

        WHY: Validates that healthy sensor data passes the gap check.
        Expected: TestResult with passed=True.
        """
        ts = pd.Series(pd.date_range("2024-01-01", periods=100, freq="1min"))
        result = assert_no_time_gaps(ts, max_gap_seconds=120)
        assert result.passed is True
        assert result.duration_ms > 0  # timing was recorded

    def test_missing_data_fails(self) -> None:
        """FAIL: 1-hour gap in 1-minute data.

        WHY: Catches sensor outages in IoT pipelines.
        Expected: MltkAssertionError raised.
        """
        ts = pd.Series([
            *pd.date_range("2024-01-01 00:00", periods=30, freq="1min"),
            *pd.date_range("2024-01-01 02:00", periods=30, freq="1min"),
        ])
        with pytest.raises(MltkAssertionError) as exc_info:
            assert_no_time_gaps(ts, max_gap_seconds=120)
        assert exc_info.value.result.details["num_violations"] == 1
```

---

## 3. Plugin System

mltk supports third-party assertion plugins via a naming convention and decorator-based registry.

### `@register_assertion` Decorator

Register any callable as a discoverable assertion:

```python
from mltk.core.plugin import register_assertion
from mltk.core.assertion import assert_true, timed_assertion
from mltk.core.result import Severity, TestResult


# Option 1: Explicit name
@register_assertion("finance.sharpe_ratio")
@timed_assertion
def assert_sharpe_ratio(returns, min_ratio=1.0) -> TestResult:
    """Assert portfolio Sharpe ratio exceeds minimum."""
    import numpy as np

    mean_return = np.mean(returns)
    std_return = np.std(returns)
    sharpe = mean_return / std_return if std_return > 0 else 0.0

    return assert_true(
        sharpe >= min_ratio,
        name="finance.sharpe_ratio",
        message=f"Sharpe ratio {sharpe:.4f} >= {min_ratio}",
        severity=Severity.CRITICAL,
        sharpe_ratio=round(sharpe, 4),
        threshold=min_ratio,
    )


# Option 2: Inferred name (uses function.__name__)
@register_assertion()
@timed_assertion
def assert_portfolio_diversified(weights, max_concentration=0.4) -> TestResult:
    ...


# Option 3: No parentheses (shorthand)
@register_assertion
@timed_assertion
def assert_drawdown(equity_curve, max_drawdown=0.2) -> TestResult:
    ...
```

### Querying the Registry

```python
from mltk.core.plugin import get_registered_assertions

# After imports/discovery, all registered assertions are available:
assertions = get_registered_assertions()
print(assertions.keys())
# dict_keys(['finance.sharpe_ratio', 'assert_portfolio_diversified', 'assert_drawdown'])

# Call any registered assertion by name:
result = assertions["finance.sharpe_ratio"]([0.01, 0.02, -0.005, 0.03])
```

### `discover_plugins()` -- Auto-Discovery

mltk auto-discovers installed packages whose name starts with `mltk_plugin_`:

```python
from mltk.core.plugin import discover_plugins

discovered = discover_plugins()
# Returns: ['mltk_plugin_finance', 'mltk_plugin_audio']
```

Under the hood, `discover_plugins()`:

1. Scans `importlib.metadata.packages_distributions()` for package names matching the prefix.
2. Imports each matching package (`importlib.import_module(pkg)`).
3. Importing triggers `@register_assertion` decorators at module level.
4. Returns the list of successfully imported package names.

### Creating a Plugin Package

#### Directory structure

```text
mltk-plugin-finance/
  pyproject.toml
  src/
    mltk_plugin_finance/
      __init__.py       # Assertions registered here on import
      sharpe.py
      drawdown.py
```

#### `pyproject.toml`

```toml
[project]
name = "mltk-plugin-finance"
version = "0.1.0"
description = "Financial ML assertions for mltk"
requires-python = ">=3.10"
dependencies = ["mltk>=0.6.0", "numpy>=1.24"]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]
```

#### `src/mltk_plugin_finance/__init__.py`

```python
"""mltk plugin -- financial ML assertions."""

from mltk.core.plugin import register_assertion
from mltk.core.assertion import assert_true, timed_assertion
from mltk.core.result import Severity, TestResult

# Importing this module automatically registers all assertions below.


@register_assertion("finance.sharpe_ratio")
@timed_assertion
def assert_sharpe_ratio(returns, min_ratio=1.0) -> TestResult:
    """Assert Sharpe ratio meets minimum threshold."""
    import numpy as np
    ratio = float(np.mean(returns) / max(np.std(returns), 1e-10))
    return assert_true(
        ratio >= min_ratio,
        name="finance.sharpe_ratio",
        message=f"Sharpe {ratio:.4f} >= {min_ratio}",
        severity=Severity.CRITICAL,
        sharpe_ratio=round(ratio, 4),
    )


@register_assertion("finance.max_drawdown")
@timed_assertion
def assert_max_drawdown(equity_curve, max_dd=0.2) -> TestResult:
    """Assert maximum drawdown does not exceed threshold."""
    import numpy as np
    peak = np.maximum.accumulate(equity_curve)
    dd = (peak - equity_curve) / np.where(peak > 0, peak, 1.0)
    max_observed = float(np.max(dd))
    return assert_true(
        max_observed <= max_dd,
        name="finance.max_drawdown",
        message=f"Max drawdown {max_observed:.4f} <= {max_dd}",
        severity=Severity.CRITICAL,
        max_drawdown=round(max_observed, 4),
    )
```

#### Publishing

```bash
# Build
python -m build

# Publish to PyPI (or private index)
twine upload dist/*

# Users install it:
pip install mltk-plugin-finance

# Auto-discovered:
from mltk.core.plugin import discover_plugins
discover_plugins()  # ['mltk_plugin_finance']
```

### YAML Integration

Registered plugin assertions are automatically available in YAML test definitions. The YAML runner checks the plugin registry as a fallback whenever an assertion key does not match a built-in assertion -- no additional wiring required.

```yaml
# mltk-tests.yaml
data_source: data/portfolio.csv

tests:
  - name: "Sharpe ratio above 1.0"
    assertion: finance.sharpe_ratio
    params:
      min_ratio: 1.0
```

Plugin assertion functions can accept the loaded DataFrame via a `df` keyword argument. If the function does not accept `df`, the runner retries with only the YAML-provided params. See the [YAML Test Definitions API](../api/yaml-tests.md#custom-assertions-via-plugins) for full details.

---

## 4. Python API Reference

### Core Types

```python
from mltk.core import (
    TestResult,           # Single assertion result (dataclass)
    TestSuite,            # Collection of results with .score, .passed, etc.
    Severity,             # Enum: CRITICAL, WARNING, INFO
    MltkConfig,           # Configuration loader
    MltkAssertionError,   # Raised on CRITICAL failure
    assert_true,          # Base assertion function
    timed_assertion,      # Timing decorator
    register_assertion,   # Plugin registry decorator
    discover_plugins,     # Auto-discover installed plugins
    get_registered_assertions,  # Query the registry
)
```

**TestResult fields:**

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Dotted test identifier |
| `passed` | `bool` | Whether the assertion passed |
| `severity` | `Severity` | CRITICAL, WARNING, or INFO |
| `message` | `str` | Human-readable outcome |
| `details` | `dict[str, Any]` | Structured diagnostic data |
| `duration_ms` | `float` | Wall-clock execution time |
| `timestamp` | `datetime` | When the test ran |

**TestSuite properties:**

| Property | Type | Description |
|----------|------|-------------|
| `results` | `list[TestResult]` | All collected results |
| `passed` | `bool` | True if all CRITICAL tests passed |
| `total` | `int` | Total number of results |
| `passed_count` | `int` | Number of passing results |
| `failed_count` | `int` | Number of failing results |
| `score` | `float` | Pass rate as percentage (0-100) |

Both `TestResult` and `TestSuite` implement `_repr_html_()` for rich display in Jupyter notebooks.

### Data Assertions

```python
from mltk.data import (
    # Schema validation
    assert_schema,           # Column names match expected set
    assert_no_nulls,         # No null/NaN values in specified columns
    assert_dtypes,           # Column dtypes match expected mapping

    # Distribution and range
    assert_range,            # Values within [min, max] bounds
    assert_unique,           # Column has no duplicates
    assert_no_outliers,      # No extreme outliers (IQR or z-score)

    # Drift detection
    assert_no_drift,         # Statistical drift (KS, PSI, chi-squared, JS, Wasserstein)
    assert_no_embedding_drift,  # Cosine-distance drift for embeddings

    # Freshness and volume
    assert_freshness,        # Data is recent enough
    assert_row_count,        # Row count within expected range

    # PII detection
    assert_no_pii,           # No PII patterns found
    scan_pii,                # Scan and return PII matches

    # Labels
    assert_label_balance,    # Class distribution within bounds
    assert_label_coverage,   # All expected labels are present

    # Statistics
    assert_column_mean,      # Column mean within range
    assert_column_median,    # Column median within range
    assert_column_stdev,     # Column stdev within range
    assert_quantiles,        # Quantile values match expectations

    # Validation
    assert_datetime_format,  # Datetime strings match expected format
    assert_values_in_set,    # Values from an allowed set
    assert_no_conflicting_labels,  # Same features, different labels
    assert_feature_label_correlation_stable,

    # Lineage
    LineageGraph,            # DAG of data transformations
    track_lineage,           # Decorator: auto-record lineage nodes
    assert_lineage_complete, # Graph has expected number of steps

    # Preset (one-call sweep)
    assert_data_quality,     # Runs schema + nulls + range + drift + PII
    data_quality_report,     # Returns a TestSuite with all checks
)
```

### Model Assertions

```python
from mltk.model import (
    assert_metric,           # Any sklearn metric >= threshold
    assert_no_regression,    # Current metrics >= saved baseline
    save_baseline,           # Persist baseline metrics to disk
    assert_slice_performance,  # Per-slice metric thresholds
    assert_calibration,      # Probability calibration (Brier score)
    assert_no_bias,          # Demographic parity / equalized odds
    assert_robust,           # Adversarial perturbation robustness
    assert_no_overfitting,   # Train-val metric gap
    assert_label_drift,      # Prediction distribution shift
)
```

### Training Assertions

```python
from mltk.training import (
    # Gradient health
    assert_gradient_flow,        # No dead layers (mean |grad| > threshold)
    assert_no_vanishing_gradient,  # No layer gradient norms near zero
    assert_no_exploding_gradient,  # No gradient norms above ceiling
    assert_loss_finite,          # Loss is not NaN or Inf

    # Numerical stability
    assert_no_nan_inf,           # No NaN/Inf in tensors
    assert_loss_decreasing,      # Loss trend is downward
    assert_no_loss_divergence,   # Loss not diverging
    assert_softmax_valid,        # Softmax outputs sum to 1, all in [0,1]

    # Data leakage
    assert_no_train_test_overlap,  # No shared samples between splits
    assert_temporal_split,       # Time-ordered split is respected
    assert_no_target_leakage,    # Features do not leak target info

    # Checkpointing
    assert_checkpoint_complete,  # Checkpoint file contains all expected keys
    assert_resume_loss_continuous,  # Loss after resume matches pre-save value

    # Augmentation
    assert_no_augmentation_on_test,  # Test set is not augmented
    assert_augmentation_preserves_signal,  # Augmented data preserves labels

    # Distributed training
    assert_effective_batch_size,  # batch_size * world_size matches expected
    assert_gradient_sync,        # Gradients are synchronized across workers

    # Memory
    assert_no_memory_leak,       # Memory usage stable across iterations
    assert_loss_is_detached,     # Loss tensor is detached from graph

    # Serving skew
    assert_no_training_serving_skew,  # Feature distributions match
)
```

### Domain Kits

#### Computer Vision

```python
from mltk.domains.cv import (
    compute_iou,              # Raw IoU computation
    assert_iou,               # IoU >= threshold
    assert_map,               # Mean Average Precision
    assert_frame_accuracy,    # Video frame classification accuracy
    assert_temporal_consistency,  # Prediction stability across frames
    assert_topk_accuracy,     # Top-K classification accuracy
    assert_face_far,          # Face recognition False Acceptance Rate
    assert_mota,              # Multi-Object Tracking Accuracy
    assert_motp,              # Multi-Object Tracking Precision
    assert_idf1,              # ID F1 score for tracking
)
```

#### NLP

```python
from mltk.domains.nlp import (
    assert_bleu,              # BLEU translation quality
    assert_rouge,             # ROUGE summarization quality
    assert_ner_f1,            # Named Entity Recognition F1
    assert_no_prompt_injection,  # Prompt injection detection
    assert_sentiment_positive,   # Sentiment polarity check
    assert_no_sentiment_drift,   # Sentiment distribution stability
)
```

#### Speech

```python
from mltk.domains.speech import (
    assert_wer,               # Word Error Rate
    assert_cer,               # Character Error Rate
    assert_rtf,               # Real-Time Factor (speed)
    assert_accent_coverage,   # Coverage across accent groups
)
```

#### Tabular

```python
from mltk.domains.tabular import (
    assert_feature_drift,              # Per-feature drift detection
    assert_feature_importance_stable,  # Importance rankings stable
    assert_class_balance,              # Target class distribution
)
```

#### LLM Evaluation

```python
from mltk.domains.llm import (
    # Similarity and quality
    assert_bertscore,          # BERTScore P/R/F1
    assert_semantic_similarity,  # Embedding cosine similarity
    assert_coherence,          # Text coherence score

    # Safety
    assert_no_toxicity,        # Toxicity detection
    assert_no_hallucination,   # Factual grounding check

    # Latency
    assert_ttft,               # Time to First Token
    assert_itl,                # Inter-Token Latency

    # Text quality
    assert_text_length,        # Output length bounds
    assert_output_format,      # Format validation (JSON, etc.)
    assert_readability,        # Flesch-Kincaid readability

    # RAG evaluation
    assert_faithfulness,       # Answer grounded in context
    assert_context_relevancy,  # Retrieved context relevance
    assert_answer_relevancy,   # Answer relevance to question
    assert_context_precision,  # Precision of retrieved contexts
    assert_context_recall,     # Recall of relevant contexts

    # RAGAS composite
    compute_ragas_score,       # Compute all RAGAS sub-scores
    assert_ragas_score,        # Assert composite RAGAS score

    # Agentic evaluation
    assert_task_completion,    # Agent completed the task
    assert_tool_selection,     # Correct tool was chosen
    assert_tool_call_correctness,  # Tool call params are correct

    # Multi-turn conversation
    assert_knowledge_retention,     # Context maintained across turns
    assert_turn_relevancy,          # Each turn is relevant
    assert_conversation_completeness,  # Conversation reached resolution
)
```

### Inference and Pipeline

```python
from mltk.inference import (
    assert_latency,       # P50/P95/P99 latency thresholds
    assert_cold_start,    # First-request latency
    assert_throughput,    # Requests per second
    assert_api_contract,  # Output schema validation
)

from mltk.pipeline import (
    assert_reproducible,  # Same seed, same output
    assert_checksum,      # File/data integrity
    assert_pipeline,      # End-to-end pipeline validation
)
```

### Production Monitoring

```python
from mltk.monitor import (
    assert_no_degradation,     # Gradual metric decline detection
    assert_sla,                # Latency and error rate SLA
    assert_no_output_drift,    # Output distribution shift

    # On-prem / Prometheus
    assert_prometheus_metric,  # PromQL threshold check
    assert_gpu_utilization,    # GPU util via DCGM/Prometheus
    assert_triton_healthy,     # Triton inference server health
)

# Cloud providers (import directly -- heavy optional dependencies)
from mltk.monitor.aws import assert_endpoint_healthy, assert_endpoint_latency
from mltk.monitor.gcp import assert_prediction_latency
from mltk.monitor.azure import assert_endpoint_latency
```

### Import Patterns

```python
# Convenience: most common assertions at top level
from mltk import assert_schema, assert_no_drift, assert_metric

# Full module path for less common assertions
from mltk.training import assert_gradient_flow
from mltk.domains.llm import assert_bertscore

# Core types
from mltk.core import TestResult, TestSuite, Severity, MltkConfig

# Plugin system
from mltk.core.plugin import register_assertion, discover_plugins
```

---

## 5. Integrating with ML Frameworks

All mltk assertions accept **plain numpy arrays** or Python built-ins. This makes them framework-agnostic by design. Here is how to bridge each major framework.

### PyTorch

#### Gradient inspection

```python
import torch
import numpy as np
from mltk.training import (
    assert_gradient_flow,
    assert_no_vanishing_gradient,
    assert_no_exploding_gradient,
)


def test_gradient_health(model, loss_fn, sample_batch):
    """Run one forward/backward pass and check gradient health."""
    x, y = sample_batch
    output = model(x)
    loss = loss_fn(output, y)
    loss.backward()

    # Extract gradients as numpy arrays
    gradients = [
        p.grad.detach().cpu().numpy()
        for p in model.parameters()
        if p.grad is not None
    ]

    assert_gradient_flow(gradients, min_mean_grad=1e-7)
    assert_no_vanishing_gradient(gradients, threshold=1e-6)
    assert_no_exploding_gradient(gradients, threshold=1e3)
```

#### Checkpoint validation

```python
import torch
from mltk.training import assert_checkpoint_complete

def test_checkpoint_integrity(checkpoint_path):
    """Verify checkpoint contains all required keys."""
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    assert_checkpoint_complete(
        checkpoint,
        required_keys=["model_state_dict", "optimizer_state_dict", "epoch", "loss"],
    )
```

#### Memory leak detection

```python
from mltk.training import assert_no_memory_leak

def test_training_memory(model, train_loader, loss_fn, optimizer):
    """Run several training steps and check for memory leaks."""
    memory_snapshots = []
    for i, (x, y) in enumerate(train_loader):
        if i >= 20:
            break
        output = model(x)
        loss = loss_fn(output, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Record memory at each step
        memory_snapshots.append(torch.cuda.memory_allocated())

    assert_no_memory_leak(memory_snapshots, max_growth_factor=1.2)
```

### TensorFlow / Keras

```python
import numpy as np
from mltk.model import assert_metric, assert_no_overfitting

def test_keras_model(model, x_test, y_test, x_train, y_train):
    """Validate a trained Keras model."""
    # Get predictions as numpy
    y_pred = (model.predict(x_test) > 0.5).astype(int).flatten()
    y_pred_train = (model.predict(x_train) > 0.5).astype(int).flatten()

    # Metric check
    assert_metric(
        y_true=y_test.numpy() if hasattr(y_test, 'numpy') else y_test,
        y_pred=y_pred,
        metric="f1",
        threshold=0.85,
    )

    # Overfitting check: compare train vs validation performance
    from sklearn.metrics import accuracy_score
    train_acc = accuracy_score(y_train, y_pred_train)
    val_acc = accuracy_score(y_test, y_pred)

    assert_no_overfitting(
        train_metric=train_acc,
        val_metric=val_acc,
        max_gap=0.1,
    )
```

### scikit-learn

```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from mltk.model import assert_metric, assert_slice_performance
from mltk.training import assert_no_train_test_overlap
from mltk.data import assert_no_drift

def test_sklearn_pipeline(X, y):
    """End-to-end validation of a scikit-learn pipeline."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Verify no data leakage
    assert_no_train_test_overlap(
        train_ids=list(range(len(X_train))),
        test_ids=list(range(len(X_train), len(X_train) + len(X_test))),
    )

    # Check feature drift between splits
    for col in range(X.shape[1]):
        assert_no_drift(
            reference=X_train[:, col].tolist(),
            current=X_test[:, col].tolist(),
            method="ks",
            threshold=0.1,
        )

    # Train and evaluate
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    assert_metric(y_true=y_test, y_pred=y_pred, metric="accuracy", threshold=0.9)
```

### Hugging Face Transformers

```python
from mltk.domains.llm import (
    assert_bertscore,
    assert_no_toxicity,
    assert_text_length,
    assert_faithfulness,
)

def test_llm_output(model, tokenizer, prompts, reference_answers):
    """Validate LLM generation quality."""
    for prompt, ref in zip(prompts, reference_answers):
        inputs = tokenizer(prompt, return_tensors="pt")
        output_ids = model.generate(**inputs, max_new_tokens=256)
        generated = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        # Quality checks
        assert_text_length(generated, min_length=20, max_length=500)
        assert_no_toxicity(generated, threshold=0.1)

        # BERTScore against reference
        assert_bertscore(
            reference=ref,
            hypothesis=generated,
            threshold=0.85,
        )
```

### MLflow Integration

mltk integrates natively with MLflow for logging test results as experiment metrics.

#### Via pytest (automatic)

```bash
# Log all test results to MLflow experiment "model-quality"
pytest tests/ --mltk-mlflow "model-quality"

# With a remote tracking server
MLFLOW_TRACKING_URI=http://mlflow.internal:5000 \
  pytest tests/ --mltk-mlflow "production-gate"
```

#### Via Python API (programmatic)

```python
from mltk.core.result import TestSuite
from mltk.integrations.mlflow_logger import MlflowLogger

# Collect results into a suite
suite = TestSuite()
suite.add(assert_metric(y_true, y_pred, metric="f1", threshold=0.9))
suite.add(assert_no_drift(ref_data, cur_data))

# Log to MLflow
logger = MlflowLogger(experiment_name="my-model-tests")
logger.log_results(suite)

# Attach HTML report as artifact
logger.log_report("mltk-reports/report.html")

# Log individual results
result = assert_metric(y_true, y_pred, metric="accuracy", threshold=0.95)
logger.log_test_result(result)
```

Logged metrics:

| Metric | Description |
|--------|-------------|
| `mltk_total_tests` | Total assertion count |
| `mltk_passed` | Number passing |
| `mltk_failed` | Number failing |
| `mltk_score` | Pass rate (0-100) |
| `mltk_duration_ms` | Total execution time |
| `mltk_{test_name}` | Per-test: 1.0 (pass) or 0.0 (fail) |

---

## 6. Server API Client

mltk ships a self-hosted test results platform (FastAPI + SQLite). Think of it as a lightweight quality dashboard for ML tests.

### Starting the Server

```bash
# Install server dependencies
pip install "mltk[server]"

# Start the server
uvicorn mltk.server.app:create_app --factory --host 0.0.0.0 --port 8080
```

### Auto-Push via pytest

```bash
# Push results to a running server after every test session
pytest tests/ --mltk-server http://localhost:8080
```

### Python Client (programmatic)

```python
import json
import urllib.request


class MltkServerClient:
    """Minimal client for the mltk server API."""

    def __init__(self, base_url: str, api_key: str | None = None):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _get(self, path: str) -> dict:
        req = urllib.request.Request(
            f"{self.base_url}{path}",
            headers=self._headers(),
        )
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read())

    def _post(self, path: str, data: dict) -> dict:
        payload = json.dumps(data).encode()
        req = urllib.request.Request(
            f"{self.base_url}{path}",
            data=payload,
            headers=self._headers(),
            method="POST",
        )
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read())

    def submit_results(self, project: str, results: list[dict]) -> dict:
        """Submit test results from a run.

        Args:
            project: Project identifier.
            results: List of result dicts with keys:
                name, passed, severity, message, details, duration_ms.

        Returns:
            Server response with run_id.
        """
        return self._post("/api/runs", {"project": project, "results": results})

    def get_trends(self, project: str, limit: int = 20) -> dict:
        """Fetch score trends for a project."""
        return self._get(f"/api/trends/{project}?limit={limit}")

    def get_summary(self, project: str, limit: int = 20) -> dict:
        """Fetch test history summary with failure analysis."""
        return self._get(f"/api/summary/{project}?limit={limit}")

    def compare_runs(self, run_a: int, run_b: int) -> dict:
        """Compare two test runs and get a structured diff."""
        return self._get(f"/api/compare?run_a={run_a}&run_b={run_b}")

    def list_runs(self, project: str | None = None, limit: int = 50) -> list:
        """List recent test runs."""
        url = f"/api/runs?limit={limit}"
        if project:
            url += f"&project={project}"
        return self._get(url)["runs"]
```

#### Usage Example

```python
from mltk.core.result import TestSuite
from mltk.data import assert_schema, assert_no_nulls
import pandas as pd

# Run assertions
df = pd.DataFrame({"age": [25, 30, 35], "name": ["A", "B", "C"]})
suite = TestSuite()
suite.add(assert_schema(df, expected_columns=["age", "name"]))
suite.add(assert_no_nulls(df, columns=["age", "name"]))

# Convert to server-compatible records
records = [
    {
        "name": r.name,
        "passed": r.passed,
        "severity": r.severity.value,
        "message": r.message,
        "details": r.details,
        "duration_ms": r.duration_ms,
    }
    for r in suite.results
]

# Submit to server
client = MltkServerClient("http://localhost:8080", api_key="mltk_your_key_here")
response = client.submit_results("my-project", records)
print(f"Run ID: {response['run_id']}")

# Fetch trends
trends = client.get_trends("my-project")
for t in trends["trends"]:
    print(f"Run {t['id']}: score={t['score']}%")
```

### Server API Endpoints

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `GET` | `/health` | No | Health check |
| `POST` | `/api/runs` | Bearer | Submit test run results |
| `GET` | `/api/runs` | No | List recent runs |
| `GET` | `/api/runs/{id}` | No | Get run details |
| `GET` | `/api/trends/{project}` | No | Score trends over time |
| `GET` | `/api/summary/{project}` | No | Test history summary |
| `GET` | `/api/compare?run_a=X&run_b=Y` | No | Diff two runs |
| `POST` | `/api/webhooks` | Bearer | Register a webhook |
| `GET` | `/api/webhooks` | No | List webhooks |
| `DELETE` | `/api/webhooks/{id}` | Bearer | Remove a webhook |

### Webhooks

Register webhooks to receive notifications on test events:

```python
client._post("/api/webhooks", {
    "url": "https://hooks.slack.com/services/...",
    "events": ["on_failure"],
    "project": "production",
})
```

Events: `on_success`, `on_failure`.

---

## 7. Rust Acceleration

### When Rust is Used

The Rust extension accelerates 10 compute-intensive functions:

| Function | Purpose | Fallback |
|----------|---------|----------|
| `ks_test` | Kolmogorov-Smirnov test | scipy `ks_2samp`, then `ImportError` |
| `psi` | Population Stability Index | numpy histogram |
| `kl_divergence` | KL divergence | numpy histogram |
| `chi_squared` | Chi-squared test | scipy `chisquare`, then numpy+Wilson-Hilferty |
| `js_divergence` | Jensen-Shannon divergence | numpy histogram |
| `wasserstein` | Earth Mover's Distance | scipy `wasserstein_distance`, then numpy CDF |
| `cosine_similarity` | Cosine similarity | numpy dot product |
| `centroid_cosine_distance` | Centroid cosine distance | numpy |
| `bertscore_precision_recall` | BERTScore P/R/F1 | numpy nested loops |
| `scan_pii_rust` | PII regex scanning | Python `re` module |

### Building from Source

```bash
# Prerequisites
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
pip install maturin

# Development build (fast, unoptimized)
maturin develop

# Release build (optimized, LTO enabled)
maturin develop --release

# Verify
python -c "from mltk._rust import RUST_AVAILABLE; print(RUST_AVAILABLE)"
```

### Adding a New Rust Function

#### Step 1: Implement in `rust/src/lib.rs`

```rust
/// Compute the Gini impurity of a label distribution.
#[pyfunction]
fn gini_impurity(labels: Vec<i64>) -> f64 {
    if labels.is_empty() {
        return 0.0;
    }
    let n = labels.len() as f64;
    let mut counts: std::collections::HashMap<i64, usize> =
        std::collections::HashMap::new();
    for &label in &labels {
        *counts.entry(label).or_insert(0) += 1;
    }
    let sum_sq: f64 = counts.values()
        .map(|&c| (c as f64 / n).powi(2))
        .sum();
    1.0 - sum_sq
}
```

#### Step 2: Register in the PyO3 module

```rust
#[pymodule]
fn _mltk_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // ... existing functions ...
    m.add_function(wrap_pyfunction!(gini_impurity, m)?)?;
    Ok(())
}
```

#### Step 3: Add the Python bridge in `src/mltk/_rust.py`

```python
try:
    from mltk._mltk_rust import gini_impurity as _gini_impurity_rust
    RUST_AVAILABLE = True
except ImportError:
    pass


def gini_impurity(labels: list[int]) -> float:
    """Compute Gini impurity of a label distribution.

    Args:
        labels: List of integer class labels.

    Returns:
        Gini impurity in [0, 1]. 0 = pure, 1 = maximally impure.
    """
    if RUST_AVAILABLE:
        return _gini_impurity_rust(labels)
    # Pure Python fallback
    from collections import Counter
    n = len(labels)
    if n == 0:
        return 0.0
    counts = Counter(labels)
    return 1.0 - sum((c / n) ** 2 for c in counts.values())
```

#### Step 4: Build and test

```bash
maturin develop --release
python -c "from mltk._rust import gini_impurity; print(gini_impurity([0, 0, 1, 1, 2]))"
```

### Fallback Behavior

The fallback chain is:

```text
Rust extension (fastest, compiled with LTO)
    |-- not available -->
scipy (if installed)
    |-- not available -->
pure numpy / Python stdlib (always works)
```

Users who install mltk with `pip install mltk` get numpy-based fallbacks. The Rust extension is compiled into the wheel on PyPI for supported platforms. Building from source gives users the Rust path even on unsupported platforms.

### Memory Behavior at the PyO3 Boundary

When calling Rust functions, PyO3 converts Python `list[float]` into a Rust
`Vec<f64>`.  This is a **full copy** --- peak memory during the call is
roughly **2x the input array size**.  For the three highest-impact hot-path
functions (`ks_test`, `psi`, `cosine_similarity`) this matters at scale:

| Array size | Copy overhead | Impact |
|------------|---------------|--------|
| < 10 000   | < 80 KB       | Negligible |
| 100 000    | ~800 KB       | Acceptable |
| 1 000 000  | ~8 MB         | Noticeable |
| 2 000 000+ | 16 MB+        | Monitor peak RSS |

**Architecture:**

The Rust crate separates each hot-path function into two layers:

1. **Core function** (`ks_test_core`, `psi_core`, `cosine_similarity_core`) ---
   operates on borrowed `&[f64]` / `&mut [f64]` slices with zero internal
   allocation beyond what the algorithm needs.  These are also used by
   `centroid_cosine_distance` and `bertscore_precision_recall` internally,
   eliminating redundant `Vec` clones.
2. **PyO3 wrapper** (`#[pyfunction] fn ks_test(...)`) --- accepts `Vec<f64>`
   from the Python boundary and delegates to the core.  The Vec is the *only*
   copy; the core never re-copies.

**Python bridge (`_rust.py`):**

The `_to_list()` helper avoids a redundant copy when the caller already
passes a plain `list`.  If a numpy array is passed, `.tolist()` is called
once.  The converted list is reused for both the Rust path and fallback paths.

**If memory is critical at 2M+ rows:**

- Use the pure-numpy fallback by setting `mltk._rust.RUST_AVAILABLE = False`
  (numpy operates on memory-mapped buffers without a full copy).
- Process data in chunks (e.g., sample 100K rows for drift tests).
- Adding PyO3 numpy buffer protocol (`pyo3/numpy` feature) for true zero-copy
  is on the roadmap but not yet implemented to keep the dependency footprint
  minimal.

---

## 8. Testing Your Tests

mltk provides meta-testing utilities in `mltk.testing` for dealing with the inherent non-determinism of ML tests.

### Flaky Test Detection

```python
from mltk.testing import detect_flaky, FlakySummary

def my_stochastic_test():
    """A test that sometimes fails due to randomness."""
    import numpy as np
    predictions = np.random.rand(100) > 0.05  # ~5% false rate
    assert predictions.all()

# Run 20 times and check stability
summary: FlakySummary = detect_flaky(
    my_stochastic_test,
    runs=20,
    threshold=0.8,  # pass rate below 80% = flaky
)

print(f"Pass rate: {summary.pass_rate:.1%}")
print(f"Is flaky: {summary.is_flaky}")
# FlakySummary(test_name='my_stochastic_test', pass_count=14,
#              fail_count=6, pass_rate=0.7, is_flaky=True)
```

A test is flaky when `0 < pass_rate < threshold`. Always-pass (1.0) and always-fail (0.0) are not flaky -- they are stable or broken.

### Golden Test Sets

Version and compare baseline data across releases:

```python
from mltk.testing import save_golden, load_golden, assert_matches_golden
import numpy as np

# Save a baseline once (version-tracked in your repo)
save_golden(
    data={"accuracy": 0.934, "f1": 0.891, "auc": 0.967},
    path="tests/golden/model_v2.json",
    version="2.0.0",
)

# In tests, compare current results against the golden baseline
current_metrics = {"accuracy": 0.936, "f1": 0.889, "auc": 0.965}
result = assert_matches_golden(
    current=current_metrics,
    golden_path="tests/golden/model_v2.json",
    tolerance=0.01,  # max absolute deviation per metric
)
# Passes: max_diff=0.002 <= 0.01

# Works with numpy arrays too
current_preds = np.array([0.1, 0.9, 0.8, 0.2])
save_golden(current_preds, "tests/golden/predictions.json")
assert_matches_golden(current_preds, "tests/golden/predictions.json", tolerance=0.05)
```

The golden file format:

```json
{
  "version": "2.0.0",
  "timestamp": "2025-03-26T10:30:00",
  "data": {"accuracy": 0.934, "f1": 0.891, "auc": 0.967}
}
```

### Non-Deterministic Retry

For tests with inherent randomness, use statistical confidence intervals instead of brittle single-run assertions:

```python
from mltk.testing import retry_until_confident, RetryResult

def stochastic_model_check():
    """Check that a stochastic model predicts above threshold."""
    import numpy as np
    preds = np.random.normal(0.9, 0.05, size=100)
    accuracy = (preds > 0.5).mean()
    assert accuracy > 0.85

result: RetryResult = retry_until_confident(
    stochastic_model_check,
    min_runs=3,       # at least 3 runs before making a verdict
    max_runs=10,      # stop after 10 runs regardless
    confidence=0.95,  # 95% Wilson confidence interval
    failure_threshold=0.5,  # CI lower bound must exceed this
)

print(f"Pass rate: {result.pass_rate:.1%}")
print(f"95% CI: [{result.confidence_lower:.3f}, {result.confidence_upper:.3f}]")
print(f"Verdict: {'PASSING' if result.is_passing else 'FAILING'}")
```

The function uses Wilson score confidence intervals (no scipy dependency) and supports early termination when the verdict is statistically clear.

### Smart Test Selection

Only re-run tests affected by your code changes:

```python
from mltk.testing import build_test_map, select_affected_tests

# Build a dependency graph (test_file -> source_files it imports)
test_map = build_test_map(test_dir="tests/", src_dir="src/")

# Given a list of changed files (e.g., from git diff):
changed = ["src/mltk/data/drift.py", "src/mltk/data/schema.py"]
affected = select_affected_tests(changed, test_map)
# ['tests/test_data/test_drift.py', 'tests/test_data/test_schema.py']

# Run only affected tests:
import subprocess
if affected:
    subprocess.run(["pytest", *affected, "-q"])
```

The dependency map is built using `ast.parse` to trace import statements. It works across the entire source tree without executing any code.

---

## 9. Data Lineage

Track and verify data transformations from raw ingestion to model-ready features.

### `@track_lineage` Decorator

Automatically record input/output hashes for every function call:

```python
from mltk.data import LineageGraph, track_lineage, assert_lineage_complete

# Create a lineage graph
graph = LineageGraph()

# Decorate pipeline functions
@track_lineage(graph, "load_raw")
def load_raw(path):
    import pandas as pd
    return pd.read_csv(path)

@track_lineage(graph, "clean")
def clean(df):
    return df.dropna()

@track_lineage(graph, "normalize")
def normalize(df):
    return (df - df.mean()) / df.std()

@track_lineage(graph, "split")
def split(df, ratio=0.8):
    n = int(len(df) * ratio)
    return df.iloc[:n], df.iloc[n:]

# Run the pipeline
raw = load_raw("data/features.csv")
cleaned = clean(raw)
normalized = normalize(cleaned)
train, test = split(normalized)

# Verify the pipeline executed all expected steps
assert_lineage_complete(graph, expected_steps=4)
```

### LineageGraph

The `LineageGraph` records each transformation as a `LineageNode`:

```python
for node in graph.nodes:
    print(f"{node.name}: {node.input_hash} -> {node.output_hash} @ {node.timestamp}")
# load_raw:  a1b2c3d4e5f6 -> f6e5d4c3b2a1 @ 2025-03-26T10:00:00+00:00
# clean:     f6e5d4c3b2a1 -> 1a2b3c4d5e6f @ 2025-03-26T10:00:01+00:00
# normalize: 1a2b3c4d5e6f -> 6f5e4d3c2b1a @ 2025-03-26T10:00:01+00:00
# split:     6f5e4d3c2b1a -> 9a8b7c6d5e4f @ 2025-03-26T10:00:02+00:00
```

Each node stores:

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Human-readable transformation label |
| `input_hash` | `str` | SHA-256 hash of stringified input (first 12 chars) |
| `output_hash` | `str` | SHA-256 hash of stringified output (first 12 chars) |
| `timestamp` | `str` | ISO-8601 UTC timestamp |

### Exporting Lineage Records

Export the lineage graph as JSON for external tracking systems:

```python
import json

records = graph.export()
# [
#   {"name": "load_raw", "input_hash": "a1b2...", "output_hash": "f6e5...", "timestamp": "..."},
#   {"name": "clean", ...},
#   ...
# ]

# Write to file
with open("lineage.json", "w") as f:
    json.dump(records, f, indent=2)
```

### Manual Lineage Recording

You can also add lineage steps manually without the decorator:

```python
graph = LineageGraph()

# Record a manual step
graph.add(
    name="feature_engineering",
    input_data=raw_features,
    output_data=engineered_features,
)

# The graph computes SHA-256 hashes of str(input_data) and str(output_data)
```

### Using Lineage in Tests

```python
import pytest
from mltk.core.assertion import MltkAssertionError

def test_pipeline_lineage():
    """Assert that the full pipeline executed all transformation steps.

    WHY: Catches pipeline misconfigurations where a step is silently
         skipped, producing stale or unprocessed features.
    Expected: assert_lineage_complete passes with 4 steps.
    """
    graph = LineageGraph()

    @track_lineage(graph, "extract")
    def extract(source):
        return source[:100]

    @track_lineage(graph, "transform")
    def transform(data):
        return [x * 2 for x in data]

    @track_lineage(graph, "load")
    def load(data):
        return {"rows": len(data)}

    raw = extract(list(range(200)))
    transformed = transform(raw)
    loaded = load(transformed)

    assert_lineage_complete(graph, expected_steps=3)
```

---

## 10. Contributing

### Dev Setup

```bash
# Clone
git clone https://github.com/Liorrr/mltk.git
cd mltk

# Install in development mode with all dependencies
pip install -e ".[dev,scipy,sklearn,cli,report]"

# Install Rust toolchain (optional, for Rust acceleration)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Verify everything works
pytest -q              # Run Python tests
ruff check src/ tests/ # Run linter
cargo test             # Run Rust tests (in rust/)
```

### Code Style

- **Python**: PEP 8, enforced by `ruff`
- **Line length**: 100 characters
- **Type hints**: Required on all function signatures
- **Docstrings**: Required on all public functions with `Args`, `Returns`, `Example` sections
- **Imports**: Sorted by `ruff` (isort-compatible)

```bash
# Format code
ruff format src/ tests/

# Check linting
ruff check src/ tests/

# Auto-fix lint issues
ruff check --fix src/ tests/

# Type check
mypy src/mltk/
```

#### PEP 561 `py.typed` marker

mltk ships a `py.typed` marker file (`src/mltk/py.typed`) per [PEP 561](https://peps.python.org/pep-0561/). This tells mypy and other type checkers that mltk provides inline type annotations. If you depend on mltk and run mypy with `--strict`, types will be resolved automatically without requiring a `types-mltk` stub package.

Ruff configuration (from `pyproject.toml`):

```toml
[tool.ruff]
target-version = "py310"
line-length = 100
src = ["src"]

[tool.ruff.lint]
select = ["E", "F", "W", "I", "UP", "B", "A", "C4", "PT"]
```

### Test Conventions

Every test follows the `SCENARIO / WHY / Expected` pattern:

```python
def test_drift_detected_with_shifted_distribution(self) -> None:
    """FAIL: Mean-shifted distribution triggers KS drift detection.

    WHY: Catch silent data pipeline regressions where upstream changes
         shift feature distributions without changing schema.
    Expected: MltkAssertionError raised with drift details.
    """
    ref = [1.0, 2.0, 3.0, 4.0, 5.0] * 20
    shifted = [x + 10 for x in ref]

    with pytest.raises(MltkAssertionError) as exc_info:
        assert_no_drift(ref, shifted, method="ks", threshold=0.05)

    result = exc_info.value.result
    assert result.details["method"] == "ks"
    assert result.details["statistic"] > 0.5
```

**Pattern rules:**

1. Test name describes the scenario: `test_{what}_{condition}`.
2. Docstring first line: `PASS:` or `FAIL:` followed by scenario description.
3. `WHY:` explains the ML context -- what real-world bug does this catch?
4. `Expected:` states the expected outcome.
5. Always verify the `TestResult.details` dict, not just pass/fail.

### Assertion Naming Convention

- Functions: `assert_{domain}_{check}` (e.g., `assert_no_drift`, `assert_gradient_flow`).
- Names in TestResult: `{module}.{check}` (e.g., `"data.drift"`, `"training.gradient_flow"`).
- File placement: Domain determines the module path.

### Adding a New Assertion Module

1. **Create the file**: `src/mltk/{module}/{check}.py`
2. **Follow the pattern**: `@timed_assertion` + `assert_true()` + `TestResult`
3. **Export in `__init__.py`**: Add to the module's `__init__.py` and `__all__`
4. **Write tests**: `tests/test_{module}/test_{check}.py` with PASS + FAIL scenarios
5. **Write docs**: `docs/api/{check}.md` with examples
6. **Update mkdocs.yml**: Add to the nav section

### PR Process

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/my-feature`).
3. Write code + tests + docs.
4. Run `pytest -q` and `ruff check src/ tests/`.
5. Submit PR with description of changes.

**PR Checklist:**

- [ ] All tests pass (`pytest -q`)
- [ ] Lint clean (`ruff check src/ tests/`)
- [ ] New functions have docstrings (Args, Returns, Example)
- [ ] New tests have SCENARIO + WHY docstrings
- [ ] Doc page updated or created for new assertions
- [ ] CHANGELOG.md updated
- [ ] Type hints on all new function signatures

### Architecture Decisions

When proposing significant changes, consider these principles:

- **Framework agnosticism**: Assertions accept numpy arrays, not framework-specific tensors.
- **Fallback chain**: Rust > scipy > numpy. Never require a specific backend.
- **Severity semantics**: CRITICAL = gate (raises), WARNING = flag, INFO = metric.
- **Zero-config**: Assertions should work with sensible defaults. Configuration is opt-in.
- **Composability**: Every assertion returns `TestResult`. They compose into `TestSuite`. Suites feed reports, MLflow, server, etc.
- **Timing by default**: Use `@timed_assertion` on every assertion function.
- **Details for diagnostics**: Always pass structured data via `**details` kwargs.

### License

By contributing, you agree that your contributions will be licensed under Apache-2.0.
