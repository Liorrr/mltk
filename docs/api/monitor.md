# Production Monitoring

Continuous validation assertions for production ML systems. Detect metric degradation over time and validate SLA compliance.

**Module:** `mltk.monitor`

---

## assert_no_degradation

Sliding window check on metric history. Detects gradual performance decline that aggregate metrics miss.

```python
from mltk.monitor import assert_no_degradation

assert_no_degradation(metric_history, window=7, max_decline=0.05)
```

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `metric_history` | `list[float]` | *(required)* | Time-ordered metric values (oldest first) |
| `window` | `int` | `7` | Number of recent values to compare against earlier values |
| `max_decline` | `float` | `0.05` | Maximum allowed decline from earlier mean to recent mean |

### Returns

`TestResult` with details:
- `decline` -- actual decline (earlier_mean - recent_mean)
- `max_decline` -- configured threshold
- `recent_mean` -- mean of the last `window` values
- `earlier_mean` -- mean of all values before the window
- `window` -- window size used
- `history_length` -- total number of values in history

### How it works

```
decline = mean(earlier_values) - mean(recent_window)
PASS if decline <= max_decline
FAIL if decline > max_decline
```

### Example

```python
import pytest
from mltk.monitor import assert_no_degradation

def test_model_not_degrading():
    """Accuracy has not dropped over the last week."""
    daily_accuracy = [0.95, 0.94, 0.95, 0.93, 0.94, 0.92, 0.91, 0.90, 0.89, 0.88]
    assert_no_degradation(daily_accuracy, window=3, max_decline=0.03)
```

### Edge Cases

- **Insufficient history**: If `len(metric_history) < window`, returns a passing result with `INFO` severity (not enough data to judge).
- **Improving metric**: A negative decline (metric improving) always passes.

---

## assert_sla

Validate latency and error rate against SLA thresholds.

```python
from mltk.monitor import assert_sla

assert_sla(latency_p99=120.0, error_rate=0.005, thresholds={"latency_p99_ms": 200.0, "error_rate": 0.01})
```

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `latency_p99` | `float \| None` | `None` | Observed P99 latency in milliseconds |
| `error_rate` | `float \| None` | `None` | Observed error rate (0.0-1.0) |
| `thresholds` | `dict \| None` | `None` | Dict with `latency_p99_ms` and/or `error_rate` limits. Default: `{"latency_p99_ms": 500.0, "error_rate": 0.01}` |

### Returns

`TestResult` with details:
- `latency_p99` -- observed latency (if provided)
- `max_latency` -- latency threshold from config
- `error_rate` -- observed error rate (if provided)
- `max_error_rate` -- error rate threshold from config
- `violations` -- list of SLA violation strings

### Example

```python
import pytest
from mltk.monitor import assert_sla

def test_sla_compliance():
    """Production metrics are within SLA bounds."""
    assert_sla(
        latency_p99=150.0,
        error_rate=0.002,
        thresholds={"latency_p99_ms": 200.0, "error_rate": 0.01},
    )
```

### Edge Cases

- **Default thresholds**: If `thresholds` is `None`, defaults to `{"latency_p99_ms": 500.0, "error_rate": 0.01}`.
- **Partial checks**: You can pass only `latency_p99` or only `error_rate` -- the other is skipped.

---

## assert_no_output_drift

Detect behavioral drift in model outputs. Compares output score/probability distributions between a reference window and the current window using KS test or PSI. Catches cases where model predictions shift even if input features look stable.

```python
from mltk.monitor import assert_no_output_drift

ref_scores = [0.8, 0.82, 0.79, 0.81, 0.83]
cur_scores = [0.6, 0.58, 0.62, 0.55, 0.59]
assert_no_output_drift(ref_scores, cur_scores, method="ks", threshold=0.05)
```

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `ref_outputs` | `list[float] \| np.ndarray` | *(required)* | Baseline model outputs (e.g., from a reference window) |
| `cur_outputs` | `list[float] \| np.ndarray` | *(required)* | Current model outputs to compare against baseline |
| `method` | `str` | `"ks"` | Comparison method: `"ks"` (KS test p-value) or `"psi"` (Population Stability Index) |
| `threshold` | `float` | `0.05` | Significance threshold. For KS: pass if p > threshold. For PSI: pass if PSI < threshold. |

### Returns

`TestResult` with details:
- `method` -- comparison method used
- `statistic` -- test statistic (KS statistic or PSI value)
- `p_value` -- p-value (KS method only)
- `threshold` -- configured threshold
- `drift_detected` -- boolean flag

### Example

```python
import pytest
from mltk.monitor import assert_no_output_drift

def test_prediction_distribution_stable():
    """Model output distribution hasn't shifted from the baseline."""
    ref_scores = load_baseline_scores()
    cur_scores = get_current_scores()
    assert_no_output_drift(ref_scores, cur_scores, method="psi", threshold=0.1)
```

### Edge Cases

- **Empty arrays**: Returns a passing result with `drift_detected=False` if either array is empty.
- **Unknown method**: Fails with a descriptive error if method is not `"ks"` or `"psi"`.

---
