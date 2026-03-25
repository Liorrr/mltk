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
