# Inference Latency Testing

Latency testing validates that ML model inference meets response time SLAs. Industry benchmarks: P95 <50ms for real-time classification, <200ms for NLP, <1s for LLM TTFT.

**Module:** `mltk.inference.latency`

---

## assert_latency

Assert inference latency percentiles are within bounds.

```python
from mltk.inference import assert_latency

assert_latency(model.predict, X_test, p95=50.0, p99=100.0, warmup=5)
```

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `func` | `Callable` | *(required)* | Function to benchmark |
| `*args` | `Any` | | Positional arguments passed through to `func` on each call |
| `p50` | `float \| None` | `None` | Max P50 latency in ms |
| `p95` | `float \| None` | `None` | Max P95 latency in ms |
| `p99` | `float \| None` | `None` | Max P99 latency in ms |
| `iterations` | `int` | `100` | Number of measurement iterations |
| `warmup` | `int` | `5` | Warmup iterations (excluded from measurement) |

!!! note "At least one percentile threshold required"
    You must specify at least one of `p50`, `p95`, or `p99`. If all are `None`, the assertion fails immediately with an error.

### Returns

`TestResult` with details:
- `p50` -- actual P50 latency in ms
- `p95` -- actual P95 latency in ms
- `p99` -- actual P99 latency in ms
- `min` -- minimum latency observed
- `max` -- maximum latency observed
- `mean` -- mean latency
- `std` -- standard deviation of latencies
- `iterations` -- number of measurement iterations
- `warmup` -- number of warmup iterations
- `thresholds` -- dict of configured thresholds (`{"p50": ..., "p95": ..., "p99": ...}`)

### Example

```python
import pytest
from mltk.inference import assert_latency

@pytest.mark.ml_inference
def test_model_latency(model, X_test):
    """Model must respond within 50ms at P95."""
    assert_latency(model.predict, X_test, p95=50.0, p99=100.0)

@pytest.mark.ml_inference
def test_batch_latency(model, batch):
    """Batch inference under 200ms at P99."""
    assert_latency(model.predict_batch, batch, p99=200.0, iterations=50)
```

### Edge Cases

- **No thresholds**: If all of `p50`, `p95`, `p99` are `None`, the assertion fails with `CRITICAL` severity immediately.
- **Warmup phase**: The first `warmup` calls are executed but excluded from measurement. This avoids JIT compilation, cache warming, and GPU kernel compilation inflating results.
- **Variadic args**: Additional positional arguments after `func` are passed through to each call (e.g., `assert_latency(model.predict, X_test, p95=50.0)` calls `model.predict(X_test)` on each iteration).

---

## assert_cold_start

Assert first-call latency (cold start) is within bounds. Measures model loading time plus first inference.

```python
from mltk.inference import assert_cold_start

assert_cold_start(load_and_predict, X_test, max_ms=2000.0)
```

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `func` | `Callable` | *(required)* | Function to benchmark (should include model loading) |
| `*args` | `Any` | | Positional arguments passed through to `func` |
| `max_ms` | `float` | `2000.0` | Maximum allowed cold start time in milliseconds |

### Returns

`TestResult` with details:
- `cold_start_ms` -- actual cold start time in ms
- `max_ms` -- configured threshold

### Example

```python
import pytest
from mltk.inference import assert_cold_start

@pytest.mark.ml_inference
def test_model_cold_start():
    """Model must load and produce first prediction within 2 seconds."""
    def load_and_predict(x):
        import joblib
        model = joblib.load("model.pkl")
        return model.predict(x)

    assert_cold_start(load_and_predict, sample_input, max_ms=2000.0)
```

### Edge Cases

- **No warmup**: Unlike `assert_latency`, cold start measures exactly one call with no warmup -- that is the point.
- **Variadic args**: Same as `assert_latency`, additional positional arguments are passed through to `func`.

---
