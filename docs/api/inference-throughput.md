# Inference Throughput Testing

Throughput testing validates that ML endpoints can handle the required request volume. Duration-based measurement (not count-based) gives realistic RPS numbers.

**Module:** `mltk.inference.throughput`

---

## assert_throughput

Assert model serves at least N requests per second.

```python
from mltk.inference import assert_throughput

assert_throughput(model.predict, X_single, min_rps=100, duration=5.0)
```

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `func` | `Callable` | *(required)* | Function to benchmark |
| `*args` | `Any` | | Positional arguments passed through to `func` on each call |
| `min_rps` | `float` | `100.0` | Minimum required requests per second |
| `duration` | `float` | `5.0` | Test duration in seconds |
| `concurrency` | `int` | `1` | Number of concurrent workers (uses ThreadPoolExecutor) |

### Returns

`TestResult` with details:
- `actual_rps` -- measured requests per second
- `min_rps` -- configured threshold
- `completed` -- total completed requests
- `errors` -- number of requests that raised exceptions
- `error_rate` -- fraction of failed requests
- `duration` -- test duration in seconds
- `concurrency` -- number of concurrent workers used

### Example

```python
import pytest
from mltk.inference import assert_throughput

@pytest.mark.ml_inference
def test_model_throughput(model, sample_input):
    """Model must handle at least 100 requests per second."""
    assert_throughput(model.predict, sample_input, min_rps=100, duration=5.0)

@pytest.mark.ml_inference
def test_concurrent_throughput(model, sample_input):
    """Model must handle 500 RPS with 4 concurrent workers."""
    assert_throughput(model.predict, sample_input, min_rps=500, duration=5.0, concurrency=4)
```

### Edge Cases

- **Sequential mode**: When `concurrency=1` (default), calls are made in a tight loop for the specified duration.
- **Concurrent mode**: When `concurrency > 1`, a `ThreadPoolExecutor` spawns the specified number of workers, each making calls in a tight loop. Total RPS is the combined throughput of all workers.
- **Errors counted**: If `func` raises an exception, it is counted as an error but does not stop the test. The `error_rate` is reported in results.
- **Variadic args**: Additional positional arguments after `func` are passed through to each call.

---
