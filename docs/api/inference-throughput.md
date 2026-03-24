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
| `args` | `tuple` | `()` | Arguments to pass to func |
| `min_rps` | `float` | `100` | Minimum required requests per second |
| `duration` | `float` | `5.0` | Test duration in seconds |
| `concurrency` | `int` | `1` | Number of concurrent workers |

---
