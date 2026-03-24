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
| `args` | `tuple` | `()` | Arguments to pass to func |
| `p50` | `float \| None` | `None` | Max P50 latency in ms |
| `p95` | `float \| None` | `None` | Max P95 latency in ms |
| `p99` | `float \| None` | `None` | Max P99 latency in ms |
| `iterations` | `int` | `100` | Number of measurement iterations |
| `warmup` | `int` | `5` | Warmup iterations (excluded from measurement) |

## assert_cold_start

Assert first-call latency is within bounds (model loading time).

```python
from mltk.inference import assert_cold_start

assert_cold_start(load_and_predict, X_test, max_ms=2000.0)
```

---
