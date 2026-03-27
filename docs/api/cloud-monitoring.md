# Cloud Monitoring

Monitor ML model endpoints deployed on AWS SageMaker, GCP Vertex AI, Azure ML, or on-prem (Prometheus/Triton).

**Module:** `mltk.monitor`

**Install:** `pip install mltk[aws]`, `mltk[gcp]`, or `mltk[azure]`

---

## AWS SageMaker

```python
from mltk.monitor.aws import assert_endpoint_healthy, assert_endpoint_latency

assert_endpoint_healthy("my-model-endpoint", region="us-east-1")
assert_endpoint_latency("my-model-endpoint", max_p99_ms=500)
assert_endpoint_error_rate("my-model-endpoint", max_rate=0.01)
```

| Function | Description |
|----------|-------------|
| `assert_endpoint_healthy` | SageMaker endpoint is InService |
| `assert_endpoint_latency` | CloudWatch ModelLatency P99 within threshold |
| `assert_endpoint_error_rate` | CloudWatch 4XX+5XX error rate within threshold |

---

## GCP Vertex AI

```python
from mltk.monitor.gcp import assert_endpoint_healthy, assert_prediction_latency

assert_endpoint_healthy("my-endpoint", project="my-project", location="us-central1")
assert_prediction_latency("my-endpoint", max_p99_ms=500)
```

| Function | Description |
|----------|-------------|
| `assert_endpoint_healthy` | Vertex AI endpoint is deployed and serving |
| `assert_prediction_latency` | Cloud Monitoring prediction latency within threshold |

---

## Azure ML

```python
from mltk.monitor.azure import assert_endpoint_healthy, assert_endpoint_latency

assert_endpoint_healthy("my-endpoint", resource_group="my-rg")
assert_endpoint_latency("my-endpoint", max_p99_ms=500)
```

| Function | Description |
|----------|-------------|
| `assert_endpoint_healthy` | Azure managed endpoint is healthy |
| `assert_endpoint_latency` | Azure Monitor request latency within threshold |

---

## Local GPU (nvidia-smi)

No Prometheus or DCGM stack required — queries `nvidia-smi` directly.

```python
from mltk.monitor.gpu import assert_gpu_utilization_local, assert_gpu_memory_local

assert_gpu_utilization_local(max_util=0.95)
assert_gpu_memory_local(max_util=0.90)
```

| Function | Description |
|----------|-------------|
| `assert_gpu_utilization_local` | GPU compute utilization below threshold (nvidia-smi) |
| `assert_gpu_memory_local` | GPU memory usage below threshold (nvidia-smi) |

---

## Prometheus / On-prem

```python
from mltk.monitor.prometheus import (
    assert_prometheus_metric,
    assert_gpu_utilization,
    assert_triton_healthy,
)

assert_prometheus_metric("http://prometheus:9090", "up{job='model'}", threshold=1.0)
assert_gpu_utilization("http://prometheus:9090", max_util=0.95)
assert_triton_healthy("http://triton:8000")
```

| Function | Description |
|----------|-------------|
| `assert_prometheus_metric` | Run PromQL query, check result against threshold |
| `assert_gpu_utilization` | DCGM GPU utilization below limit |
| `assert_triton_healthy` | Triton /v2/health/ready returns 200 |

---
