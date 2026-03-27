"""Production monitoring — drift alerts, degradation, SLA, and cloud/on-prem health.

Core assertions (no extra dependencies):
    assert_no_degradation — detect gradual metric decline over a sliding window
    assert_sla            — gate on latency and error rate SLA thresholds

Local GPU monitoring (nvidia-smi, no Prometheus required):
    assert_gpu_utilization_local — GPU compute utilization via nvidia-smi
    assert_gpu_memory_local      — GPU memory usage via nvidia-smi

On-prem / Prometheus monitoring (stdlib only — no extra dependencies):
    assert_prometheus_metric  — PromQL threshold check (lte/gte/eq)
    assert_gpu_utilization    — DCGM_FI_DEV_GPU_UTIL via Prometheus (0-1 scale)
    assert_triton_healthy     — Triton /v2/health/ready readiness probe

Cloud monitoring (optional — requires provider-specific extras):
    AWS / SageMaker   → mltk.monitor.aws    (pip install mltk[aws])
        assert_endpoint_healthy      — endpoint InService check
        assert_endpoint_latency      — CloudWatch ModelLatency P99
        assert_endpoint_error_rate   — 4XX+5XX error rate

    GCP / Vertex AI   → mltk.monitor.gcp    (pip install mltk[gcp])
        assert_endpoint_healthy      — deployed model count check
        assert_prediction_latency    — Cloud Monitoring latency P99

    Azure / AzureML   → mltk.monitor.azure  (pip install mltk[azure])
        assert_endpoint_healthy      — managed endpoint provisioning state
        assert_endpoint_latency      — Azure Monitor RequestLatency_P99

Cloud modules are NOT imported at the package level because their SDKs are
heavy optional dependencies. Import them directly when needed:

    from mltk.monitor.aws import assert_endpoint_healthy
    from mltk.monitor.gcp import assert_prediction_latency
    from mltk.monitor.azure import assert_endpoint_latency
"""

from mltk.monitor.anomaly import assert_no_test_anomaly
from mltk.monitor.concept_drift import assert_no_concept_drift
from mltk.monitor.drift_monitor import assert_no_degradation, assert_no_output_drift, assert_sla
from mltk.monitor.gpu import assert_gpu_memory_local, assert_gpu_utilization_local
from mltk.monitor.prometheus import (
    assert_gpu_utilization,
    assert_prometheus_metric,
    assert_triton_healthy,
)
from mltk.monitor.streaming_drift import (
    ADWINDetector,
    BaseDriftDetector,
    CUSUMDetector,
    assert_no_streaming_drift,
)

__all__ = [
    "assert_no_degradation",
    "assert_sla",
    "assert_no_output_drift",
    "assert_no_concept_drift",
    "assert_no_streaming_drift",
    "assert_prometheus_metric",
    "assert_gpu_utilization",
    "assert_triton_healthy",
    "assert_gpu_utilization_local",
    "assert_gpu_memory_local",
    "ADWINDetector",
    "BaseDriftDetector",
    "CUSUMDetector",
    "assert_no_test_anomaly",
]
