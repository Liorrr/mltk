# mltk.monitor

## Overview
Production monitoring assertions: metric degradation, SLA, output/concept/streaming drift, test-metric anomalies, and endpoint/GPU health across cloud and on-prem.

## Interface
- `assert_no_degradation`, `assert_sla`, `assert_no_output_drift` (drift_monitor.py)
- `assert_no_concept_drift` (concept_drift.py)
- `BaseDriftDetector`, `CUSUMDetector`, `ADWINDetector`, `assert_no_streaming_drift` (streaming_drift.py)
- `assert_no_test_anomaly` (anomaly.py)
- AWS SageMaker: `assert_endpoint_healthy`, `assert_endpoint_latency`, `assert_endpoint_error_rate` (aws.py)
- Azure ML: `assert_endpoint_healthy`, `assert_endpoint_latency` (azure.py)
- GCP Vertex AI: `assert_endpoint_healthy`, `assert_prediction_latency` (gcp.py)
- Local GPU: `assert_gpu_utilization_local`, `assert_gpu_memory_local` (gpu.py)
- Prometheus: `assert_prometheus_metric`, `assert_gpu_utilization`, `assert_triton_healthy` (prometheus.py)

## Invariants
- Assertions return `TestResult`; a failed check at `Severity.CRITICAL` (the default) raises `MltkAssertionError` via `mltk.core.assert_true`. They are plain functions usable inside pytest.
- Same function names (e.g. `assert_endpoint_healthy`) exist per provider module; import from the specific module.

## Side Effects
- aws/azure/gcp/prometheus modules make network calls to external services.
- gpu.py shells out to `nvidia-smi`.

## Verification
- Run: `python -m pytest tests/test_monitor -q`
