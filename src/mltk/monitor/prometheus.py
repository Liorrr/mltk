"""Prometheus + on-prem monitoring — PromQL queries, GPU utilization, Triton health.

Bridges mltk assertions to live infrastructure metrics, enabling the same
pass/fail gates used in offline testing to be applied against running services.
Covers three on-prem pillars:
- General PromQL: any metric threshold check via Prometheus query API.
- GPU utilization: DCGM_FI_DEV_GPU_UTIL via the NVIDIA DCGM exporter.
- Triton readiness: NVIDIA Triton Inference Server /v2/health/ready endpoint.

No external dependencies — uses stdlib urllib.request only.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any
from urllib.parse import quote

from mltk.core.assertion import assert_true, timed_assertion
from mltk.core.result import Severity, TestResult

# ---------------------------------------------------------------------------
# Internal HTTP helper
# ---------------------------------------------------------------------------

def _http_get(url: str, timeout: float = 10.0) -> dict[str, Any]:
    """Simple HTTP GET returning parsed JSON.

    Args:
        url: Full URL to request.
        timeout: Socket timeout in seconds.

    Returns:
        Parsed JSON response as a dict.

    Raises:
        urllib.error.URLError: On network or HTTP errors.
        json.JSONDecodeError: If the response body is not valid JSON.
    """
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


# ---------------------------------------------------------------------------
# Public assertions
# ---------------------------------------------------------------------------

@timed_assertion
def assert_prometheus_metric(
    url: str,
    query: str,
    threshold: float,
    comparison: str = "lte",
) -> TestResult:
    """Assert a Prometheus metric meets threshold via PromQL query.

    Executes a PromQL instant query against the Prometheus HTTP API and
    compares the first result value against the given threshold.

    Args:
        url: Prometheus server base URL (e.g., "http://prometheus:9090").
        query: PromQL query string (e.g., "rate(http_requests_total[5m])").
        threshold: Numeric threshold to compare against.
        comparison: Comparison operator — "lte" (<=), "gte" (>=), or "eq" (==).

    Returns:
        TestResult indicating pass/fail with query details.

    Example:
        >>> assert_prometheus_metric(
        ...     url="http://prometheus:9090",
        ...     query='model_error_rate{job="inference"}',
        ...     threshold=0.05,
        ...     comparison="lte",
        ... )
    """
    api_url = f"{url.rstrip('/')}/api/v1/query?query={quote(query)}"

    try:
        data = _http_get(api_url)
    except Exception as exc:
        return assert_true(
            False,
            name="monitor.prometheus.metric",
            message=f"Prometheus query failed: {exc}",
            severity=Severity.CRITICAL,
            url=url,
            query=query,
        )

    try:
        results = data["data"]["result"]
        if not results:
            return assert_true(
                False,
                name="monitor.prometheus.metric",
                message=f"PromQL returned no results for query: {query!r}",
                severity=Severity.CRITICAL,
                url=url,
                query=query,
            )
        raw_value = results[0]["value"][1]
        value = float(raw_value)
    except (KeyError, IndexError, ValueError, TypeError) as exc:
        return assert_true(
            False,
            name="monitor.prometheus.metric",
            message=f"Unexpected Prometheus response shape: {exc}",
            severity=Severity.CRITICAL,
            url=url,
            query=query,
        )

    if comparison == "lte":
        passed = value <= threshold
        op_str = "<="
    elif comparison == "gte":
        passed = value >= threshold
        op_str = ">="
    elif comparison == "eq":
        passed = value == threshold
        op_str = "=="
    else:
        return assert_true(
            False,
            name="monitor.prometheus.metric",
            message=f"Unknown comparison operator: {comparison!r}. Use 'lte', 'gte', or 'eq'.",
            severity=Severity.CRITICAL,
            url=url,
            query=query,
        )

    message = (
        f"Metric {value:.6g} {op_str} {threshold:.6g}"
        if passed
        else f"Metric threshold violated: {value:.6g} not {op_str} {threshold:.6g}"
    )

    return assert_true(
        passed,
        name="monitor.prometheus.metric",
        message=message,
        severity=Severity.CRITICAL,
        url=url,
        query=query,
        value=value,
        threshold=threshold,
        comparison=comparison,
    )


@timed_assertion
def assert_gpu_utilization(
    url: str,
    max_util: float = 0.95,
    gpu_id: str | None = None,
) -> TestResult:
    """Assert GPU utilization is below threshold via DCGM Prometheus metrics.

    Queries the DCGM_FI_DEV_GPU_UTIL metric from a Prometheus instance fed
    by the NVIDIA DCGM exporter. The raw metric is on a 0-100 scale; this
    function converts it to 0.0-1.0 before comparing against max_util.

    Args:
        url: Prometheus server base URL (e.g., "http://prometheus:9090").
        max_util: Maximum allowed GPU utilization as a fraction (0.0-1.0).
            Default 0.95 (95%).
        gpu_id: Optional GPU UUID or device index to scope the query.
            When None, the first result from DCGM_FI_DEV_GPU_UTIL is used.

    Returns:
        TestResult indicating pass/fail with utilization details.

    Example:
        >>> assert_gpu_utilization(url="http://prometheus:9090", max_util=0.90)
        >>> assert_gpu_utilization(url="http://prometheus:9090", gpu_id="GPU-0")
    """
    if gpu_id is not None:
        query = f'DCGM_FI_DEV_GPU_UTIL{{gpu="{gpu_id}"}}'
    else:
        query = "DCGM_FI_DEV_GPU_UTIL"

    api_url = f"{url.rstrip('/')}/api/v1/query?query={quote(query)}"

    try:
        data = _http_get(api_url)
    except Exception as exc:
        return assert_true(
            False,
            name="monitor.prometheus.gpu_utilization",
            message=f"Prometheus query failed: {exc}",
            severity=Severity.CRITICAL,
            url=url,
            gpu_id=gpu_id,
        )

    try:
        results = data["data"]["result"]
        if not results:
            return assert_true(
                False,
                name="monitor.prometheus.gpu_utilization",
                message="DCGM_FI_DEV_GPU_UTIL returned no results — DCGM exporter may be offline",
                severity=Severity.CRITICAL,
                url=url,
                gpu_id=gpu_id,
            )
        raw_value = results[0]["value"][1]
        # DCGM reports 0-100; normalise to 0.0-1.0
        utilization = float(raw_value) / 100.0
    except (KeyError, IndexError, ValueError, TypeError) as exc:
        return assert_true(
            False,
            name="monitor.prometheus.gpu_utilization",
            message=f"Unexpected DCGM metric response shape: {exc}",
            severity=Severity.CRITICAL,
            url=url,
            gpu_id=gpu_id,
        )

    passed = utilization <= max_util
    message = (
        f"GPU utilization {utilization:.1%} within limit {max_util:.1%}"
        if passed
        else f"GPU utilization {utilization:.1%} exceeds limit {max_util:.1%}"
    )

    return assert_true(
        passed,
        name="monitor.prometheus.gpu_utilization",
        message=message,
        severity=Severity.CRITICAL,
        url=url,
        gpu_id=gpu_id,
        utilization=utilization,
        max_util=max_util,
        raw_dcgm_value=float(raw_value),
    )


@timed_assertion
def assert_triton_healthy(url: str) -> TestResult:
    """Assert NVIDIA Triton Inference Server is ready to serve requests.

    Calls the Triton readiness probe (GET /v2/health/ready). An HTTP 200
    response means all loaded models are ready. Any other status or network
    error is treated as a failure.

    Args:
        url: Triton server base URL (e.g., "http://triton:8000").

    Returns:
        TestResult indicating pass/fail with server URL detail.

    Example:
        >>> assert_triton_healthy(url="http://triton:8000")
    """
    health_url = f"{url.rstrip('/')}/v2/health/ready"

    try:
        req = urllib.request.Request(health_url)
        with urllib.request.urlopen(req, timeout=10.0) as resp:
            http_status = resp.status
    except urllib.error.HTTPError as exc:
        return assert_true(
            False,
            name="monitor.prometheus.triton_health",
            message=f"Triton not ready: HTTP {exc.code} from {health_url}",
            severity=Severity.CRITICAL,
            url=url,
            http_status=exc.code,
        )
    except Exception as exc:
        return assert_true(
            False,
            name="monitor.prometheus.triton_health",
            message=f"Triton health check failed: {exc}",
            severity=Severity.CRITICAL,
            url=url,
        )

    passed = http_status == 200
    message = (
        f"Triton ready (HTTP {http_status})"
        if passed
        else f"Triton not ready: unexpected HTTP {http_status}"
    )

    return assert_true(
        passed,
        name="monitor.prometheus.triton_health",
        message=message,
        severity=Severity.CRITICAL,
        url=url,
        http_status=http_status,
    )
