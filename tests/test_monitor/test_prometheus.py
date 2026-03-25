"""Tests for mltk.monitor.prometheus — Prometheus/on-prem monitoring assertions.

All HTTP calls are mocked via unittest.mock.patch so tests run fully offline.
Three assertion surfaces are covered:
- assert_prometheus_metric: PromQL threshold checks (lte/gte/eq).
- assert_gpu_utilization: DCGM_FI_DEV_GPU_UTIL via Prometheus.
- assert_triton_healthy: Triton /v2/health/ready readiness probe.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from mltk.core.assertion import MltkAssertionError
from mltk.monitor.prometheus import (
    assert_gpu_utilization,
    assert_prometheus_metric,
    assert_triton_healthy,
)

# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------

PROMETHEUS_URL = "http://prometheus:9090"
TRITON_URL = "http://triton:8000"


def _mock_prometheus_response(value: float) -> MagicMock:
    """Create a mock urllib response for the Prometheus instant-query API.

    Builds a minimal payload matching the real /api/v1/query response shape:
    {"status": "success", "data": {"result": [{"value": [<timestamp>, "<value>"]}]}}
    The value is encoded as a string, matching Prometheus JSON serialisation.
    """
    data = {
        "status": "success",
        "data": {
            "result": [
                {"metric": {}, "value": [1700000000, str(value)]}
            ]
        },
    }
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps(data).encode()
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


def _mock_empty_prometheus_response() -> MagicMock:
    """Create a mock urllib response with an empty Prometheus result list."""
    data = {"status": "success", "data": {"result": []}}
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps(data).encode()
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


def _mock_triton_response(status: int) -> MagicMock:
    """Create a mock urllib response for a Triton health-check call."""
    mock_resp = MagicMock()
    mock_resp.status = status
    mock_resp.read.return_value = b""
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


# ---------------------------------------------------------------------------
# assert_prometheus_metric
# ---------------------------------------------------------------------------

class TestAssertPrometheusMetric:
    """PromQL threshold assertion tests.

    Each test represents a realistic CI gate scenario: a team uses PromQL
    to poll a live metric and decides whether to proceed with a deployment.
    """

    @patch("urllib.request.urlopen")
    def test_prometheus_metric_lte_pass(self, mock_urlopen: MagicMock) -> None:
        """SCENARIO: Model error rate is 0.5, threshold is 1.0 (lte).
        WHY: Healthy service — error rate is well within the SLA limit.
        EXPECTED: result.passed is True; value and threshold captured in details.
        """
        mock_urlopen.return_value = _mock_prometheus_response(0.5)

        result = assert_prometheus_metric(
            url=PROMETHEUS_URL,
            query='model_error_rate{job="inference"}',
            threshold=1.0,
            comparison="lte",
        )

        assert result.passed is True
        assert "0.5" in result.message or "<=" in result.message
        assert result.details["value"] == 0.5
        assert result.details["threshold"] == 1.0
        assert result.details["comparison"] == "lte"

    @patch("urllib.request.urlopen")
    def test_prometheus_metric_lte_fail(self, mock_urlopen: MagicMock) -> None:
        """SCENARIO: Model error rate is 2.0, threshold is 1.0 (lte).
        WHY: Error rate has doubled the SLA limit — deployment should be blocked.
        EXPECTED: MltkAssertionError raised; message mentions threshold violation.
        """
        mock_urlopen.return_value = _mock_prometheus_response(2.0)

        with pytest.raises(MltkAssertionError) as exc_info:
            assert_prometheus_metric(
                url=PROMETHEUS_URL,
                query='model_error_rate{job="inference"}',
                threshold=1.0,
                comparison="lte",
            )

        result = exc_info.value.result
        assert result.passed is False
        assert "threshold violated" in result.message or "not <=" in result.message

    @patch("urllib.request.urlopen")
    def test_prometheus_metric_gte_pass(self, mock_urlopen: MagicMock) -> None:
        """SCENARIO: Model accuracy is 0.9, minimum threshold is 0.5 (gte).
        WHY: Gate checks that accuracy has not collapsed below a floor value.
            0.9 comfortably exceeds 0.5, so deployment should proceed.
        EXPECTED: result.passed is True.
        """
        mock_urlopen.return_value = _mock_prometheus_response(0.9)

        result = assert_prometheus_metric(
            url=PROMETHEUS_URL,
            query='model_accuracy{job="inference"}',
            threshold=0.5,
            comparison="gte",
        )

        assert result.passed is True
        assert result.details["value"] == 0.9
        assert result.details["comparison"] == "gte"

    @patch("urllib.request.urlopen")
    def test_prometheus_metric_empty_result(self, mock_urlopen: MagicMock) -> None:
        """SCENARIO: PromQL returns an empty result list.
        WHY: The metric does not exist in Prometheus (exporter offline, wrong
            label selector). This is an infrastructure fault, not a pass.
        EXPECTED: MltkAssertionError raised; message mentions "no results".
        """
        mock_urlopen.return_value = _mock_empty_prometheus_response()

        with pytest.raises(MltkAssertionError) as exc_info:
            assert_prometheus_metric(
                url=PROMETHEUS_URL,
                query="nonexistent_metric",
                threshold=1.0,
            )

        assert "no results" in exc_info.value.result.message

    @patch("urllib.request.urlopen")
    def test_prometheus_metric_network_error(self, mock_urlopen: MagicMock) -> None:
        """SCENARIO: urllib raises an exception (Prometheus unreachable).
        WHY: If Prometheus is down the assertion must fail gracefully rather
            than propagate an unhandled exception.
        EXPECTED: MltkAssertionError raised; message mentions query failure.
        """
        import urllib.error
        mock_urlopen.side_effect = urllib.error.URLError("Connection refused")

        with pytest.raises(MltkAssertionError) as exc_info:
            assert_prometheus_metric(
                url=PROMETHEUS_URL,
                query='up{job="prometheus"}',
                threshold=1.0,
            )

        assert "failed" in exc_info.value.result.message.lower()


# ---------------------------------------------------------------------------
# assert_gpu_utilization
# ---------------------------------------------------------------------------

class TestAssertGpuUtilization:
    """GPU utilization assertion tests.

    DCGM reports utilisation on a 0-100 integer scale. The assertion must
    normalise to 0.0-1.0 before comparing against max_util.
    """

    @patch("urllib.request.urlopen")
    def test_gpu_utilization_ok(self, mock_urlopen: MagicMock) -> None:
        """SCENARIO: DCGM reports 60 (60%), max_util is 0.95 (95%).
        WHY: GPU is comfortably within budget — training or inference is
            proceeding normally without starving other workloads.
        EXPECTED: result.passed is True; utilization stored as 0.6.
        """
        # DCGM raw value is 60 (out of 100)
        mock_urlopen.return_value = _mock_prometheus_response(60)

        result = assert_gpu_utilization(
            url=PROMETHEUS_URL,
            max_util=0.95,
        )

        assert result.passed is True
        assert result.details["utilization"] == pytest.approx(0.60)
        assert result.details["raw_dcgm_value"] == pytest.approx(60.0)
        assert result.details["max_util"] == 0.95

    @patch("urllib.request.urlopen")
    def test_gpu_utilization_high(self, mock_urlopen: MagicMock) -> None:
        """SCENARIO: DCGM reports 98 (98%), max_util is 0.95 (95%).
        WHY: GPU is saturated — at 98% utilisation the inference server may
            queue requests, causing latency spikes. Deployment should be blocked
            or alerting triggered.
        EXPECTED: MltkAssertionError raised; message mentions "exceeds".
        """
        mock_urlopen.return_value = _mock_prometheus_response(98)

        with pytest.raises(MltkAssertionError) as exc_info:
            assert_gpu_utilization(
                url=PROMETHEUS_URL,
                max_util=0.95,
            )

        result = exc_info.value.result
        assert result.passed is False
        assert "exceeds" in result.message
        assert result.details["utilization"] == pytest.approx(0.98)

    @patch("urllib.request.urlopen")
    def test_gpu_utilization_with_gpu_id(self, mock_urlopen: MagicMock) -> None:
        """SCENARIO: Checking a specific GPU by id, utilization at 50%.
        WHY: Multi-GPU hosts require per-device checks to catch a single
            overloaded GPU masked by a low cluster average.
        EXPECTED: result.passed is True; gpu_id stored in details.
        """
        mock_urlopen.return_value = _mock_prometheus_response(50)

        result = assert_gpu_utilization(
            url=PROMETHEUS_URL,
            max_util=0.90,
            gpu_id="GPU-0",
        )

        assert result.passed is True
        assert result.details["gpu_id"] == "GPU-0"
        assert result.details["utilization"] == pytest.approx(0.50)

    @patch("urllib.request.urlopen")
    def test_gpu_utilization_no_results(self, mock_urlopen: MagicMock) -> None:
        """SCENARIO: DCGM exporter is offline — Prometheus returns empty result.
        WHY: An empty result means the GPU fleet is invisible to the monitoring
            stack. This must fail loudly rather than silently pass.
        EXPECTED: MltkAssertionError raised; message mentions "no results".
        """
        mock_urlopen.return_value = _mock_empty_prometheus_response()

        with pytest.raises(MltkAssertionError) as exc_info:
            assert_gpu_utilization(url=PROMETHEUS_URL)

        msg = exc_info.value.result.message.lower()
        assert "no results" in msg or "offline" in msg


# ---------------------------------------------------------------------------
# assert_triton_healthy
# ---------------------------------------------------------------------------

class TestAssertTritonHealthy:
    """Triton Inference Server readiness tests.

    Triton's /v2/health/ready returns 200 when all models are loaded and
    ready. Any other status or network error must be treated as a failure.
    """

    @patch("urllib.request.urlopen")
    def test_triton_healthy_ok(self, mock_urlopen: MagicMock) -> None:
        """SCENARIO: Triton returns HTTP 200 from /v2/health/ready.
        WHY: All models are loaded and the server is accepting requests.
            This is the expected steady state in a healthy deployment.
        EXPECTED: result.passed is True; message contains "ready".
        """
        mock_urlopen.return_value = _mock_triton_response(200)

        result = assert_triton_healthy(url=TRITON_URL)

        assert result.passed is True
        assert "ready" in result.message.lower()
        assert result.details["http_status"] == 200

    @patch("urllib.request.urlopen")
    def test_triton_unhealthy_http_error(self, mock_urlopen: MagicMock) -> None:
        """SCENARIO: Triton returns HTTP 503 (models not yet loaded).
        WHY: A 503 means Triton is still initialising or a model failed to load.
            Routing traffic to it would cause inference errors for all users.
        EXPECTED: MltkAssertionError raised; result.passed is False.
        """
        import urllib.error
        mock_urlopen.side_effect = urllib.error.HTTPError(
            url=f"{TRITON_URL}/v2/health/ready",
            code=503,
            msg="Service Unavailable",
            hdrs=None,   # type: ignore[arg-type]
            fp=None,
        )

        with pytest.raises(MltkAssertionError) as exc_info:
            assert_triton_healthy(url=TRITON_URL)

        result = exc_info.value.result
        assert result.passed is False
        assert "503" in result.message
        assert result.details["http_status"] == 503

    @patch("urllib.request.urlopen")
    def test_triton_unhealthy_network_error(self, mock_urlopen: MagicMock) -> None:
        """SCENARIO: urllib raises URLError — Triton is completely unreachable.
        WHY: Network partition, DNS failure, or the pod crashed. The assertion
            must fail gracefully and surface a clear error message.
        EXPECTED: MltkAssertionError raised; message mentions failure.
        """
        import urllib.error
        mock_urlopen.side_effect = urllib.error.URLError("Connection refused")

        with pytest.raises(MltkAssertionError) as exc_info:
            assert_triton_healthy(url=TRITON_URL)

        result = exc_info.value.result
        assert result.passed is False
        assert "failed" in result.message.lower() or "connection" in result.message.lower()

    @patch("urllib.request.urlopen")
    def test_triton_result_has_timing(self, mock_urlopen: MagicMock) -> None:
        """SCENARIO: Triton returns HTTP 200; caller needs timing data.
        WHY: The @timed_assertion decorator must populate duration_ms so
            callers can detect slow health-check round-trips (e.g., > 1s
            indicates a network issue between monitor and Triton).
        EXPECTED: result.duration_ms is a non-negative float.
        """
        mock_urlopen.return_value = _mock_triton_response(200)

        result = assert_triton_healthy(url=TRITON_URL)

        assert isinstance(result.duration_ms, float)
        assert result.duration_ms >= 0.0
