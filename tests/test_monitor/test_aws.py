"""Tests for mltk.monitor.aws — SageMaker endpoint monitoring assertions.

All boto3 calls are fully mocked; no AWS credentials or network access are
required. Each test re-imports the module inside its mock context to prevent
module-level import caching from polluting between scenarios.

Test scenarios cover:
- Endpoint health: InService (pass) and Failed (fail)
- Endpoint latency: within threshold (pass) and no datapoints (INFO pass)
- Error rate: low rate (pass) and no invocations (INFO pass)
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from mltk.core.assertion import MltkAssertionError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_boto3_mock() -> MagicMock:
    """Return a fresh MagicMock that stands in for the boto3 module."""
    return MagicMock()


def _fresh_aws_module(boto3_mock: MagicMock):
    """Import mltk.monitor.aws with boto3 replaced by *boto3_mock*.

    Forces a fresh import every time by removing cached module entries first.
    """
    for key in list(sys.modules.keys()):
        if key.startswith("mltk.monitor.aws"):
            del sys.modules[key]

    with patch.dict("sys.modules", {"boto3": boto3_mock}):
        import mltk.monitor.aws as aws_mod
        return aws_mod


# ---------------------------------------------------------------------------
# assert_endpoint_healthy
# ---------------------------------------------------------------------------

class TestAssertEndpointHealthy:
    """SageMaker endpoint InService health check."""

    def test_endpoint_healthy_in_service(self) -> None:
        """SCENARIO: SageMaker endpoint reports EndpointStatus = InService.
        WHY: An InService endpoint is fully ready to serve predictions. This
             is the expected healthy-system baseline check run after deploys.
        EXPECTED: result.passed is True, status detail = 'InService'.
        """
        boto3_mock = _make_boto3_mock()
        boto3_mock.client.return_value.describe_endpoint.return_value = {
            "EndpointStatus": "InService",
            "EndpointName": "my-endpoint",
        }

        with patch.dict("sys.modules", {"boto3": boto3_mock}):
            for key in list(sys.modules.keys()):
                if "mltk.monitor.aws" in key:
                    del sys.modules[key]
            from mltk.monitor.aws import assert_endpoint_healthy

            result = assert_endpoint_healthy("my-endpoint", region="us-east-1")

        assert result.passed is True
        assert result.details["status"] == "InService"
        assert "InService" in result.message

    def test_endpoint_healthy_failed_state(self) -> None:
        """SCENARIO: SageMaker endpoint reports EndpointStatus = Failed.
        WHY: A Failed endpoint cannot serve predictions. Common after a bad
             model package or misconfigured instance type. Must raise CRITICAL.
        EXPECTED: MltkAssertionError raised, 'Failed' in message.
        """
        boto3_mock = _make_boto3_mock()
        boto3_mock.client.return_value.describe_endpoint.return_value = {
            "EndpointStatus": "Failed",
            "EndpointName": "broken-endpoint",
        }

        with patch.dict("sys.modules", {"boto3": boto3_mock}):
            for key in list(sys.modules.keys()):
                if "mltk.monitor.aws" in key:
                    del sys.modules[key]
            from mltk.monitor.aws import assert_endpoint_healthy

            with pytest.raises(MltkAssertionError) as exc_info:
                assert_endpoint_healthy("broken-endpoint")

        assert "Failed" in str(exc_info.value)

    def test_endpoint_healthy_updating_state(self) -> None:
        """SCENARIO: SageMaker endpoint is Updating (rolling deploy in progress).
        WHY: An Updating endpoint may drop requests during the rollout window.
             Monitoring should flag this as not yet stable.
        EXPECTED: MltkAssertionError raised, 'Updating' visible in message.
        """
        boto3_mock = _make_boto3_mock()
        boto3_mock.client.return_value.describe_endpoint.return_value = {
            "EndpointStatus": "Updating",
            "EndpointName": "updating-endpoint",
        }

        with patch.dict("sys.modules", {"boto3": boto3_mock}):
            for key in list(sys.modules.keys()):
                if "mltk.monitor.aws" in key:
                    del sys.modules[key]
            from mltk.monitor.aws import assert_endpoint_healthy

            with pytest.raises(MltkAssertionError) as exc_info:
                assert_endpoint_healthy("updating-endpoint")

        assert "Updating" in str(exc_info.value)


# ---------------------------------------------------------------------------
# assert_endpoint_latency
# ---------------------------------------------------------------------------

class TestAssertEndpointLatency:
    """CloudWatch ModelLatency P99 threshold checks."""

    def test_endpoint_latency_ok(self) -> None:
        """SCENARIO: CloudWatch returns P99 latency of 120 000 µs (= 120 ms).
        WHY: 120 ms is well within the default 500 ms SLA. This validates the
             µs-to-ms conversion and the passing path.
        EXPECTED: result.passed is True, p99_latency_ms ≈ 120.0.
        """
        boto3_mock = _make_boto3_mock()
        # SageMaker reports ModelLatency in microseconds.
        boto3_mock.client.return_value.get_metric_statistics.return_value = {
            "Datapoints": [
                {"ExtendedStatistics": {"p99": 120_000.0}, "Timestamp": "2026-01-01T00:00:00Z"},
            ]
        }

        with patch.dict("sys.modules", {"boto3": boto3_mock}):
            for key in list(sys.modules.keys()):
                if "mltk.monitor.aws" in key:
                    del sys.modules[key]
            from mltk.monitor.aws import assert_endpoint_latency

            result = assert_endpoint_latency("my-endpoint", max_p99_ms=500.0)

        assert result.passed is True
        assert result.details["p99_latency_ms"] == pytest.approx(120.0, abs=0.1)

    def test_endpoint_latency_exceeds_threshold(self) -> None:
        """SCENARIO: CloudWatch returns P99 latency of 800 000 µs (= 800 ms).
        WHY: 800 ms exceeds the 500 ms default SLA. Inference is too slow —
             likely an underpowered instance or model not warmed up.
        EXPECTED: MltkAssertionError raised, 'exceeds' in message.
        """
        boto3_mock = _make_boto3_mock()
        boto3_mock.client.return_value.get_metric_statistics.return_value = {
            "Datapoints": [
                {"ExtendedStatistics": {"p99": 800_000.0}, "Timestamp": "2026-01-01T00:00:00Z"},
            ]
        }

        with patch.dict("sys.modules", {"boto3": boto3_mock}):
            for key in list(sys.modules.keys()):
                if "mltk.monitor.aws" in key:
                    del sys.modules[key]
            from mltk.monitor.aws import assert_endpoint_latency

            with pytest.raises(MltkAssertionError) as exc_info:
                assert_endpoint_latency("my-endpoint", max_p99_ms=500.0)

        assert "exceeds" in str(exc_info.value)

    def test_endpoint_latency_no_datapoints(self) -> None:
        """SCENARIO: CloudWatch returns an empty Datapoints list.
        WHY: A newly deployed endpoint with no traffic yet has no metric data.
             This should return INFO (not CRITICAL) so it doesn't block CI.
        EXPECTED: result.passed is True, severity is INFO.
        """
        from mltk.core.result import Severity

        boto3_mock = _make_boto3_mock()
        boto3_mock.client.return_value.get_metric_statistics.return_value = {
            "Datapoints": []
        }

        with patch.dict("sys.modules", {"boto3": boto3_mock}):
            for key in list(sys.modules.keys()):
                if "mltk.monitor.aws" in key:
                    del sys.modules[key]
            from mltk.monitor.aws import assert_endpoint_latency

            result = assert_endpoint_latency("quiet-endpoint")

        assert result.passed is True
        assert result.severity == Severity.INFO


# ---------------------------------------------------------------------------
# assert_endpoint_error_rate
# ---------------------------------------------------------------------------

class TestAssertEndpointErrorRate:
    """CloudWatch 4XX + 5XX error rate threshold checks."""

    def test_endpoint_error_rate_ok(self) -> None:
        """SCENARIO: 5 errors across 1 000 invocations (0.5% rate).
        WHY: 0.5% is below the 1% default threshold. A healthy endpoint in
             production with only transient client errors.
        EXPECTED: result.passed is True, error_rate ≈ 0.005.
        """
        boto3_mock = _make_boto3_mock()

        # get_metric_statistics is called three times: Invocations, 4XX, 5XX.
        boto3_mock.client.return_value.get_metric_statistics.side_effect = [
            {"Datapoints": [{"Sum": 1000.0}]},  # Invocations
            {"Datapoints": [{"Sum": 3.0}]},      # 4XX errors
            {"Datapoints": [{"Sum": 2.0}]},      # 5XX errors
        ]

        with patch.dict("sys.modules", {"boto3": boto3_mock}):
            for key in list(sys.modules.keys()):
                if "mltk.monitor.aws" in key:
                    del sys.modules[key]
            from mltk.monitor.aws import assert_endpoint_error_rate

            result = assert_endpoint_error_rate("my-endpoint", max_rate=0.01)

        assert result.passed is True
        assert result.details["error_rate"] == pytest.approx(0.005, abs=1e-6)

    def test_endpoint_error_rate_exceeds_threshold(self) -> None:
        """SCENARIO: 50 errors across 500 invocations (10% rate).
        WHY: 10% error rate means 1 in 10 requests is failing. Far above
             the 1% SLA — likely a model crash or bad input pipeline.
        EXPECTED: MltkAssertionError raised.
        """
        boto3_mock = _make_boto3_mock()
        boto3_mock.client.return_value.get_metric_statistics.side_effect = [
            {"Datapoints": [{"Sum": 500.0}]},   # Invocations
            {"Datapoints": [{"Sum": 30.0}]},    # 4XX
            {"Datapoints": [{"Sum": 20.0}]},    # 5XX
        ]

        with patch.dict("sys.modules", {"boto3": boto3_mock}):
            for key in list(sys.modules.keys()):
                if "mltk.monitor.aws" in key:
                    del sys.modules[key]
            from mltk.monitor.aws import assert_endpoint_error_rate

            with pytest.raises(MltkAssertionError):
                assert_endpoint_error_rate("my-endpoint", max_rate=0.01)

    def test_endpoint_error_rate_no_invocations(self) -> None:
        """SCENARIO: Zero invocations in the observation window.
        WHY: A silent endpoint (no traffic) should not be flagged as having
             an error rate. Division-by-zero must be guarded and return INFO.
        EXPECTED: result.passed is True, severity is INFO.
        """
        from mltk.core.result import Severity

        boto3_mock = _make_boto3_mock()
        boto3_mock.client.return_value.get_metric_statistics.side_effect = [
            {"Datapoints": []},  # Invocations — empty
            {"Datapoints": []},  # 4XX
            {"Datapoints": []},  # 5XX
        ]

        with patch.dict("sys.modules", {"boto3": boto3_mock}):
            for key in list(sys.modules.keys()):
                if "mltk.monitor.aws" in key:
                    del sys.modules[key]
            from mltk.monitor.aws import assert_endpoint_error_rate

            result = assert_endpoint_error_rate("quiet-endpoint")

        assert result.passed is True
        assert result.severity == Severity.INFO
