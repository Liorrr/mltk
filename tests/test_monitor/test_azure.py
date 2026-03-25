"""Tests for mltk.monitor.azure — Azure ML endpoint monitoring assertions.

All Azure SDK calls are fully mocked via sys.modules patching; no Azure
credentials or network access are required. Each test re-imports the module
inside its mock context to prevent import-caching side effects.

Test scenarios cover:
- Endpoint health: Succeeded + deployments (pass), Failed state (fail),
  Succeeded but no deployments (fail)
- Endpoint latency: within threshold (pass) and no datapoints (INFO pass)
"""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

from mltk.core.assertion import MltkAssertionError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Shared environment overrides so MLClient resolution never needs real env vars.
_AZURE_ENV = {
    "AZURE_SUBSCRIPTION_ID": "test-sub-id",
    "AZURE_RESOURCE_GROUP": "test-rg",
    "AZURE_WORKSPACE_NAME": "test-ws",
}


def _make_azure_mocks() -> tuple[MagicMock, MagicMock, MagicMock, MagicMock]:
    """Return (ml_mock, identity_mock, monitor_query_mock, credential_mock)."""
    ml_mock = MagicMock()
    identity_mock = MagicMock()
    monitor_query_mock = MagicMock()
    credential_mock = MagicMock()
    identity_mock.DefaultAzureCredential.return_value = credential_mock
    return ml_mock, identity_mock, monitor_query_mock, credential_mock


def _patch_azure(ml_mock, identity_mock, monitor_query_mock):
    """Return a patch.dict context that replaces all Azure SDK modules."""
    return patch.dict(
        "sys.modules",
        {
            "azure": MagicMock(),
            "azure.ai": MagicMock(),
            "azure.ai.ml": ml_mock,
            "azure.identity": identity_mock,
            "azure.monitor": MagicMock(),
            "azure.monitor.query": monitor_query_mock,
        },
    )


def _fresh_azure_module(ml_mock, identity_mock, monitor_query_mock):
    """Force a fresh import of mltk.monitor.azure with mocked Azure SDKs."""
    for key in list(sys.modules.keys()):
        if "mltk.monitor.azure" in key:
            del sys.modules[key]

    with _patch_azure(ml_mock, identity_mock, monitor_query_mock):
        import mltk.monitor.azure as azure_mod
        return azure_mod


# ---------------------------------------------------------------------------
# assert_endpoint_healthy
# ---------------------------------------------------------------------------

class TestAzureAssertEndpointHealthy:
    """Azure managed online endpoint provisioning-state health checks."""

    def test_endpoint_healthy_succeeded_with_deployments(self) -> None:
        """SCENARIO: Endpoint provisioning_state = Succeeded, 1 deployment active.
        WHY: A Succeeded endpoint with at least one deployment is fully
             operational. This is the expected post-deploy healthy baseline.
        EXPECTED: result.passed is True, deployment_count = 1.
        """
        ml_mock, identity_mock, monitor_query_mock, _ = _make_azure_mocks()

        # Mock endpoint object.
        mock_endpoint = MagicMock()
        mock_endpoint.provisioning_state = "Succeeded"
        ml_mock.MLClient.return_value.online_endpoints.get.return_value = mock_endpoint

        # Mock deployments list.
        ml_mock.MLClient.return_value.online_deployments.list.return_value = iter(
            [MagicMock()]
        )

        for key in list(sys.modules.keys()):
            if "mltk.monitor.azure" in key:
                del sys.modules[key]

        with _patch_azure(ml_mock, identity_mock, monitor_query_mock), \
             patch.dict(os.environ, _AZURE_ENV):
            from mltk.monitor.azure import assert_endpoint_healthy

            result = assert_endpoint_healthy("my-endpoint")

        assert result.passed is True
        assert result.details["provisioning_state"] == "Succeeded"
        assert result.details["deployment_count"] == 1

    def test_endpoint_healthy_failed_state(self) -> None:
        """SCENARIO: Endpoint provisioning_state = Failed.
        WHY: A Failed managed endpoint means the last deploy or update did
             not complete successfully. All prediction traffic is affected.
             Must raise CRITICAL immediately.
        EXPECTED: MltkAssertionError raised, 'Failed' in message.
        """
        ml_mock, identity_mock, monitor_query_mock, _ = _make_azure_mocks()

        mock_endpoint = MagicMock()
        mock_endpoint.provisioning_state = "Failed"
        ml_mock.MLClient.return_value.online_endpoints.get.return_value = mock_endpoint
        ml_mock.MLClient.return_value.online_deployments.list.return_value = iter([])

        for key in list(sys.modules.keys()):
            if "mltk.monitor.azure" in key:
                del sys.modules[key]

        with _patch_azure(ml_mock, identity_mock, monitor_query_mock), \
             patch.dict(os.environ, _AZURE_ENV):
            from mltk.monitor.azure import assert_endpoint_healthy

            with pytest.raises(MltkAssertionError) as exc_info:
                assert_endpoint_healthy("broken-endpoint")

        assert "Failed" in str(exc_info.value)

    def test_endpoint_healthy_succeeded_no_deployments(self) -> None:
        """SCENARIO: Endpoint provisioning_state = Succeeded but 0 deployments.
        WHY: An endpoint can provision successfully but have all deployments
             removed (e.g., after a rollback that deleted traffic allocation).
             It cannot serve requests without active deployments.
        EXPECTED: MltkAssertionError raised, 'no active deployments' in message.
        """
        ml_mock, identity_mock, monitor_query_mock, _ = _make_azure_mocks()

        mock_endpoint = MagicMock()
        mock_endpoint.provisioning_state = "Succeeded"
        ml_mock.MLClient.return_value.online_endpoints.get.return_value = mock_endpoint
        ml_mock.MLClient.return_value.online_deployments.list.return_value = iter([])

        for key in list(sys.modules.keys()):
            if "mltk.monitor.azure" in key:
                del sys.modules[key]

        with _patch_azure(ml_mock, identity_mock, monitor_query_mock), \
             patch.dict(os.environ, _AZURE_ENV):
            from mltk.monitor.azure import assert_endpoint_healthy

            with pytest.raises(MltkAssertionError) as exc_info:
                assert_endpoint_healthy("empty-endpoint")

        assert "no active deployments" in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# assert_endpoint_latency
# ---------------------------------------------------------------------------

class TestAzureAssertEndpointLatency:
    """Azure Monitor RequestLatency_P99 threshold checks."""

    def test_endpoint_latency_ok(self) -> None:
        """SCENARIO: Azure Monitor returns P99 latency of 200.0 ms.
        WHY: 200 ms is within the default 500 ms SLA. Validates the Azure
             Monitor query path and the passing result shape.
        EXPECTED: result.passed is True, p99_latency_ms ≈ 200.0.
        """
        ml_mock, identity_mock, monitor_query_mock, _ = _make_azure_mocks()

        # Build a fake MetricsQueryResult structure.
        mock_data_point = MagicMock()
        mock_data_point.average = 200.0

        mock_ts = MagicMock()
        mock_ts.data = [mock_data_point]

        mock_metric = MagicMock()
        mock_metric.timeseries = [mock_ts]

        mock_response = MagicMock()
        mock_response.metrics = [mock_metric]

        mock_monitor_client = MagicMock()
        mock_monitor_client.query_resource.return_value = mock_response
        monitor_query_mock.MetricsQueryClient.return_value = mock_monitor_client
        monitor_query_mock.MetricAggregationType = MagicMock()

        for key in list(sys.modules.keys()):
            if "mltk.monitor.azure" in key:
                del sys.modules[key]

        with _patch_azure(ml_mock, identity_mock, monitor_query_mock), \
             patch.dict(os.environ, _AZURE_ENV):
            from mltk.monitor.azure import assert_endpoint_latency

            result = assert_endpoint_latency("my-endpoint", max_p99_ms=500.0)

        assert result.passed is True
        assert result.details["p99_latency_ms"] == pytest.approx(200.0, abs=0.1)

    def test_endpoint_latency_exceeds_threshold(self) -> None:
        """SCENARIO: Azure Monitor returns P99 latency of 750.0 ms.
        WHY: 750 ms far exceeds the 500 ms SLA. The endpoint is overloaded
             or the model is too large for the SKU. Must alert immediately.
        EXPECTED: MltkAssertionError raised, 'exceeds' in message.
        """
        ml_mock, identity_mock, monitor_query_mock, _ = _make_azure_mocks()

        mock_data_point = MagicMock()
        mock_data_point.average = 750.0

        mock_ts = MagicMock()
        mock_ts.data = [mock_data_point]

        mock_metric = MagicMock()
        mock_metric.timeseries = [mock_ts]

        mock_response = MagicMock()
        mock_response.metrics = [mock_metric]

        mock_monitor_client = MagicMock()
        mock_monitor_client.query_resource.return_value = mock_response
        monitor_query_mock.MetricsQueryClient.return_value = mock_monitor_client
        monitor_query_mock.MetricAggregationType = MagicMock()

        for key in list(sys.modules.keys()):
            if "mltk.monitor.azure" in key:
                del sys.modules[key]

        with _patch_azure(ml_mock, identity_mock, monitor_query_mock), \
             patch.dict(os.environ, _AZURE_ENV):
            from mltk.monitor.azure import assert_endpoint_latency

            with pytest.raises(MltkAssertionError) as exc_info:
                assert_endpoint_latency("my-endpoint", max_p99_ms=500.0)

        assert "exceeds" in str(exc_info.value)

    def test_endpoint_latency_no_datapoints(self) -> None:
        """SCENARIO: Azure Monitor returns metrics with all None average values.
        WHY: A new endpoint with no traffic yet has no RequestLatency_P99 data.
             This is a normal cold-start state and must not fail the CI gate.
        EXPECTED: result.passed is True, severity is INFO.
        """
        from mltk.core.result import Severity

        ml_mock, identity_mock, monitor_query_mock, _ = _make_azure_mocks()

        mock_data_point = MagicMock()
        mock_data_point.average = None  # No data

        mock_ts = MagicMock()
        mock_ts.data = [mock_data_point]

        mock_metric = MagicMock()
        mock_metric.timeseries = [mock_ts]

        mock_response = MagicMock()
        mock_response.metrics = [mock_metric]

        mock_monitor_client = MagicMock()
        mock_monitor_client.query_resource.return_value = mock_response
        monitor_query_mock.MetricsQueryClient.return_value = mock_monitor_client
        monitor_query_mock.MetricAggregationType = MagicMock()

        for key in list(sys.modules.keys()):
            if "mltk.monitor.azure" in key:
                del sys.modules[key]

        with _patch_azure(ml_mock, identity_mock, monitor_query_mock), \
             patch.dict(os.environ, _AZURE_ENV):
            from mltk.monitor.azure import assert_endpoint_latency

            result = assert_endpoint_latency("quiet-endpoint")

        assert result.passed is True
        assert result.severity == Severity.INFO
