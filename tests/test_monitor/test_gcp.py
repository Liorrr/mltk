"""Tests for mltk.monitor.gcp — Vertex AI endpoint monitoring assertions.

All Google Cloud SDK calls are fully mocked via sys.modules patching; no GCP
credentials or network access are required. Each test re-imports the module
inside its mock context to prevent import-caching side effects.

Test scenarios cover:
- Endpoint health: models deployed (pass) and empty deploy list (fail)
- Prediction latency: within threshold (pass) and no datapoints (INFO pass)
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from mltk.core.assertion import MltkAssertionError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_gcp_mocks() -> tuple[MagicMock, MagicMock, MagicMock]:
    """Return (aiplatform_mock, monitoring_v3_mock, google_auth_mock)."""
    aiplatform_mock = MagicMock()
    monitoring_v3_mock = MagicMock()
    google_auth_mock = MagicMock()
    google_auth_mock.default.return_value = (MagicMock(), "test-project")
    return aiplatform_mock, monitoring_v3_mock, google_auth_mock


def _patch_gcp(aiplatform_mock, monitoring_v3_mock, google_auth_mock):
    """Return a patch.dict context that replaces all GCP SDK modules."""
    google_mock = MagicMock()
    google_cloud_mock = MagicMock()
    google_cloud_mock.aiplatform = aiplatform_mock
    google_cloud_mock.monitoring_v3 = monitoring_v3_mock
    google_mock.cloud = google_cloud_mock
    google_mock.auth = google_auth_mock
    return patch.dict(
        "sys.modules",
        {
            "google": google_mock,
            "google.cloud": google_cloud_mock,
            "google.cloud.aiplatform": aiplatform_mock,
            "google.cloud.monitoring_v3": monitoring_v3_mock,
            "google.auth": google_auth_mock,
        },
    )


def _fresh_gcp_module(aiplatform_mock, monitoring_v3_mock, google_auth_mock):
    """Force a fresh import of mltk.monitor.gcp with mocked GCP SDKs."""
    for key in list(sys.modules.keys()):
        if "mltk.monitor.gcp" in key:
            del sys.modules[key]

    with _patch_gcp(aiplatform_mock, monitoring_v3_mock, google_auth_mock):
        import mltk.monitor.gcp as gcp_mod
        return gcp_mod


# ---------------------------------------------------------------------------
# assert_endpoint_healthy
# ---------------------------------------------------------------------------

class TestGcpAssertEndpointHealthy:
    """Vertex AI endpoint deployment-state health checks."""

    def test_endpoint_healthy_with_deployed_models(self) -> None:
        """SCENARIO: Vertex AI endpoint has 2 deployed models.
        WHY: An endpoint with active deployed models is ready to serve
             predictions. This is the normal post-deployment healthy state.
        EXPECTED: result.passed is True, deployed_model_count = 2.
        """
        aiplatform_mock, monitoring_v3_mock, google_auth_mock = _make_gcp_mocks()

        # Mock: endpoint.deployed_models returns a list of two model objects.
        mock_endpoint_instance = MagicMock()
        mock_endpoint_instance.deployed_models = [MagicMock(), MagicMock()]
        aiplatform_mock.Endpoint.return_value = mock_endpoint_instance

        for key in list(sys.modules.keys()):
            if "mltk.monitor.gcp" in key:
                del sys.modules[key]

        with _patch_gcp(aiplatform_mock, monitoring_v3_mock, google_auth_mock):
            from mltk.monitor.gcp import assert_endpoint_healthy

            result = assert_endpoint_healthy(
                "projects/test/locations/us-central1/endpoints/123",
                project="test-project",
                location="us-central1",
            )

        assert result.passed is True
        assert result.details["deployed_model_count"] == 2

    def test_endpoint_healthy_no_deployed_models(self) -> None:
        """SCENARIO: Vertex AI endpoint exists but has no deployed models.
        WHY: An endpoint with zero deployed models returns errors for all
             prediction requests. Happens after accidentally undeploying all
             traffic splits. Must be caught immediately.
        EXPECTED: MltkAssertionError raised, 'no deployed models' in message.
        """
        aiplatform_mock, monitoring_v3_mock, google_auth_mock = _make_gcp_mocks()

        mock_endpoint_instance = MagicMock()
        mock_endpoint_instance.deployed_models = []
        aiplatform_mock.Endpoint.return_value = mock_endpoint_instance

        for key in list(sys.modules.keys()):
            if "mltk.monitor.gcp" in key:
                del sys.modules[key]

        with _patch_gcp(aiplatform_mock, monitoring_v3_mock, google_auth_mock):
            from mltk.monitor.gcp import assert_endpoint_healthy

            with pytest.raises(MltkAssertionError) as exc_info:
                assert_endpoint_healthy(
                    "projects/test/locations/us-central1/endpoints/456"
                )

        assert "no deployed models" in str(exc_info.value).lower()

    def test_endpoint_healthy_none_deployed_models(self) -> None:
        """SCENARIO: endpoint.deployed_models returns None (SDK edge case).
        WHY: Some SDK versions return None instead of an empty list when no
             models are deployed. The assertion must handle this gracefully
             rather than raising an unhandled AttributeError.
        EXPECTED: MltkAssertionError raised (treated as empty, i.e., unhealthy).
        """
        aiplatform_mock, monitoring_v3_mock, google_auth_mock = _make_gcp_mocks()

        mock_endpoint_instance = MagicMock()
        mock_endpoint_instance.deployed_models = None
        aiplatform_mock.Endpoint.return_value = mock_endpoint_instance

        for key in list(sys.modules.keys()):
            if "mltk.monitor.gcp" in key:
                del sys.modules[key]

        with _patch_gcp(aiplatform_mock, monitoring_v3_mock, google_auth_mock):
            from mltk.monitor.gcp import assert_endpoint_healthy

            with pytest.raises(MltkAssertionError):
                assert_endpoint_healthy(
                    "projects/test/locations/us-central1/endpoints/789"
                )


# ---------------------------------------------------------------------------
# assert_prediction_latency
# ---------------------------------------------------------------------------

class TestGcpAssertPredictionLatency:
    """Cloud Monitoring prediction latency P99 threshold checks."""

    def test_prediction_latency_ok(self) -> None:
        """SCENARIO: Cloud Monitoring returns a P99 latency of 85.0 ms.
        WHY: 85 ms is comfortably within the default 500 ms SLA. Validates
             the Cloud Monitoring query path and the passing result shape.
        EXPECTED: result.passed is True, p99_latency_ms ≈ 85.0.
        """
        aiplatform_mock, monitoring_v3_mock, google_auth_mock = _make_gcp_mocks()

        # Build a fake time-series response.
        mock_point = MagicMock()
        mock_point.value.double_value = 85.0

        mock_series = MagicMock()
        mock_series.points = [mock_point]

        mock_client = MagicMock()
        mock_client.list_time_series.return_value = [mock_series]
        monitoring_v3_mock.MetricServiceClient.return_value = mock_client

        for key in list(sys.modules.keys()):
            if "mltk.monitor.gcp" in key:
                del sys.modules[key]

        with _patch_gcp(aiplatform_mock, monitoring_v3_mock, google_auth_mock):
            from mltk.monitor.gcp import assert_prediction_latency

            result = assert_prediction_latency(
                "projects/test/locations/us-central1/endpoints/123",
                max_p99_ms=500.0,
                project="test-project",
            )

        assert result.passed is True
        assert result.details["p99_latency_ms"] == pytest.approx(85.0, abs=0.1)

    def test_prediction_latency_exceeds_threshold(self) -> None:
        """SCENARIO: Cloud Monitoring returns a P99 latency of 650.0 ms.
        WHY: 650 ms exceeds the 500 ms threshold. The model is too slow —
             possibly due to cold-start, under-provisioned replicas, or a
             very large model that needs batching.
        EXPECTED: MltkAssertionError raised, 'exceeds' in message.
        """
        aiplatform_mock, monitoring_v3_mock, google_auth_mock = _make_gcp_mocks()

        mock_point = MagicMock()
        mock_point.value.double_value = 650.0

        mock_series = MagicMock()
        mock_series.points = [mock_point]

        mock_client = MagicMock()
        mock_client.list_time_series.return_value = [mock_series]
        monitoring_v3_mock.MetricServiceClient.return_value = mock_client

        for key in list(sys.modules.keys()):
            if "mltk.monitor.gcp" in key:
                del sys.modules[key]

        with _patch_gcp(aiplatform_mock, monitoring_v3_mock, google_auth_mock):
            from mltk.monitor.gcp import assert_prediction_latency

            with pytest.raises(MltkAssertionError) as exc_info:
                assert_prediction_latency(
                    "projects/test/locations/us-central1/endpoints/123",
                    max_p99_ms=500.0,
                )

        assert "exceeds" in str(exc_info.value)

    def test_prediction_latency_no_datapoints(self) -> None:
        """SCENARIO: Cloud Monitoring returns no time-series data.
        WHY: A freshly deployed endpoint with zero traffic has no metrics yet.
             This must not block CI — return INFO so pipelines continue.
        EXPECTED: result.passed is True, severity is INFO.
        """
        from mltk.core.result import Severity

        aiplatform_mock, monitoring_v3_mock, google_auth_mock = _make_gcp_mocks()

        mock_client = MagicMock()
        mock_client.list_time_series.return_value = []  # No data
        monitoring_v3_mock.MetricServiceClient.return_value = mock_client

        for key in list(sys.modules.keys()):
            if "mltk.monitor.gcp" in key:
                del sys.modules[key]

        with _patch_gcp(aiplatform_mock, monitoring_v3_mock, google_auth_mock):
            from mltk.monitor.gcp import assert_prediction_latency

            result = assert_prediction_latency(
                "projects/test/locations/us-central1/endpoints/123",
                project="test-project",
            )

        assert result.passed is True
        assert result.severity == Severity.INFO
