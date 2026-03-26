"""GCP Vertex AI monitoring — endpoint health, prediction latency.

Verifies that Vertex AI endpoints have live deployed models and that
prediction latency reported by Cloud Monitoring stays within SLA thresholds.
All Google Cloud SDK imports are lazy so this module is importable without
the ``mltk[gcp]`` extras installed.
"""

from __future__ import annotations

from typing import Any

from mltk.core.assertion import assert_true, timed_assertion
from mltk.core.result import Severity, TestResult


def _require_aiplatform() -> Any:
    """Lazy-import ``google.cloud.aiplatform``.

    Raises:
        ImportError: When the optional ``mltk[gcp]`` extra is not installed.
    """
    try:
        from google.cloud import aiplatform  # type: ignore[import]
        return aiplatform
    except ImportError as exc:
        raise ImportError(
            "google-cloud-aiplatform is required for GCP monitoring. "
            "Install it with: pip install mltk[gcp]"
        ) from exc


def _require_monitoring() -> Any:
    """Lazy-import ``google.cloud.monitoring_v3``.

    Raises:
        ImportError: When the optional ``mltk[gcp]`` extra is not installed.
    """
    try:
        from google.cloud import monitoring_v3  # type: ignore[import]
        return monitoring_v3
    except ImportError as exc:
        raise ImportError(
            "google-cloud-monitoring is required for GCP latency monitoring. "
            "Install it with: pip install mltk[gcp]"
        ) from exc


@timed_assertion
def assert_endpoint_healthy(
    endpoint_name: str,
    project: str | None = None,
    location: str | None = None,
) -> TestResult:
    """Assert a Vertex AI endpoint is deployed and serving.

    Checks that the endpoint exists and has at least one deployed model. An
    endpoint with no deployed models cannot serve predictions; this condition
    is treated as CRITICAL.

    Args:
        endpoint_name: Full resource name
            (``projects/.../locations/.../endpoints/...``) or short ID.
        project: GCP project ID. Inferred from application default credentials
            when *None*.
        location: GCP region (e.g. ``"us-central1"``). Defaults to
            ``"us-central1"`` when *None*.

    Returns:
        TestResult capturing deployment status and timing.

    Example:
        >>> assert_endpoint_healthy(
        ...     "projects/my-project/locations/us-central1/endpoints/12345"
        ... )
    """
    aiplatform = _require_aiplatform()

    init_kwargs: dict[str, Any] = {}
    if project:
        init_kwargs["project"] = project
    if location:
        init_kwargs["location"] = location
    if init_kwargs:
        aiplatform.init(**init_kwargs)

    endpoint = aiplatform.Endpoint(endpoint_name)
    deployed_models = endpoint.deployed_models

    has_models = bool(deployed_models)
    model_count = len(deployed_models) if deployed_models else 0

    message = (
        f"Endpoint '{endpoint_name}' has {model_count} deployed model(s)"
        if has_models
        else f"Endpoint '{endpoint_name}' has no deployed models — cannot serve predictions"
    )

    return assert_true(
        has_models,
        name="gcp.endpoint.health",
        message=message,
        severity=Severity.CRITICAL,
        endpoint_name=endpoint_name,
        deployed_model_count=model_count,
        project=project or "default",
        location=location or "us-central1",
    )


@timed_assertion
def assert_prediction_latency(
    endpoint_name: str,
    max_p99_ms: float = 500.0,
    project: str | None = None,
    location: str | None = None,
    minutes: int = 5,
) -> TestResult:
    """Assert prediction latency via Cloud Monitoring is within threshold.

    Queries the ``aiplatform.googleapis.com/prediction/online/response_latencies``
    metric for the p99 value over the last *minutes* minutes.

    Args:
        endpoint_name: Vertex AI endpoint resource name or short numeric ID.
        max_p99_ms: Maximum allowed P99 latency in milliseconds.
        project: GCP project ID. Inferred from application default credentials
            when *None*.
        location: GCP region. Defaults to ``"us-central1"`` when *None*.
        minutes: Look-back window in minutes for the latency query (default 5).

    Returns:
        TestResult with observed P99 latency and threshold details.

    Example:
        >>> assert_prediction_latency(
        ...     "projects/my-project/locations/us-central1/endpoints/12345",
        ...     max_p99_ms=300.0,
        ... )
    """
    import datetime

    monitoring_v3 = _require_monitoring()

    resolved_project = project or _infer_project()
    resolved_location = location or "us-central1"

    client = monitoring_v3.MetricServiceClient()
    project_name = f"projects/{resolved_project}"

    now = datetime.datetime.now(datetime.UTC)
    interval = monitoring_v3.TimeInterval(
        {
            "end_time": {"seconds": int(now.timestamp())},
            "start_time": {"seconds": int((now - datetime.timedelta(minutes=minutes)).timestamp())},
        }
    )

    aggregation = monitoring_v3.Aggregation(
        {
            "alignment_period": {"seconds": minutes * 60},
            "per_series_aligner": monitoring_v3.Aggregation.Aligner.ALIGN_PERCENTILE_99,
        }
    )

    results = client.list_time_series(
        request={
            "name": project_name,
            "filter": (
                'metric.type="aiplatform.googleapis.com/prediction/online/response_latencies" '
                f'AND resource.labels.endpoint_id="{endpoint_name}"'
            ),
            "interval": interval,
            "view": monitoring_v3.ListTimeSeriesRequest.TimeSeriesView.FULL,
            "aggregation": aggregation,
        }
    )

    series_list = list(results)
    if not series_list or not series_list[0].points:
        return assert_true(
            True,
            name="gcp.endpoint.latency",
            message=(
                f"No latency datapoints for '{endpoint_name}' in last {minutes}m "
                "(no traffic or metric not yet available)"
            ),
            severity=Severity.INFO,
            endpoint_name=endpoint_name,
            window_minutes=minutes,
        )

    # Latency values are in milliseconds in Cloud Monitoring.
    p99_ms: float = max(
        point.value.double_value
        for series in series_list
        for point in series.points
    )

    passed = p99_ms <= max_p99_ms
    message = (
        f"P99 latency {p99_ms:.1f}ms within {max_p99_ms}ms threshold"
        if passed
        else f"P99 latency {p99_ms:.1f}ms exceeds {max_p99_ms}ms threshold"
    )

    return assert_true(
        passed,
        name="gcp.endpoint.latency",
        message=message,
        severity=Severity.CRITICAL,
        endpoint_name=endpoint_name,
        p99_latency_ms=round(p99_ms, 2),
        max_p99_ms=max_p99_ms,
        window_minutes=minutes,
        project=resolved_project,
        location=resolved_location,
    )


def _infer_project() -> str:
    """Best-effort: infer the GCP project from application default credentials."""
    try:
        import google.auth  # type: ignore[import]
        _, project = google.auth.default()
        return project or "unknown"
    except Exception:
        return "unknown"
