"""Azure ML monitoring — managed endpoint health, latency.

Validates Azure managed online endpoints are healthy and that request
latency measured by Azure Monitor stays within SLA thresholds. All Azure SDK
imports are lazy so this module is importable without the ``mltk[azure]``
extras installed.
"""

from __future__ import annotations

from typing import Any

from mltk.core.assertion import assert_true, timed_assertion
from mltk.core.result import Severity, TestResult


def _require_ml_client(
    subscription_id: str | None,
    resource_group: str | None,
    workspace_name: str | None,
) -> Any:
    """Lazy-import azure-ai-ml and return an MLClient.

    Args:
        subscription_id: Azure subscription ID. Reads ``AZURE_SUBSCRIPTION_ID``
            env var when *None*.
        resource_group: Resource group name. Reads ``AZURE_RESOURCE_GROUP``
            when *None*.
        workspace_name: Azure ML workspace name. Reads ``AZURE_WORKSPACE_NAME``
            when *None*.

    Raises:
        ImportError: When the optional ``mltk[azure]`` extra is not installed.
        ValueError: When required parameters cannot be resolved.
    """
    try:
        from azure.ai.ml import MLClient  # type: ignore[import]
        from azure.identity import DefaultAzureCredential  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "azure-ai-ml and azure-identity are required for Azure monitoring. "
            "Install them with: pip install mltk[azure]"
        ) from exc

    import os

    sub = subscription_id or os.environ.get("AZURE_SUBSCRIPTION_ID")
    rg = resource_group or os.environ.get("AZURE_RESOURCE_GROUP")
    ws = workspace_name or os.environ.get("AZURE_WORKSPACE_NAME")

    missing = [k for k, v in [
        ("subscription_id", sub),
        ("resource_group", rg),
        ("workspace_name", ws),
    ] if not v]
    if missing:
        raise ValueError(
            f"Missing required Azure parameters: {missing}. "
            "Pass them as arguments or set the corresponding environment variables."
        )

    credential = DefaultAzureCredential()
    return MLClient(
        credential=credential,
        subscription_id=sub,
        resource_group_name=rg,
        workspace_name=ws,
    )


@timed_assertion
def assert_endpoint_healthy(
    endpoint_name: str,
    resource_group: str | None = None,
    subscription_id: str | None = None,
    workspace_name: str | None = None,
) -> TestResult:
    """Assert an Azure managed online endpoint is healthy.

    Checks the endpoint's ``provisioning_state`` is ``"Succeeded"`` and that
    at least one deployment is active. An endpoint in ``"Failed"`` state or
    with no deployments cannot serve requests.

    Args:
        endpoint_name: Name of the Azure managed online endpoint.
        resource_group: Azure resource group. Falls back to the
            ``AZURE_RESOURCE_GROUP`` environment variable when *None*.
        subscription_id: Azure subscription ID. Falls back to
            ``AZURE_SUBSCRIPTION_ID`` when *None*.
        workspace_name: Azure ML workspace name. Falls back to
            ``AZURE_WORKSPACE_NAME`` when *None*.

    Returns:
        TestResult with provisioning state and deployment count details.

    Example:
        >>> assert_endpoint_healthy(
        ...     "my-online-endpoint",
        ...     resource_group="my-rg",
        ...     subscription_id="00000000-...",
        ...     workspace_name="my-workspace",
        ... )
    """
    ml_client = _require_ml_client(subscription_id, resource_group, workspace_name)
    endpoint = ml_client.online_endpoints.get(name=endpoint_name)

    provisioning_state: str = getattr(endpoint, "provisioning_state", "Unknown")
    is_succeeded = provisioning_state == "Succeeded"

    # Also check that at least one deployment exists.
    deployments = list(ml_client.online_deployments.list(endpoint_name=endpoint_name))
    has_deployments = bool(deployments)
    deployment_count = len(deployments)

    passed = is_succeeded and has_deployments
    if not is_succeeded:
        message = (
            f"Endpoint '{endpoint_name}' provisioning state is '{provisioning_state}', "
            "expected 'Succeeded'"
        )
    elif not has_deployments:
        message = f"Endpoint '{endpoint_name}' has no active deployments"
    else:
        message = (
            f"Endpoint '{endpoint_name}' is healthy "
            f"(state=Succeeded, deployments={deployment_count})"
        )

    return assert_true(
        passed,
        name="azure.endpoint.health",
        message=message,
        severity=Severity.CRITICAL,
        endpoint_name=endpoint_name,
        provisioning_state=provisioning_state,
        deployment_count=deployment_count,
        resource_group=resource_group or "env",
        workspace_name=workspace_name or "env",
    )


@timed_assertion
def assert_endpoint_latency(
    endpoint_name: str,
    max_p99_ms: float = 500.0,
    resource_group: str | None = None,
    subscription_id: str | None = None,
    workspace_name: str | None = None,
    minutes: int = 5,
) -> TestResult:
    """Assert Azure Monitor request latency for a managed endpoint is within threshold.

    Queries the ``RequestLatency_P99`` metric from Azure Monitor for the
    managed online endpoint resource over the last *minutes* minutes.

    Args:
        endpoint_name: Name of the Azure managed online endpoint.
        max_p99_ms: Maximum allowed P99 latency in milliseconds.
        resource_group: Azure resource group. Falls back to
            ``AZURE_RESOURCE_GROUP`` when *None*.
        subscription_id: Azure subscription ID. Falls back to
            ``AZURE_SUBSCRIPTION_ID`` when *None*.
        workspace_name: Azure ML workspace name. Falls back to
            ``AZURE_WORKSPACE_NAME`` when *None*.
        minutes: Look-back window in minutes (default 5).

    Returns:
        TestResult with observed P99 latency and threshold details.

    Example:
        >>> assert_endpoint_latency(
        ...     "my-online-endpoint",
        ...     max_p99_ms=300.0,
        ...     resource_group="my-rg",
        ...     subscription_id="00000000-...",
        ... )
    """
    import datetime
    import os

    try:
        from azure.identity import DefaultAzureCredential  # type: ignore[import]
        from azure.monitor.query import (  # type: ignore[import]
            MetricAggregationType,
            MetricsQueryClient,
        )
    except ImportError as exc:
        raise ImportError(
            "azure-monitor-query and azure-identity are required for Azure latency monitoring. "
            "Install them with: pip install mltk[azure]"
        ) from exc

    sub = subscription_id or os.environ.get("AZURE_SUBSCRIPTION_ID")
    rg = resource_group or os.environ.get("AZURE_RESOURCE_GROUP")
    ws = workspace_name or os.environ.get("AZURE_WORKSPACE_NAME")

    # Build the resource URI for the managed endpoint.
    resource_uri = (
        f"/subscriptions/{sub}/resourceGroups/{rg}"
        f"/providers/Microsoft.MachineLearningServices/workspaces/{ws}"
        f"/onlineEndpoints/{endpoint_name}"
    )

    credential = DefaultAzureCredential()
    monitor_client = MetricsQueryClient(credential)

    end_time = datetime.datetime.now(datetime.UTC)
    start_time = end_time - datetime.timedelta(minutes=minutes)

    response = monitor_client.query_resource(
        resource_uri,
        metric_names=["RequestLatency_P99"],
        timespan=(start_time, end_time),
        granularity=datetime.timedelta(minutes=minutes),
        aggregations=[MetricAggregationType.AVERAGE],
    )

    # Extract the maximum P99 value across all time series / time buckets.
    p99_ms: float | None = None
    for metric in response.metrics:
        for ts in metric.timeseries:
            for data_point in ts.data:
                value = data_point.average
                if value is not None:
                    p99_ms = max(p99_ms, value) if p99_ms is not None else value

    if p99_ms is None:
        return assert_true(
            True,
            name="azure.endpoint.latency",
            message=(
                f"No latency datapoints for '{endpoint_name}' in last {minutes}m "
                "(no traffic or metric not yet available)"
            ),
            severity=Severity.INFO,
            endpoint_name=endpoint_name,
            window_minutes=minutes,
        )

    passed = p99_ms <= max_p99_ms
    message = (
        f"P99 latency {p99_ms:.1f}ms within {max_p99_ms}ms threshold"
        if passed
        else f"P99 latency {p99_ms:.1f}ms exceeds {max_p99_ms}ms threshold"
    )

    return assert_true(
        passed,
        name="azure.endpoint.latency",
        message=message,
        severity=Severity.CRITICAL,
        endpoint_name=endpoint_name,
        p99_latency_ms=round(p99_ms, 2),
        max_p99_ms=max_p99_ms,
        window_minutes=minutes,
        resource_group=rg or "env",
        subscription_id=sub or "env",
    )
