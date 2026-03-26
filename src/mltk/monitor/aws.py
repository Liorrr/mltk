"""AWS SageMaker monitoring — endpoint health, latency, error rate.

Catches the most common cloud ML failure modes: endpoints silently stuck in
a non-InService state, latency creeping above SLA, and error rates spiking
after a deployment. All metrics are pulled from CloudWatch so no test
traffic is required.
"""

from __future__ import annotations

from typing import Any

from mltk.core.assertion import assert_true, timed_assertion
from mltk.core.result import Severity, TestResult


def _get_boto3_client(service: str, region: str | None) -> Any:
    """Lazy-import boto3 and return a client for *service*.

    Raises a clear ImportError when the optional ``mltk[aws]`` extra is not
    installed so the user gets an actionable message rather than a bare
    ModuleNotFoundError.
    """
    try:
        import boto3  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "boto3 is required for AWS monitoring. "
            "Install it with: pip install mltk[aws]"
        ) from exc

    kwargs: dict[str, Any] = {}
    if region:
        kwargs["region_name"] = region
    return boto3.client(service, **kwargs)


@timed_assertion
def assert_endpoint_healthy(
    endpoint_name: str,
    region: str | None = None,
) -> TestResult:
    """Assert a SageMaker endpoint is InService.

    Args:
        endpoint_name: Name of the SageMaker endpoint to check.
        region: AWS region name (e.g. ``"us-east-1"``). Uses the default
            credential chain region when *None*.

    Returns:
        TestResult capturing endpoint status and timing.

    Example:
        >>> assert_endpoint_healthy("my-model-endpoint", region="us-east-1")
    """
    client = _get_boto3_client("sagemaker", region)
    response = client.describe_endpoint(EndpointName=endpoint_name)
    status: str = response.get("EndpointStatus", "Unknown")

    passed = status == "InService"
    message = (
        f"Endpoint '{endpoint_name}' is InService"
        if passed
        else f"Endpoint '{endpoint_name}' is not InService: status={status}"
    )

    return assert_true(
        passed,
        name="aws.endpoint.health",
        message=message,
        severity=Severity.CRITICAL,
        endpoint_name=endpoint_name,
        status=status,
        region=region or "default",
    )


@timed_assertion
def assert_endpoint_latency(
    endpoint_name: str,
    max_p99_ms: float = 500.0,
    region: str | None = None,
    period: int = 300,
) -> TestResult:
    """Assert CloudWatch ModelLatency P99 is within threshold.

    Queries the ``AWS/SageMaker`` namespace for the ``ModelLatency`` metric
    using the ``p99`` extended statistic over *period* seconds.

    Args:
        endpoint_name: Name of the SageMaker endpoint.
        max_p99_ms: Maximum allowed P99 latency in milliseconds.
        region: AWS region name. Uses the default credential chain when *None*.
        period: CloudWatch aggregation window in seconds (default 300 = 5 min).

    Returns:
        TestResult with observed P99 latency and threshold details.

    Example:
        >>> assert_endpoint_latency("my-model-endpoint", max_p99_ms=300.0)
    """
    import datetime

    cloudwatch = _get_boto3_client("cloudwatch", region)
    end_time = datetime.datetime.now(datetime.UTC)
    start_time = end_time - datetime.timedelta(seconds=period)

    response = cloudwatch.get_metric_statistics(
        Namespace="AWS/SageMaker",
        MetricName="ModelLatency",
        Dimensions=[{"Name": "EndpointName", "Value": endpoint_name}],
        StartTime=start_time,
        EndTime=end_time,
        Period=period,
        ExtendedStatistics=["p99"],
    )

    datapoints = response.get("Datapoints", [])
    if not datapoints:
        # No datapoints means no traffic — treat as INFO, not a failure.
        return assert_true(
            True,
            name="aws.endpoint.latency",
            message=(
                f"No latency datapoints for '{endpoint_name}' in last {period}s "
                "(no traffic or metric not yet published)"
            ),
            severity=Severity.INFO,
            endpoint_name=endpoint_name,
            period_seconds=period,
            datapoints=0,
        )

    # SageMaker reports latency in microseconds; convert to ms.
    p99_us: float = max(dp["ExtendedStatistics"]["p99"] for dp in datapoints)
    p99_ms = p99_us / 1000.0

    passed = p99_ms <= max_p99_ms
    message = (
        f"P99 latency {p99_ms:.1f}ms within {max_p99_ms}ms threshold"
        if passed
        else f"P99 latency {p99_ms:.1f}ms exceeds {max_p99_ms}ms threshold"
    )

    return assert_true(
        passed,
        name="aws.endpoint.latency",
        message=message,
        severity=Severity.CRITICAL,
        endpoint_name=endpoint_name,
        p99_latency_ms=round(p99_ms, 2),
        max_p99_ms=max_p99_ms,
        period_seconds=period,
        datapoints=len(datapoints),
    )


@timed_assertion
def assert_endpoint_error_rate(
    endpoint_name: str,
    max_rate: float = 0.01,
    region: str | None = None,
    period: int = 300,
) -> TestResult:
    """Assert CloudWatch error rate (4XX + 5XX) is within threshold.

    Computes ``(Invocation4XXErrors + Invocation5XXErrors) / Invocations``
    over *period* seconds. Returns INFO (not CRITICAL) when there are no
    invocations in the window.

    Args:
        endpoint_name: Name of the SageMaker endpoint.
        max_rate: Maximum allowed error rate as a fraction (0.0–1.0).
        region: AWS region name. Uses the default credential chain when *None*.
        period: CloudWatch aggregation window in seconds (default 300 = 5 min).

    Returns:
        TestResult with computed error rate and threshold details.

    Example:
        >>> assert_endpoint_error_rate("my-model-endpoint", max_rate=0.005)
    """
    import datetime

    cloudwatch = _get_boto3_client("cloudwatch", region)
    end_time = datetime.datetime.now(datetime.UTC)
    start_time = end_time - datetime.timedelta(seconds=period)

    def _sum_metric(metric_name: str) -> float:
        resp = cloudwatch.get_metric_statistics(
            Namespace="AWS/SageMaker",
            MetricName=metric_name,
            Dimensions=[{"Name": "EndpointName", "Value": endpoint_name}],
            StartTime=start_time,
            EndTime=end_time,
            Period=period,
            Statistics=["Sum"],
        )
        datapoints = resp.get("Datapoints", [])
        return sum(dp["Sum"] for dp in datapoints) if datapoints else 0.0

    invocations = _sum_metric("Invocations")
    errors_4xx = _sum_metric("Invocation4XXErrors")
    errors_5xx = _sum_metric("Invocation5XXErrors")
    total_errors = errors_4xx + errors_5xx

    if invocations == 0:
        return assert_true(
            True,
            name="aws.endpoint.error_rate",
            message=(
                f"No invocations for '{endpoint_name}' in last {period}s "
                "(cannot compute error rate)"
            ),
            severity=Severity.INFO,
            endpoint_name=endpoint_name,
            period_seconds=period,
            invocations=0,
        )

    error_rate = total_errors / invocations
    passed = error_rate <= max_rate
    message = (
        f"Error rate {error_rate:.4f} within {max_rate} threshold"
        if passed
        else f"Error rate {error_rate:.4f} exceeds {max_rate} threshold "
        f"({int(total_errors)} errors / {int(invocations)} invocations)"
    )

    return assert_true(
        passed,
        name="aws.endpoint.error_rate",
        message=message,
        severity=Severity.CRITICAL,
        endpoint_name=endpoint_name,
        error_rate=round(error_rate, 6),
        max_rate=max_rate,
        invocations=int(invocations),
        errors_4xx=int(errors_4xx),
        errors_5xx=int(errors_5xx),
        period_seconds=period,
    )
