"""SageMaker Pipelines integration -- verify pipeline executions and step status.

SageMaker Pipelines orchestrate multi-step ML workflows: data processing,
training, evaluation, model registration, and deployment.  Unlike simple
SageMaker endpoints (covered in ``mltk.monitor.aws``), pipelines are DAGs
of steps that can fail independently, silently, or partially.

**Why this matters for ML teams:**

A SageMaker Pipeline execution can enter a ``Failed`` state because one
step timed out, ran out of disk, or received malformed input -- but the
pipeline service does not send push notifications by default.  Your model
registry still holds the last "good" model, your serving endpoint still
runs the old version, and nobody notices until the next manual check.
These assertions close that gap.

**Architecture:**

Uses ``boto3`` (AWS SDK for Python) to call the SageMaker API.  ``boto3``
is imported lazily so that importing this module does not fail when the
dependency is absent.  An ``ImportError`` with an installation hint is
raised only when the functions are actually called.

Typical usage::

    from mltk.integrations.sagemaker_pipeline import (
        assert_sagemaker_pipeline_success,
        assert_sagemaker_step_status,
    )

    # Verify the latest pipeline execution succeeded
    result = assert_sagemaker_pipeline_success(
        pipeline_name="my-training-pipeline",
        region="us-east-1",
    )

    # Verify a specific step in a known execution
    result = assert_sagemaker_step_status(
        execution_arn="arn:aws:sagemaker:us-east-1:123456:pipeline/my-pipeline/execution/abc",
        step_name="TrainModel",
    )
"""

from __future__ import annotations

from typing import Any

from mltk.core.assertion import assert_true, timed_assertion
from mltk.core.result import Severity, TestResult


def _get_sagemaker_client(region: str) -> Any:
    """Lazy-import boto3 and return a SageMaker client.

    ``boto3`` is an optional dependency.  This function raises a clear
    ``ImportError`` with an installation hint when the package is missing,
    rather than letting a bare ``ModuleNotFoundError`` propagate.

    Args:
        region: AWS region name (e.g. ``"us-east-1"``).

    Returns:
        A ``boto3`` SageMaker client configured for the given region.

    Raises:
        ImportError: When ``boto3`` is not installed.
    """
    try:
        import boto3  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "boto3 is required for SageMaker Pipeline assertions. "
            "Install it with: pip install mltk[aws]"
        ) from exc

    return boto3.client("sagemaker", region_name=region)


@timed_assertion
def assert_sagemaker_pipeline_success(
    pipeline_name: str,
    execution_arn: str | None = None,
    region: str = "us-east-1",
) -> TestResult:
    """Assert that a SageMaker Pipeline execution completed successfully.

    Queries the SageMaker API for the execution status and asserts it is
    ``"Succeeded"``.  Any other terminal state (``Failed``, ``Stopped``)
    or a still-running state (``Executing``, ``Stopping``) means the
    assertion fails.

    **When to use this:**

    - After triggering a retraining pipeline (manually or via EventBridge).
    - In CI/CD gates that prevent model promotion when training fails.
    - As part of a nightly health check on scheduled pipelines.

    **What can go wrong:**

    - A processing step runs out of disk and the execution enters
      ``Failed``, but no CloudWatch alarm is configured for pipeline
      failures.
    - The pipeline is ``Stopped`` due to a manual cancellation that
      nobody communicated to the team.
    - The ``execution_arn`` refers to an old execution while a newer one
      has failed silently.

    **How it works:**

    If ``execution_arn`` is provided, that specific execution is checked.
    Otherwise, the function lists executions for the pipeline sorted by
    creation time (descending) and picks the most recent one.  This
    "latest by default" behavior mirrors how teams typically think about
    pipeline health: "is the most recent run OK?"

    Args:
        pipeline_name: Name of the SageMaker Pipeline.  This is the
            pipeline identifier visible in the SageMaker Studio UI and
            used in ``create_pipeline()`` calls.
        execution_arn: ARN of a specific execution to check.  When
            ``None``, the latest execution is used.
        region: AWS region where the pipeline is deployed.

    Returns:
        TestResult with ``passed=True`` when the execution status is
        ``"Succeeded"``.  Details include ``pipeline_name``,
        ``execution_arn``, ``status``, ``start_time``, and ``end_time``.

    Example::

        >>> result = assert_sagemaker_pipeline_success(
        ...     pipeline_name="nightly-retrain",
        ...     region="us-west-2",
        ... )
    """
    try:
        client = _get_sagemaker_client(region)
    except ImportError as exc:
        return assert_true(
            False,
            name="integrations.sagemaker.pipeline_success",
            message=str(exc),
            severity=Severity.CRITICAL,
            pipeline_name=pipeline_name,
            error="boto3_not_installed",
        )

    try:
        # If no execution_arn provided, find the latest execution.
        if execution_arn is None:
            list_response = client.list_pipeline_executions(
                PipelineName=pipeline_name,
                SortBy="CreationTime",
                SortOrder="Descending",
                MaxResults=1,
            )
            executions = list_response.get("PipelineExecutionSummaries", [])
            if not executions:
                return assert_true(
                    False,
                    name="integrations.sagemaker.pipeline_success",
                    message=f"No executions found for pipeline '{pipeline_name}'",
                    severity=Severity.CRITICAL,
                    pipeline_name=pipeline_name,
                )
            execution_arn = executions[0]["PipelineExecutionArn"]

        # Describe the specific execution.
        describe_response = client.describe_pipeline_execution(
            PipelineExecutionArn=execution_arn,
        )

        status = describe_response.get("PipelineExecutionStatus", "Unknown")
        start_time = str(describe_response.get("CreationTime", "unknown"))
        end_time = str(describe_response.get("LastModifiedTime", "unknown"))

    except Exception as exc:
        return assert_true(
            False,
            name="integrations.sagemaker.pipeline_success",
            message=f"Failed to query SageMaker API: {exc}",
            severity=Severity.CRITICAL,
            pipeline_name=pipeline_name,
            execution_arn=execution_arn or "unknown",
            error=str(exc),
        )

    passed = status == "Succeeded"
    message = (
        f"Pipeline '{pipeline_name}' execution succeeded"
        if passed
        else f"Pipeline '{pipeline_name}' execution did not succeed: status={status}"
    )

    return assert_true(
        passed,
        name="integrations.sagemaker.pipeline_success",
        message=message,
        severity=Severity.CRITICAL,
        pipeline_name=pipeline_name,
        execution_arn=execution_arn,
        status=status,
        start_time=start_time,
        end_time=end_time,
    )


@timed_assertion
def assert_sagemaker_step_status(
    execution_arn: str,
    step_name: str,
    expected_status: str = "Succeeded",
    region: str = "us-east-1",
) -> TestResult:
    """Assert that a specific SageMaker Pipeline step has the expected status.

    While ``assert_sagemaker_pipeline_success`` checks the overall
    execution, this function drills into individual steps.  A pipeline
    might "succeed" overall (if configured with ``FailStepConfig`` to
    continue), but a specific step could be in ``Failed`` or ``Stopped``
    state.

    **When to use this:**

    - Verify that the ``TrainModel`` step completed, not just the pipeline.
    - Check that an evaluation step reached ``Succeeded`` before trusting
      its output metrics.
    - Detect ``Stopped`` steps in pipelines that use conditional logic
      (e.g., skip deployment if accuracy < threshold).

    **What can go wrong:**

    - A ``RegisterModel`` step fails because of a permissions error, but
      the pipeline continues because it has error handling.  The model
      is never registered, but no alert fires.
    - A ``QualityCheck`` step times out and enters ``Failed``, but the
      pipeline's conditional logic routes around it.

    **How it works:**

    Calls ``list_pipeline_execution_steps()`` for the given execution ARN
    and searches for a step matching ``step_name``.  Compares the step's
    ``StepStatus`` against ``expected_status``.

    Args:
        execution_arn: ARN of the pipeline execution containing the step.
        step_name: Name of the step to check (must match exactly).
        expected_status: Expected status string.  Common values are
            ``"Succeeded"``, ``"Failed"``, ``"Stopped"``, ``"Executing"``.
            Defaults to ``"Succeeded"``.
        region: AWS region where the pipeline is deployed.

    Returns:
        TestResult with ``passed=True`` when the step status matches
        ``expected_status``.  Details include ``step_name``,
        ``actual_status``, ``expected_status``, and ``execution_arn``.

    Example::

        >>> result = assert_sagemaker_step_status(
        ...     execution_arn="arn:aws:sagemaker:us-east-1:123:pipeline/p/execution/e",
        ...     step_name="TrainModel",
        ... )
    """
    try:
        client = _get_sagemaker_client(region)
    except ImportError as exc:
        return assert_true(
            False,
            name="integrations.sagemaker.step_status",
            message=str(exc),
            severity=Severity.CRITICAL,
            execution_arn=execution_arn,
            step_name=step_name,
            error="boto3_not_installed",
        )

    try:
        response = client.list_pipeline_execution_steps(
            PipelineExecutionArn=execution_arn,
        )
        steps = response.get("PipelineExecutionSteps", [])
    except Exception as exc:
        return assert_true(
            False,
            name="integrations.sagemaker.step_status",
            message=f"Failed to list pipeline steps: {exc}",
            severity=Severity.CRITICAL,
            execution_arn=execution_arn,
            step_name=step_name,
            error=str(exc),
        )

    # Find the target step by name.
    target_step = None
    for step in steps:
        if step.get("StepName") == step_name:
            target_step = step
            break

    if target_step is None:
        available_steps = [s.get("StepName", "unknown") for s in steps]
        return assert_true(
            False,
            name="integrations.sagemaker.step_status",
            message=(
                f"Step '{step_name}' not found in execution. "
                f"Available steps: {available_steps}"
            ),
            severity=Severity.CRITICAL,
            execution_arn=execution_arn,
            step_name=step_name,
            available_steps=available_steps,
        )

    actual_status = target_step.get("StepStatus", "Unknown")
    start_time = str(target_step.get("StartTime", "unknown"))
    end_time = str(target_step.get("EndTime", "unknown"))

    passed = actual_status == expected_status
    message = (
        f"Step '{step_name}' status is '{actual_status}' (expected '{expected_status}')"
        if passed
        else (
            f"Step '{step_name}' status is '{actual_status}', "
            f"expected '{expected_status}'"
        )
    )

    return assert_true(
        passed,
        name="integrations.sagemaker.step_status",
        message=message,
        severity=Severity.CRITICAL,
        execution_arn=execution_arn,
        step_name=step_name,
        actual_status=actual_status,
        expected_status=expected_status,
        start_time=start_time,
        end_time=end_time,
    )
