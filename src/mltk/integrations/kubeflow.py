"""Kubeflow Pipelines integration -- verify pipeline runs and step outputs.

ML pipelines orchestrated by Kubeflow can fail silently: a training step
might time out, an evaluation step might produce empty metrics, or the
entire run might stall in a "RUNNING" state indefinitely.  These assertions
let you verify pipeline health as part of your ML test suite.

**Why this matters for ML teams:**

A failed Kubeflow pipeline means your model was not retrained, your
evaluation metrics were not refreshed, or your data preprocessing did
not complete.  Downstream systems (model registries, serving endpoints,
dashboards) will continue using stale artifacts without any notification.
These assertions catch that gap.

**Architecture:**

All communication with the Kubeflow Pipelines API uses ``urllib`` from the
Python standard library -- no extra dependency is needed.  The REST API
follows the KFP v2 (``/apis/v2beta1/``) schema.

Typical usage::

    from mltk.integrations.kubeflow import (
        assert_kubeflow_pipeline_success,
        assert_kubeflow_step_outputs,
    )

    # After triggering a pipeline run, verify it succeeded
    result = assert_kubeflow_pipeline_success(run_id="abc-123")

    # Verify the training step produced a model artifact
    result = assert_kubeflow_step_outputs(
        run_id="abc-123",
        step_name="train-model",
        expected_artifacts=["model"],
    )
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from mltk.core.assertion import assert_true, timed_assertion
from mltk.core.result import Severity, TestResult


def _kfp_api_get(url: str, timeout_seconds: int = 60) -> dict[str, Any]:
    """Perform a GET request against the Kubeflow Pipelines REST API.

    Uses ``urllib.request`` (stdlib) so there is no external dependency.
    Parses the JSON response body and returns it as a dictionary.

    Args:
        url: Full URL to the KFP API endpoint.
        timeout_seconds: HTTP request timeout in seconds.

    Returns:
        Parsed JSON response as a dict.

    Raises:
        urllib.error.URLError: On network-level failures (DNS, connection
            refused, TLS errors).
        urllib.error.HTTPError: On non-2xx HTTP responses from the API.
        json.JSONDecodeError: If the response body is not valid JSON.
    """
    request = urllib.request.Request(url, method="GET")
    request.add_header("Accept", "application/json")
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        body = response.read().decode("utf-8")
    return json.loads(body)


@timed_assertion
def assert_kubeflow_pipeline_success(
    run_id: str,
    host: str = "http://localhost:8080",
    namespace: str = "kubeflow",
    timeout_seconds: int = 60,
) -> TestResult:
    """Assert that a Kubeflow Pipeline run completed successfully.

    Queries the KFP v2 REST API for the run status and asserts it is
    ``"SUCCEEDED"``.  Any other terminal state (``FAILED``, ``SKIPPED``,
    ``CANCELING``, ``CANCELED``) or a still-running state means the
    assertion fails.

    **When to use this:**

    - After triggering a retraining pipeline via the KFP SDK or REST API.
    - In nightly CI jobs that validate the training infrastructure.
    - As a gate before promoting a new model version to serving.

    **What can go wrong:**

    - Pipeline stays ``RUNNING`` because a step is stuck (OOM, node
      scheduling failure, image pull error).
    - Pipeline reports ``FAILED`` due to a data validation error in the
      first step, but no alert fires.
    - The KFP host is unreachable (network issue, namespace mismatch).

    Args:
        run_id: The UUID of the pipeline run to check.  Obtained from the
            KFP SDK's ``create_run_from_pipeline_func()`` return value or
            from the Kubeflow UI URL.
        host: Base URL of the Kubeflow Pipelines API server.  Defaults to
            ``http://localhost:8080`` which matches a port-forwarded local
            setup.  In production this might be
            ``https://kubeflow.example.com``.
        namespace: Kubernetes namespace where KFP is deployed.  Used as a
            query parameter in the API call.  Defaults to ``"kubeflow"``.
        timeout_seconds: HTTP request timeout in seconds.

    Returns:
        TestResult with ``passed=True`` when the run state is
        ``"SUCCEEDED"``, otherwise ``passed=False``.  Details include
        ``run_id``, ``state``, ``created_at``, ``finished_at``, and
        ``pipeline_name``.

    Example::

        >>> result = assert_kubeflow_pipeline_success(
        ...     run_id="550e8400-e29b-41d4-a716-446655440000",
        ...     host="https://kubeflow.internal.example.com",
        ... )
    """
    url = f"{host.rstrip('/')}/apis/v2beta1/runs/{run_id}?namespace={namespace}"

    try:
        data = _kfp_api_get(url, timeout_seconds=timeout_seconds)
    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError) as exc:
        return assert_true(
            False,
            name="integrations.kubeflow.pipeline_success",
            message=f"Failed to query Kubeflow API: {exc}",
            severity=Severity.CRITICAL,
            run_id=run_id,
            host=host,
            error=str(exc),
        )

    state = data.get("state", data.get("run_details", {}).get("state", "UNKNOWN"))
    created_at = data.get("created_at", "unknown")
    finished_at = data.get("finished_at", "unknown")
    pipeline_name = data.get(
        "display_name",
        data.get("pipeline_spec", {}).get("pipeline_name", "unknown"),
    )

    passed = state == "SUCCEEDED"
    message = (
        f"Pipeline run '{run_id}' succeeded"
        if passed
        else f"Pipeline run '{run_id}' did not succeed: state={state}"
    )

    return assert_true(
        passed,
        name="integrations.kubeflow.pipeline_success",
        message=message,
        severity=Severity.CRITICAL,
        run_id=run_id,
        state=state,
        created_at=created_at,
        finished_at=finished_at,
        pipeline_name=pipeline_name,
    )


@timed_assertion
def assert_kubeflow_step_outputs(
    run_id: str,
    step_name: str,
    expected_artifacts: list[str],
    host: str = "http://localhost:8080",
    timeout_seconds: int = 60,
) -> TestResult:
    """Assert that a Kubeflow pipeline step produced expected output artifacts.

    After a pipeline completes, each step should produce specific artifacts.
    A training step should produce a ``model`` artifact; an evaluation step
    should produce ``metrics``; a preprocessing step should produce a
    ``dataset``.  Missing artifacts mean the step ran but did not produce
    the output that downstream steps or model registries expect.

    **When to use this:**

    - Validate that the training step actually wrote a model artifact.
    - Confirm evaluation metrics were generated (not just that the step
      "succeeded" -- a step can succeed but write no outputs).
    - Detect broken artifact paths after infrastructure changes (bucket
      moves, permission changes).

    **How it works:**

    Queries the KFP v2 artifacts API filtered by the run ID, then looks
    for the specified step's output artifacts.  Compares the set of
    artifact names produced against the ``expected_artifacts`` list.

    Args:
        run_id: UUID of the pipeline run.
        step_name: Display name of the pipeline step (task) to inspect.
        expected_artifacts: List of artifact names expected as outputs
            from this step (e.g. ``["model", "metrics"]``).
        host: Base URL of the KFP API server.
        timeout_seconds: HTTP request timeout in seconds.

    Returns:
        TestResult with ``passed=True`` when all expected artifacts are
        present.  Details include ``step_name``, ``expected_artifacts``,
        ``actual_artifacts``, and ``missing_artifacts``.

    Example::

        >>> result = assert_kubeflow_step_outputs(
        ...     run_id="abc-123",
        ...     step_name="train-model",
        ...     expected_artifacts=["model"],
        ... )
    """
    url = f"{host.rstrip('/')}/apis/v2beta1/runs/{run_id}"

    try:
        data = _kfp_api_get(url, timeout_seconds=timeout_seconds)
    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError) as exc:
        return assert_true(
            False,
            name="integrations.kubeflow.step_outputs",
            message=f"Failed to query Kubeflow API: {exc}",
            severity=Severity.CRITICAL,
            run_id=run_id,
            step_name=step_name,
            error=str(exc),
        )

    # Navigate the run detail structure to find the target step.
    # KFP v2 nests task details under run_details -> task_details (list).
    task_details = data.get("run_details", {}).get("task_details", [])
    step_detail = None
    for task in task_details:
        if task.get("display_name") == step_name or task.get("task_id") == step_name:
            step_detail = task
            break

    if step_detail is None:
        return assert_true(
            False,
            name="integrations.kubeflow.step_outputs",
            message=f"Step '{step_name}' not found in pipeline run '{run_id}'",
            severity=Severity.CRITICAL,
            run_id=run_id,
            step_name=step_name,
            expected_artifacts=expected_artifacts,
            available_steps=[t.get("display_name", "unknown") for t in task_details],
        )

    # Extract output artifact names from the step detail.
    outputs = step_detail.get("outputs", {})
    artifact_list = outputs.get("artifacts", [])
    actual_names = [a.get("name", a.get("display_name", "")) for a in artifact_list]

    expected_set = set(expected_artifacts)
    actual_set = set(actual_names)
    missing = expected_set - actual_set

    passed = len(missing) == 0
    message = (
        f"Step '{step_name}' produced all expected artifacts: {sorted(expected_set)}"
        if passed
        else (
            f"Step '{step_name}' missing artifacts: {sorted(missing)} "
            f"(found: {sorted(actual_set)})"
        )
    )

    return assert_true(
        passed,
        name="integrations.kubeflow.step_outputs",
        message=message,
        severity=Severity.CRITICAL,
        run_id=run_id,
        step_name=step_name,
        expected_artifacts=sorted(expected_set),
        actual_artifacts=sorted(actual_set),
        missing_artifacts=sorted(missing),
    )
