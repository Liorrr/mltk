"""Tests for mltk.integrations.kubeflow -- Kubeflow Pipelines assertions.

All HTTP calls to the Kubeflow Pipelines REST API are fully mocked using
``unittest.mock.patch`` on ``urllib.request.urlopen``.  No network access
or running KFP instance is required.

Each test follows the pattern:

    # SCENARIO: <what situation is being tested>
    # WHY: <reason this behaviour matters for ML teams>
    # EXPECTED: <what the test asserts>

Test coverage:
    1. Pipeline success -- SUCCEEDED state passes
    2. Pipeline failed -- FAILED state raises CRITICAL
    3. Pipeline still running -- non-terminal state raises CRITICAL
    4. Step outputs present -- all expected artifacts found
    5. Step outputs missing -- subset of artifacts missing
    6. Step not found -- step_name does not exist in run
    7. Network error -- urllib raises, graceful failure
    8. Malformed JSON response -- unparseable body, graceful failure
"""

from __future__ import annotations

import json
import urllib.error
from unittest.mock import MagicMock, patch

import pytest

from mltk.core.assertion import MltkAssertionError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_urlopen_response(data: dict) -> MagicMock:
    """Build a mock that makes ``urllib.request.urlopen`` return *data* as JSON.

    The mock implements the context-manager protocol (``__enter__`` /
    ``__exit__``) because production code uses ``with urlopen(...) as r:``.
    The ``read()`` method returns the UTF-8 encoded JSON body.

    Args:
        data: Dictionary to serialize as the HTTP response body.

    Returns:
        MagicMock suitable for ``patch("urllib.request.urlopen", return_value=...)``.
    """
    body = json.dumps(data).encode("utf-8")
    response = MagicMock()
    response.read.return_value = body
    response.__enter__ = MagicMock(return_value=response)
    response.__exit__ = MagicMock(return_value=False)
    return response


# ---------------------------------------------------------------------------
# assert_kubeflow_pipeline_success
# ---------------------------------------------------------------------------

class TestKubeflowPipelineSuccess:
    """Verify that assert_kubeflow_pipeline_success correctly interprets
    the KFP v2 run state field."""

    def test_pipeline_succeeded(self) -> None:
        """SCENARIO: KFP API returns state=SUCCEEDED for the run.

        WHY: The happy path -- a completed retraining pipeline should
        produce a passing assertion so that downstream gates (model
        promotion, endpoint update) can proceed.  If this wrongly fails,
        valid model updates are blocked.

        EXPECTED: result.passed is True, state detail is "SUCCEEDED",
        pipeline_name and timing details are captured.
        """
        from mltk.integrations.kubeflow import assert_kubeflow_pipeline_success

        api_response = {
            "state": "SUCCEEDED",
            "created_at": "2025-01-15T10:00:00Z",
            "finished_at": "2025-01-15T10:30:00Z",
            "display_name": "nightly-retrain",
        }
        mock_resp = _mock_urlopen_response(api_response)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = assert_kubeflow_pipeline_success(
                run_id="run-001",
                host="http://kubeflow.test:8080",
            )

        assert result.passed is True
        assert result.details["state"] == "SUCCEEDED"
        assert result.details["run_id"] == "run-001"
        assert result.details["pipeline_name"] == "nightly-retrain"
        assert result.name == "integrations.kubeflow.pipeline_success"

    def test_pipeline_failed(self) -> None:
        """SCENARIO: KFP API returns state=FAILED for the run.

        WHY: A failed pipeline means the model was not retrained.  The
        assertion must raise a CRITICAL error so that CI/CD pipelines
        stop and the team is notified.  If this silently passes, stale
        models continue serving predictions.

        EXPECTED: MltkAssertionError is raised, state is "FAILED".
        """
        from mltk.integrations.kubeflow import assert_kubeflow_pipeline_success

        api_response = {
            "state": "FAILED",
            "created_at": "2025-01-15T10:00:00Z",
            "finished_at": "2025-01-15T10:05:00Z",
            "display_name": "broken-pipeline",
        }
        mock_resp = _mock_urlopen_response(api_response)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            with pytest.raises(MltkAssertionError) as exc_info:
                assert_kubeflow_pipeline_success(run_id="run-002")

        assert "FAILED" in str(exc_info.value)
        assert exc_info.value.result.details["state"] == "FAILED"

    def test_pipeline_still_running(self) -> None:
        """SCENARIO: KFP API returns state=RUNNING (not yet complete).

        WHY: A pipeline that is still running has not produced final
        artifacts.  Treating it as "succeeded" would let downstream
        gates pass prematurely.  The assertion should fail clearly so
        callers know to wait or investigate why the run is stuck.

        EXPECTED: MltkAssertionError raised, state is "RUNNING".
        """
        from mltk.integrations.kubeflow import assert_kubeflow_pipeline_success

        api_response = {
            "state": "RUNNING",
            "created_at": "2025-01-15T10:00:00Z",
            "display_name": "slow-pipeline",
        }
        mock_resp = _mock_urlopen_response(api_response)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            with pytest.raises(MltkAssertionError) as exc_info:
                assert_kubeflow_pipeline_success(run_id="run-003")

        assert exc_info.value.result.details["state"] == "RUNNING"

    def test_pipeline_network_error(self) -> None:
        """SCENARIO: The KFP API is unreachable (network error).

        WHY: Infrastructure failures (DNS, firewall, namespace mismatch)
        should not crash the test runner with an unhandled exception.
        The assertion must return a clear CRITICAL failure with the error
        details so operators can diagnose the connectivity issue.

        EXPECTED: MltkAssertionError raised, error detail contains
        the network error message.
        """
        from mltk.integrations.kubeflow import assert_kubeflow_pipeline_success

        with patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.URLError("Connection refused"),
        ):
            with pytest.raises(MltkAssertionError) as exc_info:
                assert_kubeflow_pipeline_success(run_id="run-004")

        assert "Connection refused" in exc_info.value.result.details["error"]
        assert exc_info.value.result.passed is False


# ---------------------------------------------------------------------------
# assert_kubeflow_step_outputs
# ---------------------------------------------------------------------------

class TestKubeflowStepOutputs:
    """Verify that assert_kubeflow_step_outputs checks artifact presence."""

    def test_step_outputs_all_present(self) -> None:
        """SCENARIO: Step produced all expected artifacts (model + metrics).

        WHY: A training step that outputs both a model artifact and
        evaluation metrics is the happy path.  The assertion confirms
        the step did not just "succeed" (exit code 0) but actually
        produced the outputs that downstream steps depend on.

        EXPECTED: result.passed is True, missing_artifacts is empty.
        """
        from mltk.integrations.kubeflow import assert_kubeflow_step_outputs

        api_response = {
            "run_details": {
                "task_details": [
                    {
                        "display_name": "train-model",
                        "outputs": {
                            "artifacts": [
                                {"name": "model"},
                                {"name": "metrics"},
                            ],
                        },
                    },
                ],
            },
        }
        mock_resp = _mock_urlopen_response(api_response)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = assert_kubeflow_step_outputs(
                run_id="run-010",
                step_name="train-model",
                expected_artifacts=["model", "metrics"],
            )

        assert result.passed is True
        assert result.details["missing_artifacts"] == []
        assert result.name == "integrations.kubeflow.step_outputs"

    def test_step_outputs_missing_artifact(self) -> None:
        """SCENARIO: Step produced 'model' but is missing 'metrics'.

        WHY: A training step that succeeds but does not write evaluation
        metrics means the model was trained but not evaluated.  Downstream
        model registry logic that depends on metrics will silently use
        stale values.  The assertion must fail and list what is missing.

        EXPECTED: MltkAssertionError raised, missing_artifacts
        contains "metrics".
        """
        from mltk.integrations.kubeflow import assert_kubeflow_step_outputs

        api_response = {
            "run_details": {
                "task_details": [
                    {
                        "display_name": "train-model",
                        "outputs": {
                            "artifacts": [
                                {"name": "model"},
                            ],
                        },
                    },
                ],
            },
        }
        mock_resp = _mock_urlopen_response(api_response)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            with pytest.raises(MltkAssertionError) as exc_info:
                assert_kubeflow_step_outputs(
                    run_id="run-011",
                    step_name="train-model",
                    expected_artifacts=["model", "metrics"],
                )

        assert "metrics" in exc_info.value.result.details["missing_artifacts"]
        assert "model" not in exc_info.value.result.details["missing_artifacts"]

    def test_step_not_found(self) -> None:
        """SCENARIO: The specified step_name does not exist in the run.

        WHY: A renamed or removed pipeline step will silently "pass"
        artifact checks if the code does not handle missing steps.
        This commonly happens after pipeline refactors where step names
        change (e.g., "train" becomes "train-xgboost").

        EXPECTED: MltkAssertionError raised, message mentions step not
        found, available_steps lists what does exist.
        """
        from mltk.integrations.kubeflow import assert_kubeflow_step_outputs

        api_response = {
            "run_details": {
                "task_details": [
                    {
                        "display_name": "preprocess-data",
                        "outputs": {"artifacts": []},
                    },
                ],
            },
        }
        mock_resp = _mock_urlopen_response(api_response)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            with pytest.raises(MltkAssertionError) as exc_info:
                assert_kubeflow_step_outputs(
                    run_id="run-012",
                    step_name="train-model",
                    expected_artifacts=["model"],
                )

        assert "not found" in exc_info.value.result.message
        assert "preprocess-data" in exc_info.value.result.details["available_steps"]

    def test_step_outputs_malformed_json(self) -> None:
        """SCENARIO: KFP API returns a body that is not valid JSON.

        WHY: API version mismatches, proxy servers, or authentication
        pages can return HTML instead of JSON.  The assertion must not
        crash with an unhandled JSONDecodeError -- it should produce a
        clean CRITICAL failure that tells the operator what went wrong.

        EXPECTED: MltkAssertionError raised, error detail present.
        """
        from mltk.integrations.kubeflow import assert_kubeflow_step_outputs

        # Return a mock whose read() gives invalid JSON
        bad_response = MagicMock()
        bad_response.read.return_value = b"<html>Not Found</html>"
        bad_response.__enter__ = MagicMock(return_value=bad_response)
        bad_response.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=bad_response):
            with pytest.raises(MltkAssertionError) as exc_info:
                assert_kubeflow_step_outputs(
                    run_id="run-013",
                    step_name="train-model",
                    expected_artifacts=["model"],
                )

        assert exc_info.value.result.passed is False
        assert "error" in exc_info.value.result.details
