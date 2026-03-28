"""Tests for mltk.integrations.sagemaker_pipeline -- SageMaker Pipeline assertions.

All boto3 calls are fully mocked using ``unittest.mock``.  No AWS
credentials, network access, or running SageMaker service is required.

Each test re-imports the module inside its ``patch.dict`` context to prevent
module-level import caching from polluting between scenarios.  The import
AND the function call both happen inside the ``with`` block so that the
lazy ``import boto3`` inside ``_get_sagemaker_client`` resolves to the mock.

This follows the same pattern used in ``tests/test_monitor/test_aws.py``.

Each test follows the pattern:

    # SCENARIO: <what situation is being tested>
    # WHY: <reason this behaviour matters for ML teams>
    # EXPECTED: <what the test asserts>

Test coverage:
    1. Pipeline success with explicit ARN -- Succeeded status passes
    2. Pipeline failed -- Failed status raises CRITICAL
    3. Pipeline success with latest execution -- no ARN, picks latest
    4. No executions found -- empty execution list raises CRITICAL
    5. Step status matches -- specific step Succeeded
    6. Step status mismatch -- step Failed when Succeeded expected
    7. Step not found -- step_name absent from execution
    8. Missing boto3 -- ImportError raised with installation hint
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
    """Return a fresh MagicMock standing in for the entire boto3 package.

    The mock is configured so that ``boto3.client("sagemaker", ...)``
    returns a MagicMock client whose methods can be further configured
    per test.
    """
    return MagicMock()


def _clear_module_cache() -> None:
    """Remove cached sagemaker_pipeline module entries from sys.modules.

    Must be called inside the ``patch.dict`` context, before the fresh
    import, so that Python re-executes the module code with the mocked
    boto3.
    """
    for key in list(sys.modules.keys()):
        if "mltk.integrations.sagemaker_pipeline" in key:
            del sys.modules[key]


# ---------------------------------------------------------------------------
# assert_sagemaker_pipeline_success
# ---------------------------------------------------------------------------

class TestSagemakerPipelineSuccess:
    """Verify that assert_sagemaker_pipeline_success correctly interprets
    the SageMaker Pipeline execution status."""

    def test_pipeline_succeeded_with_arn(self) -> None:
        """SCENARIO: Describe execution returns PipelineExecutionStatus=Succeeded.

        WHY: The happy path -- a completed pipeline execution should pass
        the assertion.  This validates that the model retraining, data
        processing, or evaluation pipeline completed end-to-end.  If this
        wrongly fails, valid model promotions are blocked.

        EXPECTED: result.passed is True, status is "Succeeded",
        execution_arn matches input.
        """
        boto3_mock = _make_boto3_mock()
        client = boto3_mock.client.return_value
        client.describe_pipeline_execution.return_value = {
            "PipelineExecutionStatus": "Succeeded",
            "CreationTime": "2025-01-15T10:00:00Z",
            "LastModifiedTime": "2025-01-15T10:30:00Z",
        }

        with patch.dict("sys.modules", {"boto3": boto3_mock}):
            _clear_module_cache()
            from mltk.integrations.sagemaker_pipeline import assert_sagemaker_pipeline_success

            result = assert_sagemaker_pipeline_success(
                pipeline_name="my-pipeline",
                execution_arn="arn:aws:sagemaker:us-east-1:123:pipeline/my-pipeline/execution/e1",
                region="us-east-1",
            )

        assert result.passed is True
        assert result.details["status"] == "Succeeded"
        assert result.details["pipeline_name"] == "my-pipeline"
        assert "e1" in result.details["execution_arn"]
        assert result.name == "integrations.sagemaker.pipeline_success"

    def test_pipeline_failed_status(self) -> None:
        """SCENARIO: Pipeline execution reports PipelineExecutionStatus=Failed.

        WHY: A failed pipeline means the model was not updated.  Common
        causes: a processing step ran out of memory, a training job hit a
        NaN loss and was stopped, or an evaluation step found accuracy
        below threshold and raised an error.  The assertion must raise
        CRITICAL.

        EXPECTED: MltkAssertionError raised, status is "Failed".
        """
        boto3_mock = _make_boto3_mock()
        client = boto3_mock.client.return_value
        client.describe_pipeline_execution.return_value = {
            "PipelineExecutionStatus": "Failed",
            "CreationTime": "2025-01-15T10:00:00Z",
            "LastModifiedTime": "2025-01-15T10:05:00Z",
        }

        with patch.dict("sys.modules", {"boto3": boto3_mock}):
            _clear_module_cache()
            from mltk.integrations.sagemaker_pipeline import assert_sagemaker_pipeline_success

            with pytest.raises(MltkAssertionError) as exc_info:
                assert_sagemaker_pipeline_success(
                    pipeline_name="broken-pipeline",
                    execution_arn="arn:aws:sagemaker:us-east-1:123:pipeline/bp/execution/e2",
                )

        assert exc_info.value.result.details["status"] == "Failed"
        assert "Failed" in str(exc_info.value)

    def test_pipeline_success_latest_execution(self) -> None:
        """SCENARIO: No execution_arn provided; function finds the latest.

        WHY: The most common check is "did the most recent run succeed?"
        Teams should not need to look up the ARN manually.  This test
        verifies the list-then-describe flow works correctly.

        EXPECTED: result.passed is True, the latest execution ARN is used.
        """
        boto3_mock = _make_boto3_mock()
        client = boto3_mock.client.return_value

        latest_arn = "arn:aws:sagemaker:us-east-1:123:pipeline/p/execution/latest"
        client.list_pipeline_executions.return_value = {
            "PipelineExecutionSummaries": [
                {"PipelineExecutionArn": latest_arn},
            ],
        }
        client.describe_pipeline_execution.return_value = {
            "PipelineExecutionStatus": "Succeeded",
            "CreationTime": "2025-01-15T11:00:00Z",
            "LastModifiedTime": "2025-01-15T11:45:00Z",
        }

        with patch.dict("sys.modules", {"boto3": boto3_mock}):
            _clear_module_cache()
            from mltk.integrations.sagemaker_pipeline import assert_sagemaker_pipeline_success

            result = assert_sagemaker_pipeline_success(
                pipeline_name="auto-retrain",
            )

        assert result.passed is True
        assert result.details["execution_arn"] == latest_arn
        # Verify list was called with correct pipeline name
        client.list_pipeline_executions.assert_called_once()
        call_kwargs = client.list_pipeline_executions.call_args[1]
        assert call_kwargs["PipelineName"] == "auto-retrain"

    def test_pipeline_no_executions_found(self) -> None:
        """SCENARIO: Pipeline has no executions (newly created or purged).

        WHY: A pipeline with zero executions cannot have a "latest
        succeeded" run.  The assertion must fail clearly with a message
        indicating no executions exist, rather than crashing with an
        IndexError on an empty list.

        EXPECTED: MltkAssertionError raised, message mentions no executions.
        """
        boto3_mock = _make_boto3_mock()
        client = boto3_mock.client.return_value
        client.list_pipeline_executions.return_value = {
            "PipelineExecutionSummaries": [],
        }

        with patch.dict("sys.modules", {"boto3": boto3_mock}):
            _clear_module_cache()
            from mltk.integrations.sagemaker_pipeline import assert_sagemaker_pipeline_success

            with pytest.raises(MltkAssertionError) as exc_info:
                assert_sagemaker_pipeline_success(
                    pipeline_name="empty-pipeline",
                )

        assert "No executions found" in exc_info.value.result.message


# ---------------------------------------------------------------------------
# assert_sagemaker_step_status
# ---------------------------------------------------------------------------

class TestSagemakerStepStatus:
    """Verify that assert_sagemaker_step_status inspects individual steps."""

    def test_step_succeeded(self) -> None:
        """SCENARIO: The target step has StepStatus=Succeeded.

        WHY: Confirms a specific step (e.g., TrainModel) completed
        successfully.  Even when the overall pipeline succeeds, individual
        steps might have been skipped or conditionally bypassed.  This
        assertion verifies the exact step you care about.

        EXPECTED: result.passed is True, actual_status matches expected.
        """
        boto3_mock = _make_boto3_mock()
        client = boto3_mock.client.return_value
        client.list_pipeline_execution_steps.return_value = {
            "PipelineExecutionSteps": [
                {
                    "StepName": "PreprocessData",
                    "StepStatus": "Succeeded",
                    "StartTime": "2025-01-15T10:00:00Z",
                    "EndTime": "2025-01-15T10:10:00Z",
                },
                {
                    "StepName": "TrainModel",
                    "StepStatus": "Succeeded",
                    "StartTime": "2025-01-15T10:10:00Z",
                    "EndTime": "2025-01-15T10:40:00Z",
                },
            ],
        }

        with patch.dict("sys.modules", {"boto3": boto3_mock}):
            _clear_module_cache()
            from mltk.integrations.sagemaker_pipeline import assert_sagemaker_step_status

            result = assert_sagemaker_step_status(
                execution_arn="arn:aws:sagemaker:us-east-1:123:pipeline/p/execution/e1",
                step_name="TrainModel",
            )

        assert result.passed is True
        assert result.details["actual_status"] == "Succeeded"
        assert result.details["step_name"] == "TrainModel"
        assert result.name == "integrations.sagemaker.step_status"

    def test_step_failed(self) -> None:
        """SCENARIO: The target step has StepStatus=Failed.

        WHY: A training step that fails means the model was not updated.
        Even if the pipeline has error-handling logic that continues past
        failures, the ML team needs to know that training specifically
        failed so they can investigate the root cause (bad data, OOM,
        hyperparameter issue).

        EXPECTED: MltkAssertionError raised, actual_status is "Failed".
        """
        boto3_mock = _make_boto3_mock()
        client = boto3_mock.client.return_value
        client.list_pipeline_execution_steps.return_value = {
            "PipelineExecutionSteps": [
                {
                    "StepName": "TrainModel",
                    "StepStatus": "Failed",
                    "StartTime": "2025-01-15T10:10:00Z",
                    "EndTime": "2025-01-15T10:12:00Z",
                },
            ],
        }

        with patch.dict("sys.modules", {"boto3": boto3_mock}):
            _clear_module_cache()
            from mltk.integrations.sagemaker_pipeline import assert_sagemaker_step_status

            with pytest.raises(MltkAssertionError) as exc_info:
                assert_sagemaker_step_status(
                    execution_arn="arn:aws:sagemaker:us-east-1:123:pipeline/p/execution/e2",
                    step_name="TrainModel",
                )

        assert exc_info.value.result.details["actual_status"] == "Failed"
        assert exc_info.value.result.details["expected_status"] == "Succeeded"

    def test_step_not_found(self) -> None:
        """SCENARIO: The step_name does not exist in the execution.

        WHY: Pipeline refactors often rename steps (e.g., "Train" becomes
        "TrainXGBoost").  If the assertion silently passes because it
        cannot find the step, you have a false sense of security.  It
        must fail and list available steps to help diagnose the mismatch.

        EXPECTED: MltkAssertionError raised, available_steps listed.
        """
        boto3_mock = _make_boto3_mock()
        client = boto3_mock.client.return_value
        client.list_pipeline_execution_steps.return_value = {
            "PipelineExecutionSteps": [
                {
                    "StepName": "PreprocessData",
                    "StepStatus": "Succeeded",
                },
                {
                    "StepName": "EvaluateModel",
                    "StepStatus": "Succeeded",
                },
            ],
        }

        with patch.dict("sys.modules", {"boto3": boto3_mock}):
            _clear_module_cache()
            from mltk.integrations.sagemaker_pipeline import assert_sagemaker_step_status

            with pytest.raises(MltkAssertionError) as exc_info:
                assert_sagemaker_step_status(
                    execution_arn="arn:aws:sagemaker:us-east-1:123:pipeline/p/execution/e3",
                    step_name="TrainModel",
                )

        assert "not found" in exc_info.value.result.message
        assert "PreprocessData" in exc_info.value.result.details["available_steps"]
        assert "EvaluateModel" in exc_info.value.result.details["available_steps"]

    def test_missing_boto3_raises_helpful_error(self) -> None:
        """SCENARIO: boto3 is not installed.

        WHY: SageMaker assertions require the AWS SDK.  When it is
        missing, the error must be actionable: tell the user exactly what
        package to install (``pip install mltk[aws]``), not just show a
        bare ModuleNotFoundError traceback.

        EXPECTED: MltkAssertionError raised, message contains
        installation hint.
        """
        # Remove any cached module and boto3 from sys.modules, then make
        # import boto3 raise ImportError via a patched __import__.
        import builtins
        original_import = builtins.__import__

        def _import_no_boto3(name, *args, **kwargs):
            if name == "boto3":
                raise ImportError("No module named 'boto3'")
            return original_import(name, *args, **kwargs)

        # Ensure boto3 is not in sys.modules so the lazy import runs
        sys.modules.pop("boto3", None)
        _clear_module_cache()

        with patch("builtins.__import__", side_effect=_import_no_boto3):
            import mltk.integrations.sagemaker_pipeline as sm_mod

            with pytest.raises(MltkAssertionError) as exc_info:
                sm_mod.assert_sagemaker_step_status(
                    execution_arn="arn:test",
                    step_name="Train",
                )

        assert "mltk[aws]" in exc_info.value.result.message


# -------------------------------------------------------------------
# Parametrized & edge-case tests (hardening)
# -------------------------------------------------------------------


class TestSagemakerPipelineStatusParametrized:
    """Parametrize execution status variants."""

    @pytest.mark.parametrize(
        "status,should_pass",
        [
            ("Succeeded", True),
            ("Failed", False),
            ("Executing", False),
            ("Stopped", False),
        ],
    )
    def test_execution_status_variants(
        self, status: str, should_pass: bool
    ) -> None:
        """Each SageMaker status is handled correctly."""
        boto3_mock = _make_boto3_mock()
        client = boto3_mock.client.return_value
        arn = (
            "arn:aws:sagemaker:us-east-1:123:"
            "pipeline/p/execution/e-param"
        )
        client.describe_pipeline_execution.return_value = {
            "PipelineExecutionStatus": status,
            "CreationTime": "2025-06-01T00:00:00Z",
            "LastModifiedTime": "2025-06-01T00:30:00Z",
        }
        with patch.dict("sys.modules", {"boto3": boto3_mock}):
            _clear_module_cache()
            from mltk.integrations.sagemaker_pipeline import (
                assert_sagemaker_pipeline_success,
            )

            if should_pass:
                r = assert_sagemaker_pipeline_success(
                    pipeline_name="param-pipe",
                    execution_arn=arn,
                )
                assert r.passed is True
                assert r.details["status"] == status
            else:
                with pytest.raises(MltkAssertionError) as ei:
                    assert_sagemaker_pipeline_success(
                        pipeline_name="param-pipe",
                        execution_arn=arn,
                    )
                d = ei.value.result.details
                assert d["status"] == status


class TestSagemakerLatestNoArn:
    """Latest execution when no ARN provided."""

    def test_latest_execution_no_arn(self) -> None:
        """Picks latest when execution_arn is omitted."""
        boto3_mock = _make_boto3_mock()
        client = boto3_mock.client.return_value
        latest = (
            "arn:aws:sagemaker:us-east-1:123:"
            "pipeline/p/execution/latest-2"
        )
        client.list_pipeline_executions.return_value = {
            "PipelineExecutionSummaries": [
                {"PipelineExecutionArn": latest},
            ],
        }
        client.describe_pipeline_execution.return_value = {
            "PipelineExecutionStatus": "Succeeded",
            "CreationTime": "2025-06-01T00:00:00Z",
            "LastModifiedTime": "2025-06-01T00:30:00Z",
        }
        with patch.dict("sys.modules", {"boto3": boto3_mock}):
            _clear_module_cache()
            from mltk.integrations.sagemaker_pipeline import (
                assert_sagemaker_pipeline_success,
            )

            r = assert_sagemaker_pipeline_success(
                pipeline_name="auto-pipe",
            )
        assert r.passed is True
        assert r.details["execution_arn"] == latest


class TestSagemakerEmptyStepList:
    """Empty step list from execution."""

    def test_empty_step_list(self) -> None:
        """Step not found when execution has zero steps."""
        boto3_mock = _make_boto3_mock()
        client = boto3_mock.client.return_value
        client.list_pipeline_execution_steps.return_value = {
            "PipelineExecutionSteps": [],
        }
        with patch.dict("sys.modules", {"boto3": boto3_mock}):
            _clear_module_cache()
            from mltk.integrations.sagemaker_pipeline import (
                assert_sagemaker_step_status,
            )

            with pytest.raises(MltkAssertionError) as ei:
                assert_sagemaker_step_status(
                    execution_arn="arn:empty",
                    step_name="Train",
                )
        assert "not found" in ei.value.result.message
        avail = ei.value.result.details["available_steps"]
        assert avail == []


class TestSagemakerStepNotFoundLargeList:
    """Step not found in a 10-step execution."""

    def test_step_not_found_in_large_list(self) -> None:
        """Missing step among 10 existing steps."""
        boto3_mock = _make_boto3_mock()
        client = boto3_mock.client.return_value
        steps = [
            {
                "StepName": f"Step{i}",
                "StepStatus": "Succeeded",
            }
            for i in range(10)
        ]
        client.list_pipeline_execution_steps.return_value = {
            "PipelineExecutionSteps": steps,
        }
        with patch.dict("sys.modules", {"boto3": boto3_mock}):
            _clear_module_cache()
            from mltk.integrations.sagemaker_pipeline import (
                assert_sagemaker_step_status,
            )

            with pytest.raises(MltkAssertionError) as ei:
                assert_sagemaker_step_status(
                    execution_arn="arn:big",
                    step_name="MissingStep",
                )
        avail = ei.value.result.details["available_steps"]
        assert len(avail) == 10
        assert "MissingStep" not in avail
        assert "Step0" in avail
        assert "Step9" in avail
