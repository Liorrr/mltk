# SageMaker Pipeline Testing

Validate SageMaker pipeline executions, step-level status, and cross-step data flow. Catches orchestration failures, step dependency breakdowns, and silent data corruption in multi-step ML workflows.

**Module:** `mltk.integrations.sagemaker_pipeline`

**Install:** `pip install mltk[aws]`

---

## Why Test SageMaker Pipelines?

SageMaker Pipelines is an orchestration service for multi-step ML workflows. A typical pipeline chains together processing, training, evaluation, condition checks, registration, and deployment steps. Each step runs in its own container, passes data through S3 artifacts, and reports status independently.

The orchestration complexity creates failure modes that do not exist in single-step training:

**Step dependency failures.** Step B depends on Step A's output artifact in S3. Step A completes but writes to a different S3 prefix than Step B expects (a common misconfiguration after refactoring pipeline parameters). Step B reads an empty directory, trains on zero rows, and produces a model with random weights. Both steps report `Succeeded`. The pipeline reports `Succeeded`. The model is garbage.

**Conditional branch mismatch.** The pipeline has a `ConditionStep` that routes to either "register-model" or "send-alert" based on evaluation metrics. A code change modifies the evaluation output format from `{"accuracy": 0.95}` to `{"metrics": {"accuracy": 0.95}}`. The condition step cannot parse the new format, defaults to the else-branch, and sends an alert every night. No model is registered. The production model goes stale. The alert is ignored because it has been firing for weeks and everyone assumes it is a known issue.

**Partial execution.** A pipeline run is triggered, but a quota limit causes the training step to fail after 3 hours of preprocessing. The preprocessing artifacts are valid. The next nightly run skips preprocessing (cache hit) and proceeds to training with the previous night's features. If the data changed significantly between nights, the model trains on stale features. The pipeline shows green because each individual step succeeded -- it does not know the features are from the wrong date.

mltk's SageMaker pipeline assertions validate the actual outcome of pipeline executions: overall status, per-step status, and the data flow between steps.

---

## assert_sagemaker_pipeline_success

Assert that a SageMaker pipeline execution completed successfully. Checks the execution status, duration, and reports which steps failed if the pipeline did not succeed.

```python
from mltk.integrations.sagemaker_pipeline import assert_sagemaker_pipeline_success

result = assert_sagemaker_pipeline_success(
    pipeline_name="nightly-training-pipeline",
    execution_arn="arn:aws:sagemaker:us-east-1:123456789:pipeline/nightly/execution/abc123",
    region="us-east-1",
)
```

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `pipeline_name` | `str` | *(required)* | Name of the SageMaker pipeline |
| `execution_arn` | `str \| None` | `None` | Specific execution ARN to check. When `None`, checks the most recent execution of the named pipeline. |
| `region` | `str \| None` | `None` | AWS region name. Uses the default credential chain region when `None`. |
| `timeout_seconds` | `int` | `7200` | Maximum allowed execution duration. Executions exceeding this while still in `Executing` state are flagged as failures. |
| `severity` | `Severity` | `CRITICAL` | Severity level |

### Returns

`TestResult` with details:

- `pipeline_name` -- the pipeline name
- `execution_arn` -- the execution ARN that was checked
- `status` -- final execution status (`"Succeeded"`, `"Failed"`, `"Executing"`, `"Stopped"`)
- `duration_seconds` -- total execution duration
- `start_time` -- execution start timestamp
- `end_time` -- execution end timestamp (or `None` if still running)
- `failed_steps` -- list of step names that failed (empty if pipeline succeeded)
- `step_count` -- total number of steps in the execution

### How It Works

1. Calls `sagemaker_client.describe_pipeline_execution(PipelineExecutionArn=...)` to fetch execution details.
2. If `execution_arn` is `None`, calls `list_pipeline_executions` sorted by creation time descending, and takes the first result.
3. Checks the execution status. `"Succeeded"` is a pass. `"Failed"` and `"Stopped"` are failures. `"Executing"` is checked against `timeout_seconds`.
4. On failure, calls `list_pipeline_execution_steps` to identify which specific steps failed, and includes their names and failure reasons in the result details.

### Example

```python
import pytest
from mltk.integrations.sagemaker_pipeline import assert_sagemaker_pipeline_success


def test_nightly_pipeline_succeeded(sagemaker_client_fixture):
    """Most recent nightly pipeline execution completed successfully."""
    assert_sagemaker_pipeline_success(
        pipeline_name="nightly-training-pipeline",
        region="us-east-1",
        timeout_seconds=7200,  # 2 hour maximum
    )


def test_specific_execution_succeeded():
    """A specific triggered execution completed."""
    assert_sagemaker_pipeline_success(
        pipeline_name="ad-hoc-retrain",
        execution_arn="arn:aws:sagemaker:us-east-1:123456789:pipeline/ad-hoc-retrain/execution/xyz789",
        region="us-east-1",
    )
```

---

## assert_sagemaker_step_status

Assert that a specific step within a SageMaker pipeline execution reached the expected status. Use this to validate individual steps in complex pipelines where overall success is not enough -- you need to know that the right path was taken through conditional branches.

```python
from mltk.integrations.sagemaker_pipeline import assert_sagemaker_step_status

result = assert_sagemaker_step_status(
    pipeline_name="nightly-training-pipeline",
    execution_arn="arn:aws:sagemaker:us-east-1:123456789:pipeline/nightly/execution/abc123",
    step_name="RegisterModel",
    expected_status="Succeeded",
    region="us-east-1",
)
```

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `pipeline_name` | `str` | *(required)* | Name of the SageMaker pipeline |
| `execution_arn` | `str \| None` | `None` | Specific execution ARN. When `None`, uses the most recent execution. |
| `step_name` | `str` | *(required)* | The display name of the step to inspect |
| `expected_status` | `str` | `"Succeeded"` | Expected step status. Common values: `"Succeeded"`, `"Failed"`, `"Stopped"`, `"Starting"`, `"Executing"`. |
| `region` | `str \| None` | `None` | AWS region name |
| `check_outputs` | `bool` | `False` | When `True`, verifies the step produced non-empty output artifacts. Catches steps that succeed but write nothing. |
| `severity` | `Severity` | `CRITICAL` | Severity level |

### Returns

`TestResult` with details:

- `pipeline_name` -- the pipeline name
- `execution_arn` -- the execution ARN
- `step_name` -- the step that was inspected
- `actual_status` -- the step's actual status
- `expected_status` -- the expected status
- `step_type` -- the step type (e.g., `"Training"`, `"Processing"`, `"Condition"`, `"RegisterModel"`)
- `start_time` -- step start timestamp
- `end_time` -- step end timestamp
- `failure_reason` -- failure reason string (empty if the step succeeded)
- `has_outputs` -- whether the step produced output artifacts (only checked when `check_outputs=True`)

### How It Works

1. Calls `list_pipeline_execution_steps` with the execution ARN.
2. Finds the step matching `step_name` (case-sensitive match on `StepName`).
3. Compares the step's `StepStatus` against `expected_status`.
4. If `check_outputs=True`, inspects the step's `Metadata` field to verify output artifacts or model artifacts exist.
5. If the step is not found in the execution, the assertion fails with a message listing the available step names (to help debug typos in `step_name`).

### Example

```python
import pytest
from mltk.integrations.sagemaker_pipeline import assert_sagemaker_step_status


def test_model_registration_step(sm_region):
    """The RegisterModel step must succeed (model was good enough)."""
    assert_sagemaker_step_status(
        pipeline_name="nightly-training-pipeline",
        step_name="RegisterModel",
        expected_status="Succeeded",
        check_outputs=True,  # Verify it actually registered something
        region=sm_region,
    )


def test_evaluation_step_completed(sm_region):
    """Evaluation step must complete and produce outputs."""
    assert_sagemaker_step_status(
        pipeline_name="nightly-training-pipeline",
        step_name="EvaluateModel",
        expected_status="Succeeded",
        check_outputs=True,
        region=sm_region,
    )


def test_conditional_branch_taken(sm_region):
    """The condition step must have succeeded (meaning it evaluated the branch).

    This catches the case where the condition step fails to parse
    evaluation metrics and silently takes the wrong branch.
    """
    assert_sagemaker_step_status(
        pipeline_name="nightly-training-pipeline",
        step_name="CheckModelQuality",
        expected_status="Succeeded",
        region=sm_region,
    )
```

---

## pytest Integration: boto3 Fixture Pattern

### conftest.py

```python
# conftest.py
import os
import pytest
import boto3


@pytest.fixture(scope="session")
def sm_region():
    """AWS region from environment or default."""
    return os.environ.get("AWS_REGION", "us-east-1")


@pytest.fixture(scope="session")
def sagemaker_client(sm_region):
    """Session-scoped SageMaker client.

    Uses the standard AWS credential chain:
    1. Environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
    2. Shared credentials file (~/.aws/credentials)
    3. IAM role (EC2 instance profile, ECS task role, Lambda execution role)
    """
    return boto3.client("sagemaker", region_name=sm_region)


@pytest.fixture(scope="session")
def latest_execution_arn(sagemaker_client):
    """Get the most recent execution ARN for the nightly pipeline.

    Session-scoped so all tests validate the same execution.
    """
    response = sagemaker_client.list_pipeline_executions(
        PipelineName="nightly-training-pipeline",
        SortBy="CreationTime",
        SortOrder="Descending",
        MaxResults=1,
    )
    executions = response.get("PipelineExecutionSummaries", [])
    if not executions:
        pytest.skip("No executions found for nightly-training-pipeline")
    return executions[0]["PipelineExecutionArn"]
```

### LocalStack for Local Testing

For local development and CI environments without AWS access, use LocalStack to emulate SageMaker. This lets you run pipeline tests without incurring AWS costs or needing production credentials.

```bash
# Start LocalStack with SageMaker support
docker run -d --name localstack \
  -p 4566:4566 \
  -e SERVICES=sagemaker \
  localstack/localstack:latest
```

```python
# conftest.py -- LocalStack override
import os
import pytest
import boto3


@pytest.fixture(scope="session")
def sagemaker_client():
    """SageMaker client pointing at LocalStack for local testing."""
    use_local = os.environ.get("MLTK_USE_LOCALSTACK", "false").lower() == "true"

    if use_local:
        return boto3.client(
            "sagemaker",
            endpoint_url="http://localhost:4566",
            region_name="us-east-1",
            aws_access_key_id="test",
            aws_secret_access_key="test",
        )

    return boto3.client("sagemaker", region_name=os.environ.get("AWS_REGION", "us-east-1"))
```

```bash
# Run tests against LocalStack
MLTK_USE_LOCALSTACK=true pytest tests/test_sagemaker_pipeline.py -v

# Run tests against real AWS
pytest tests/test_sagemaker_pipeline.py -v
```

---

## CI/CD Pattern: Run Pipeline, Wait, Verify

The most powerful pattern combines pipeline triggering with mltk verification in a single CI/CD job. The job starts a pipeline execution, polls until completion, then runs mltk assertions as a deployment gate.

### GitHub Actions example

```yaml
# .github/workflows/ml-pipeline-verify.yml
name: ML Pipeline Verification

on:
  schedule:
    - cron: "0 2 * * *"  # 2 AM nightly
  workflow_dispatch:

jobs:
  verify-pipeline:
    runs-on: ubuntu-latest
    permissions:
      id-token: write  # For OIDC auth to AWS
    steps:
      - uses: actions/checkout@v4

      - uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: arn:aws:iam::123456789:role/ci-ml-pipeline
          aws-region: us-east-1

      - name: Start pipeline execution
        id: start
        run: |
          EXECUTION_ARN=$(aws sagemaker start-pipeline-execution \
            --pipeline-name nightly-training-pipeline \
            --query 'PipelineExecutionArn' \
            --output text)
          echo "execution_arn=$EXECUTION_ARN" >> "$GITHUB_OUTPUT"

      - name: Wait for pipeline completion
        run: |
          aws sagemaker wait pipeline-execution-complete \
            --pipeline-execution-arn ${{ steps.start.outputs.execution_arn }}
        timeout-minutes: 120

      - name: Install mltk
        run: pip install "mltk[aws]"

      - name: Verify pipeline with mltk
        env:
          EXECUTION_ARN: ${{ steps.start.outputs.execution_arn }}
        run: |
          pytest tests/test_sagemaker_pipeline.py -v \
            --mltk-report \
            -k "test_pipeline_success or test_model_registered or test_evaluation_step"

      - name: Upload mltk report
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: mltk-pipeline-report
          path: mltk-reports/
```

### Test file used by CI

```python
# tests/test_sagemaker_pipeline.py
import os
import pytest
from mltk.integrations.sagemaker_pipeline import (
    assert_sagemaker_pipeline_success,
    assert_sagemaker_step_status,
)


@pytest.fixture
def execution_arn():
    """Execution ARN from environment (set by CI) or latest."""
    return os.environ.get("EXECUTION_ARN")


def test_pipeline_success(execution_arn):
    """Pipeline execution completed within 2 hours."""
    assert_sagemaker_pipeline_success(
        pipeline_name="nightly-training-pipeline",
        execution_arn=execution_arn,
        timeout_seconds=7200,
        region="us-east-1",
    )


def test_model_registered(execution_arn):
    """RegisterModel step succeeded and produced outputs."""
    assert_sagemaker_step_status(
        pipeline_name="nightly-training-pipeline",
        execution_arn=execution_arn,
        step_name="RegisterModel",
        expected_status="Succeeded",
        check_outputs=True,
        region="us-east-1",
    )


def test_evaluation_step(execution_arn):
    """Evaluation step completed (metrics were computed)."""
    assert_sagemaker_step_status(
        pipeline_name="nightly-training-pipeline",
        execution_arn=execution_arn,
        step_name="EvaluateModel",
        expected_status="Succeeded",
        region="us-east-1",
    )
```

### The verification sequence

```
CI trigger (cron or manual)
    |
    v
Start pipeline execution (aws sagemaker start-pipeline-execution)
    |
    v
Wait for completion (aws sagemaker wait pipeline-execution-complete)
    |
    v
Run mltk assertions (pytest tests/test_sagemaker_pipeline.py)
    |
    +-- assert_sagemaker_pipeline_success   -> Did the pipeline finish OK?
    +-- assert_sagemaker_step_status        -> Did each step reach the right state?
    |
    v
Gate decision: deploy model or block and alert
```

This pattern ensures that no model is deployed without passing external verification. The pipeline's own success status is necessary but not sufficient -- mltk assertions provide the independent check that catches the silent failure modes described above.

---
