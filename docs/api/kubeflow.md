# Kubeflow Pipelines Testing

Validate Kubeflow pipeline runs, step outputs, and artifact integrity from your test suite. Catches silent failures, missing artifacts, zombie runs, and step-level regressions that the Kubeflow UI does not surface.

**Module:** `mltk.integrations.kubeflow`

**Install:** `pip install mltk[kubeflow]`

---

## Why Test Kubeflow Pipelines?

Consider a team running a training pipeline on Kubeflow Pipelines (KFP) every night. The pipeline has five steps: data ingestion, preprocessing, training, evaluation, and model registration. The Kubeflow UI shows a green checkmark on the run. Everyone assumes the model is fresh.

Three weeks later, a data scientist notices the production model has not improved since March. Investigation reveals:

1. **Silent evaluation failure.** The evaluation step caught an exception internally, logged a warning, and returned a default metric dict with `accuracy: 0.0`. KFP marked the step as "Succeeded" because the container exited with code 0. No alert fired.

2. **Missing artifact.** The training step was supposed to produce a `model.pkl` artifact. A disk pressure event caused the artifact upload to fail silently. The registration step received an empty path, registered a stale model from cache, and exited successfully.

3. **Zombie run.** A previous run got stuck in "Running" state due to a node preemption. It held a lock on shared storage, causing the nightly run to queue indefinitely. The cron trigger created a new run every night, but none could start. The Kubeflow dashboard showed 23 "Running" runs, none of which were actually running.

These are not hypothetical scenarios. They are the three most common failure modes in production Kubeflow deployments, and all three share the same characteristic: KFP reports success when the pipeline has actually failed.

mltk's Kubeflow assertions address this gap by inspecting the actual state of pipeline runs and their outputs after execution, from your pytest suite, where failures are loud and blocking.

---

## assert_kubeflow_pipeline_success

Assert that a Kubeflow pipeline run completed successfully within a timeout window. Checks the run's final status, duration, and optionally verifies that the run is not a stale cached result.

```python
from mltk.integrations.kubeflow import assert_kubeflow_pipeline_success

result = assert_kubeflow_pipeline_success(
    client=kfp_client,
    run_id="a1b2c3d4-5678-90ab-cdef-1234567890ab",
    timeout_seconds=3600,
)
```

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `client` | `kfp.Client` | *(required)* | Authenticated KFP client instance |
| `run_id` | `str` | *(required)* | The pipeline run ID to check (UUID from KFP) |
| `timeout_seconds` | `int` | `3600` | Maximum allowed run duration in seconds. Runs exceeding this are flagged as failures even if still "Running". |
| `check_cache` | `bool` | `False` | When `True`, fails if the run was served entirely from cache (all steps show `Cached`). Catches stale-model-from-cache scenarios. |
| `severity` | `Severity` | `CRITICAL` | Severity level for the assertion |

### Returns

`TestResult` with details:

- `run_id` -- the pipeline run ID
- `status` -- final run status string (e.g., `"Succeeded"`, `"Failed"`, `"Running"`)
- `duration_seconds` -- actual run duration
- `created_at` -- run creation timestamp
- `finished_at` -- run completion timestamp (or `None` if still running)
- `all_cached` -- `True` if every step was served from cache

### How It Works

1. Calls `client.get_run(run_id)` to fetch the run details.
2. Checks the run status. Valid success statuses: `"Succeeded"`. Any other terminal status (`"Failed"`, `"Error"`, `"Skipped"`) is a failure.
3. If the run is still `"Running"`, compares the elapsed time against `timeout_seconds`. Exceeding the timeout produces a failure with the message indicating a potential zombie run.
4. If `check_cache=True`, inspects each step's status. If all steps show `"Cached"`, the assertion fails with a warning that the run produced no fresh results.

### Example

```python
import pytest
from kfp import Client as KfpClient
from mltk.integrations.kubeflow import assert_kubeflow_pipeline_success


@pytest.fixture(scope="session")
def kfp_client():
    """KFP client pointing at the cluster's Kubeflow endpoint."""
    return KfpClient(host="https://kubeflow.internal/pipeline")


@pytest.fixture
def latest_run_id(kfp_client):
    """Get the most recent run from the nightly experiment."""
    experiment = kfp_client.get_experiment(experiment_name="nightly-training")
    runs = kfp_client.list_runs(
        experiment_id=experiment.experiment_id,
        sort_by="created_at desc",
        page_size=1,
    )
    assert runs.runs, "No runs found in nightly-training experiment"
    return runs.runs[0].run_id


def test_nightly_pipeline_succeeded(kfp_client, latest_run_id):
    """Nightly training pipeline completed successfully within 1 hour."""
    assert_kubeflow_pipeline_success(
        client=kfp_client,
        run_id=latest_run_id,
        timeout_seconds=3600,
        check_cache=True,  # Fail if all steps were cached (stale model)
    )
```

---

## assert_kubeflow_step_outputs

Assert that specific pipeline steps produced expected output artifacts. Validates artifact existence, size, and optionally content hash. This catches the "missing artifact" failure mode where a step exits successfully but fails to upload its output.

```python
from mltk.integrations.kubeflow import assert_kubeflow_step_outputs

result = assert_kubeflow_step_outputs(
    client=kfp_client,
    run_id="a1b2c3d4-5678-90ab-cdef-1234567890ab",
    step_name="train-model",
    expected_artifacts=["model.pkl", "metrics.json"],
)
```

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `client` | `kfp.Client` | *(required)* | Authenticated KFP client instance |
| `run_id` | `str` | *(required)* | The pipeline run ID |
| `step_name` | `str` | *(required)* | Display name of the pipeline step to inspect |
| `expected_artifacts` | `list[str]` | *(required)* | List of artifact names that the step must have produced |
| `min_size_bytes` | `int` | `1` | Minimum artifact size in bytes. Set to `0` to allow empty artifacts. Default `1` catches zero-byte uploads. |
| `expected_hashes` | `dict[str, str] \| None` | `None` | Optional dict mapping artifact name to expected SHA-256 hash. When provided, downloads the artifact and verifies its hash. |
| `severity` | `Severity` | `CRITICAL` | Severity level |

### Returns

`TestResult` with details:

- `run_id` -- the pipeline run ID
- `step_name` -- the step that was inspected
- `expected_artifacts` -- list of expected artifact names
- `found_artifacts` -- list of artifact names that were actually present
- `missing_artifacts` -- list of expected artifacts that were not found
- `artifact_sizes` -- dict mapping artifact name to size in bytes
- `hash_mismatches` -- dict of artifacts where the hash did not match (empty if all matched or hashes were not checked)

### How It Works

1. Fetches the run's workflow manifest to locate the specified step.
2. Lists the step's output artifacts from the KFP artifact store.
3. For each expected artifact, checks: (a) it exists, (b) its size meets `min_size_bytes`, (c) its hash matches `expected_hashes` if provided.
4. The assertion passes only if all expected artifacts are present, meet the size threshold, and pass hash verification.

### Example

```python
import pytest
from mltk.integrations.kubeflow import assert_kubeflow_step_outputs


def test_training_step_produces_model(kfp_client, latest_run_id):
    """Training step must produce a model artifact and metrics."""
    assert_kubeflow_step_outputs(
        client=kfp_client,
        run_id=latest_run_id,
        step_name="train-model",
        expected_artifacts=["model.pkl", "metrics.json"],
        min_size_bytes=1024,  # Model must be at least 1KB (not an empty file)
    )


def test_evaluation_step_produces_report(kfp_client, latest_run_id):
    """Evaluation step must produce a report with actual content."""
    assert_kubeflow_step_outputs(
        client=kfp_client,
        run_id=latest_run_id,
        step_name="evaluate-model",
        expected_artifacts=["evaluation_report.json", "confusion_matrix.png"],
        min_size_bytes=100,
    )


def test_preprocessing_output_hash(kfp_client, latest_run_id):
    """Preprocessed data must match the expected hash (deterministic pipeline)."""
    assert_kubeflow_step_outputs(
        client=kfp_client,
        run_id=latest_run_id,
        step_name="preprocess-data",
        expected_artifacts=["features.parquet"],
        expected_hashes={
            "features.parquet": "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        },
    )
```

---

## pytest Integration: KFP Client Fixture Pattern

The recommended pattern uses a session-scoped KFP client fixture and a helper fixture to resolve the latest run. This avoids creating multiple client connections and keeps test files clean.

### conftest.py

```python
# conftest.py
import os
import pytest
from kfp import Client as KfpClient


@pytest.fixture(scope="session")
def kfp_client():
    """Authenticated KFP client from environment variables.

    Set KFP_HOST to your Kubeflow Pipelines endpoint:
        export KFP_HOST=https://kubeflow.internal/pipeline

    For multi-user deployments with Dex/OIDC, also set:
        export KFP_TOKEN=<bearer-token>
    """
    host = os.environ.get("KFP_HOST", "http://localhost:8080")
    token = os.environ.get("KFP_TOKEN")

    client_kwargs = {"host": host}
    if token:
        client_kwargs["existing_token"] = token

    return KfpClient(**client_kwargs)


@pytest.fixture(scope="session")
def nightly_run_id(kfp_client):
    """Resolve the most recent run from the nightly experiment.

    This fixture is session-scoped so all tests in the session validate
    the same run. If the nightly run has not executed, the fixture
    raises a clear skip message.
    """
    experiment = kfp_client.get_experiment(experiment_name="nightly-training")
    runs = kfp_client.list_runs(
        experiment_id=experiment.experiment_id,
        sort_by="created_at desc",
        page_size=1,
    )
    if not runs.runs:
        pytest.skip("No runs found in nightly-training experiment")
    return runs.runs[0].run_id
```

### Test file

```python
# tests/test_kubeflow_pipeline.py
from mltk.integrations.kubeflow import (
    assert_kubeflow_pipeline_success,
    assert_kubeflow_step_outputs,
)


def test_pipeline_completed(kfp_client, nightly_run_id):
    assert_kubeflow_pipeline_success(
        client=kfp_client,
        run_id=nightly_run_id,
        timeout_seconds=3600,
        check_cache=True,
    )


def test_model_artifact_exists(kfp_client, nightly_run_id):
    assert_kubeflow_step_outputs(
        client=kfp_client,
        run_id=nightly_run_id,
        step_name="train-model",
        expected_artifacts=["model.pkl"],
        min_size_bytes=1024,
    )


def test_metrics_artifact_exists(kfp_client, nightly_run_id):
    assert_kubeflow_step_outputs(
        client=kfp_client,
        run_id=nightly_run_id,
        step_name="evaluate-model",
        expected_artifacts=["metrics.json"],
    )
```

Run with:

```bash
KFP_HOST=https://kubeflow.internal/pipeline pytest tests/test_kubeflow_pipeline.py -v
```

---

## In-Pipeline vs Post-Pipeline Testing

There are two distinct approaches to testing Kubeflow pipelines. They are complementary, not competing.

### In-pipeline testing (KFP component tests)

Tests run inside the pipeline as dedicated steps. The Kubeflow SDK's `@component` decorator defines lightweight containers that execute validation logic as part of the pipeline DAG.

```python
from kfp import dsl

@dsl.component(base_image="python:3.11-slim", packages_to_install=["pandas"])
def validate_features(features_path: str) -> bool:
    """KFP component that validates features inside the pipeline."""
    import pandas as pd
    df = pd.read_parquet(features_path)
    assert df.shape[0] > 0, "Empty feature set"
    assert df.isnull().mean().max() < 0.05, "Too many nulls"
    return True
```

**Strengths**: Runs at pipeline execution time. Can halt downstream steps on failure (conditional execution). Has direct access to pipeline artifacts without downloading them.

**Weaknesses**: Failures are reported in the Kubeflow UI, which requires manual checking or custom alerting. Validation logic is embedded in the pipeline definition, making it harder to maintain and test independently. No pytest integration, no HTML reports, no CI/CD gating.

### Post-pipeline testing (mltk assertions)

Tests run after the pipeline completes, from an external pytest suite. They query the KFP API to inspect run status and artifacts.

```python
def test_pipeline_completed(kfp_client, latest_run_id):
    assert_kubeflow_pipeline_success(client=kfp_client, run_id=latest_run_id)

def test_model_artifact(kfp_client, latest_run_id):
    assert_kubeflow_step_outputs(
        client=kfp_client,
        run_id=latest_run_id,
        step_name="train-model",
        expected_artifacts=["model.pkl"],
    )
```

**Strengths**: Runs in your standard pytest workflow. Integrates with CI/CD gates (block deployment if tests fail). Produces mltk HTML reports. Tests are version-controlled alongside your application code, not embedded in pipeline definitions.

**Weaknesses**: Runs after the pipeline completes, so it cannot halt in-progress steps. Requires network access to the KFP API. Artifact hash verification requires downloading artifacts, which adds latency.

### Recommendation

Use both. In-pipeline validation for fast feedback during execution (halt the pipeline before wasting GPU hours on bad data). Post-pipeline mltk assertions for CI/CD gating, reporting, and long-term test result tracking. The in-pipeline checks are your first line of defense; the mltk assertions are your external verification that the first line of defense actually worked.

```
Pipeline execution                         CI/CD (post-pipeline)
===================                         =====================
ingest -> validate_features -> train        pytest tests/test_kubeflow_pipeline.py
           ^                                  |
           |                                  +-- assert_kubeflow_pipeline_success
           +-- KFP component test             +-- assert_kubeflow_step_outputs
               (halts pipeline on failure)    +-- assert_no_drift(features)
                                              +-- assert_model_regression(...)
```

---
