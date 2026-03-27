# DVC Integration

Verify DVC-tracked files, data versioning integrity, and pipeline reproducibility. Catches forgotten `dvc add` commands, stale hashes, uncommitted data changes, and broken remote links before they corrupt your training runs.

**Module:** `mltk.integrations.dvc`

**Install:** `pip install mltk[dvc]`

---

## Why Test Data Versioning?

DVC (Data Version Control) solves a real problem: tracking large datasets and model files alongside your Git code. But DVC introduces a new class of failure modes that are invisible to `git status` and `pytest` unless you explicitly test for them.

**Forgotten `dvc add`.** A data scientist updates `data/training.csv` by re-running the data pipeline. They commit the code changes to Git and push. But they forget to run `dvc add data/training.csv`, so the `.dvc` file still points to the old hash. The next person who runs `dvc pull` gets the old data. They train on stale data without knowing it. The model works. The metrics look fine. But the model does not reflect the latest data.

**Stale hashes after manual edits.** Someone opens `features.parquet` in a notebook, fixes a few values manually, and saves. The file on disk no longer matches the hash in `features.parquet.dvc`. DVC does not detect this automatically -- it only checks hashes during `dvc status` or `dvc push`, which are not part of most CI pipelines. The corrupted file gets used for training. If someone later runs `dvc checkout`, they overwrite the manual fix with the old version.

**Team collaboration drift.** Developer A adds a new column to the preprocessing output and runs `dvc add`. Developer B, working on a different branch, adds a different column. Both push their DVC files to different remote storage paths. When the branches merge, the `.dvc` file from the merge resolution points to one developer's version. The other developer's data is orphaned in remote storage. The merged code expects both columns but only gets one.

**Broken remote links.** The DVC remote is configured to use an S3 bucket. Someone changes the bucket lifecycle policy, and objects older than 90 days are archived to Glacier. A researcher runs `dvc pull` for a 4-month-old experiment and gets an error. The data is technically still there, but inaccessible without a restore request. Nothing in the CI pipeline caught this because DVC remote health is never tested.

mltk's DVC assertions catch these problems by verifying the actual state of DVC-tracked files at test time, in your pytest suite, where failures block the pipeline.

---

## assert_dvc_file_tracked

Assert that a file is tracked by DVC and its `.dvc` file exists in the repository. This is the most basic DVC health check: does DVC know about this file?

```python
from mltk.integrations.dvc import assert_dvc_file_tracked

result = assert_dvc_file_tracked(
    file_path="data/training.csv",
    repo_path=".",
)
```

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `file_path` | `str` | *(required)* | Path to the data file, relative to the repository root |
| `repo_path` | `str` | `"."` | Path to the Git/DVC repository root |
| `check_remote` | `bool` | `False` | When `True`, also verifies that the file's hash exists in the configured DVC remote. Catches broken remote links and purged cache entries. |
| `severity` | `Severity` | `CRITICAL` | Severity level |

### Returns

`TestResult` with details:

- `file_path` -- the file path that was checked
- `dvc_file` -- path to the corresponding `.dvc` file (e.g., `data/training.csv.dvc`)
- `dvc_file_exists` -- whether the `.dvc` file was found
- `tracked` -- whether the file is tracked by DVC
- `in_remote` -- whether the hash exists in the DVC remote (only checked when `check_remote=True`)
- `hash` -- the MD5 hash from the `.dvc` file (if it exists)

### How It Works

1. Constructs the expected `.dvc` file path by appending `.dvc` to `file_path`.
2. Checks if the `.dvc` file exists on disk.
3. If it exists, parses the YAML content to extract the `md5` hash.
4. If `check_remote=True`, runs a DVC API call to verify the hash exists in the configured remote storage.

### Edge Cases

- **File tracked via `dvc.yaml` (pipeline output)**: Files that are outputs of DVC pipeline stages do not have individual `.dvc` files. They are tracked in `dvc.lock`. The assertion checks both `.dvc` files and `dvc.lock` entries.
- **File not in DVC at all**: Returns a failing `TestResult` with a message suggesting `dvc add <file_path>`.

### Example

```python
import pytest
from mltk.integrations.dvc import assert_dvc_file_tracked


def test_training_data_tracked():
    """Training data must be tracked by DVC (not raw Git)."""
    assert_dvc_file_tracked("data/training.csv")


def test_model_artifact_tracked():
    """Trained model must be in DVC, not Git LFS or untracked."""
    assert_dvc_file_tracked("models/production/model.pkl")


def test_data_available_in_remote():
    """Training data must be pushed to DVC remote (team can pull)."""
    assert_dvc_file_tracked(
        "data/training.csv",
        check_remote=True,  # Verify it is actually in S3/GCS/Azure
    )
```

---

## assert_dvc_data_version

Assert that a DVC-tracked file matches an expected hash. This catches stale data, manual edits, and version mismatches between what the code expects and what is on disk.

```python
from mltk.integrations.dvc import assert_dvc_data_version

result = assert_dvc_data_version(
    file_path="data/training.csv",
    expected_hash="d41d8cd98f00b204e9800998ecf8427e",
    repo_path=".",
)
```

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `file_path` | `str` | *(required)* | Path to the DVC-tracked file, relative to the repository root |
| `expected_hash` | `str` | *(required)* | Expected MD5 hash of the file content. This is the hash stored in the `.dvc` file after `dvc add`. |
| `repo_path` | `str` | `"."` | Path to the Git/DVC repository root |
| `verify_on_disk` | `bool` | `True` | When `True`, computes the actual MD5 of the file on disk and compares it to `expected_hash`. When `False`, only checks the hash in the `.dvc` file (faster but does not catch manual edits). |
| `severity` | `Severity` | `CRITICAL` | Severity level |

### Returns

`TestResult` with details:

- `file_path` -- the file path that was checked
- `expected_hash` -- the expected MD5 hash
- `dvc_hash` -- the hash stored in the `.dvc` file
- `disk_hash` -- the actual MD5 of the file on disk (only computed when `verify_on_disk=True`)
- `dvc_matches_expected` -- whether the `.dvc` hash matches the expected hash
- `disk_matches_expected` -- whether the on-disk hash matches (only when `verify_on_disk=True`)
- `file_size_bytes` -- size of the file on disk

### How It Works

1. Parses the `.dvc` file to extract the stored MD5 hash.
2. Compares the stored hash against `expected_hash`. If they differ, the `.dvc` file is out of date (someone ran `dvc add` with different data than expected).
3. If `verify_on_disk=True`, computes the MD5 of the actual file on disk and compares it to `expected_hash`. This catches manual edits that bypass `dvc add`.
4. The assertion passes only if both comparisons succeed (when `verify_on_disk=True`) or the `.dvc` hash matches (when `verify_on_disk=False`).

### Hash Verification Scenarios

| `.dvc` hash | Disk hash | Expected hash | Result | Diagnosis |
|-------------|-----------|---------------|--------|-----------|
| `abc123` | `abc123` | `abc123` | PASS | Everything consistent |
| `abc123` | `abc123` | `def456` | FAIL | Data was re-versioned; code expects old version |
| `abc123` | `def456` | `abc123` | FAIL | File was manually edited after `dvc add` |
| `def456` | `def456` | `abc123` | FAIL | Both `.dvc` and disk have unexpected data |

### Example

```python
import pytest
from mltk.integrations.dvc import assert_dvc_data_version


# Store expected hashes in a constants file or fixture
EXPECTED_DATA_VERSIONS = {
    "data/training.csv": "a3f2b8c91d4e5f6071829304a5b6c7d8",
    "data/validation.csv": "b4c3d9e02f5a6b7182930415c6d7e8f9",
    "features/engineered.parquet": "c5d4e0f13a6b7c8293041526d7e8f900",
}


@pytest.mark.parametrize("file_path,expected_hash", EXPECTED_DATA_VERSIONS.items())
def test_data_version_matches(file_path, expected_hash):
    """DVC-tracked data files match expected hashes."""
    assert_dvc_data_version(
        file_path=file_path,
        expected_hash=expected_hash,
        verify_on_disk=True,  # Catch manual edits too
    )


def test_training_data_not_stale():
    """Training data is the version this model was validated against.

    The hash below corresponds to the March 2024 data refresh.
    Update it when the data pipeline produces a new version.
    """
    assert_dvc_data_version(
        file_path="data/training.csv",
        expected_hash="a3f2b8c91d4e5f6071829304a5b6c7d8",
        verify_on_disk=True,
    )
```

---

## Integration with CI: Verify DVC State Before Training

The most valuable pattern is running DVC assertions in CI before any training job starts. This prevents wasting GPU hours on training with wrong data.

### conftest.py

```python
# conftest.py
import os
import pytest


@pytest.fixture(scope="session")
def repo_root():
    """Repository root path."""
    return os.environ.get("REPO_ROOT", ".")


@pytest.fixture(scope="session")
def expected_versions():
    """Load expected data versions from a version lock file.

    The lock file is committed to Git and updated whenever
    the data pipeline produces a new version.
    """
    import json
    lock_path = os.path.join(
        os.environ.get("REPO_ROOT", "."),
        "data/versions.lock.json",
    )
    if not os.path.exists(lock_path):
        pytest.skip("No data version lock file found")
    with open(lock_path) as f:
        return json.load(f)
```

### Version lock file

Commit a `data/versions.lock.json` to Git that records the expected state:

```json
{
  "data/training.csv": {
    "hash": "a3f2b8c91d4e5f6071829304a5b6c7d8",
    "updated": "2024-03-15",
    "rows": 150000
  },
  "data/validation.csv": {
    "hash": "b4c3d9e02f5a6b7182930415c6d7e8f9",
    "updated": "2024-03-15",
    "rows": 30000
  },
  "features/engineered.parquet": {
    "hash": "c5d4e0f13a6b7c8293041526d7e8f900",
    "updated": "2024-03-15",
    "columns": 48
  }
}
```

### Test file

```python
# tests/test_dvc_state.py
import pytest
from mltk.integrations.dvc import assert_dvc_file_tracked, assert_dvc_data_version


def test_all_data_tracked(expected_versions, repo_root):
    """Every file in the version lock must be tracked by DVC."""
    for file_path in expected_versions:
        assert_dvc_file_tracked(file_path, repo_path=repo_root)


@pytest.mark.parametrize("file_path", [
    "data/training.csv",
    "data/validation.csv",
    "features/engineered.parquet",
])
def test_data_version_matches(file_path, expected_versions, repo_root):
    """DVC-tracked data matches the version lock."""
    expected = expected_versions[file_path]
    assert_dvc_data_version(
        file_path=file_path,
        expected_hash=expected["hash"],
        repo_path=repo_root,
        verify_on_disk=True,
    )


def test_data_available_in_remote(expected_versions, repo_root):
    """All data files are pushed to the DVC remote (team can pull)."""
    for file_path in expected_versions:
        assert_dvc_file_tracked(
            file_path,
            repo_path=repo_root,
            check_remote=True,
        )
```

### GitHub Actions integration

```yaml
# .github/workflows/train.yml
name: Training Pipeline

on:
  push:
    branches: [main]
  workflow_dispatch:

jobs:
  verify-data:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install dependencies
        run: pip install "mltk[dvc]" dvc[s3]

      - name: Configure DVC remote
        run: dvc remote modify myremote access_key_id ${{ secrets.AWS_ACCESS_KEY_ID }}
        env:
          AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}

      - name: Pull DVC data
        run: dvc pull

      - name: Verify data integrity
        run: pytest tests/test_dvc_state.py -v --mltk-report

      - name: Upload mltk report
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: dvc-verification-report
          path: mltk-reports/

  train:
    needs: verify-data  # Only train if data verification passes
    runs-on: [self-hosted, gpu]
    steps:
      - uses: actions/checkout@v4
      - run: dvc pull
      - run: python train.py
```

The `needs: verify-data` dependency ensures training never starts with incorrect data.

---

## DVC + mltk Pattern: Data Quality Gates in DVC Pipelines

DVC pipelines (`dvc.yaml`) define reproducible multi-stage workflows. mltk assertions can be embedded as a dedicated validation stage that gates downstream steps.

### dvc.yaml

```yaml
stages:
  preprocess:
    cmd: python src/preprocess.py
    deps:
      - src/preprocess.py
      - data/raw/
    outs:
      - data/processed/features.parquet

  validate-data:
    cmd: pytest tests/test_data_quality.py -v --tb=short
    deps:
      - tests/test_data_quality.py
      - data/processed/features.parquet
    metrics:
      - mltk-reports/results.json:
          cache: false

  train:
    cmd: python src/train.py
    deps:
      - src/train.py
      - data/processed/features.parquet
      - mltk-reports/results.json  # Depends on validation output
    outs:
      - models/model.pkl
    metrics:
      - metrics/train_metrics.json:
          cache: false

  validate-model:
    cmd: pytest tests/test_model_quality.py -v --tb=short
    deps:
      - tests/test_model_quality.py
      - models/model.pkl
    metrics:
      - mltk-reports/model_results.json:
          cache: false
```

### Data quality test

```python
# tests/test_data_quality.py
import json
import pandas as pd
from mltk.data import assert_no_nulls, assert_schema, assert_no_drift, assert_range
from mltk.integrations.dvc import assert_dvc_file_tracked


def test_features_tracked():
    """Processed features must be tracked by DVC."""
    assert_dvc_file_tracked("data/processed/features.parquet")


def test_feature_schema():
    """Feature file has the expected columns and types."""
    df = pd.read_parquet("data/processed/features.parquet")
    assert_schema(df, {
        "user_id": "int64",
        "age": "float64",
        "income": "float64",
        "label": "int64",
    })


def test_no_nulls_in_features():
    """No null values in critical features."""
    df = pd.read_parquet("data/processed/features.parquet")
    for col in ["age", "income", "label"]:
        assert_no_nulls(df[col])


def test_feature_ranges():
    """Feature values are within expected ranges."""
    df = pd.read_parquet("data/processed/features.parquet")
    assert_range(df["age"], min_val=0, max_val=120)
    assert_range(df["income"], min_val=0, max_val=10_000_000)


def test_no_drift_from_reference():
    """Features have not drifted from the reference distribution."""
    current = pd.read_parquet("data/processed/features.parquet")
    reference = pd.read_parquet("data/reference/features.parquet")

    for col in ["age", "income"]:
        assert_no_drift(reference[col], current[col], method="psi", threshold=0.2)
```

### Pipeline execution

```bash
# Run the full pipeline -- validation gates prevent training on bad data
dvc repro

# If validate-data fails, train never runs:
#   preprocess     -> Succeeded
#   validate-data  -> FAILED (assert_no_nulls failed for 'income')
#   train          -> Skipped (dependency failed)
#   validate-model -> Skipped (dependency failed)
```

### The gating pattern

```
dvc repro
    |
    v
preprocess (produces features.parquet)
    |
    v
validate-data (mltk assertions on features.parquet)
    |
    +-- PASS --> train --> validate-model --> done
    |
    +-- FAIL --> pipeline stops, no training runs, no GPU wasted
```

This pattern ensures that data quality issues are caught before training, not after. A null column in the features file stops the pipeline immediately instead of producing a broken model that passes training but fails in production.

---
