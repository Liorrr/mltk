# Weights & Biases Integration

Log mltk test results to Weights & Biases dashboards. Unifies training metrics and testing outcomes in a single view -- see "did the model train well AND pass quality gates?" in one place.

**Module:** `mltk.integrations.wandb_adapter`

**Install:** `pip install mltk[wandb]`

---

## Why Log Test Results to W&B?

Most ML teams already have W&B dashboards for experiment tracking. Training metrics (loss curves, learning rates, hyperparameters) live in W&B. Test results from mltk (drift detection, bias audits, regression checks) live in pytest output -- ephemeral, gone after the CI job finishes.

This creates a visibility gap. A model finishes training with excellent validation metrics. The W&B run shows a clean loss curve and 0.95 accuracy. But the pytest suite that runs afterward catches a fairness violation: the model's accuracy on a protected subgroup dropped by 12 percentage points. The training engineer sees the green W&B run and starts a deployment. The QA engineer sees the red pytest output in a different tab. If they are not the same person (and they usually are not), the deployment proceeds.

With the W&B integration, the same W&B run that shows training metrics also shows mltk test results. The training engineer sees both signals in one dashboard:

```
Run: nightly-v2.1-abc123
  Training:
    final_loss: 0.032
    val_accuracy: 0.951
    epochs: 45
  mltk Tests:
    mltk/summary/total: 12
    mltk/summary/passed_count: 11
    mltk/summary/failed_count: 1
    mltk/summary/pass_rate: 91.7
    mltk/bias_gender/passed: 0         <-- visible in the same view
    mltk/bias_gender/severity: 2       <-- critical severity
```

The W&B Table view provides a browsable, sortable breakdown of every test result. Over time, the time-series metrics reveal trends: pass rates declining, specific assertions becoming flaky, test durations creeping up.

---

## WandbLogger

```python
class WandbLogger:
    def __init__(
        self,
        project: str = "mltk-tests",
        entity: str | None = None,
        run_name: str | None = None,
        tags: list[str] | None = None,
    ) -> None: ...
```

The central class for logging mltk test results to W&B. Each `WandbLogger` instance creates a W&B run that stays open until `finish()` is called. Multiple `log_result` and `log_suite` calls accumulate in the same run.

### Constructor Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `project` | `str` | `"mltk-tests"` | W&B project name. All runs logged to this project appear together in the W&B UI. Use a descriptive name like `"model-qa"` or `"nightly-validation"`. |
| `entity` | `str \| None` | `None` | W&B team or user entity. When `None`, uses the entity from your `wandb login` session or `WANDB_ENTITY` environment variable. |
| `run_name` | `str \| None` | `None` | Human-readable name for the run (shown in the W&B dashboard). When `None`, W&B auto-generates a random name like `"amber-sunset-42"`. |
| `tags` | `list[str] \| None` | `None` | Tags for filtering runs in the W&B UI. Example: `["nightly", "v2.1", "regression"]`. |

### Raises

`ImportError` -- if `wandb` is not installed. The error message includes a `pip install wandb` hint.

---

### log_result

```python
def log_result(self, result: dict[str, Any]) -> None
```

Log a single test result as flat W&B metrics. Each result becomes a set of scalar metrics at the current W&B step.

#### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `result` | `dict[str, Any]` | *(required)* | Dict with at least `name` and `passed` keys. Optional: `duration_ms`, `severity`, `message`, `details`. |

#### Logged Metrics

For a result with `name="drift_psi"`:

| Metric Key | Value | Description |
|------------|-------|-------------|
| `mltk/drift_psi/passed` | `1` or `0` | 1 if passed, 0 if failed |
| `mltk/drift_psi/duration_ms` | `float` | Assertion execution time |
| `mltk/drift_psi/severity` | `0`, `1`, or `2` | Numeric severity: info=0, warning=1, critical=2 |
| `mltk/drift_psi/{detail_key}` | `float` | Each numeric value from the `details` dict is flattened into a metric |

The slash-separated keys create auto-grouped panels in the W&B dashboard. All metrics under `mltk/drift_*` cluster together without manual configuration.

#### Example

```python
from mltk.integrations.wandb_adapter import WandbLogger

logger = WandbLogger(project="model-qa", run_name="nightly-2024-03-15")

logger.log_result({
    "name": "drift_psi",
    "passed": True,
    "severity": "warning",
    "duration_ms": 142.5,
    "message": "PSI 0.08 within threshold 0.2",
    "details": {"psi_score": 0.08, "threshold": 0.2, "feature": "age"},
})

logger.log_result({
    "name": "bias_gender",
    "passed": False,
    "severity": "critical",
    "duration_ms": 89.3,
    "message": "DPR 0.72 below threshold 0.80",
    "details": {"dpr": 0.72, "threshold": 0.80},
})
```

---

### log_suite

```python
def log_suite(self, results: list[dict[str, Any]]) -> None
```

Log an entire test suite as a W&B Table plus summary metrics. This method does two things:

1. **Table.** Creates a `wandb.Table` with columns (name, passed, severity, duration_ms, message) and logs it as `"mltk/test_results"`. Tables are browsable in the W&B UI with sorting, filtering, and grouping.

2. **Summary metrics.** Logs aggregate numbers as `mltk/summary/*` metrics for dashboard widgets and trend monitoring.

#### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `results` | `list[dict[str, Any]]` | *(required)* | List of result dicts, each with at least `name` and `passed`. Optional: `severity`, `duration_ms`, `message`. |

#### Summary Metrics

| Metric Key | Type | Description |
|------------|------|-------------|
| `mltk/summary/total` | `int` | Total test count |
| `mltk/summary/passed_count` | `int` | Number of passing tests |
| `mltk/summary/failed_count` | `int` | Number of failing tests |
| `mltk/summary/pass_rate` | `float` | Pass percentage (0-100) |
| `mltk/summary/total_duration_ms` | `float` | Sum of all assertion durations |

#### Example

```python
from mltk.integrations.wandb_adapter import WandbLogger

results = [
    {"name": "schema_valid", "passed": True, "severity": "critical", "duration_ms": 12.0, "message": "ok"},
    {"name": "no_nulls", "passed": True, "severity": "critical", "duration_ms": 8.5, "message": "ok"},
    {"name": "drift_psi", "passed": True, "severity": "warning", "duration_ms": 142.5, "message": "PSI 0.08 < 0.2"},
    {"name": "bias_gender", "passed": False, "severity": "critical", "duration_ms": 89.3, "message": "DPR 0.72 < 0.80"},
]

logger = WandbLogger(project="model-qa", run_name="nightly-2024-03-15")
logger.log_suite(results)
```

In the W&B UI, the `mltk/test_results` table looks like this:

| name | passed | severity | duration_ms | message |
|------|--------|----------|-------------|---------|
| schema_valid | True | critical | 12.0 | ok |
| no_nulls | True | critical | 8.5 | ok |
| drift_psi | True | warning | 142.5 | PSI 0.08 < 0.2 |
| bias_gender | False | critical | 89.3 | DPR 0.72 < 0.80 |

---

### finish

```python
def finish(self) -> str
```

Finish the W&B run and return the run URL. Must be called when all logging is complete.

W&B runs hold network connections and background threads for async metric upload. Not calling `finish()` leaves orphaned runs in the "running" state on the dashboard, which confuses team members and wastes quota.

#### Returns

`str` -- the W&B run URL (e.g., `"https://wandb.ai/team/model-qa/runs/abc123"`).

---

## Complete pytest + W&B Workflow

This example shows the full integration: a conftest.py that creates a W&B logger, a test file that runs mltk assertions, and automatic logging of results at session end.

### conftest.py

```python
# conftest.py
import os
import pytest
from mltk.integrations.wandb_adapter import WandbLogger


@pytest.fixture(scope="session")
def wandb_logger():
    """Session-scoped W&B logger. All test results go to one run.

    Set WANDB_PROJECT and WANDB_ENTITY via environment, or use defaults.
    Set WANDB_MODE=offline to disable network calls during local dev.
    """
    project = os.environ.get("WANDB_PROJECT", "mltk-tests")
    entity = os.environ.get("WANDB_ENTITY")

    logger = WandbLogger(
        project=project,
        entity=entity,
        run_name=os.environ.get("WANDB_RUN_NAME"),
        tags=["ci", os.environ.get("GIT_BRANCH", "unknown")],
    )
    yield logger
    logger.finish()


@pytest.fixture(scope="session")
def collected_results():
    """Accumulate all mltk results for suite-level logging."""
    return []


@pytest.fixture(autouse=True)
def _log_mltk_to_wandb(wandb_logger, collected_results, request):
    """After each test, log any mltk results to W&B."""
    yield
    for result in getattr(request.node, "mltk_results", []):
        result_dict = {
            "name": result.name,
            "passed": result.passed,
            "severity": result.severity.value if hasattr(result.severity, "value") else str(result.severity),
            "duration_ms": result.duration_ms,
            "message": result.message,
            "details": result.details,
        }
        wandb_logger.log_result(result_dict)
        collected_results.append(result_dict)


def pytest_sessionfinish(session, exitstatus):
    """Log the full suite summary at session end."""
    all_results = []
    for item in session.items:
        for result in getattr(item, "mltk_results", []):
            all_results.append({
                "name": result.name,
                "passed": result.passed,
                "severity": result.severity.value if hasattr(result.severity, "value") else str(result.severity),
                "duration_ms": result.duration_ms,
                "message": result.message,
            })

    if all_results:
        logger = WandbLogger(project=os.environ.get("WANDB_PROJECT", "mltk-tests"))
        logger.log_suite(all_results)
        url = logger.finish()
        print(f"\nmltk results logged to W&B: {url}")
```

### Test file

```python
# tests/test_model_quality.py
import pandas as pd
import numpy as np
from mltk.data import assert_no_nulls, assert_no_drift
from mltk.model import assert_accuracy, assert_bias_ratio


def test_feature_quality():
    """Features have no nulls and no drift from reference."""
    df = pd.read_parquet("features/latest.parquet")
    ref = pd.read_parquet("features/reference.parquet")

    assert_no_nulls(df["user_age"])
    assert_no_drift(ref["user_age"], df["user_age"], method="psi", threshold=0.2)


def test_model_accuracy():
    """Model accuracy meets minimum threshold."""
    y_true = np.load("eval/y_true.npy")
    y_pred = np.load("eval/y_pred.npy")
    assert_accuracy(y_true, y_pred, min_accuracy=0.90)


def test_fairness():
    """Model fairness across protected groups."""
    y_true = np.load("eval/y_true.npy")
    y_pred = np.load("eval/y_pred.npy")
    groups = np.load("eval/groups.npy")
    assert_bias_ratio(y_true, y_pred, groups, min_ratio=0.80)
```

### Running

```bash
# Local development (offline mode -- no W&B network calls)
WANDB_MODE=offline pytest tests/test_model_quality.py -v

# CI (logs to W&B cloud)
WANDB_PROJECT=model-qa \
WANDB_API_KEY=${{ secrets.WANDB_API_KEY }} \
GIT_BRANCH=$(git branch --show-current) \
pytest tests/test_model_quality.py -v --mltk-report
```

---

## W&B Table Visualization

The `log_suite` method creates a `wandb.Table` object that appears as an interactive table in the W&B UI. Tables support:

- **Sorting** by any column (click the column header). Sort by `passed` to see all failures at the top.
- **Filtering** by column values. Filter `severity == "critical"` to focus on blocking failures.
- **Grouping** by column. Group by `severity` to see how many tests failed at each level.
- **Cross-run comparison.** Compare tables across runs to see which tests changed status.

### Building custom tables

For more control over table contents, build the `wandb.Table` manually:

```python
import wandb
from mltk.integrations.wandb_adapter import WandbLogger

logger = WandbLogger(project="model-qa")

# Build a custom table with additional columns
columns = ["name", "passed", "severity", "duration_ms", "message", "threshold", "actual"]
table = wandb.Table(columns=columns)

results = [
    {"name": "accuracy", "passed": True, "severity": "critical",
     "duration_ms": 45.0, "message": "0.94 >= 0.90",
     "details": {"threshold": 0.90, "actual": 0.94}},
    {"name": "drift_age", "passed": False, "severity": "warning",
     "duration_ms": 120.0, "message": "PSI 0.35 > 0.20",
     "details": {"threshold": 0.20, "actual": 0.35}},
]

for r in results:
    table.add_data(
        r["name"], r["passed"], r["severity"], r["duration_ms"],
        r["message"], r["details"].get("threshold"), r["details"].get("actual"),
    )

wandb.log({"mltk/detailed_results": table})
url = logger.finish()
```

### Plotting test results over time

W&B automatically creates line charts for scalar metrics. After several runs, the `mltk/summary/pass_rate` metric produces a trend line showing how your test pass rate evolves. Combine with W&B's alerting feature to get notified when pass rate drops below a threshold:

1. Go to the W&B project page.
2. Click on the `mltk/summary/pass_rate` chart.
3. Click "Create Alert" and set a threshold (e.g., alert when pass_rate < 95).

---

## Comparison: W&B vs MLflow Integration

mltk provides both a W&B integration (`mltk.integrations.wandb_adapter`) and an MLflow integration (`mltk.integrations.mlflow_logger`). They solve the same problem (logging test results to experiment trackers) but serve different ecosystems.

| Aspect | W&B (`WandbLogger`) | MLflow (`MlflowLogger`) |
|--------|---------------------|------------------------|
| **Hosting** | SaaS by default (wandb.ai), self-hosted available | Self-hosted by default, managed options available |
| **Setup** | `wandb login` + API key | MLflow tracking server URL |
| **Best for** | Teams already using W&B for experiment tracking | Teams using MLflow or needing fully self-hosted |
| **Tables** | Native W&B Tables with sorting, filtering, grouping | No table equivalent -- metrics only |
| **Artifacts** | `log_report` not yet implemented (use MLflow for artifact attachment) | `log_report(path)` attaches HTML reports as artifacts |
| **Cost** | Free tier (100GB), paid tiers for teams | Free (open source), paid for managed MLflow |
| **Offline mode** | `WANDB_MODE=offline` -- logs locally, syncs later | `mlflow.set_tracking_uri("file:///...")` -- fully local |

### When to use W&B

- Your team already tracks training experiments in W&B.
- You want interactive tables for browsing test results.
- You want built-in alerting on metric thresholds.
- You are comfortable with SaaS (or run the self-hosted server).

### When to use MLflow

- You need fully self-hosted experiment tracking with no SaaS dependency.
- You want to attach mltk HTML reports as artifacts alongside the model.
- You use MLflow Model Registry and want test results tied to registered model versions.
- Your organization standardized on MLflow.

### Using both

There is no conflict. You can log to both W&B and MLflow in the same test session:

```python
# conftest.py
from mltk.integrations.wandb_adapter import WandbLogger
from mltk.integrations.mlflow_logger import MlflowLogger

@pytest.fixture(scope="session")
def wandb_logger():
    logger = WandbLogger(project="model-qa")
    yield logger
    logger.finish()

@pytest.fixture(scope="session")
def mlflow_logger():
    return MlflowLogger(experiment_name="model-qa")

@pytest.fixture(autouse=True)
def _log_everywhere(wandb_logger, mlflow_logger, request):
    yield
    for result in getattr(request.node, "mltk_results", []):
        result_dict = {
            "name": result.name,
            "passed": result.passed,
            "severity": str(result.severity),
            "duration_ms": result.duration_ms,
            "message": result.message,
        }
        wandb_logger.log_result(result_dict)
        mlflow_logger.log_test_result(result)
```

---
