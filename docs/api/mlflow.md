# MLflow Integration

Log mltk test results as MLflow metrics and attach HTML reports as artifacts.

**Module:** `mltk.integrations.mlflow_logger`

**Install:** `pip install mltk[mlflow]`

---

## Quick Start

```python
from mltk.integrations.mlflow_logger import MlflowLogger

logger = MlflowLogger()
logger.log_results(test_suite)  # Logs metrics to active MLflow run
logger.log_report("mltk-reports/report.html")  # Attaches as artifact
```

## pytest Integration

```bash
pytest --mltk-report --mltk-mlflow
```

Automatically logs test results to the active MLflow experiment after the session.

## API

### MlflowLogger

```python
MlflowLogger(experiment_name: str | None = None, tracking_uri: str | None = None)
```

| Method | Description |
|--------|-------------|
| `log_results(suite)` | Log pass/fail counts, per-assertion metrics, score, duration |
| `log_report(path)` | Attach HTML report as MLflow artifact |
| `log_test_result(result)` | Log a single TestResult as metric |

### Logged Metrics

| Metric | Description |
|--------|-------------|
| `mltk.total_tests` | Total test count |
| `mltk.passed` | Passed count |
| `mltk.failed` | Failed count |
| `mltk.score` | Suite score (0-100) |
| `mltk.duration_ms` | Total duration |
| `mltk.{name}` | Per-assertion pass (1) or fail (0) |

---
