# pytest Plugin

mltk includes a pytest plugin that auto-registers when you `pip install mlspec`. It provides ML-specific markers for test categorization, fixtures for configuration, and a `--mltk-report` flag for test summary output.

**Module:** `mltk.pytest_plugin`

**Auto-registered:** via `pyproject.toml` entry-point. No configuration needed.

---

## Markers

Use markers to categorize and selectively run ML tests.

```python
@pytest.mark.ml_data
def test_data_quality():
    ...

@pytest.mark.ml_model
def test_model_accuracy():
    ...

@pytest.mark.ml_smoke
def test_pipeline_runs():
    ...
```

### Available Markers

| Marker | Purpose | CI/CD Usage |
|--------|---------|-------------|
| `ml_data` | Data quality tests (schema, drift, PII) | Run on every PR |
| `ml_model` | Model quality tests (metrics, bias, regression) | Run on merge |
| `ml_drift` | Drift detection tests | Run on merge + nightly |
| `ml_inference` | Inference performance tests | Run on deploy |
| `ml_smoke` | Fast smoke tests (<5 min) | Run on every commit |
| `ml_slow` | Long-running tests (full training) | Run nightly only |
| `ml_nondeterministic` | Tests with inherent randomness | Run with retries |
| `ml_gpu` | Tests requiring GPU hardware | Run on GPU runners only |

### Running subsets

```bash
# Only data quality tests (fast, every PR)
pytest -m ml_data

# Only smoke tests (fastest, every commit)
pytest -m ml_smoke

# Skip slow tests in CI
pytest -m "not ml_slow"

# Model tests with report
pytest -m ml_model --mltk-report
```

---

## Fixtures

### ml_config

Loads `MltkConfig` from project configuration.

```python
def test_with_config(ml_config):
    assert ml_config.drift_threshold == 0.05
```

### ml_report

Collects test results during a session for report generation.

```python
def test_something(ml_report):
    # Results are auto-collected by the plugin
    ...
```

---

## --mltk-report

Generate a test summary at the end of the pytest session.

```bash
pytest --mltk-report
```

Output includes:
- Total pass/fail/warning counts
- Per-module breakdown (data, model, inference)
- Failed assertion details with TestResult info
- Timing information per test

## Assertion export

Use `--mltk-export-json PATH` or `--mltk-report` to enable
per-test assertion recording. Each exported test record includes an additive
`assertions` list. Every entry contains `name`, `passed`, `severity`, `message`,
`details`, and `duration_ms`. The `caught` field is `true` when a critical
failed assertion raised `MltkAssertionError` but was caught by the test; a
critical failure that failed the test has `caught: false`.

Recording is inactive unless one of those two flags is set. Assertions made
in threads spawned by a test are not captured because threads do not inherit
the recording context. Asyncio tasks share the context and are captured.
Async fixtures or tasks shared across tests are captured separately for each
test.
Assertions from a session-scoped fixture land in the first test that requests
that fixture, and teardown-phase assertions are discarded. The plugin does not
collect assertion results under pytest-xdist. The server push intentionally
contains only each test's headline result and does not include `assertions`.

If the optional HTML-report dependency is unavailable, pytest prints a visible
warning with the missing module and `pip install mlspec[report]`. Other HTML
report errors are also printed without changing the pytest session exit status.

---

## CI/CD Integration

### GitHub Actions example

```yaml
jobs:
  ml-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: pip install mlspec[sklearn]
      - run: pytest -m "ml_data or ml_smoke" --mltk-report  # Fast tests on PR

  ml-full:
    runs-on: ubuntu-latest
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    steps:
      - run: pytest -m "not ml_slow" --mltk-report  # Full suite on merge
```

---
