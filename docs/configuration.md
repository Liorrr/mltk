# Configuration

mltk loads configuration with a cascade priority:

**environment variables > function arguments > `mltk.yaml` > `pyproject.toml [tool.mltk]` > defaults**

This means you can set project-wide defaults in a config file, override them per-assertion in code, and override everything with environment variables in CI/CD.

---

## Config Cascade Explained

mltk resolves each configuration option by checking these sources in order. The first match wins:

```text
Priority 1 (highest): MLTK_* environment variables
Priority 2:           Function arguments (e.g., threshold=0.1 in assert_no_outliers)
Priority 3:           mltk.yaml in the current directory
Priority 4:           pyproject.toml [tool.mltk] in the current directory
Priority 5 (lowest):  Built-in defaults
```

**Typical usage patterns:**

| Scenario | Approach |
|----------|----------|
| Project-wide defaults | `pyproject.toml [tool.mltk]` -- checked into git, shared by the team |
| Local overrides | `mltk.yaml` -- add to `.gitignore`, each developer can tune locally |
| CI/CD overrides | `MLTK_*` environment variables -- set in pipeline config |
| Per-test overrides | Function arguments -- passed directly to assertion calls |

---

## pyproject.toml

Add a `[tool.mltk]` section to your existing `pyproject.toml`:

```toml
[tool.mltk]
drift_method = "ks"
drift_threshold = 0.05
report_dir = "./mltk-reports"
report_format = "html"
baseline_dir = "./mltk-baselines"
seed = 42
pii_patterns = ["email", "phone", "ssn", "credit_card"]
```

This is the recommended approach for most projects since `pyproject.toml` already exists.

### How it works

mltk uses `tomllib` (Python 3.11+) or `tomli` (Python 3.10) to parse the TOML file. If neither is available, it falls back to default config values.

---

## mltk.yaml

For teams that prefer a standalone config file, create `mltk.yaml` in your project root:

### Flat format

```yaml
drift_method: psi
drift_threshold: 0.1
report_dir: ./custom-reports
report_format: html
baseline_dir: ./baselines
seed: 123
pii_patterns:
  - email
  - phone
  - ssn
  - credit_card
```

### Nested format

```yaml
mltk:
  drift_method: psi
  drift_threshold: 0.1
  seed: 123
```

Both formats are supported. The nested format uses the `mltk:` top-level key for namespacing in shared config files.

!!! note "YAML parsing"
    mltk uses PyYAML if installed. If PyYAML is not available, it falls back to a built-in parser that handles simple `key: value` files and JSON-formatted YAML.

!!! tip "Local overrides"
    Add `mltk.yaml` to your `.gitignore` and use it for local developer overrides. The team shares defaults via `pyproject.toml`, and each developer can tune thresholds locally without affecting others.

---

## Environment Variables

All environment variables are prefixed with `MLTK_` and take the **highest priority** in the cascade. This makes them ideal for CI/CD pipelines where you want to override project defaults without modifying files.

### Core config variables

| Variable | Maps to | Type | Example |
|----------|---------|------|---------|
| `MLTK_DRIFT_METHOD` | `drift_method` | `str` | `export MLTK_DRIFT_METHOD=psi` |
| `MLTK_DRIFT_THRESHOLD` | `drift_threshold` | `float` | `export MLTK_DRIFT_THRESHOLD=0.1` |
| `MLTK_REPORT_DIR` | `report_dir` | `str` | `export MLTK_REPORT_DIR=./ci-reports` |
| `MLTK_REPORT_FORMAT` | `report_format` | `str` | `export MLTK_REPORT_FORMAT=html` |
| `MLTK_BASELINE_DIR` | `baseline_dir` | `str` | `export MLTK_BASELINE_DIR=./baselines` |
| `MLTK_SEED` | `seed` | `int` | `export MLTK_SEED=99` |
| `MLTK_PII_PATTERNS` | `pii_patterns` | `str` (comma-separated) | `export MLTK_PII_PATTERNS=email,phone,ssn` |

### Other environment variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `MLTK_REGISTRY_DIR` | Override the test registry directory | `~/.mltk/registry/` |
| `MLTK_DOCS_PORT` | Port for `mltk docs-serve` | `8000` |
| `MLTK_DOCS_HOST` | Host for `mltk docs-serve` | `127.0.0.1` |
| `MLTK_SLACK_WEBHOOK` | Slack incoming webhook URL for `mltk slack-notify` | *(none)* |

### Example: CI/CD with environment overrides

```yaml
# GitHub Actions example
- name: Run ML tests
  env:
    MLTK_DRIFT_METHOD: psi
    MLTK_DRIFT_THRESHOLD: "0.1"
    MLTK_REPORT_DIR: ./ci-reports
    MLTK_SEED: "42"
  run: pytest --mltk-report -v
```

---

## All Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `drift_method` | `str` | `"ks"` | Statistical test for drift detection. Supported: `"ks"` (Kolmogorov-Smirnov), `"psi"` (Population Stability Index), `"kl"` (KL divergence), `"chi2"` (chi-squared). |
| `drift_threshold` | `float` | `0.05` | P-value or score threshold for drift detection. Values below this trigger a drift alert. |
| `report_dir` | `str` | `"./mltk-reports"` | Directory for generated HTML test reports. Created automatically if it does not exist. |
| `report_format` | `str` | `"html"` | Report output format. |
| `baseline_dir` | `str` | `"./mltk-baselines"` | Directory for storing distribution baselines used in drift detection. |
| `seed` | `int` | `42` | Random seed for reproducible tests. Used by nondeterministic assertions and the `ml_nondeterministic` marker. |
| `pii_patterns` | `list[str]` | `["email", "phone", "ssn", "credit_card"]` | PII pattern names for `assert_no_pii`. Determines which patterns are scanned. |

---

## Loading Config in Code

```python
from mltk.core import MltkConfig

# Auto-detect: tries env vars → mltk.yaml → pyproject.toml → defaults
config = MltkConfig.load()

# Explicit YAML path
config = MltkConfig.load(path="configs/mltk-staging.yaml")

# Direct construction (override everything)
config = MltkConfig(
    drift_method="psi",
    drift_threshold=0.1,
    seed=99,
)

# Serialize to dict (for logging or report metadata)
print(config.to_dict())
```

---

## Scaffolding a Config

Use the CLI to generate a starter config:

```bash
mltk init
# Created mltk.yaml
# Created tests/test_mltk_example.py
```

This creates an `mltk.yaml` with sensible defaults and an example test file. See [Getting Started](getting-started.md) for a full walkthrough.

---

## Unknown Keys

Unknown configuration keys are silently ignored. This provides forward compatibility: if a newer version of mltk adds a config option, older versions will not crash when reading the file.

```yaml
# This works even if future_option doesn't exist yet
drift_method: ks
future_option: some_value  # ignored by current version
```
