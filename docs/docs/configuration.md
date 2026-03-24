# Configuration

mltk loads configuration with a cascade priority:

**function arguments > `mltk.yaml` > `pyproject.toml [tool.mltk]` > defaults**

This means you can set project-wide defaults in a config file and override them per-assertion when needed.

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
```

This is the recommended approach for most projects since `pyproject.toml` already exists.

### How it works

mltk uses `tomllib` (Python 3.11+) or `tomli` (Python 3.10) to parse the TOML file. If neither is available, it falls back to default config values.

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

## Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `drift_method` | `str` | `"ks"` | Statistical test for drift detection. Supported: `"ks"` (Kolmogorov-Smirnov), `"psi"` (Population Stability Index), `"kl"` (KL divergence), `"chi2"` (chi-squared). |
| `drift_threshold` | `float` | `0.05` | P-value or score threshold for drift detection. Values below this trigger a drift alert. |
| `report_dir` | `str` | `"./mltk-reports"` | Directory for generated HTML test reports. |
| `report_format` | `str` | `"html"` | Report output format. |
| `baseline_dir` | `str` | `"./mltk-baselines"` | Directory for storing distribution baselines used in drift detection. |
| `seed` | `int` | `42` | Random seed for reproducible tests. Used by nondeterministic assertions. |
| `pii_patterns` | `list[str]` | `["email", "phone", "ssn", "credit_card"]` | PII pattern names for `assert_no_pii`. |

## Loading Config in Code

```python
from mltk.core import MltkConfig

# Auto-detect: tries mltk.yaml, then pyproject.toml, then defaults
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

## Cascade Priority

mltk resolves configuration in this order, where higher priority wins:

1. **Function arguments** -- values passed directly to assertion functions (e.g., `threshold=0.1` in `assert_no_outliers`)
2. **`mltk.yaml`** -- if present in the current directory
3. **`pyproject.toml [tool.mltk]`** -- if present in the current directory
4. **Built-in defaults** -- always available, no config files needed

This means a project can define defaults in `pyproject.toml` and a developer can override them locally with `mltk.yaml` (which should be `.gitignore`d for local overrides).

## Unknown Keys

Unknown configuration keys are silently ignored. This provides forward compatibility: if a newer version of mltk adds a config option, older versions will not crash when reading the file.

```yaml
# This works even if future_option doesn't exist yet
drift_method: ks
future_option: some_value  # ignored by current version
```
