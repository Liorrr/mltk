"""Tests for mltk.core.config.

Config loading tests verify the cascade: env vars > YAML > TOML > defaults.
This ensures users can configure mltk via their preferred method
and that missing/invalid configs fall back gracefully.
"""

from pathlib import Path

import pytest

from mltk.core.config import MltkConfig

# --- Existing tests ---


def test_default_config() -> None:
    """Default config uses sensible ML testing defaults.

    WHY: When no config file exists, mltk should still work out of the box
    with safe defaults (KS test, 0.05 threshold, seed=42 for reproducibility).
    """
    config = MltkConfig()
    assert config.drift_method == "ks"
    assert config.drift_threshold == 0.05
    assert config.seed == 42


def test_config_to_dict() -> None:
    """Config serializes to dict for report metadata and logging.

    WHY: Test reports need to record which config was used for reproducibility.
    """
    config = MltkConfig()
    d = config.to_dict()
    assert d["drift_method"] == "ks"
    assert isinstance(d["pii_patterns"], list)


def test_config_load_defaults() -> None:
    """load() returns defaults when no config files are present.

    WHY: Fresh projects without any config should still work.
    """
    config = MltkConfig.load()
    assert config.drift_method == "ks"


# --- Sprint 1: TOML/YAML loading tests ---


def test_config_from_pyproject(tmp_path: Path) -> None:
    """Loads [tool.mltk] section from pyproject.toml.

    WHY: Most Python projects already have pyproject.toml. mltk config
    should live alongside build config without a separate file.
    """
    toml_content = """\
[tool.mltk]
drift_method = "psi"
drift_threshold = 0.1
seed = 123
"""
    toml_file = tmp_path / "pyproject.toml"
    toml_file.write_text(toml_content)

    config = MltkConfig._from_pyproject(toml_file)
    assert config.drift_method == "psi"
    assert config.drift_threshold == 0.1
    assert config.seed == 123
    # Unset values should keep defaults
    assert config.report_format == "html"


def test_config_from_yaml(tmp_path: Path) -> None:
    """Loads config from mltk.yaml file.

    WHY: Standalone config is cleaner for teams that want mltk-specific
    settings separate from their build tool config.
    """
    yaml_content = """\
drift_method: psi
drift_threshold: 0.15
seed: 99
report_dir: ./custom-reports
"""
    yaml_file = tmp_path / "mltk.yaml"
    yaml_file.write_text(yaml_content)

    config = MltkConfig._from_yaml(yaml_file)
    assert config.drift_method == "psi"
    assert config.drift_threshold == 0.15
    assert config.seed == 99
    assert config.report_dir == "./custom-reports"


def test_config_from_yaml_nested(tmp_path: Path) -> None:
    """Supports nested format: mltk: { ... } in YAML.

    WHY: Some users may nest under an 'mltk' key for namespacing.
    """
    yaml_content = '{"mltk": {"drift_method": "kl", "seed": 7}}'
    yaml_file = tmp_path / "mltk.yaml"
    yaml_file.write_text(yaml_content)

    config = MltkConfig._from_yaml(yaml_file)
    assert config.drift_method == "kl"
    assert config.seed == 7


def test_config_load_missing_path() -> None:
    """load() with a non-existent path returns defaults.

    WHY: Graceful degradation — never crash because a config file is missing.
    """
    config = MltkConfig.load(path="/nonexistent/mltk.yaml")
    assert config.drift_method == "ks"
    assert config.seed == 42


def test_config_from_dict_ignores_unknown_keys() -> None:
    """_from_dict ignores keys not in the dataclass.

    WHY: Forward compatibility — new config keys in newer mltk versions
    shouldn't crash older versions.
    """
    config = MltkConfig._from_dict({"drift_method": "chi2", "unknown_future_key": True})
    assert config.drift_method == "chi2"


def test_config_from_pyproject_no_mltk_section(tmp_path: Path) -> None:
    """pyproject.toml without [tool.mltk] returns defaults.

    WHY: Not every project using mltk will have configured it in pyproject.toml.
    """
    toml_file = tmp_path / "pyproject.toml"
    toml_file.write_text('[project]\nname = "myproject"\n')

    config = MltkConfig._from_pyproject(toml_file)
    assert config.drift_method == "ks"
    assert config.seed == 42


# --- Sprint 17: Environment variable override tests ---


def test_env_var_drift_method(monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO: MLTK_DRIFT_METHOD is set to 'psi' in the environment.
    WHY: CI pipelines often configure mltk via env vars so config files
         don't need to be committed per environment.
    EXPECTED: config.drift_method == 'psi' even when no config file exists.
    """
    monkeypatch.setenv("MLTK_DRIFT_METHOD", "psi")

    config = MltkConfig.load()
    assert config.drift_method == "psi"


def test_env_var_drift_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO: MLTK_DRIFT_THRESHOLD is set to '0.01' in the environment.
    WHY: Production environments may require a stricter threshold than the
         default 0.05 without modifying committed config files.
    EXPECTED: config.drift_threshold == 0.01 (parsed as float).
    """
    monkeypatch.setenv("MLTK_DRIFT_THRESHOLD", "0.01")

    config = MltkConfig.load()
    assert config.drift_threshold == pytest.approx(0.01)
    assert isinstance(config.drift_threshold, float)


def test_env_var_pii_patterns(monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO: MLTK_PII_PATTERNS is set to a comma-separated list.
    WHY: Different deployments may scan for different PII types
         (e.g., healthcare adds 'npi', finance adds 'iban').
    EXPECTED: config.pii_patterns is ['email', 'iban', 'npi'] — split on commas.
    """
    monkeypatch.setenv("MLTK_PII_PATTERNS", "email,iban,npi")

    config = MltkConfig.load()
    assert config.pii_patterns == ["email", "iban", "npi"]


def test_env_var_pii_patterns_trims_whitespace(monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO: MLTK_PII_PATTERNS contains extra whitespace around items.
    WHY: Shell env vars typed by hand often have spaces ('email, phone, ssn').
         The parser must strip whitespace to avoid patterns like ' phone'.
    EXPECTED: Each pattern is stripped; empty strings from trailing commas excluded.
    """
    monkeypatch.setenv("MLTK_PII_PATTERNS", " email , phone , ssn ")

    config = MltkConfig.load()
    assert config.pii_patterns == ["email", "phone", "ssn"]


def test_env_var_seed(monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO: MLTK_SEED is set to '0' in the environment.
    WHY: Reproducibility audits may require a fixed seed different from
         the default 42 without changing committed config.
    EXPECTED: config.seed == 0 (parsed as int).
    """
    monkeypatch.setenv("MLTK_SEED", "0")

    config = MltkConfig.load()
    assert config.seed == 0
    assert isinstance(config.seed, int)


def test_env_var_report_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO: MLTK_REPORT_DIR is set to '/tmp/ci-reports'.
    WHY: CI runners write to ephemeral directories. Env var avoids committing
         CI-specific paths to the shared mltk.yaml.
    EXPECTED: config.report_dir == '/tmp/ci-reports'.
    """
    monkeypatch.setenv("MLTK_REPORT_DIR", "/tmp/ci-reports")

    config = MltkConfig.load()
    assert config.report_dir == "/tmp/ci-reports"


def test_env_var_highest_priority(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO: mltk.yaml sets drift_method='chi2' AND MLTK_DRIFT_METHOD='ks'.
    WHY: The env var cascade rule says env vars beat all file-based config.
         This is the critical priority test — if env vars don't win, CI
         environment overrides silently fail.
    EXPECTED: config.drift_method == 'ks' (env var wins over yaml).
    """
    yaml_content = "drift_method: chi2\ndrift_threshold: 0.20\n"
    yaml_file = tmp_path / "mltk.yaml"
    yaml_file.write_text(yaml_content)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MLTK_DRIFT_METHOD", "ks")

    config = MltkConfig.load()
    # Env var wins
    assert config.drift_method == "ks"
    # Non-overridden value still comes from yaml
    assert config.drift_threshold == pytest.approx(0.20)


def test_env_var_unset_leaves_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO: No MLTK_* env vars are set in the environment.
    WHY: When no env vars are present, _apply_env_overrides must be a no-op
         and not overwrite defaults with None or empty strings.
    EXPECTED: All config fields retain their default values.
    """
    # Ensure no MLTK_* vars bleed in from the test environment
    for key in [
        "MLTK_DRIFT_METHOD", "MLTK_DRIFT_THRESHOLD", "MLTK_REPORT_DIR",
        "MLTK_REPORT_FORMAT", "MLTK_BASELINE_DIR", "MLTK_SEED", "MLTK_PII_PATTERNS",
    ]:
        monkeypatch.delenv(key, raising=False)

    config = MltkConfig.load()
    assert config.drift_method == "ks"
    assert config.drift_threshold == pytest.approx(0.05)
    assert config.seed == 42
    assert "email" in config.pii_patterns


def test_env_var_apply_env_overrides_direct(monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO: _apply_env_overrides is called directly on a known config.
    WHY: Unit test of the method in isolation — verifies it mutates only
         the fields whose env vars are set, leaving others untouched.
    EXPECTED: Only drift_method changes; all other fields keep original values.
    """
    monkeypatch.setenv("MLTK_DRIFT_METHOD", "wasserstein")
    for key in [
        "MLTK_DRIFT_THRESHOLD", "MLTK_REPORT_DIR", "MLTK_REPORT_FORMAT",
        "MLTK_BASELINE_DIR", "MLTK_SEED", "MLTK_PII_PATTERNS",
    ]:
        monkeypatch.delenv(key, raising=False)

    base = MltkConfig(drift_method="ks", drift_threshold=0.10, seed=99)
    result = MltkConfig._apply_env_overrides(base)

    assert result.drift_method == "wasserstein"
    assert result.drift_threshold == pytest.approx(0.10)  # unchanged
    assert result.seed == 99  # unchanged
