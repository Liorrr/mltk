"""Tests for mltk.core.config.

Config loading tests verify the cascade: YAML > TOML > defaults.
This ensures users can configure mltk via their preferred method
and that missing/invalid configs fall back gracefully.
"""

from pathlib import Path

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
