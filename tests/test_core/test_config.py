"""Tests for mltk.core.config."""

from mltk.core.config import MltkConfig


def test_default_config() -> None:
    config = MltkConfig()
    assert config.drift_method == "ks"
    assert config.drift_threshold == 0.05
    assert config.seed == 42


def test_config_to_dict() -> None:
    config = MltkConfig()
    d = config.to_dict()
    assert d["drift_method"] == "ks"
    assert isinstance(d["pii_patterns"], list)


def test_config_load_defaults() -> None:
    config = MltkConfig.load()
    assert config.drift_method == "ks"
