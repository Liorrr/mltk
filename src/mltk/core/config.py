"""MLTK configuration — loaded from pyproject.toml or mltk.yaml."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class MltkConfig:
    """Global configuration for mltk test runs.

    Configuration is loaded with a cascade priority:
    env vars > function args > mltk.yaml > pyproject.toml [tool.mltk] > defaults.

    Environment variables (all prefixed with ``MLTK_``):

    - ``MLTK_DRIFT_METHOD``    — drift_method (str)
    - ``MLTK_DRIFT_THRESHOLD`` — drift_threshold (float)
    - ``MLTK_REPORT_DIR``      — report_dir (str)
    - ``MLTK_REPORT_FORMAT``   — report_format (str)
    - ``MLTK_BASELINE_DIR``    — baseline_dir (str)
    - ``MLTK_SEED``            — seed (int)
    - ``MLTK_PII_PATTERNS``    — pii_patterns (comma-separated list)
    - ``MLTK_API_KEY``         — api_key (str, for server auth)
    """

    drift_method: str = "ks"
    drift_threshold: float = 0.05
    report_dir: str = "./mltk-reports"
    report_format: str = "html"
    baseline_dir: str = "./mltk-baselines"
    seed: int = 42
    api_key: str = ""
    pii_patterns: list[str] = field(
        default_factory=lambda: ["email", "phone", "ssn", "credit_card"]
    )

    @classmethod
    def load(cls, path: str | Path | None = None) -> MltkConfig:
        """Load config from mltk.yaml, pyproject.toml, or defaults.

        Search order:
        1. Explicit path (if provided) -- treated as YAML
        2. mltk.yaml in current directory
        3. pyproject.toml [tool.mltk] in current directory
        4. Default config
        5. MLTK_* environment variables (applied last, highest priority)

        Args:
            path: Explicit path to a YAML config file. None triggers auto-discovery.

        Returns:
            MltkConfig instance populated from the first config source found,
            with any MLTK_* environment variables applied on top.

        Example:
            >>> config = MltkConfig.load()
            >>> config = MltkConfig.load("custom-mltk.yaml")
        """
        if path is not None:
            p = Path(path)
            if not p.exists():
                base = cls()
            else:
                base = cls._from_yaml(p)
        elif (yaml_path := Path("mltk.yaml")).exists():
            base = cls._from_yaml(yaml_path)
        elif (pyproject := Path("pyproject.toml")).exists():
            base = cls._from_pyproject(pyproject)
        else:
            base = cls()

        return cls._apply_env_overrides(base)

    @classmethod
    def _apply_env_overrides(cls, config: MltkConfig) -> MltkConfig:
        """Override config fields from MLTK_* environment variables.

        Environment variables always take the highest priority in the cascade.
        Unset variables are silently ignored — only *present* vars override.

        Mapping:
        - ``MLTK_DRIFT_METHOD``    → drift_method (str)
        - ``MLTK_DRIFT_THRESHOLD`` → drift_threshold (float)
        - ``MLTK_REPORT_DIR``      → report_dir (str)
        - ``MLTK_REPORT_FORMAT``   → report_format (str)
        - ``MLTK_BASELINE_DIR``    → baseline_dir (str)
        - ``MLTK_SEED``            → seed (int)
        - ``MLTK_PII_PATTERNS``    → pii_patterns (comma-separated, e.g. "email,phone")
        - ``MLTK_API_KEY``         → api_key (str, for server auth)

        Args:
            config: Base config to apply overrides to.

        Returns:
            New MltkConfig with environment overrides applied.
        """
        drift_method = os.environ.get("MLTK_DRIFT_METHOD")
        if drift_method is not None:
            config.drift_method = drift_method

        drift_threshold = os.environ.get("MLTK_DRIFT_THRESHOLD")
        if drift_threshold is not None:
            config.drift_threshold = float(drift_threshold)

        report_dir = os.environ.get("MLTK_REPORT_DIR")
        if report_dir is not None:
            config.report_dir = report_dir

        report_format = os.environ.get("MLTK_REPORT_FORMAT")
        if report_format is not None:
            config.report_format = report_format

        baseline_dir = os.environ.get("MLTK_BASELINE_DIR")
        if baseline_dir is not None:
            config.baseline_dir = baseline_dir

        seed = os.environ.get("MLTK_SEED")
        if seed is not None:
            config.seed = int(seed)

        pii_patterns = os.environ.get("MLTK_PII_PATTERNS")
        if pii_patterns is not None:
            config.pii_patterns = [p.strip() for p in pii_patterns.split(",") if p.strip()]

        api_key = os.environ.get("MLTK_API_KEY")
        if api_key is not None:
            config.api_key = api_key

        return config

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> MltkConfig:
        """Create config from a flat dict, ignoring unknown keys."""
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered)

    @classmethod
    def _from_yaml(cls, path: Path) -> MltkConfig:
        """Load from mltk.yaml.

        Supports two formats:
        - Flat: ``drift_method: psi``
        - Nested: ``mltk:\\n  drift_method: psi``

        Falls back to JSON parsing if PyYAML is not installed
        (works for simple key: value files).
        """
        text = path.read_text(encoding="utf-8")
        try:
            import yaml

            raw = yaml.safe_load(text)
        except ImportError:
            # Fallback: try JSON (mltk.yaml could be JSON-compatible)
            try:
                raw = json.loads(text)
            except json.JSONDecodeError:
                # Simple key: value parser for basic YAML
                raw = _parse_simple_yaml(text)

        if not isinstance(raw, dict):
            return cls()

        # Support nested: mltk: { ... }
        if "mltk" in raw and isinstance(raw["mltk"], dict):
            raw = raw["mltk"]

        return cls._from_dict(raw)

    @classmethod
    def _from_pyproject(cls, path: Path) -> MltkConfig:
        """Load from pyproject.toml [tool.mltk] section."""
        text = path.read_text(encoding="utf-8")

        if sys.version_info >= (3, 11):
            import tomllib

            data = tomllib.loads(text)
        else:
            try:
                import tomli

                data = tomli.loads(text)
            except ImportError:
                return cls()

        mltk_section = data.get("tool", {}).get("mltk", {})
        if not mltk_section:
            return cls()

        return cls._from_dict(mltk_section)

    def to_dict(self) -> dict[str, Any]:
        """Serialize config to a flat dict suitable for JSON/YAML output.

        Returns:
            Dict with all config fields.

        Example:
            >>> MltkConfig().to_dict()
            {'drift_method': 'ks', 'drift_threshold': 0.05, ...}
        """
        return {
            "drift_method": self.drift_method,
            "drift_threshold": self.drift_threshold,
            "report_dir": self.report_dir,
            "report_format": self.report_format,
            "baseline_dir": self.baseline_dir,
            "seed": self.seed,
            "api_key": self.api_key,
            "pii_patterns": self.pii_patterns,
        }


def _parse_simple_yaml(text: str) -> dict[str, Any]:
    """Minimal YAML parser for flat key: value files (no nesting)."""
    result: dict[str, Any] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip()
        value = value.strip()
        # Try numeric conversion
        try:
            result[key] = int(value)
        except ValueError:
            try:
                result[key] = float(value)
            except ValueError:
                if value.lower() in ("true", "false"):
                    result[key] = value.lower() == "true"
                else:
                    result[key] = value
    return result
