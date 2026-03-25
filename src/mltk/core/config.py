"""MLTK configuration — loaded from pyproject.toml or mltk.yaml."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class MltkConfig:
    """Global configuration for mltk test runs.

    Configuration is loaded with a cascade priority:
    function args > mltk.yaml > pyproject.toml [tool.mltk] > defaults.
    """

    drift_method: str = "ks"
    drift_threshold: float = 0.05
    report_dir: str = "./mltk-reports"
    report_format: str = "html"
    baseline_dir: str = "./mltk-baselines"
    seed: int = 42
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

        Args:
            path: Explicit path to a YAML config file. None triggers auto-discovery.

        Returns:
            MltkConfig instance populated from the first config source found.

        Example:
            >>> config = MltkConfig.load()
            >>> config = MltkConfig.load("custom-mltk.yaml")
        """
        if path is not None:
            p = Path(path)
            if not p.exists():
                return cls()
            return cls._from_yaml(p)
        yaml_path = Path("mltk.yaml")
        if yaml_path.exists():
            return cls._from_yaml(yaml_path)
        pyproject = Path("pyproject.toml")
        if pyproject.exists():
            return cls._from_pyproject(pyproject)
        return cls()

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
