"""MLTK configuration — loaded from pyproject.toml or mltk.yaml."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class MltkConfig:
    """Global configuration for mltk test runs."""

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
        """Load config from mltk.yaml, pyproject.toml, or defaults."""
        if path is not None:
            return cls._from_yaml(Path(path))
        yaml_path = Path("mltk.yaml")
        if yaml_path.exists():
            return cls._from_yaml(yaml_path)
        pyproject = Path("pyproject.toml")
        if pyproject.exists():
            return cls._from_pyproject(pyproject)
        return cls()

    @classmethod
    def _from_yaml(cls, path: Path) -> MltkConfig:
        """Load from mltk.yaml."""
        # TODO: Sprint 1 — implement YAML loading
        return cls()

    @classmethod
    def _from_pyproject(cls, path: Path) -> MltkConfig:
        """Load from pyproject.toml [tool.mltk] section."""
        # TODO: Sprint 1 — implement TOML loading
        return cls()

    def to_dict(self) -> dict[str, Any]:
        """Serialize config to dict."""
        return {
            "drift_method": self.drift_method,
            "drift_threshold": self.drift_threshold,
            "report_dir": self.report_dir,
            "report_format": self.report_format,
            "baseline_dir": self.baseline_dir,
            "seed": self.seed,
            "pii_patterns": self.pii_patterns,
        }
