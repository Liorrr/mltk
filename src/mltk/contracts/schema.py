"""Contract schema parsing — load and validate mltk contract YAML files."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ColumnSpec:
    """Specification for a single column in a data contract."""

    name: str
    type: str = "object"
    nullable: bool = True
    unique: bool = False
    range: tuple[float, float] | None = None
    pii_class: str | None = None


@dataclass
class QualitySpec:
    """Quality requirements for the dataset."""

    min_rows: int | None = None
    max_rows: int | None = None
    max_nulls_pct: float | None = None
    freshness_days: int | None = None
    freshness_column: str | None = None


@dataclass
class Contract:
    """Parsed data contract specification.

    Args:
        name: Contract name.
        version: Contract version string.
        columns: List of column specifications.
        quality: Quality requirements.

    Example:
        >>> contract = Contract.from_yaml("contract.yaml")
        >>> print(contract.name, len(contract.columns))
    """

    name: str = "unnamed"
    version: str = "1.0"
    columns: list[ColumnSpec] = field(default_factory=list)
    quality: QualitySpec = field(default_factory=QualitySpec)

    @classmethod
    def from_yaml(cls, path: str | Path) -> Contract:
        """Parse a contract from a YAML file.

        Args:
            path: Path to the YAML contract file.

        Returns:
            Parsed Contract instance.
        """
        p = Path(path)
        text = p.read_text(encoding="utf-8")

        try:
            import yaml

            raw = yaml.safe_load(text)
        except ImportError:
            raw = json.loads(text)

        if not isinstance(raw, dict):
            return cls()

        # Parse columns
        columns = []
        for col_name, col_spec in raw.get("columns", {}).items():
            if not isinstance(col_spec, dict):
                col_spec = {"type": str(col_spec)}
            col_range = col_spec.get("range")
            if col_range and isinstance(col_range, list) and len(col_range) == 2:
                col_range = (float(col_range[0]), float(col_range[1]))
            else:
                col_range = None

            columns.append(ColumnSpec(
                name=col_name,
                type=col_spec.get("type", "object"),
                nullable=col_spec.get("nullable", True),
                unique=col_spec.get("unique", False),
                range=col_range,
                pii_class=col_spec.get("pii_class"),
            ))

        # Parse quality
        q = raw.get("quality", {})
        quality = QualitySpec(
            min_rows=q.get("min_rows"),
            max_rows=q.get("max_rows"),
            max_nulls_pct=q.get("max_nulls_pct"),
            freshness_days=q.get("freshness_days"),
            freshness_column=q.get("freshness_column"),
        )

        return cls(
            name=raw.get("name", "unnamed"),
            version=raw.get("version", "1.0"),
            columns=columns,
            quality=quality,
        )
