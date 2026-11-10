"""Canonical parity artifacts, with a bounded assertion preview."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any

from mltk.parity.models import ParityReport, Run
from mltk.parity.presentation import (
    finalize_report,
    is_failing,
    offending_rows,
    ordered_rates,
    row_key,
)


def normalize(value: Any) -> Any:
    """JSON-safe values without implicit stringification or callable imports."""
    if is_dataclass(value) and not isinstance(value, type):
        return normalize(asdict(value))
    if isinstance(value, dict):
        if any(not isinstance(k, str) for k in value):
            raise ValueError("Artifact object keys must be strings")
        return {k: normalize(v) for k, v in sorted(value.items())}
    if isinstance(value, (set, frozenset)):
        if any(not isinstance(v, str) for v in value):
            raise ValueError("ID sets require string values")
        return {"type": "id_set", "values": sorted(value)}
    if isinstance(value, (tuple, list)):
        return [normalize(v) for v in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise ValueError(f"Unsupported artifact type: {type(value).__name__}")


def _json(value: Any) -> str:
    return json.dumps(
        normalize(value), ensure_ascii=False, sort_keys=True, allow_nan=False, separators=(",", ":")
    )


def run_to_dict(run: Run) -> dict:
    """Canonical serializable recording, including tagged unordered ID sets."""
    obj = normalize(run)
    obj["items"] = sorted(obj["items"], key=lambda item: item["item_id"])
    for item in obj["items"]:
        item["trials"].sort(key=lambda trial: trial["trial_id"])
    _json(obj)  # Validate finite numbers before returning.
    return obj


def report_to_dict(report: ParityReport) -> dict:
    obj = normalize(report)
    obj["items"] = normalize(sorted(report.items, key=lambda row: row_key(report, row)))
    obj["rates"] = normalize(ordered_rates(report))
    obj["reasons"] = sorted(report.reasons)
    return obj


@dataclass(frozen=True)
class ArtifactPaths:
    rates: Path
    items: Path
    gaps: Path
    hashes: dict[str, str]
    items_jsonl: Path | None = None


def artifact_names(artifacts: ArtifactPaths) -> dict[str, str]:
    """Expose paths relative to the common output directory, never host paths."""
    directory = artifacts.rates.parent
    paths = {"rates": artifacts.rates, "items": artifacts.items, "gaps": artifacts.gaps}
    if artifacts.items_jsonl is not None:
        paths["items_jsonl"] = artifacts.items_jsonl
    return {key: path.relative_to(directory).as_posix() for key, path in paths.items()}


def write_report(report: ParityReport, output_dir: str | Path) -> ArtifactPaths:
    """Write the report envelope/rates and canonical typed rows separately.

    ``items.jsonl`` begins with a header containing the report envelope, followed
    by ``type=item`` records. CSV is a row-only view; gaps retain Unconcluded rows.
    Stable identity ordering and relative artifact names make bytes independent
    of the output directory. All payloads are validated before writing any file.
    """
    names = {
        "rates": "rates.json",
        "items": "items.csv",
        "gaps": "gaps.json",
        "items_jsonl": "items.jsonl",
    }
    report.artifacts = names
    finalize_report(report)
    obj = report_to_dict(report)
    rows = obj.pop("items")
    envelope = {key: value for key, value in obj.items() if key != "rates"}
    primary = ["item_id", "stratum", "layer", "classification", "group", "reason"]
    columns = primary + sorted({k for row in rows for k in row} - set(primary))
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({k: _json(v) if not isinstance(v, str) else v for k, v in row.items()})
    gaps = [row for row in rows if str(row.get("group", "")).lower() == "unconcluded"]
    gap_payload = {**envelope, "group": "Unconcluded", "items": gaps}
    jsonl = [_json({"type": "header", **envelope})]
    jsonl.extend(_json({"type": "item", "item": row}) for row in rows)
    payloads = {
        "rates.json": (_json(obj) + "\n").encode("utf-8"),
        "items.jsonl": ("\n".join(jsonl) + "\n").encode("utf-8"),
        "items.csv": stream.getvalue().encode("utf-8"),
        "gaps.json": (_json(gap_payload) + "\n").encode("utf-8"),
    }
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    for name, content in payloads.items():
        (directory / name).write_bytes(content)
    return ArtifactPaths(
        directory / "rates.json",
        directory / "items.csv",
        directory / "gaps.json",
        {name: hashlib.sha256(content).hexdigest() for name, content in payloads.items()},
        directory / "items.jsonl",
    )


def report_summary(report: ParityReport, artifacts: ArtifactPaths | None = None) -> dict:
    """Bounded, failing-first evidence for TestResult; full rows stay in artifacts."""
    offenders = offending_rows(report)
    rates = ordered_rates(report, failing_first=True)
    fields = (
        "layer",
        "stratum",
        "verdict",
        "reason",
        "changed_rate",
        "floor",
        "mdc",
        "b",
        "c",
        "n_compared",
    )
    compact_rates = [
        {key: rate.get(key, rate.get("status") if key == "verdict" else None) for key in fields}
        for rate in rates[:10]
    ]
    result = {
        "mode": report.mode,
        "status": report.status,
        "item_count": len(report.items),
        "offender_count": len(offenders),
        "preview": [
            {key: row.get(key) for key in ("item_id", "stratum", "layer", "classification")}
            for row in offenders[:10]
        ],
        "failing": [f"{r.get('layer', '')}@{r.get('stratum', '')}" for r in rates if is_failing(r)],
        "reason_count": len(report.reasons),
        "reasons": sorted(report.reasons)[:10],
        "rate_count": len(report.rates),
        "rates": compact_rates,
    }
    if artifacts is not None:
        result["artifacts"] = {**artifact_names(artifacts), "sha256": artifacts.hashes}
    elif report.artifacts:
        result["artifacts"] = dict(report.artifacts)
    return result
