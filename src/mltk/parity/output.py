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
    return json.dumps(normalize(value), ensure_ascii=False, sort_keys=True,
                      allow_nan=False, separators=(",", ":"))


def _sorted_rows(rows: list[dict]) -> list[dict]:
    return sorted((normalize(row) for row in rows), key=_json)


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
    obj["items"] = _sorted_rows(report.items)
    obj["rates"] = _sorted_rows(report.rates)
    obj["reasons"] = sorted(report.reasons)
    return obj


@dataclass(frozen=True)
class ArtifactPaths:
    rates: Path
    items: Path
    gaps: Path
    hashes: dict[str, str]


def write_report(report: ParityReport, output_dir: str | Path) -> ArtifactPaths:
    """Write rates.json, items.csv and gaps.json deterministically.

    CSV rows repeat schema/engine versions and refs. JSON stores the complete
    report, including all item evidence. Gaps preserve Unconcluded evidence for
    later review. Output paths and wall-clock timing never enter these payloads.
    """
    obj = report_to_dict(report)
    metadata = {k: obj[k] for k in ("schema_version", "engine_version", "mode", "refs")}
    rows = [{**row, **metadata} for row in obj["items"]]
    primary = ["item_id", "stratum", "layer", "classification", "group", "reason"]
    columns = primary + sorted(set(metadata) | {k for row in rows for k in row} - set(primary))
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({k: _json(v) if not isinstance(v, str) else v for k, v in row.items()})
    gaps = [row for row in obj["items"] if str(row.get("group", "")).lower() == "unconcluded"]
    gap_payload = {**metadata, "group": "Unconcluded", "items": gaps}
    payloads = {
        "rates.json": (_json(obj) + "\n").encode("utf-8"),
        "items.csv": stream.getvalue().encode("utf-8"),
        "gaps.json": (_json(gap_payload) + "\n").encode("utf-8"),
    }
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    for name, content in payloads.items():
        (directory / name).write_bytes(content)
    return ArtifactPaths(directory / "rates.json", directory / "items.csv", directory / "gaps.json",
                         {name: hashlib.sha256(content).hexdigest()
                          for name, content in payloads.items()})


def report_summary(report: ParityReport, artifacts: ArtifactPaths | None = None) -> dict:
    """Bounded summary for TestResult/pytest; full evidence stays in artifacts."""
    offenders = [r for r in _sorted_rows(report.items)
                 if r.get("classification") not in {"UNCHANGED", "EXPECTED_DIFF"}
                 or r.get("execution_regression") or r.get("exclusion_reason")
                 or r.get("group") == "Unconcluded"]
    result = {"mode": report.mode, "status": report.status, "item_count": len(report.items),
              "offender_count": len(offenders), "preview": offenders[:10],
              "reason_count": len(report.reasons), "reasons": sorted(report.reasons)[:10],
              "rate_count": len(report.rates), "rates": _sorted_rows(report.rates)[:10]}
    if artifacts is not None:
        result["artifacts"] = {"rates": str(artifacts.rates), "items": str(artifacts.items),
                               "gaps": str(artifacts.gaps), "sha256": artifacts.hashes}
    return result
