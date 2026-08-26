"""Committed coverage baseline vs current mapping — compliance drift gate.

Internal drift (a control silently lost covering assertions, or a
control disappeared from the mapper) is a separate category from
external drift (framework text version changed). They never merge
into one score. Confirming a new framework version does **not**
rewrite the baseline file.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mltk.compliance.eu_ai_act import ARTICLE_MAPPING, map_results_to_articles
from mltk.compliance.iso_42001 import ANNEX_A_IDS, map_results_to_clauses
from mltk.compliance.nist_ai_rmf import NIST_RMF_FUNCTION_IDS, map_results_to_measures
from mltk.compliance.owasp_llm import OWASP_LLM_IDS, owasp_llm_scan
from mltk.core.assertion import assert_true, timed_assertion
from mltk.core.result import Severity, TestResult

SUPPORTED_FRAMEWORKS = ("nist_ai_rmf", "iso_42001", "owasp_llm", "eu_ai_act")

FRAMEWORK_VERSIONS: dict[str, str] = {
    "nist_ai_rmf": "AI-100-1-2023-01",
    "iso_42001": "ISO-IEC-42001-2023",
    "owasp_llm": "OWASP-LLM-TOP10-2025",
    "eu_ai_act": "EU-AI-ACT-2024",
}

_UNCAT = "uncategorised"


def _require_framework(framework: str) -> str:
    if framework not in SUPPORTED_FRAMEWORKS:
        raise ValueError(
            f"unknown framework {framework!r}. "
            f"Supported: {', '.join(SUPPORTED_FRAMEWORKS)}"
        )
    return framework


def _control_ids(framework: str) -> list[str]:
    if framework == "nist_ai_rmf":
        return list(NIST_RMF_FUNCTION_IDS)
    if framework == "iso_42001":
        return list(ANNEX_A_IDS)
    if framework == "owasp_llm":
        return list(OWASP_LLM_IDS)
    seen: list[str] = []
    for meta in ARTICLE_MAPPING.values():
        article = str(meta["article"])
        if article not in seen:
            seen.append(article)
    return seen


def _names_by_control(framework: str, results: list[dict]) -> dict[str, list[str]]:
    buckets: dict[str, list[str]] = {cid: [] for cid in _control_ids(framework)}
    if framework == "nist_ai_rmf":
        grouped = map_results_to_measures(results)
        for control, items in grouped.items():
            if control == _UNCAT or control not in buckets:
                continue
            buckets[control] = sorted({str(r.get("name", "")) for r in items})
        return buckets
    if framework == "iso_42001":
        grouped = map_results_to_clauses(results)
        for control, items in grouped.items():
            if control == _UNCAT or control not in buckets:
                continue
            buckets[control] = sorted({str(r.get("name", "")) for r in items})
        return buckets
    if framework == "owasp_llm":
        scanned = owasp_llm_scan(results)
        for control, payload in scanned.items():
            if control not in buckets:
                continue
            covering = payload.get("tests") or []
            names = [
                str(item.get("name", ""))
                for item in covering
                if isinstance(item, dict)
            ]
            buckets[control] = sorted({n for n in names if n})
        return buckets
    grouped = map_results_to_articles(results)
    for control, items in grouped.items():
        if control == _UNCAT or control not in buckets:
            continue
        buckets[control] = sorted({str(r.get("name", "")) for r in items})
    return buckets


def snapshot_coverage(framework: str, results: list[dict]) -> dict[str, list[str]]:
    """Return ``control_id -> sorted unique assertion names`` for *framework*."""
    _require_framework(framework)
    return _names_by_control(framework, results)


def write_coverage_baseline(
    path: str | Path,
    *,
    framework: str,
    framework_version: str,
    controls: dict[str, list[str]],
) -> None:
    """Write a committed coverage baseline JSON file."""
    _require_framework(framework)
    payload = {
        "framework": framework,
        "framework_version": framework_version,
        "controls": {
            key: list(names) for key, names in controls.items()
        },
    }
    dest = Path(path)
    dest.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_coverage_baseline(path: str | Path) -> dict[str, Any]:
    """Load a coverage baseline JSON file."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("coverage baseline must be a JSON object")
    if "framework" not in data or "framework_version" not in data or "controls" not in data:
        raise ValueError(
            "coverage baseline must contain 'framework', "
            "'framework_version', and 'controls'"
        )
    return data


@timed_assertion
def assert_no_compliance_drift(
    results: list[dict],
    baseline_path: str | Path,
    *,
    framework: str,
    confirmed_framework_version: str | None = None,
) -> TestResult:
    """Fail if coverage mapping drifted internally or the framework version did.

    Internal drift
        A control that had covering assertions in the baseline now has
        none, or a baseline control is gone from the live mapper
        (``dropped_control``). Adding coverage is not drift.

    External drift
        ``baseline.framework_version`` differs from the live module
        version and ``confirmed_framework_version`` does not equal the
        live version. Confirmation never rewrites the baseline file.
    """
    _require_framework(framework)
    baseline = load_coverage_baseline(baseline_path)
    live_version = FRAMEWORK_VERSIONS[framework]
    current = snapshot_coverage(framework, results)

    internal_drift: list[dict[str, Any]] = []
    external_drift: list[dict[str, Any]] = []

    live_controls = set(current)
    for control, names in baseline.get("controls", {}).items():
        if control not in live_controls:
            internal_drift.append({
                "control": control,
                "kind": "dropped_control",
                "was": list(names),
            })
            continue
        if len(names) >= 1 and len(current.get(control, [])) == 0:
            internal_drift.append({
                "control": control,
                "kind": "lost_coverage",
                "was": list(names),
            })

    baseline_version = str(baseline.get("framework_version", ""))
    if baseline_version != live_version:
        if confirmed_framework_version != live_version:
            external_drift.append({
                "baseline_version": baseline_version,
                "live_version": live_version,
            })

    passed = not internal_drift and not external_drift
    if not passed:
        parts: list[str] = []
        if internal_drift:
            parts.append(f"internal_drift={len(internal_drift)}")
        if external_drift:
            parts.append(f"external_drift={len(external_drift)}")
        message = "compliance drift: " + ", ".join(parts)
    else:
        message = "no compliance drift"

    return assert_true(
        passed,
        name="compliance.drift",
        message=message,
        severity=Severity.CRITICAL,
        internal_drift=internal_drift,
        external_drift=external_drift,
        framework=framework,
        framework_version=live_version,
    )
