"""Model card generator — auto-generate Google Model Cards from mltk test results."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


def _load_results(results_path: str | Path) -> list[dict[str, Any]]:
    """Load test results from a JSON file.

    Accepts a JSON list or a dict with a ``"results"`` key.
    """
    path = Path(results_path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict) and "results" in raw:
        return raw["results"]
    raise ValueError(
        f"Cannot parse results from {path}. "
        "Expected a JSON list or a dict with a 'results' key."
    )


def _group_by_prefix(results: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group results by their dot-prefix (e.g. 'model.metric', 'data.schema')."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for r in results:
        name: str = r.get("name", "")
        parts = name.split(".")
        prefix = ".".join(parts[:2]) if len(parts) >= 2 else parts[0]
        groups.setdefault(prefix, []).append(r)
    return groups


def _infer_intended_uses(groups: dict[str, list[dict[str, Any]]]) -> list[str]:
    """Derive intended-use bullet points from which test categories are present."""
    use_map = {
        "model.metric": "Classification / regression performance evaluation",
        "model.bias": "Fairness and bias assessment across protected groups",
        "model.slice": "Subgroup and cohort performance analysis",
        "model.calibration": "Probability calibration for reliable confidence scores",
        "model.robust": "Robustness and adversarial resilience testing",
        "model.regression": "No-regression gating against baseline performance",
        "data.schema": "Data quality and schema validation",
        "data.drift": "Distribution drift monitoring between datasets",
        "data.pii": "PII detection and data-privacy compliance",
        "data.labels": "Label quality and class-balance checks",
    }
    uses = []
    for key, desc in use_map.items():
        if key in groups:
            uses.append(desc)
    return uses if uses else ["General-purpose ML model evaluation"]


def _format_metric_row(result: dict[str, Any]) -> str:
    """Format a single metric result as a Markdown table row."""
    name = result.get("name", "unknown")
    passed = result.get("passed", False)
    details = result.get("details") or {}
    actual = details.get("actual_value", details.get("value", "—"))
    threshold = details.get("threshold", details.get("min_value", "—"))
    status = "PASS" if passed else "FAIL"
    return f"| `{name}` | {actual} | {threshold} | {status} |"


def _format_bias_section(bias_results: list[dict[str, Any]]) -> str:
    """Render the Fairness Analysis section body."""
    if not bias_results:
        return "_No bias tests found in results._\n"

    lines: list[str] = []
    for r in bias_results:
        details = r.get("details") or {}
        method = details.get("method", details.get("metric", "unknown"))
        passed = r.get("passed", False)
        status = "PASS" if passed else "FAIL"
        msg = r.get("message", "")
        lines.append(f"- **{r.get('name', 'bias')}** | method: `{method}` | {status}")
        if msg:
            lines.append(f"  - {msg}")
        # Per-group metrics if present
        group_metrics: dict[str, Any] = details.get("group_metrics", {})
        if group_metrics:
            lines.append("  - Per-group values:")
            for group, val in group_metrics.items():
                lines.append(f"    - `{group}`: {val}")
    return "\n".join(lines) + "\n"


def _format_slice_section(slice_results: list[dict[str, Any]]) -> str:
    """Render the Subgroup Performance section body."""
    if not slice_results:
        return "_No slice tests found in results._\n"

    header = (
        "| Test | Slice | Metric | Value | Status |\n"
        "|------|-------|--------|-------|--------|\n"
    )
    rows: list[str] = []
    for r in slice_results:
        details = r.get("details") or {}
        slc = details.get("slice", details.get("group", "—"))
        metric = details.get("metric", "—")
        value = details.get("actual_value", details.get("value", "—"))
        status = "PASS" if r.get("passed", False) else "FAIL"
        rows.append(f"| `{r.get('name', '?')}` | {slc} | {metric} | {value} | {status} |")
    return header + "\n".join(rows) + "\n"


def _format_calibration_section(cal_results: list[dict[str, Any]]) -> str:
    """Render the Calibration section body."""
    if not cal_results:
        return "_No calibration tests found in results._\n"

    header = "| Test | ECE | Max Error | Status |\n|------|-----|-----------|--------|\n"
    rows: list[str] = []
    for r in cal_results:
        details = r.get("details") or {}
        ece = details.get("ece", details.get("calibration_error", "—"))
        max_err = details.get("max_error", "—")
        status = "PASS" if r.get("passed", False) else "FAIL"
        rows.append(f"| `{r.get('name', '?')}` | {ece} | {max_err} | {status} |")
    return header + "\n".join(rows) + "\n"


def _format_robust_section(robust_results: list[dict[str, Any]]) -> str:
    """Render the Robustness section body."""
    if not robust_results:
        return "_No robustness tests found in results._\n"

    lines: list[str] = []
    for r in robust_results:
        details = r.get("details") or {}
        attack = details.get("attack_type", details.get("perturbation", "unknown"))
        delta = details.get("delta", details.get("degradation", "—"))
        status = "PASS" if r.get("passed", False) else "FAIL"
        lines.append(
            f"- **{r.get('name', 'robustness')}** | attack: `{attack}` | "
            f"delta: {delta} | {status}"
        )
        msg = r.get("message", "")
        if msg:
            lines.append(f"  - {msg}")
    return "\n".join(lines) + "\n"


def generate_model_card(
    results_path: str | Path,
    model_name: str = "AI Model",
    model_version: str = "1.0",
    output_path: str | Path = "model-card.md",
) -> Path:
    """Generate a Markdown model card from mltk test results JSON.

    Reads the JSON file produced by ``--mltk-export-json`` and emits a
    Markdown file following Google's Model Cards specification.

    Args:
        results_path: Path to a JSON file containing mltk test results.
            Each result must have at minimum ``"name"`` (str) and
            ``"passed"`` (bool) keys.
        model_name: Display name of the model, written into the card header.
        model_version: Semantic version string for the model (e.g. ``"1.2.0"``).
        output_path: Destination path for the generated Markdown file.
            Parent directories are created automatically.

    Returns:
        :class:`pathlib.Path` pointing to the written Markdown file.

    Example::

        from mltk.report import generate_model_card

        card_path = generate_model_card(
            "results.json",
            model_name="Fraud Detector",
            model_version="2.3.1",
            output_path="docs/model-card.md",
        )
    """
    from mltk import __version__ as mltk_version

    results = _load_results(results_path)
    groups = _group_by_prefix(results)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ---- aggregate stats ----
    total = len(results)
    passed_count = sum(1 for r in results if r.get("passed", False))
    failed_count = total - passed_count

    # ---- intended uses ----
    intended_uses = _infer_intended_uses(groups)

    # ---- metrics section ----
    metric_results = groups.get("model.metric", [])

    # ---- data quality ----
    data_results = [r for key, rs in groups.items() if key.startswith("data.") for r in rs]
    data_passed = sum(1 for r in data_results if r.get("passed", False))
    data_failed = len(data_results) - data_passed

    # ---- limitations: all FAILED tests ----
    failed_results = [r for r in results if not r.get("passed", False)]

    # ====== build Markdown ======
    sections: list[str] = []

    # Header
    sections.append(f"# Model Card: {model_name}\n")

    # ── 1. Model Details ──────────────────────────────────────────────────────
    sections.append("## 1. Model Details\n")
    sections.append(
        f"| Field | Value |\n"
        f"|-------|-------|\n"
        f"| **Model name** | {model_name} |\n"
        f"| **Version** | {model_version} |\n"
        f"| **Date** | {timestamp[:10]} |\n"
        f"| **Framework** | mltk v{mltk_version} |\n"
        f"| **Total tests** | {total} |\n"
        f"| **Tests passed** | {passed_count} |\n"
        f"| **Tests failed** | {failed_count} |\n"
    )

    # ── 2. Intended Use ───────────────────────────────────────────────────────
    sections.append("## 2. Intended Use\n")
    sections.append("Primary use cases inferred from test coverage:\n")
    for use in intended_uses:
        sections.append(f"- {use}")
    sections.append("")

    # ── 3. Metrics ────────────────────────────────────────────────────────────
    sections.append("## 3. Metrics\n")
    if metric_results:
        sections.append(
            "| Test | Actual Value | Threshold | Status |\n"
            "|------|-------------|-----------|--------|\n"
        )
        for r in metric_results:
            sections.append(_format_metric_row(r))
        sections.append("")
    else:
        sections.append("_No `model.metric.*` tests found in results._\n")

    # ── 4. Fairness Analysis ──────────────────────────────────────────────────
    sections.append("## 4. Fairness Analysis\n")
    bias_results = groups.get("model.bias", [])
    sections.append(_format_bias_section(bias_results))

    # ── 5. Subgroup Performance ───────────────────────────────────────────────
    sections.append("## 5. Subgroup Performance\n")
    slice_results = groups.get("model.slice", [])
    sections.append(_format_slice_section(slice_results))

    # ── 6. Calibration ────────────────────────────────────────────────────────
    sections.append("## 6. Calibration\n")
    cal_results = groups.get("model.calibration", [])
    sections.append(_format_calibration_section(cal_results))

    # ── 7. Robustness ─────────────────────────────────────────────────────────
    sections.append("## 7. Robustness\n")
    robust_results = groups.get("model.robust", [])
    sections.append(_format_robust_section(robust_results))

    # ── 8. Data Quality Summary ───────────────────────────────────────────────
    sections.append("## 8. Data Quality Summary\n")
    if data_results:
        sections.append(
            f"| Category | Passed | Failed | Total |\n"
            f"|----------|--------|--------|-------|\n"
            f"| data.*   | {data_passed} | {data_failed} | {len(data_results)} |\n"
        )
        # Break down by sub-category
        data_groups = {k: v for k, v in groups.items() if k.startswith("data.")}
        if len(data_groups) > 1:
            sections.append("")
            sections.append("**By sub-category:**\n")
            sections.append("| Sub-category | Passed | Failed |")
            sections.append("|--------------|--------|--------|")
            for sub_key, sub_rs in sorted(data_groups.items()):
                sp = sum(1 for r in sub_rs if r.get("passed", False))
                sf = len(sub_rs) - sp
                sections.append(f"| `{sub_key}` | {sp} | {sf} |")
            sections.append("")
    else:
        sections.append("_No `data.*` tests found in results._\n")

    # ── 9. Limitations & Known Issues ────────────────────────────────────────
    sections.append("## 9. Limitations & Known Issues\n")
    if failed_results:
        sections.append(
            f"The following {len(failed_results)} test(s) did not pass "
            f"and represent known limitations or areas for improvement:\n"
        )
        for r in failed_results:
            name = r.get("name", "unknown")
            msg = r.get("message", "No message provided.")
            severity = r.get("severity", "")
            sev_tag = f" _(severity: {severity})_" if severity else ""
            sections.append(f"- **`{name}`**{sev_tag}: {msg}")
        sections.append("")
    else:
        sections.append("_No failing tests — all assertions passed._\n")

    # ── 10. Generated By ──────────────────────────────────────────────────────
    sections.append("## 10. Generated By\n")
    sections.append(
        f"This model card was automatically generated by "
        f"**mltk v{mltk_version}** on {timestamp}.\n\n"
        f"> mltk — pytest for ML. "
        f"<https://github.com/Liorrr/mltk>\n"
    )

    # ---- write file ----
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(sections), encoding="utf-8")
    return out
