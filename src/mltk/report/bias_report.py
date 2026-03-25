"""Bias/fairness report -- demographic breakdown from test results."""

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


def _extract_bias_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return only model.bias test results."""
    return [r for r in results if r.get("name", "").startswith("model.bias")]


def _format_summary_table(bias_results: list[dict[str, Any]]) -> str:
    """Render a summary pass/fail table for all bias tests."""
    if not bias_results:
        return "_No bias tests found in results._\n"

    passed = sum(1 for r in bias_results if r.get("passed", False))
    failed = len(bias_results) - passed

    lines: list[str] = [
        "| Metric | Value |",
        "|--------|-------|",
        f"| Total bias tests | {len(bias_results)} |",
        f"| Passed | {passed} |",
        f"| Failed | {failed} |",
        f"| Pass rate | {passed / len(bias_results) * 100:.1f}% |",
    ]
    return "\n".join(lines) + "\n"


def _format_method_breakdown(bias_results: list[dict[str, Any]]) -> str:
    """Render a per-method breakdown table."""
    if not bias_results:
        return "_No bias tests to break down._\n"

    header = (
        "| Test | Method | Status | Message |\n"
        "|------|--------|--------|---------|\n"
    )
    rows: list[str] = []
    for r in bias_results:
        name = r.get("name", "unknown")
        details = r.get("details") or {}
        method = details.get("method", details.get("metric", details.get("type", "unknown")))
        status = "PASS" if r.get("passed", False) else "FAIL"
        msg = r.get("message", "").replace("|", "/")
        rows.append(f"| `{name}` | `{method}` | {status} | {msg} |")

    return header + "\n".join(rows) + "\n"


def _format_group_metrics(bias_results: list[dict[str, Any]]) -> str:
    """Render per-group metric details for each bias test that has them."""
    sections: list[str] = []

    for r in bias_results:
        details = r.get("details") or {}
        group_metrics: dict[str, Any] = details.get("group_metrics", {})
        if not group_metrics:
            continue

        test_name = r.get("name", "unknown")
        status = "PASS" if r.get("passed", False) else "FAIL"
        sections.append(f"### `{test_name}` ({status})\n")

        header = "| Group | Value |\n|-------|-------|\n"
        rows = [f"| `{group}` | {val} |" for group, val in group_metrics.items()]
        sections.append(header + "\n".join(rows) + "\n")

    if not sections:
        return "_No per-group metrics found in bias test details._\n"

    return "\n".join(sections)


def _format_recommendations(bias_results: list[dict[str, Any]]) -> str:
    """Generate actionable recommendations based on which tests failed."""
    failed = [r for r in bias_results if not r.get("passed", False)]
    if not failed:
        return (
            "All bias tests passed. Continue monitoring for distribution shift "
            "and re-run bias evaluation when the model or training data changes.\n"
        )

    # Build recommendations keyed by method/pattern present in failed tests
    recs: list[str] = []
    failed_methods: set[str] = set()
    for r in failed:
        details = r.get("details") or {}
        m = details.get("method", details.get("metric", details.get("type", "")))
        if m:
            failed_methods.add(m)

    # Generic per-failed-test bullets
    for r in failed:
        name = r.get("name", "unknown")
        msg = r.get("message", "No details provided.")
        recs.append(f"- **`{name}`**: {msg}")

    # Method-specific guidance
    method_guidance: dict[str, str] = {
        "demographic_parity": (
            "Review training data balance across protected groups. "
            "Consider reweighting or resampling to equalise base rates."
        ),
        "equalized_odds": (
            "Investigate false positive and false negative rates per group. "
            "Post-processing threshold calibration per group may help."
        ),
        "equal_opportunity": (
            "True positive rates differ across groups. "
            "Examine recall gaps and consider recall-aware loss functions."
        ),
        "disparate_impact": (
            "Ratio of positive outcomes between groups falls below threshold. "
            "Apply adversarial debiasing or fairness constraints during training."
        ),
        "calibration": (
            "Predicted probabilities are miscalibrated for some groups. "
            "Apply Platt scaling or isotonic regression per group."
        ),
    }

    extra_guidance: list[str] = []
    for method, guidance in method_guidance.items():
        if method in failed_methods:
            extra_guidance.append(f"- **{method.replace('_', ' ').title()}**: {guidance}")

    if extra_guidance:
        recs.append("")
        recs.append("**Method-specific guidance:**")
        recs.extend(extra_guidance)

    return "\n".join(recs) + "\n"


def generate_bias_report(
    results_path: str | Path,
    output_path: str | Path = "bias-report.md",
) -> Path:
    """Generate a Markdown bias/fairness report from test results JSON.

    Extracts model.bias test results and creates a report with:

    1. Summary -- pass/fail counts for bias tests
    2. Per-method breakdown -- demographic parity, equalized odds, etc.
    3. Per-group metrics -- extracted from test result details
    4. Recommendations -- based on which tests failed
    5. Timestamp + generated-by footer

    Args:
        results_path: Path to a JSON file containing mltk test results.
            Each result must have at minimum ``"name"`` (str) and
            ``"passed"`` (bool) keys.
        output_path: Destination path for the generated Markdown file.
            Parent directories are created automatically.

    Returns:
        :class:`pathlib.Path` pointing to the written Markdown file.

    Example::

        from mltk.report import generate_bias_report

        report_path = generate_bias_report(
            "results.json",
            output_path="docs/bias-report.md",
        )
    """
    from mltk import __version__ as mltk_version

    results = _load_results(results_path)
    bias_results = _extract_bias_results(results)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    total = len(results)
    bias_count = len(bias_results)
    bias_passed = sum(1 for r in bias_results if r.get("passed", False))
    bias_failed = bias_count - bias_passed

    sections: list[str] = []

    # Header
    sections.append("# Bias / Fairness Report\n")

    # ── 1. Summary ────────────────────────────────────────────────────────────
    sections.append("## 1. Summary\n")
    sections.append(
        f"| Field | Value |\n"
        f"|-------|-------|\n"
        f"| **Report date** | {timestamp[:10]} |\n"
        f"| **Total tests in file** | {total} |\n"
        f"| **Bias tests found** | {bias_count} |\n"
        f"| **Bias tests passed** | {bias_passed} |\n"
        f"| **Bias tests failed** | {bias_failed} |\n"
    )
    sections.append(_format_summary_table(bias_results))

    # ── 2. Per-Method Breakdown ───────────────────────────────────────────────
    sections.append("## 2. Per-Method Breakdown\n")
    sections.append(_format_method_breakdown(bias_results))

    # ── 3. Per-Group Metrics ──────────────────────────────────────────────────
    sections.append("## 3. Per-Group Metrics\n")
    sections.append(_format_group_metrics(bias_results))

    # ── 4. Recommendations ────────────────────────────────────────────────────
    sections.append("## 4. Recommendations\n")
    sections.append(_format_recommendations(bias_results))

    # ── 5. Generated By ───────────────────────────────────────────────────────
    sections.append("## 5. Generated By\n")
    sections.append(
        f"This bias report was automatically generated by "
        f"**mltk v{mltk_version}** on {timestamp}.\n\n"
        f"> mltk -- pytest for ML. "
        f"<https://github.com/Liorrr/mltk>\n"
    )

    # ---- write file ----
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(sections), encoding="utf-8")
    return out
