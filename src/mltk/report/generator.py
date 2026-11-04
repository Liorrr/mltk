"""Report generator -- produces self-contained HTML from test results."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

_MAX_VALUE_CHARS = 120
_MAX_DETAIL_KEYS = 20
_MAX_ASSERTIONS = 50


def _truncate(value: str) -> str:
    """Limit a rendered value to the report's maximum display length."""
    if len(value) > _MAX_VALUE_CHARS:
        return f"{value[:_MAX_VALUE_CHARS - 1]}…"
    return value


def _format_detail_value(value: Any) -> str:
    """Format a detail value without inventing or deriving report data."""
    if isinstance(value, (dict, list)):
        value = json.dumps(value, default=str)
    else:
        value = str(value)
    return _truncate(value)


def _assertions_for_record(record: dict[str, Any]) -> list[Any]:
    """Return assertion objects using the current and legacy record contracts."""
    if "ml_results" in record:
        return list(record.get("ml_results") or [])
    ml_result = record.get("ml_result")
    return [ml_result] if ml_result is not None else []


def _prepare_assertions(record: dict[str, Any]) -> tuple[list[dict[str, Any]], int]:
    """Prepare bounded assertion data for the HTML template."""
    assertions = _assertions_for_record(record)
    prepared: list[dict[str, Any]] = []
    for assertion in assertions[:_MAX_ASSERTIONS]:
        details = getattr(assertion, "details", {}) or {}
        detail_items = [
            {"key": _truncate(str(key)), "value": _format_detail_value(value)}
            for key, value in list(details.items())[:_MAX_DETAIL_KEYS]
        ]
        prepared.append({
            "name": str(getattr(assertion, "name", "")),
            "passed": bool(getattr(assertion, "passed", False)),
            "details": detail_items,
            "details_more": max(len(details) - _MAX_DETAIL_KEYS, 0),
        })
    return prepared, max(len(assertions) - _MAX_ASSERTIONS, 0)


def generate_report(
    results: list[dict[str, Any]],
    output_dir: str | Path = "./mltk-reports",
    title: str = "MLTK Test Report",
) -> Path:
    """Generate a self-contained HTML report from test results.

    Args:
        results: List of test result dicts from MltkReportCollector.
        output_dir: Directory for HTML output.
        title: Report title.

    Returns:
        Path to the generated HTML file.

    Example:
        >>> results = [{"nodeid": "test_a", "outcome": "passed", "duration": 0.1}]
        >>> path = generate_report(results, output_dir="./reports")
    """
    try:
        from jinja2 import Template
    except ImportError as err:
        raise ImportError(
            "jinja2 is required for report generation. "
            "Install with: pip install mlspec[report]"
        ) from err

    # Load template
    template_path = Path(__file__).parent / "templates" / "report.html.j2"
    template = Template(template_path.read_text(encoding="utf-8"), autoescape=True)

    # Compute summary stats
    total = len(results)
    passed = sum(1 for r in results if r.get("outcome") == "passed")
    failed = total - passed
    duration = sum(r.get("duration", 0) for r in results)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Group by module
    modules_map: dict[str, dict[str, int]] = {}
    for r in results:
        nodeid = str(r.get("nodeid", "unknown"))
        module = nodeid.split("::")[0]
        if module not in modules_map:
            modules_map[module] = {"passed": 0, "failed": 0}
        if r.get("outcome") == "passed":
            modules_map[module]["passed"] += 1
        else:
            modules_map[module]["failed"] += 1

    modules = [
        {
            "name": name,
            "passed": counts["passed"],
            "failed": counts["failed"],
            "total": counts["passed"] + counts["failed"],
        }
        for name, counts in sorted(modules_map.items())
    ]

    # Flatten results for details table
    flat_results = []
    for r in results:
        ml_result = r.get("ml_result")
        message = ""
        if ml_result and hasattr(ml_result, "message"):
            message = ml_result.message
        assertions, assertions_more = _prepare_assertions(r)
        flat_results.append({
            "nodeid": r.get("nodeid", ""),
            "outcome": r.get("outcome", "unknown"),
            "duration": r.get("duration", 0),
            "message": message,
            "assertions": assertions,
            "assertions_more": assertions_more,
        })

    # Compute chart data for pure CSS/SVG charts (no external deps)
    pass_pct = round((passed / total) * 100, 1) if total > 0 else 0
    fail_pct = round(100 - pass_pct, 1) if total > 0 else 0

    # SVG donut needs stroke-dasharray values based on a circumference
    # Using radius=70, circumference = 2 * pi * 70 ~= 439.82
    circumference = 439.82
    pass_arc = round(circumference * pass_pct / 100, 2)
    fail_arc = round(circumference * fail_pct / 100, 2)

    # Module bar chart: compute max total for scaling
    max_module_total = max((m["total"] for m in modules), default=1)

    # Render
    html = template.render(
        title=title,
        timestamp=timestamp,
        total=total,
        passed=passed,
        failed=failed,
        duration=duration,
        modules=modules,
        results=flat_results,
        pass_pct=pass_pct,
        fail_pct=fail_pct,
        circumference=circumference,
        pass_arc=pass_arc,
        fail_arc=fail_arc,
        max_module_total=max_module_total,
    )

    # Write output
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = f"report-{datetime.now().strftime('%Y%m%d-%H%M%S')}.html"
    out_path = out_dir / filename
    out_path.write_text(html, encoding="utf-8")

    return out_path
