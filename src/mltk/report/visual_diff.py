"""Visual diff report -- side-by-side comparison of two test runs."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from mltk.server.comparison import compare_runs


def generate_diff_report(
    run_a: dict,
    run_b: dict,
    output_path: str | Path = "diff-report.html",
) -> Path:
    """Generate an HTML visual diff between two test runs.

    Uses :func:`mltk.server.comparison.compare_runs` to compute the
    structured diff, then renders a self-contained dark-themed HTML report.

    Shows: side-by-side table, color-coded changes (green=fixed, red=regressed,
    yellow=new), score comparison.

    Args:
        run_a: First test run dict (baseline). Must contain ``results``
            (list of dicts with ``name`` and ``passed``) and optionally
            ``score``.
        run_b: Second test run dict (comparison target).
        output_path: Where to write the HTML file.

    Returns:
        Path to the generated HTML file.

    Example:
        >>> run_a = {"results": [{"name": "t1", "passed": True}], "score": 90}
        >>> run_b = {"results": [{"name": "t1", "passed": False}], "score": 70}
        >>> path = generate_diff_report(run_a, run_b, "diff.html")
    """
    diff = compare_runs(run_a, run_b)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    score_a = float(run_a.get("score", 0.0))
    score_b = float(run_b.get("score", 0.0))
    score_change = diff["score_change"]

    # Build results lookup for side-by-side table
    results_a = {r["name"]: r.get("passed", False) for r in run_a.get("results", [])}
    results_b = {r["name"]: r.get("passed", False) for r in run_b.get("results", [])}
    all_names = sorted(set(results_a) | set(results_b))

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    html = _render_html(
        diff=diff,
        score_a=score_a,
        score_b=score_b,
        score_change=score_change,
        results_a=results_a,
        results_b=results_b,
        all_names=all_names,
        timestamp=timestamp,
    )

    output_path.write_text(html, encoding="utf-8")
    return output_path


# ---------------------------------------------------------------------------
# Private rendering helpers
# ---------------------------------------------------------------------------

_STATUS_PASS = '<span class="badge pass">PASS</span>'
_STATUS_FAIL = '<span class="badge fail">FAIL</span>'
_STATUS_NA = '<span class="badge na">N/A</span>'


def _status_badge(passed: bool | None) -> str:
    if passed is None:
        return _STATUS_NA
    return _STATUS_PASS if passed else _STATUS_FAIL


def _row_class(name: str, diff: dict) -> str:
    """Return a CSS class for the row based on diff category."""
    if name in diff["new_failures"]:
        return "regressed"
    if name in diff["fixed"]:
        return "fixed"
    if name in diff["new_tests"]:
        return "new-test"
    if name in diff["removed_tests"]:
        return "removed"
    return ""


def _render_html(
    *,
    diff: dict,
    score_a: float,
    score_b: float,
    score_change: float,
    results_a: dict,
    results_b: dict,
    all_names: list[str],
    timestamp: str,
) -> str:
    # Score arrow
    if score_change > 0:
        arrow = f'<span class="score-up">+{score_change}</span>'
    elif score_change < 0:
        arrow = f'<span class="score-down">{score_change}</span>'
    else:
        arrow = '<span class="score-same">0</span>'

    # Summary counts
    n_fixed = len(diff["fixed"])
    n_regressed = len(diff["new_failures"])
    n_new = len(diff["new_tests"])
    n_removed = len(diff["removed_tests"])
    n_still_pass = len(diff["still_passing"])
    n_still_fail = len(diff["still_failing"])

    # Side-by-side rows
    rows = []
    for name in all_names:
        passed_a = results_a.get(name)
        passed_b = results_b.get(name)
        css_class = _row_class(name, diff)
        rows.append(
            f'<tr class="{css_class}">'
            f'<td class="test-name">{name}</td>'
            f'<td class="status-cell">{_status_badge(passed_a)}</td>'
            f'<td class="status-cell">{_status_badge(passed_b)}</td>'
            f"</tr>"
        )
    rows_html = "\n".join(rows)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MLTK Visual Diff Report</title>
<style>
  :root {{
    --bg: #1e1e2e;
    --surface: #252538;
    --border: #3d3d5c;
    --accent: #7c3aed;
    --accent-light: #a78bfa;
    --text: #e2e8f0;
    --text-muted: #94a3b8;
    --green: #22c55e;
    --red: #ef4444;
    --yellow: #eab308;
    --blue: #3b82f6;
  }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    background: var(--bg);
    color: var(--text);
    padding: 24px;
    line-height: 1.6;
  }}
  h1 {{
    color: var(--accent-light);
    font-size: 1.6em;
    margin-bottom: 4px;
  }}
  .timestamp {{
    color: var(--text-muted);
    font-size: 0.85em;
    margin-bottom: 24px;
  }}
  .summary-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    gap: 12px;
    margin-bottom: 24px;
  }}
  .summary-card {{
    background: var(--surface);
    border-radius: 8px;
    padding: 14px 16px;
    border-left: 3px solid var(--border);
  }}
  .summary-card.fixed {{ border-left-color: var(--green); }}
  .summary-card.regressed {{ border-left-color: var(--red); }}
  .summary-card.new {{ border-left-color: var(--yellow); }}
  .summary-card.removed {{ border-left-color: var(--text-muted); }}
  .summary-card.passing {{ border-left-color: var(--green); }}
  .summary-card.failing {{ border-left-color: var(--red); }}
  .card-value {{
    font-size: 1.8em;
    font-weight: 700;
    color: var(--accent-light);
  }}
  .card-label {{
    font-size: 0.8em;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }}
  .score-section {{
    background: var(--surface);
    border-radius: 8px;
    padding: 16px 20px;
    margin-bottom: 24px;
    border-left: 3px solid var(--accent);
    display: flex;
    align-items: center;
    gap: 24px;
    flex-wrap: wrap;
  }}
  .score-label {{
    font-size: 0.8em;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }}
  .score-value {{
    font-size: 1.4em;
    font-weight: 700;
    color: var(--accent-light);
  }}
  .score-up {{ color: var(--green); font-weight: 700; font-size: 1.4em; }}
  .score-down {{ color: var(--red); font-weight: 700; font-size: 1.4em; }}
  .score-same {{ color: var(--text-muted); font-weight: 700; font-size: 1.4em; }}
  table {{
    width: 100%;
    border-collapse: collapse;
    background: var(--surface);
    border-radius: 8px;
    overflow: hidden;
    font-size: 0.9em;
  }}
  thead th {{
    background: #2a2a40;
    color: var(--accent);
    padding: 10px 14px;
    text-align: left;
    font-weight: 600;
    font-size: 0.85em;
    text-transform: uppercase;
    letter-spacing: 0.04em;
  }}
  tbody td {{
    padding: 8px 14px;
    border-top: 1px solid var(--border);
  }}
  .test-name {{
    color: var(--text);
    font-family: 'Cascadia Code', 'Fira Code', monospace;
    font-size: 0.9em;
  }}
  .status-cell {{ text-align: center; }}
  .badge {{
    display: inline-block;
    padding: 2px 10px;
    border-radius: 4px;
    font-size: 0.75em;
    font-weight: 700;
    letter-spacing: 0.04em;
  }}
  .badge.pass {{ background: var(--green); color: #fff; }}
  .badge.fail {{ background: var(--red); color: #fff; }}
  .badge.na {{ background: var(--border); color: var(--text-muted); }}
  tr.regressed {{ background: rgba(239, 68, 68, 0.1); }}
  tr.fixed {{ background: rgba(34, 197, 94, 0.1); }}
  tr.new-test {{ background: rgba(234, 179, 8, 0.1); }}
  tr.removed {{ background: rgba(148, 163, 184, 0.08); }}
  .legend {{
    display: flex;
    gap: 16px;
    flex-wrap: wrap;
    margin-bottom: 12px;
    font-size: 0.8em;
    color: var(--text-muted);
  }}
  .legend-item {{
    display: flex;
    align-items: center;
    gap: 6px;
  }}
  .legend-swatch {{
    width: 14px;
    height: 14px;
    border-radius: 3px;
  }}
  .footer {{
    margin-top: 24px;
    text-align: center;
    font-size: 0.75em;
    color: var(--text-muted);
  }}
</style>
</head>
<body>
  <h1>MLTK Visual Diff Report</h1>
  <div class="timestamp">Generated {timestamp}</div>

  <div class="score-section">
    <div>
      <div class="score-label">Run A Score</div>
      <div class="score-value">{score_a}</div>
    </div>
    <div>
      <div class="score-label">Run B Score</div>
      <div class="score-value">{score_b}</div>
    </div>
    <div>
      <div class="score-label">Change</div>
      <div>{arrow}</div>
    </div>
  </div>

  <div class="summary-grid">
    <div class="summary-card fixed">
      <div class="card-value">{n_fixed}</div>
      <div class="card-label">Fixed</div>
    </div>
    <div class="summary-card regressed">
      <div class="card-value">{n_regressed}</div>
      <div class="card-label">Regressed</div>
    </div>
    <div class="summary-card new">
      <div class="card-value">{n_new}</div>
      <div class="card-label">New Tests</div>
    </div>
    <div class="summary-card removed">
      <div class="card-value">{n_removed}</div>
      <div class="card-label">Removed</div>
    </div>
    <div class="summary-card passing">
      <div class="card-value">{n_still_pass}</div>
      <div class="card-label">Still Passing</div>
    </div>
    <div class="summary-card failing">
      <div class="card-value">{n_still_fail}</div>
      <div class="card-label">Still Failing</div>
    </div>
  </div>

  <div class="legend">
    <div class="legend-item">
      <div class="legend-swatch" style="background:rgba(239,68,68,0.25);"></div>
      Regressed
    </div>
    <div class="legend-item">
      <div class="legend-swatch" style="background:rgba(34,197,94,0.25);"></div>
      Fixed
    </div>
    <div class="legend-item">
      <div class="legend-swatch" style="background:rgba(234,179,8,0.25);"></div>
      New Test
    </div>
    <div class="legend-item">
      <div class="legend-swatch" style="background:rgba(148,163,184,0.15);"></div>
      Removed
    </div>
  </div>

  <table>
    <thead>
      <tr>
        <th>Test Name</th>
        <th style="text-align:center;">Run A</th>
        <th style="text-align:center;">Run B</th>
      </tr>
    </thead>
    <tbody>
      {rows_html}
    </tbody>
  </table>

  <div class="footer">
    MLTK Visual Diff &mdash; {len(all_names)} tests compared
  </div>
</body>
</html>"""
