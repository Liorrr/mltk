"""Grafana dashboard export -- generate a provisioned dashboard for mltk metrics.

**Why Grafana for ML testing:**

Grafana is the industry standard for operational dashboards (60K+ GitHub stars,
free open-source).  Instead of asking every team to build custom visualizations,
mltk can export a ready-made Grafana dashboard JSON that teams import with one
click.  Organizations that already run Grafana get mltk visibility for free --
no new tool to learn, no new infra to deploy.

**How it works:**

Grafana dashboards are defined as JSON documents that describe panels, queries,
and layout.  This module generates a dashboard with four key panels:

1. **Pass/fail trend** (time series) -- shows test health over time.  A sudden
   dip means something broke; a gradual decline signals model drift.
2. **Assertion duration heatmap** -- highlights slow tests.  ML assertions that
   suddenly take 10x longer often point to data pipeline issues.
3. **Failure rate by module** (bar gauge) -- identifies which modules have the
   most failures, so teams can prioritize fixes.
4. **Latest run summary** (stat panels) -- at-a-glance numbers: total tests,
   pass rate, and average duration of the most recent run.

**Provisioning:**

Grafana supports auto-loading dashboards from YAML provisioning files.  The
``generate_provisioning_yaml`` function creates this config so dashboards
appear automatically when Grafana starts -- useful for Docker/Kubernetes setups.

No external dependencies -- uses only ``json`` and ``textwrap`` from stdlib.
"""

from __future__ import annotations

import json
import textwrap
from typing import Any

# ---------------------------------------------------------------------------
# Dashboard generation
# ---------------------------------------------------------------------------

def _make_panel(
    panel_id: int,
    title: str,
    panel_type: str,
    grid_x: int,
    grid_y: int,
    grid_w: int,
    grid_h: int,
    raw_sql: str,
    datasource: str,
    *,
    description: str = "",
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a single Grafana panel definition.

    Grafana panels are JSON objects that specify *what* to display (the SQL
    query), *how* to display it (the panel type: timeseries, bargauge, stat,
    heatmap), and *where* to place it on the grid.

    Args:
        panel_id: Unique integer ID within the dashboard.
        title: Human-readable panel title shown in the header.
        panel_type: Grafana visualization type (``"timeseries"``,
            ``"bargauge"``, ``"stat"``, ``"heatmap"``).
        grid_x: Horizontal grid position (0--23, Grafana uses a 24-col grid).
        grid_y: Vertical grid position (panels stack top-to-bottom).
        grid_w: Panel width in grid units.
        grid_h: Panel height in grid units.
        raw_sql: SQL query that fetches the data for this panel.
        datasource: Name of the Grafana datasource to query against.
        description: Optional tooltip text explaining the panel.
        overrides: Extra keys merged into the panel dict (e.g. custom options).

    Returns:
        A dictionary representing one Grafana panel, ready to be placed in
        the dashboard's ``panels`` list.
    """
    panel: dict[str, Any] = {
        "id": panel_id,
        "title": title,
        "type": panel_type,
        "datasource": {"type": "sqlite", "uid": datasource},
        "gridPos": {"x": grid_x, "y": grid_y, "w": grid_w, "h": grid_h},
        "targets": [
            {
                "rawSql": raw_sql,
                "format": "time_series",
                "datasource": {"type": "sqlite", "uid": datasource},
            }
        ],
    }
    if description:
        panel["description"] = description
    if overrides:
        panel.update(overrides)
    return panel


def generate_grafana_dashboard(
    datasource: str = "mltk-sqlite",
    title: str = "mltk Test Results",
) -> dict[str, Any]:
    """Generate a Grafana dashboard JSON for mltk metrics.

    **Why Grafana for ML testing:**

    Grafana is the industry standard for dashboards (60K+ GitHub stars, free
    OSS).  Instead of building a custom visualization portal, export mltk data
    as a Grafana dashboard.  Teams that already use Grafana get mltk visibility
    for free.

    The generated dashboard includes panels for:

    - **Pass/fail trend over time** (time series) -- the most important chart
      for any ML system.  If pass rate drops, something changed.
    - **Assertion duration heatmap** -- slow assertions often precede failures.
    - **Failure rate by module** (bar gauge) -- shows which parts of the ML
      pipeline are most fragile.
    - **Latest run summary** (stat panels) -- total tests, pass rate, average
      duration at a glance.

    Args:
        datasource: The Grafana datasource UID to use in all queries.
            Must match the datasource configured in Grafana (typically
            a SQLite or PostgreSQL datasource pointing at the mltk DB).
        title: Dashboard title shown in the Grafana sidebar and header.

    Returns:
        A dictionary that, when serialized to JSON, is a valid Grafana
        dashboard import payload (``{"dashboard": {...}, "overwrite": true}``).

    Example::

        import json
        from mltk.integrations.grafana import generate_grafana_dashboard

        dashboard = generate_grafana_dashboard()
        with open("mltk-dashboard.json", "w") as f:
            json.dump(dashboard, f, indent=2)
        # Then: Grafana UI -> Dashboards -> Import -> Upload JSON
    """
    panels: list[dict[str, Any]] = []

    # --- Panel 1: Pass/fail trend (time series) ---
    # WHY: The single most important chart.  A drop in pass rate is the first
    # signal that data drift, code changes, or infra issues have broken the ML
    # pipeline.  Time series lets teams correlate failures with deployments.
    panels.append(
        _make_panel(
            panel_id=1,
            title="Pass / Fail Trend",
            panel_type="timeseries",
            grid_x=0,
            grid_y=0,
            grid_w=16,
            grid_h=8,
            raw_sql=(
                "SELECT timestamp AS time, passed, failed "
                "FROM runs ORDER BY timestamp"
            ),
            datasource=datasource,
            description=(
                "Test pass and fail counts over time.  A sudden dip in the "
                "green line (passed) usually correlates with a deployment or "
                "data change."
            ),
        )
    )

    # --- Panel 2: Assertion duration heatmap ---
    # WHY: Slow assertions are a leading indicator of failures.  If a data
    # quality check that normally takes 200ms starts taking 5s, the underlying
    # data source is likely degraded.  Heatmaps make these anomalies visible.
    panels.append(
        _make_panel(
            panel_id=2,
            title="Assertion Duration Heatmap",
            panel_type="heatmap",
            grid_x=16,
            grid_y=0,
            grid_w=8,
            grid_h=8,
            raw_sql=(
                "SELECT r.timestamp AS time, res.duration_ms "
                "FROM results res "
                "JOIN runs r ON r.id = res.run_id "
                "ORDER BY r.timestamp"
            ),
            datasource=datasource,
            description=(
                "Duration distribution of individual assertions.  Hot spots "
                "at high durations signal slow or hanging checks."
            ),
        )
    )

    # --- Panel 3: Failure rate by module (bar gauge) ---
    # WHY: Not all failures are equal.  If 90% of failures come from one
    # module, that module needs attention first.  Bar gauge ranking makes
    # this immediately obvious without digging through logs.
    panels.append(
        _make_panel(
            panel_id=3,
            title="Failure Rate by Module",
            panel_type="bargauge",
            grid_x=0,
            grid_y=8,
            grid_w=12,
            grid_h=8,
            raw_sql=(
                "SELECT name, COUNT(*) AS failures "
                "FROM results WHERE passed = 0 "
                "GROUP BY name ORDER BY failures DESC LIMIT 20"
            ),
            datasource=datasource,
            description=(
                "Top 20 assertions by failure count.  Modules at the top of "
                "this list are the most fragile parts of the ML pipeline."
            ),
        )
    )

    # --- Panel 4: Latest run summary (stat) ---
    # WHY: Executives and on-call engineers want one number: "are we healthy?"
    # Stat panels answer that instantly -- total tests, pass rate, avg duration.
    panels.append(
        _make_panel(
            panel_id=4,
            title="Latest Run Summary",
            panel_type="stat",
            grid_x=12,
            grid_y=8,
            grid_w=12,
            grid_h=8,
            raw_sql=(
                "SELECT total, passed, failed, score, duration_ms "
                "FROM runs ORDER BY id DESC LIMIT 1"
            ),
            datasource=datasource,
            description=(
                "Key metrics from the most recent test run: total assertions, "
                "pass count, fail count, overall score, and total duration."
            ),
        )
    )

    dashboard: dict[str, Any] = {
        "dashboard": {
            "title": title,
            "uid": "mltk-main",
            "version": 1,
            "schemaVersion": 39,
            "timezone": "browser",
            "refresh": "30s",
            "tags": ["mltk", "ml-testing"],
            "panels": panels,
            "time": {"from": "now-7d", "to": "now"},
            "templating": {"list": []},
            "annotations": {"list": []},
        },
        "overwrite": True,
    }
    return dashboard


# ---------------------------------------------------------------------------
# File export
# ---------------------------------------------------------------------------

def export_grafana_dashboard(
    output_path: str = "mltk-grafana-dashboard.json",
    datasource: str = "mltk-sqlite",
) -> str:
    """Export dashboard JSON to a file.  Returns the output path.

    This is the primary entry point for CLI usage::

        mltk export-grafana --output /tmp/dashboard.json

    The output file can be imported directly into Grafana via the UI
    (Dashboards -> Import -> Upload JSON file) or via the Grafana HTTP API.

    Args:
        output_path: Filesystem path where the JSON file will be written.
            Parent directories must already exist.
        datasource: Grafana datasource UID passed through to
            :func:`generate_grafana_dashboard`.

    Returns:
        The *output_path* that was written (useful when chaining commands).
    """
    dashboard = generate_grafana_dashboard(datasource=datasource)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(dashboard, fh, indent=2)
    return output_path


# ---------------------------------------------------------------------------
# Provisioning YAML
# ---------------------------------------------------------------------------

def generate_provisioning_yaml(
    dashboard_path: str = "/var/lib/grafana/dashboards/mltk.json",
    datasource_url: str = "http://localhost:8080",
) -> str:
    """Generate Grafana provisioning YAML for auto-loading the dashboard.

    **What is Grafana provisioning?**

    Grafana can auto-load dashboards and datasources from YAML config files at
    startup.  This is essential for Docker / Kubernetes deployments where you
    cannot manually import dashboards via the UI.

    The provisioning config tells Grafana:
    1. Where to find the datasource (the mltk SQLite database).
    2. Where to find the dashboard JSON file on disk.

    **How to use:**

    1. Call ``export_grafana_dashboard()`` to write the dashboard JSON.
    2. Call ``generate_provisioning_yaml()`` to get the YAML config.
    3. Write both files to Grafana's provisioning directories:
       - ``/etc/grafana/provisioning/dashboards/mltk.yml``
       - ``/etc/grafana/provisioning/datasources/mltk.yml``
    4. Restart Grafana -- the dashboard appears automatically.

    Args:
        dashboard_path: Absolute path where the dashboard JSON will be
            stored *inside the Grafana container/host*.
        datasource_url: URL of the mltk server that Grafana will query.

    Returns:
        A YAML string containing both datasource and dashboard provisioning
        configuration, separated by a ``---`` document separator.

    Example::

        yaml_str = generate_provisioning_yaml()
        with open("/etc/grafana/provisioning/dashboards/mltk.yml", "w") as f:
            f.write(yaml_str)
    """
    # Derive the directory from the dashboard file path
    # e.g. "/var/lib/grafana/dashboards/mltk.json" -> "/var/lib/grafana/dashboards"
    folder = dashboard_path.rsplit("/", 1)[0] if "/" in dashboard_path else "."

    yaml_text = textwrap.dedent(f"""\
        # -- Datasource provisioning --
        apiVersion: 1
        datasources:
          - name: mltk-sqlite
            type: frser-sqlite-datasource
            access: proxy
            url: {datasource_url}
            isDefault: false
            editable: true
        ---
        # -- Dashboard provisioning --
        apiVersion: 1
        providers:
          - name: mltk
            orgId: 1
            folder: mltk
            type: file
            disableDeletion: false
            updateIntervalSeconds: 60
            options:
              path: {folder}
              foldersFromFilesStructure: false
    """)
    return yaml_text
