"""Tests for mltk.integrations.grafana -- Grafana dashboard export.

Grafana dashboards are JSON documents that describe panels, queries, and
layout.  These tests verify that the generated JSON has the correct structure
so that importing it into Grafana works without manual fixes.

No network access or running Grafana instance is required.

Each test follows the pattern:

    # SCENARIO: <what situation is being tested>
    # WHY: <reason this behaviour matters>
    # EXPECTED: <what the test asserts>

Test coverage:
    1. generate_grafana_dashboard returns a valid JSON-serializable dict
    2. Dashboard contains the expected panel types (timeseries, bargauge, stat, heatmap)
    3. export_grafana_dashboard writes valid JSON to a file
    4. generate_provisioning_yaml returns YAML with correct structure
    5. Custom datasource name is applied to all panels
    6. Panel count matches the expected number (4 panels)
"""

from __future__ import annotations

import json

from mltk.integrations.grafana import (
    export_grafana_dashboard,
    generate_grafana_dashboard,
    generate_provisioning_yaml,
)

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestGenerateGrafanaDashboard:
    # SCENARIO: generate_grafana_dashboard called with default args
    # WHY: the primary export path -- the returned dict must be valid JSON
    #      that Grafana can import without modification
    # EXPECTED: returns a dict that round-trips through json.dumps/loads
    def test_returns_valid_json_dict(self):
        dashboard = generate_grafana_dashboard()

        assert isinstance(dashboard, dict), "return type must be dict"
        # Must be JSON-serializable (no datetime objects, no sets, etc.)
        serialized = json.dumps(dashboard)
        reloaded = json.loads(serialized)
        assert reloaded == dashboard, "round-trip through JSON must be lossless"
        # Top-level structure required by Grafana import API
        assert "dashboard" in dashboard, "must have 'dashboard' key"
        assert "overwrite" in dashboard, "must have 'overwrite' key"
        assert "panels" in dashboard["dashboard"], "dashboard must contain 'panels'"

    # SCENARIO: verify each expected Grafana panel type is present
    # WHY: if a panel type is wrong (e.g. "graph" instead of "timeseries"),
    #      Grafana will render an empty placeholder instead of the chart.
    #      These are the four visualization types the dashboard requires.
    # EXPECTED: the panels list contains timeseries, heatmap, bargauge, stat
    def test_has_expected_panel_types(self):
        dashboard = generate_grafana_dashboard()
        panels = dashboard["dashboard"]["panels"]
        panel_types = {p["type"] for p in panels}

        assert "timeseries" in panel_types, "must have a timeseries panel (pass/fail trend)"
        assert "heatmap" in panel_types, "must have a heatmap panel (assertion duration)"
        assert "bargauge" in panel_types, "must have a bargauge panel (failure rate by module)"
        assert "stat" in panel_types, "must have a stat panel (latest run summary)"

    # SCENARIO: custom datasource name is applied to all panels
    # WHY: different organizations name their Grafana datasources differently.
    #      If the datasource UID doesn't match, all panels show "No data".
    # EXPECTED: every panel and every target references the custom datasource
    def test_custom_datasource_applied(self):
        custom_ds = "my-org-sqlite-prod"
        dashboard = generate_grafana_dashboard(datasource=custom_ds)
        panels = dashboard["dashboard"]["panels"]

        for panel in panels:
            assert panel["datasource"]["uid"] == custom_ds, (
                f"panel {panel['title']!r} datasource should be {custom_ds!r}"
            )
            for target in panel.get("targets", []):
                assert target["datasource"]["uid"] == custom_ds, (
                    f"target in {panel['title']!r} should reference {custom_ds!r}"
                )

    # SCENARIO: verify the exact number of panels
    # WHY: adding or removing a panel is a deliberate design decision.
    #      This test catches accidental additions/removals during refactoring.
    # EXPECTED: exactly 4 panels (trend, heatmap, failure rate, summary)
    def test_panel_count(self):
        dashboard = generate_grafana_dashboard()
        panels = dashboard["dashboard"]["panels"]
        assert len(panels) == 4, (
            f"expected 4 panels (trend, heatmap, failure-rate, summary), got {len(panels)}"
        )


class TestExportGrafanaDashboard:
    # SCENARIO: export_grafana_dashboard writes JSON to a temp file
    # WHY: this is the CLI-facing function.  If the file isn't valid JSON,
    #      Grafana's import dialog will fail with a cryptic parse error.
    # EXPECTED: file exists, contains valid JSON matching the dashboard structure
    def test_writes_valid_json_file(self, tmp_path):
        out = str(tmp_path / "test-dashboard.json")
        result = export_grafana_dashboard(output_path=out)

        assert result == out, "should return the output path"
        with open(out, encoding="utf-8") as f:
            data = json.load(f)
        assert "dashboard" in data, "exported file must contain 'dashboard' key"
        assert len(data["dashboard"]["panels"]) == 4, "exported file must have 4 panels"


class TestGenerateProvisioningYaml:
    # SCENARIO: generate_provisioning_yaml returns valid YAML structure
    # WHY: provisioning YAML is how Grafana auto-loads dashboards in Docker /
    #      Kubernetes.  Wrong structure = dashboard never appears.
    # EXPECTED: contains apiVersion, datasources, providers, and the custom path
    def test_yaml_has_correct_structure(self):
        yaml_str = generate_provisioning_yaml(
            dashboard_path="/opt/grafana/dashboards/mltk.json",
            datasource_url="http://mltk-server:8080",
        )

        assert isinstance(yaml_str, str), "must return a string"
        # Key structural elements that Grafana requires
        assert "apiVersion: 1" in yaml_str, "must declare apiVersion"
        assert "datasources:" in yaml_str, "must have datasources section"
        assert "providers:" in yaml_str, "must have providers section"
        assert "mltk-sqlite" in yaml_str, "must reference the mltk datasource name"
        assert "http://mltk-server:8080" in yaml_str, "must contain the datasource URL"
        assert "/opt/grafana/dashboards" in yaml_str, "must contain the dashboard directory"
        # YAML document separator between datasource and dashboard configs
        assert "---" in yaml_str, "must have YAML document separator"
