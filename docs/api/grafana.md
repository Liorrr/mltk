# Grafana Integration

Export mltk test result dashboards for Grafana — the industry standard for observability visualization.

**Module:** `mltk.integrations.grafana`

---

## Why Grafana?

Grafana OSS is free, self-hosted, and used by 60K+ teams worldwide. Instead of building a custom visualization portal from scratch, export mltk data as a Grafana dashboard. Teams that already use Grafana get mltk visibility for zero additional effort.

**Design decision**: We chose Grafana over building a custom portal because:

| Option | Effort | Maintenance | Ecosystem |
|--------|--------|-------------|-----------|
| Custom portal | High (build UI) | Ongoing (bug fixes, features) | None |
| **Grafana** | Low (export JSON) | Near-zero (Grafana team maintains it) | Alerts, teams, RBAC, mobile |

---

## Quick Start

```python
from mltk.integrations.grafana import export_grafana_dashboard

# Generate and save the dashboard
path = export_grafana_dashboard("mltk-dashboard.json")
# Import into Grafana: Dashboards → Import → Upload JSON
```

---

## API Reference

### generate_grafana_dashboard

Generate a dashboard JSON dict with 4 panels:

| Panel | Type | Shows |
|-------|------|-------|
| Pass/Fail Trend | Time series | Pass rate over time |
| Duration Heatmap | Heatmap | Assertion execution times |
| Failures by Module | Bar gauge | Which modules fail most |
| Latest Run | Stat | Current pass count and rate |

```python
from mltk.integrations.grafana import generate_grafana_dashboard

dashboard = generate_grafana_dashboard(
    datasource="mltk-sqlite",
    title="ML Test Results",
)
```

### export_grafana_dashboard

Write the dashboard JSON to a file.

### generate_provisioning_yaml

Generate Grafana auto-provisioning config for Docker/Kubernetes deployments.

```python
from mltk.integrations.grafana import generate_provisioning_yaml

yaml_content = generate_provisioning_yaml(
    dashboard_path="/var/lib/grafana/dashboards/mltk.json",
)
```

---

## Docker Setup

```bash
# Start Grafana with auto-provisioned mltk dashboard
docker run -d -p 3000:3000 \
  -v ./mltk-dashboard.json:/var/lib/grafana/dashboards/mltk.json \
  grafana/grafana-oss
```
