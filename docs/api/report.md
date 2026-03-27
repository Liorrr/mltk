# HTML Reports

Generate self-contained interactive HTML reports from test results. Dark theme by default, pure CSS/SVG charts (no external dependencies).

**Module:** `mltk.report`

---

## generate_report

```python
from mltk.report import generate_report

path = generate_report(results, output_dir="./mltk-reports", title="MLTK Test Report")
```

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `results` | `list[dict]` | *(required)* | Test results from MltkReportCollector |
| `output_dir` | `str` | `"./mltk-reports"` | Directory for HTML output |
| `title` | `str` | `"MLTK Test Report"` | Report title |

### Output

Single self-contained HTML file with:
- Pass/fail summary with counts
- **Pass/fail donut chart** (SVG ring with green/red segments, percentage in center)
- **Module breakdown bar chart** (horizontal stacked bars per module, pass/fail counts)
- Per-module breakdown
- Test details table with severity, duration, messages
- Dark theme (slate background, purple accent)

Charts are pure CSS/SVG — no external dependencies, no CDN scripts, no Plotly. The report is fully self-contained and works offline.

### pytest Integration

```bash
pytest --mltk-report
# Generates: ./mltk-reports/report-20260325-120000.html
```

---
