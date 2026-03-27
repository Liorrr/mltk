# HTML Reports

Generate self-contained interactive HTML reports from test results. Dark theme by default, Plotly charts embedded inline.

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
- **Pass/fail donut chart** (green/red segments, powered by Plotly)
- **Duration distribution histogram** (bar chart of test durations)
- Per-module breakdown
- Test details table with severity, duration, messages
- Dark theme (slate background, purple accent)

The charts require `plotly` (included in the `report` extra). If Plotly is not installed or chart rendering fails, the report degrades gracefully to text-only -- no crash, no missing sections.

### pytest Integration

```bash
pytest --mltk-report
# Generates: ./mltk-reports/report-20260325-120000.html
```

---
