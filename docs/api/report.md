# HTML Reports

Generate self-contained interactive HTML reports from test results. Dark theme by default, Plotly charts embedded inline.

**Module:** `mltk.report`

---

## generate_report

```python
from mltk.report import generate_report

path = generate_report(suite, output_dir="./mltk-reports", theme="dark")
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
- Per-module breakdown
- Test details table with severity, duration, messages
- Dark theme (slate background, purple accent)

### pytest Integration

```bash
pytest --mltk-report
# Generates: ./mltk-reports/report-20260325-120000.html
```

---
