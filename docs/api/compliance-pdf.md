# Compliance PDF Export

Convert HTML compliance reports to print-ready PDF.

**Module:** `mltk.compliance`

**CLI:** `mltk compliance-pdf report.html`

---

## Quick Start

```bash
# Generate HTML compliance report first
mltk compliance results.json --risk-level high

# Convert to print-ready format
mltk compliance-pdf mltk-reports/eu-ai-act-report.html
```

## How it works

1. If `weasyprint` is installed (`pip install mltk[pdf]`): generates actual PDF
2. Otherwise: injects `@media print` CSS into the HTML for browser Print-to-PDF

## Python API

```python
from mltk.compliance import export_compliance_pdf

# Generates PDF (if weasyprint) or print-ready HTML (fallback)
output = export_compliance_pdf("report.html")
```

---
