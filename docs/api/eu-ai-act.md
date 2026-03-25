# EU AI Act Compliance Report

Generate audit-ready evidence documents that map mltk test results to EU AI Act articles. No competitor has this.

**Module:** `mltk.compliance`

**Install:** `pip install mltk[report]` (requires Jinja2)

**CLI:** `mltk compliance results.json --risk-level high --system-name "FaceID"`

---

## Quick Start

```bash
# 1. Run tests with JSON export
pytest --mltk-report --mltk-export-json results.json

# 2. Generate compliance report
mltk compliance results.json --risk-level high --system-name "Face Recognition System"
```

---

## Risk Levels

Per EU AI Act Annex III:

| Level | Description | Examples |
|-------|-------------|----------|
| `unacceptable` | Banned | Social scoring, real-time biometric in public |
| `high` | Strict obligations | Biometrics, hiring, credit scoring, law enforcement |
| `limited` | Transparency obligations | Chatbots, emotion recognition |
| `minimal` | No obligations | Spam filters, game AI |

---

## Article Mapping

mltk assertions map to EU AI Act requirements:

| Article | Requirement | mltk Assertions |
|---------|-------------|-----------------|
| Art. 10 | Data Governance | assert_schema, assert_no_nulls, assert_no_pii, assert_row_count |
| Art. 10(2f) | Bias Detection | assert_no_bias (5 fairness methods) |
| Art. 15 | Accuracy & Robustness | assert_metric, assert_no_regression, assert_robust |
| Art. 13 | Transparency | ML Test Score, model documentation |
| Art. 14 | Human Oversight | assert_calibration, assert_slice_performance |
| Art. 72 | Post-market Monitoring | assert_no_degradation, assert_sla, assert_no_drift |

---

## Report Sections

The generated HTML report includes:

1. **System Information** -- name, risk level, generation date
2. **Risk Classification** -- which Annex III category applies
3. **Data Governance (Art. 10)** -- data quality test evidence
4. **Accuracy & Robustness (Art. 15)** -- model quality evidence
5. **Bias & Fairness (Art. 10(2f))** -- fairness test evidence
6. **Transparency (Art. 13)** -- ML Test Score summary
7. **Post-market Monitoring (Art. 72)** -- monitoring test evidence
8. **Evidence Table** -- all tests mapped to articles
9. **Gaps** -- untested requirements (what to add)

---

## Python API

```python
from mltk.compliance import generate_compliance_report

report_path = generate_compliance_report(
    results_path="results.json",
    risk_level="high",
    system_name="Face Recognition v2.1",
    output_dir="./mltk-reports",
)
print(f"Report: {report_path}")
```

---
