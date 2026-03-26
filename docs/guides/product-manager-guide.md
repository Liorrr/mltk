# Product Manager Guide

You need evidence that the ML system works, compliance documentation for stakeholders, and trend visibility across releases. mltk provides all three without requiring you to write code.

---

## What mltk gives you

| Need | mltk solution |
|------|---------------|
| "Is the model safe to deploy?" | HTML reports with pass/fail for every quality check |
| "Are we EU AI Act compliant?" | Auto-generated compliance evidence documents |
| "Is the model getting worse?" | Server dashboard with trend tracking across releases |
| "What's our test coverage?" | ML Test Score -- a single number for ML test maturity |
| "Can I share this with auditors?" | PDF compliance reports with timestamped evidence |

---

## HTML reports

Every test run can produce an interactive HTML report:

```bash
pytest --mltk-report -v
```

The report includes:

- **Summary bar** -- total pass/fail/warning at a glance
- **Module breakdown** -- data, model, inference results grouped separately
- **Test details** -- each assertion with severity, timing, and messages
- **Dark theme** -- readable, shareable, self-contained

Reports are single HTML files. Share via email, Confluence, or your artifact store.

:point_right: Full reference: [HTML Reports](../api/report.md)

---

## ML Test Score

Google's ML Test Score rubric, automated. Get a single number that measures your ML testing maturity:

```bash
mltk score tests/
```

Output:

```text
ML Test Score: 3.2 / 5.0

  Data Tests:        ████████░░  4/5
  Model Tests:       ██████░░░░  3/5
  Infrastructure:    ████░░░░░░  2/5
  Monitoring:        ██████░░░░  3/5
```

Use this to track improvement over time and set goals for your team.

:point_right: Full reference: [ML Test Score](../api/ml-test-score.md)

---

## Compliance evidence

### EU AI Act

Generate audit-ready evidence documents that map test results to EU AI Act articles:

```bash
# Run tests with JSON export
pytest --mltk-report --mltk-export-json results.json

# Generate compliance report
mltk compliance results.json --risk-level high --system-name "Face Recognition System"
```

The generated report maps each test result to the relevant EU AI Act article and provides evidence of compliance.

:point_right: Full reference: [EU AI Act Compliance](../api/eu-ai-act.md)

### FDA 21 CFR Part 11

For medical/healthcare ML systems:

```bash
mltk fda-audit results.json --system-name "Medical AI v2" --operator "QA Lead"
```

:point_right: Full reference: [FDA Audit Trail](../api/fda-audit.md)

### PDF export

Generate compliance PDFs for offline distribution to auditors:

```bash
mltk compliance-pdf results.json --output compliance-report.pdf
```

:point_right: Full reference: [Compliance PDF](../api/compliance-pdf.md)

---

## Server dashboard

For ongoing monitoring across releases, deploy the mltk server:

```bash
pip install mltk[server]
mltk server-create-key --project my-project
mltk server
# Dashboard at http://localhost:8080
```

The dashboard shows:

- Test result trends over time
- Pass/fail rates per module
- Performance trends (latency, throughput)
- Webhook alerts for regressions (Slack, GitHub, PagerDuty)

:point_right: Full reference: [Server Platform](../api/server-platform.md)

---

## Model cards

Auto-generate model documentation from test results:

```bash
mltk model-card results.json --model-name "Fraud Detector v3"
```

Model cards include performance metrics, bias analysis, intended use, and limitations -- all derived from actual test runs.

:point_right: Full reference: [Model Card Generator](../api/model-card.md)

---

## No-code test definitions

Your QA team can define tests in YAML without writing Python:

```yaml
data_source: data/training.csv

tests:
  - name: "Schema matches spec"
    assert: schema
    expected:
      id: int64
      score: float64

  - name: "No nulls in labels"
    assert: no_nulls
    columns: [label]

  - name: "At least 1000 rows"
    assert: row_count
    min_rows: 1000
```

```bash
mltk test mltk-tests.yaml
```

:point_right: Full reference: [YAML Test Definitions](../api/yaml-tests.md)

---

## Next steps

- [HTML Reports](../api/report.md) -- report format and customization
- [ML Test Score](../api/ml-test-score.md) -- scoring rubric and improvement guidance
- [EU AI Act Compliance](../api/eu-ai-act.md) -- full article mapping
- [Server Platform](../api/server-platform.md) -- deployment and dashboard features
- [Getting Started](../getting-started.md) -- hands-on tutorial to try it yourself
