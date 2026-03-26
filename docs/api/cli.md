# CLI Reference

mltk provides a command-line interface for quick data quality checks, test execution, compliance reporting, and server management — all without writing Python code.

```bash
pip install mltk[cli]
```

---

## Commands

### mltk version
```bash
mltk version
# mltk v0.6.0
```

---

### mltk init
Scaffold a starter `mltk.yaml` config and example test file.
```bash
mltk init
# Created mltk.yaml
# Created tests/test_mltk_example.py
```

---

### mltk scan
Quick data quality scan on a CSV or Parquet file. Reports rows, columns, dtypes, null counts.
```bash
mltk scan data/training.csv
# MLTK Data Scan: data/training.csv
#   Rows: 10,000 | Columns: 5
#   Columns: ['id', 'age', 'income', 'score', 'label']
#   Dtypes: {'id': int64, 'age': int64, ...}
#   [PASS] No null values
#   [INFO] Row count: 10,000
```

---

### mltk drift
Compare two CSV datasets for distribution drift across all shared numeric columns.
```bash
mltk drift data/reference.csv data/current.csv --method psi
# MLTK Drift Analysis: data/reference.csv vs data/current.csv
#   Method: psi
#
#   age                  | 0.0300 | OK
#   income               | 0.1800 | DRIFT DETECTED
#   score                | 0.4200 | DRIFT DETECTED
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--method` | `psi` | Drift detection method: `ks`, `psi`, `kl`, `chi2` |

---

### mltk score
Show the ML Test Score categories (Google's 28-test rubric). Run `pytest --mltk-report` first to generate scores.
```bash
mltk score
# ML Test Score
# Run: pytest --mltk-report to generate scores
#
# Categories (Google 28-test rubric):
#   Data:           schema, distribution, drift, freshness, PII, labels
#   Model:          metrics, regression, slicing, calibration, bias, adversarial
#   Infrastructure: reproducibility, pipeline, contract, latency, throughput
#   Monitoring:     drift monitoring, degradation, SLA, alerts
```

---

### mltk doctor
Run 9 diagnostic checks on your environment: Python version, dependencies, config files, report directory, Rust extension, and pytest plugin registration. Prints actionable fix hints for each failure.

```bash
mltk doctor
# mltk doctor — environment diagnostics
# ============================================================
# [OK  ] Python version: 3.12.3
# [OK  ] numpy installed: 1.26.4
# [OK  ] pandas installed: 2.2.1
# [WARN] scipy not installed — required for KS drift test
#          -> pip install scipy
# [OK  ] mltk.yaml config found
# [OK  ] Report directory exists: ./mltk-reports
# [FAIL] Rust extension not available — using pure-Python fallback
#          -> pip install mltk[rust]
# ============================================================
# Summary: 5 OK, 1 warnings, 1 failures
```

Exits with code `1` if any check fails.

---

### mltk test
Run a YAML-defined test suite without writing Python. See [YAML Test Definitions](yaml-tests.md) for the full schema.

```bash
mltk test tests/my_tests.yaml
# mltk test — tests/my_tests.yaml
# Results: 4/5 passed
#
#   [PASS] No nulls in label column
#   [PASS] Schema matches expected dtypes
#   [PASS] No drift in age (PSI 0.03)
#   [PASS] Row count >= 1000
#   [FAIL] No nulls in income column: 42 nulls found
```

Exits with code `1` if any test fails.

---

### mltk model-card
Generate a Google Model Card in Markdown from a `--mltk-export-json` results file.

```bash
mltk model-card results.json \
  --model-name "Text Classifier v2" \
  --model-version "2.0" \
  --output model-card.md
# Model card generated: model-card.md
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--model-name` | `"AI Model"` | Display name of the model |
| `--model-version` | `"1.0"` | Version string |
| `--output` | `model-card.md` | Destination path |

See [Model Card Generator](model-card.md) for a full example.

---

### mltk compliance
Generate an EU AI Act compliance report from a `--mltk-export-json` results file.

```bash
mltk compliance results.json \
  --risk-level high \
  --system-name "Loan Approval Model"
# EU AI Act compliance report generated: mltk-reports/eu-ai-act-report.html
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--risk-level` | `high` | `high`, `limited`, or `minimal` |
| `--system-name` | `"AI System"` | Name of the AI system being assessed |

See [EU AI Act Compliance](eu-ai-act.md).

---

### mltk fda-audit
Generate an FDA 21 CFR Part 11 compliant audit trail from a `--mltk-export-json` results file. Produces a Markdown document with timestamped entries, electronic signatures, and traceability metadata required for FDA-regulated ML/AI systems.

```bash
pytest --mltk-export-json results.json
mltk fda-audit results.json \
  --system-name "Diagnostic Classifier v3" \
  --operator "Jane Smith" \
  --output fda-audit-trail.md
# FDA audit trail generated: fda-audit-trail.md
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `results_json` | *(required, positional)* | Path to JSON results from `--mltk-export-json` |
| `--system-name` | `"AI System"` | Name of the AI/ML system under audit |
| `--operator` | `"QA Engineer"` | Name of the person generating the audit trail |
| `--output` | `fda-audit-trail.md` | Destination path for the audit trail |

See [FDA Audit Trail](fda-audit.md) for the full format specification.

---

### mltk compliance-pdf
Convert an HTML compliance report (EU AI Act or OWASP) to a print-ready PDF. Requires the `weasyprint` package (`pip install mltk[pdf]`).

```bash
# First generate the HTML report
mltk compliance results.json --risk-level high

# Then convert to PDF
mltk compliance-pdf mltk-reports/eu-ai-act-report.html \
  --output compliance-report.pdf
# Compliance PDF exported: compliance-report.pdf
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `html_file` | *(required, positional)* | Path to the HTML compliance report |
| `--output` | *(auto-generated)* | Output PDF path. Defaults to the HTML filename with `.pdf` extension. |

See [Compliance PDF Export](compliance-pdf.md) for styling and formatting details.

---

## Contract Subcommands

### mltk contract init
Scaffold an example `contract.yaml` in the current directory.

```bash
mltk contract init
# Created contract.yaml
```

### mltk contract validate
Validate a CSV or Parquet file against a data contract.

```bash
mltk contract validate data/features.csv
# Contract validation: 5/5 passed
#   [PASS] id: type=int64 (required)
#   [PASS] value: range=[0, 100]
#   [PASS] label: no nulls
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--contract` | `contract.yaml` | Path to contract YAML |

### mltk contract generate-tests
Auto-generate a pytest test file from a data contract.

```bash
mltk contract generate-tests contract.yaml \
  --output tests/test_contract_gen.py
# Generated: tests/test_contract_gen.py
```

**Options:**

| Argument/Flag | Default | Description |
|--------------|---------|-------------|
| `contract` | `contract.yaml` | Path to contract YAML (positional) |
| `--output` | `tests/test_contract_gen.py` | Destination for generated file |

See [Data Contracts](contracts.md).

---

## Docs Subcommands

### mltk docs serve
Serve the MkDocs documentation locally with hot reload. Requires `mkdocs.yml` in the current directory.

```bash
mltk docs serve
# Serving docs at http://127.0.0.1:8000
# Press Ctrl+C to stop

mltk docs serve --port 9000 --host 0.0.0.0
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--port` | `8000` | Port to serve on (also `MLTK_DOCS_PORT` env var) |
| `--host` | `127.0.0.1` | Host to bind to (also `MLTK_DOCS_HOST` env var) |

### mltk docs build
Build static HTML documentation to a local directory.

```bash
mltk docs build
# Docs built to: site/

mltk docs build --output public/
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--output` | `site` | Output directory for built HTML |

### mltk docs open
Build the documentation, start a local HTTP server, and open the result in your default browser.

```bash
mltk docs open
# Building docs...
# Docs available at: http://127.0.0.1:8000
# Press Ctrl+C to stop
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--port` | `8000` | Port for the local server (also `MLTK_DOCS_PORT` env var) |

---

## Registry Subcommands

### mltk registry push
Save a directory of test fixture files to the local registry as a named collection.

```bash
mltk registry push my_fixtures
mltk registry push my_fixtures --source tests/fixtures \
  --description "Smoke-test CSVs" \
  --version 1.1 \
  --tags smoke,data
# Saved collection 'my_fixtures' to ~/.mltk/registry/my_fixtures
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--source` | `.` | Source directory to push |
| `--description` | `""` | Human-readable description |
| `--version` | `1.0` | Collection version |
| `--tags` | `""` | Comma-separated list of tags |

### mltk registry pull
Restore a named collection from the registry into a local directory.

```bash
mltk registry pull my_fixtures
mltk registry pull my_fixtures --target tests/
# Loaded collection 'my_fixtures' to tests/my_fixtures
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--target` | `.` | Destination directory |

### mltk registry list
List all collections saved in the registry.

```bash
mltk registry list
# NAME                 VERSION    TAGS                           DESCRIPTION
# --------------------------------------------------------------------------------
# my_fixtures          1.0        smoke, data                    Smoke-test CSVs
# cv_benchmarks        2.0        cv, benchmark                  COCO val subset
```

---

## Notify Subcommands

### mltk notify slack
Send test results or a custom message to a Slack channel via an incoming webhook.

```bash
# Send full results summary
mltk notify slack \
  --webhook-url https://hooks.slack.com/services/... \
  --results-json mltk-reports/results.json

# Use environment variable for webhook URL
export MLTK_SLACK_WEBHOOK=https://hooks.slack.com/services/...
mltk notify slack --results-json mltk-reports/results.json

# Send a plain message
mltk notify slack --message "Nightly ML tests passed"
# Slack notification sent
```

**Options:**

| Flag | Env Var | Description |
|------|---------|-------------|
| `--webhook-url` | `MLTK_SLACK_WEBHOOK` | Slack incoming webhook URL (required) |
| `--results-json` | — | Path to JSON from `--mltk-export-json` |
| `--message` | — | Custom plain-text message |

At least one of `--results-json` or `--message` must be provided.

See [GitHub, Slack & Plugins](github-integration.md) for the full Slack API.

---

## Server Subcommands

### mltk server
Start the self-hosted mltk server platform. Stores test results in SQLite and serves a live dashboard.

```bash
mltk server
# mltk server at http://127.0.0.1:8080

mltk server --host 0.0.0.0 --port 9000 --db /data/mltk.db
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--host` | `127.0.0.1` | Host to bind to |
| `--port` | `8080` | Port to listen on |
| `--db` | `mltk_server.db` | SQLite database path |

Requires `pip install mltk[server]`. See [Server Platform](server-platform.md).

### mltk server-create-key
Generate an API key for authenticating against the mltk server. Prints the raw key once — store it securely.

```bash
mltk server-create-key --project my-project
# API key created for project 'my-project':
#   mltk_aBcDeFgHiJkLmNoPqRsTuVwXyZ1234567890abcd
# Store this key securely — it will not be shown again.
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--project` | `default` | Project name to associate with the key |
| `--db` | `mltk_server.db` | SQLite database path |

---

## Chat Subcommand

### mltk chat
Interactive, rule-based Q&A about your test results. No external API or LLM required.

```bash
pytest --mltk-export-json results.json
mltk chat --results-json results.json
# mltk chat — ask questions about your test results
# Type 'help' for available commands, 'quit' to exit
#
# mltk> what failed?
# mltk> recommend
# mltk> quit
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--results-json` | — | Path to JSON from `--mltk-export-json` |

See [Chat Interface](chat.md).

---

## Environment Variables

All CLI commands respect these environment variables:

| Variable | Used By | Description |
|----------|---------|-------------|
| `MLTK_SLACK_WEBHOOK` | `notify slack` | Default Slack webhook URL |
| `MLTK_DOCS_PORT` | `docs serve`, `docs open` | Override default docs port |
| `MLTK_DOCS_HOST` | `docs serve` | Override default docs host |

---
