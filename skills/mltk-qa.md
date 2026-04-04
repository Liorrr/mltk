---
description: >
  mltk QA engineer persona — workflows for scanning, interpreting findings,
  writing test assertions, and using MCP tools for ML quality assurance.
---

# mltk QA Engineer Skill

## Role Summary

A QA engineer using mltk runs scans to surface ML quality issues, converts findings
into regression-preventing test assertions, verifies fixes, and generates compliance
reports. Every finding carries the exact assertion call that detected it
(`finding.suggested_test`), so findings become tests without extra work.

---

## Quick Start Workflow

```
1. Run scan          →  mltk scan (CLI) or mltk_scan (MCP)
2. Read findings     →  check severity, scanner_name, suggested_fixes
3. Triage            →  CRITICAL blocks, WARNING → test, INFO → track
4. Write assertions  →  convert each finding to a pytest test
5. Verify fix        →  re-run scan; finding should clear
6. Report            →  mltk_report (MCP) or mltk compliance (CLI)
```

### CLI
```bash
mltk scan --model model.pkl --data data.csv --output report.html
mltk scan --model model.pkl --data data.csv --format json > findings.json
```

### Python API (`src/mltk/scan/engine.py`)
```python
from __future__ import annotations
from mltk.scan.engine import ScanEngine
from mltk.scan.config import ScanConfig

engine = ScanEngine(ScanConfig(seed=42))
report = engine.scan(model.predict, X_test, y_test)
print(report.summary())
report.to_test_file("tests/test_regression.py")  # auto-generates pytest file
```

---

## Scan Interpretation Guide

Each `ScanFinding` has: `result.severity`, `result.passed`, `scanner_name`,
`suggested_fixes`, `suggested_test`. Source: `src/mltk/scan/finding.py`.

| Scanner | Fires when | Key signal |
|---------|-----------|------------|
| `data` | Nulls, outliers, schema violations, PII | `result.name` starts with `data.` |
| `drift` | Distribution shift train vs. serve | PSI / KL divergence value |
| `bias` | Fairness violations across protected groups | protected column + disparity delta |
| `overfit` | Train/test performance gap exceeds threshold | train_score vs test_score |
| `calibration` | Predicted probabilities don't match actual rates | ECE / Brier score |
| `robustness` | Model sensitive to small input perturbations | perturbation magnitude + accuracy drop |
| `leakage` | Feature correlates with target suspiciously high | mutual info or AUC-of-feature |
| `slice` | Subgroup performance below global threshold | slice key + slice accuracy |

### Triage decision tree
```
CRITICAL  →  block merge; fix immediately
WARNING   →  write regression test; schedule fix this sprint
INFO      →  log in backlog; monitor trend
suggested_fixes[0].confidence == "high"  →  apply fix directly
suggested_fixes[0].confidence == "low"   →  use mltk_experiment to rank candidates
```

---

## Writing Test Assertions

Use `finding.suggested_test` as the starting point (valid, `ast.parse()`-checked code).
Pattern: one finding → one `def test_*` function.

```python
from __future__ import annotations

# Data (src/mltk/data/) — DataScanner finding
from mltk.data import assert_no_nulls, assert_no_outliers, assert_schema
def test_no_nulls_age():
    assert assert_no_nulls(df, columns=["age"]).passed
def test_no_outliers_income():
    assert assert_no_outliers(df, columns=["income"], method="iqr").passed

# Drift (src/mltk/scan/scanners/drift.py) — DriftScanner finding
from mltk import assert_no_drift, assert_no_multivariate_drift, assert_no_embedding_drift
def test_no_feature_drift():
    assert assert_no_drift(X_train, X_serve, method="psi", threshold=0.2).passed

# Bias (src/mltk/scan/scanners/bias.py) — BiasScanner finding
from mltk import assert_no_bias, assert_intersectional_fairness
def test_no_gender_bias():
    assert assert_no_bias(y_true, y_pred, sensitive=df["gender"], metric="equalized_odds").passed

# Model quality (src/mltk/scan/scanners/) — Overfit/Calibration/Slice findings
from mltk import assert_no_overfitting, assert_calibration, assert_slice_performance
def test_no_overfitting():
    assert assert_no_overfitting(model, X_train, y_train, X_test, y_test, max_gap=0.05).passed
def test_calibration():
    assert assert_calibration(y_true, y_prob, max_ece=0.05).passed
def test_slice_performance():
    assert assert_slice_performance(y_true, y_pred, slices=df[["age_group"]], min_accuracy=0.80).passed

# LLM (src/mltk/domains/llm/) — LLM domain findings
from mltk.domains.llm import assert_no_toxicity, assert_no_hallucination, assert_faithfulness
def test_no_toxicity():
    assert assert_no_toxicity(model_outputs, threshold=0.1).passed
def test_faithfulness():
    assert assert_faithfulness(outputs, contexts, threshold=0.85).passed
```

---

## MCP Tool Workflow

Use when operating as an AI agent via MCP (`src/mltk/mcp/`).

| Tool | Purpose |
|------|---------|
| `mltk_scan` | Run full scan, returns findings list |
| `mltk_suggest` | Get `FixSuggestion` list for a finding_id |
| `mltk_experiment` | Rank competing fix suggestions via ExperimentRunner |
| `mltk_eval` | Score model output quality with solver/scorer pipeline |
| `mltk_report` | Generate HTML/JSON/PDF stakeholder report |
| `mltk_create_issue` | File Jira/GitHub ticket for a finding |
| `mltk_create_pr` | Auto-create fix PR using `suggested_test` as content |

Typical sequence: `mltk_scan` → `mltk_suggest` (per critical finding) →
`mltk_experiment` (if confidence is low) → `mltk_create_issue` → `mltk_report`.

---

## CLI Commands for QA

```bash
mltk scan --model model.pkl --data data.csv          # full project scan
mltk test                                             # run test suite
mltk score --model model.pkl --data data.csv         # ML Test Score (0-100)
mltk drift --reference train.csv --current serve.csv # point-in-time drift check
mltk scan-model --model model.pkl --data data.csv    # model-only scan
mltk compliance --framework eu_ai_act --output out.html  # compliance report
```

---

## Report Generation

Frameworks: `fda`, `nist`, `iso_42001`, `eu_ai_act`, `owasp_llm`.
Source: `src/mltk/report/`, `src/mltk/compliance/`.

```bash
# CLI
mltk compliance --framework eu_ai_act --output compliance.html
mltk compliance --framework nist
```

```python
from __future__ import annotations

# MCP
await mltk_report(format="html", frameworks=["nist", "iso_42001"])
await mltk_report(format="json", output="findings.json")

# Programmatic
report.to_html("qa_report.html")          # HTML with severity heatmap
report.to_junit("results.xml")            # JUnit XML for CI
report.to_test_file("tests/test_scan_regression.py")  # runnable pytest
```
