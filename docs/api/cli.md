# CLI Reference

mltk provides a command-line interface for quick data quality checks without writing Python code.

```bash
pip install mltk[cli]
```

---

## Commands

### mltk version
```bash
mltk version
# mltk v0.1.0
```

### mltk init
Scaffold a starter `mltk.yaml` config and example test file.
```bash
mltk init
# Created mltk.yaml
# Created tests/test_mltk_example.py
```

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
