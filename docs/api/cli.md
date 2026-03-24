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
Quick data quality scan on a CSV file.
```bash
mltk scan data/training.csv
# Schema: 5 columns, 10000 rows
# Nulls: 0 found
# Range: all features within bounds
```

### mltk drift
Compare two datasets for distribution drift.
```bash
mltk drift data/reference.csv data/current.csv --method psi
# PSI per column:
#   age: 0.03 (stable)
#   income: 0.18 (moderate)
#   score: 0.42 (DRIFT)
```

---
