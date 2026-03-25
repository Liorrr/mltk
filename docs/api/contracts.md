# Data Contracts

Define expected data quality in YAML, auto-generate tests. mltk's killer feature — no competitor has this.

**Module:** `mltk.contracts`

---

## Contract YAML Spec

```yaml
# contract.yaml
name: training_data
version: "1.0"

columns:
  id:
    type: int64
    nullable: false
    unique: true
  age:
    type: float64
    nullable: false
    range: [0, 150]
  label:
    type: int64
    nullable: false

quality:
  min_rows: 1000
  max_nulls_pct: 0.01
  freshness_days: 7
  freshness_column: created_at
```

## CLI

```bash
mltk contract init                                    # Scaffold example contract
mltk contract validate data.csv --contract contract.yaml  # Validate data
```

## Python API

```python
from mltk.contracts import validate_data, generate_tests_from_contract

# Validate directly
suite = validate_data(df, "contract.yaml")

# Generate pytest file
generate_tests_from_contract("contract.yaml", "tests/test_contract.py")
```

---
