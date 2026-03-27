# ISO 42001 Compliance

Map mltk test results to ISO/IEC 42001:2023 AI Management System Annex A controls.

**Module:** `mltk.compliance.iso_42001`

---

## Overview

[ISO/IEC 42001:2023](https://www.iso.org/standard/81230.html) specifies requirements for establishing, implementing, and improving an AI Management System (AIMS). Annex A defines 8 control areas:

| Control | Title | mltk Assertions |
|---------|-------|-----------------|
| **A.2** | AI Policies | `data.pii`, `data.schema` |
| **A.4** | AI Risk Assessment | `model.bias`, `model.adversarial`, `model.calibration` |
| **A.5** | Data Quality | `data.schema`, `data.no_nulls`, `data.dtypes`, `data.drift`, `data.freshness`, `data.no_pii` |
| **A.6** | System Performance | `model.metric`, `model.regression`, `model.slice`, `inference.*`, `monitor.*` |
| **A.7** | Third Party | `pipeline.checksum`, `pipeline.reproducible` |
| **A.8** | Documentation | *(gap — no direct assertion mapping)* |
| **A.9** | Incident Response | `monitor.degradation`, `monitor.sla` |
| **A.10** | Bias and Fairness | `model.bias`, `model.slice` |

---

## Quick Start

```python
from mltk.compliance.iso_42001 import (
    assert_iso_42001_coverage,
    map_results_to_clauses,
    find_gaps,
)

results = [
    {"name": "data.schema.check", "passed": True},
    {"name": "model.bias.dp", "passed": True},
    {"name": "model.metric.acc", "passed": True},
    {"name": "monitor.sla.p99", "passed": True},
]

# Check coverage
assert_iso_42001_coverage(results, min_coverage=0.6)

# See which clauses are covered
grouped = map_results_to_clauses(results)

# Find gaps
gaps = find_gaps(results)
# ["A.7", "A.8"] -- Third Party and Documentation not covered
```

---

## API Reference

### assert_iso_42001_coverage

```python
assert_iso_42001_coverage(results, min_coverage=0.8)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `results` | `list[dict]` | *(required)* | Test results with `name` and `passed` keys |
| `min_coverage` | `float` | `0.8` | Minimum fraction of Annex A clauses covered (0-1) |

Returns `TestResult` with details: `coverage`, `covered_count`, `total_clauses`, `gaps`.

### map_results_to_clauses

Groups results by Annex A clause ID (A.2-A.10).

### find_gaps

Returns sorted list of uncovered clause IDs. Note: A.8 (Documentation) always appears as a gap since no assertion directly maps to it.

---

## CLI

```bash
mltk compliance-gap results.json --framework iso-42001
```
