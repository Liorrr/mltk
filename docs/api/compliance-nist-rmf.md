# NIST AI RMF Compliance

Map mltk test results to the NIST AI Risk Management Framework (AI RMF 1.0) functions and subcategories.

**Module:** `mltk.compliance.nist_ai_rmf`

---

## Overview

The [NIST AI RMF](https://www.nist.gov/artificial-intelligence/risk-management-framework) provides a voluntary framework for managing AI risks. It defines 4 core functions:

| Function | Code | Description | mltk Assertions |
|----------|------|-------------|-----------------|
| **Govern** | GV | Policies, accountability, risk tolerance | `data.pii`, `data.schema`, `model.bias` |
| **Map** | MP | System context, stakeholders, risk identification | `model.metric`, `model.slice`, `data.drift` |
| **Measure** | MS | Quantitative assessment and metrics | `model.metric`, `model.regression`, `model.calibration`, `model.adversarial`, `inference.*` |
| **Manage** | MN | Risk treatment and monitoring | `monitor.degradation`, `monitor.sla`, `data.drift` |

---

## Quick Start

```python
from mltk.compliance.nist_ai_rmf import (
    assert_nist_rmf_coverage,
    map_results_to_measures,
    find_gaps,
)

# Load results from JSON or collect from pytest
results = [
    {"name": "data.pii.scan", "passed": True},
    {"name": "model.metric.accuracy", "passed": True},
    {"name": "monitor.sla.latency", "passed": True},
]

# Check coverage
assert_nist_rmf_coverage(results, min_coverage=0.75)

# See which functions are covered
grouped = map_results_to_measures(results)
# {"GV": [...], "MP": [], "MS": [...], "MN": [...]}

# Find gaps
gaps = find_gaps(results)
# ["MP"] -- no MAP function tests
```

---

## API Reference

### assert_nist_rmf_coverage

```python
assert_nist_rmf_coverage(results, min_coverage=0.8)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `results` | `list[dict]` | *(required)* | Test results with `name` and `passed` keys |
| `min_coverage` | `float` | `0.8` | Minimum fraction of RMF functions covered (0-1) |

Returns `TestResult` with details: `coverage`, `covered_count`, `total_functions`, `tier`, `gaps`.

### map_results_to_measures

Groups results by RMF function code (GV, MP, MS, MN). Results mapping to multiple functions appear in each.

### find_gaps

Returns sorted list of uncovered function codes.

---

## CLI

```bash
mltk compliance-gap results.json --framework nist-rmf
```

---

## Maturity Tiers

| Tier | Coverage | Description |
|------|----------|-------------|
| Partial | < 25% | Initial awareness, ad-hoc practices |
| Risk-Informed | 25-50% | Risk awareness integrated |
| Repeatable | 50-75% | Consistent, documented practices |
| Adaptive | > 75% | Continuous improvement, full lifecycle |
