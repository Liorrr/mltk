# ML Test Score

Implements Google's ML Test Score rubric — a standardized way to measure ML production readiness across 4 categories.

---

## mltk score

```bash
mltk score
# ML Test Score: 18/28 (64%)
#   Data:           5/7
#   Model:          6/7
#   Infrastructure: 4/7
#   Monitoring:     3/7
```

## compute_ml_test_score

```python
from mltk.report.score import compute_ml_test_score

score = compute_ml_test_score(test_results)
# {"total": 18, "max": 28, "percentage": 64.3, "categories": {...}}
```

### Categories (Google's 28-test rubric)

| Category | Tests | What it covers |
|----------|-------|----------------|
| Data (7) | Schema, distribution, drift, freshness, PII, labels, row count |
| Model (7) | Metrics, regression, slicing, calibration, bias, adversarial, robustness |
| Infrastructure (7) | Reproducibility, pipeline, contract, latency, throughput, cold start, versioning |
| Monitoring (7) | Drift monitoring, degradation, SLA, alerts, freshness monitoring, bias drift, cost |

Score = minimum across all 4 categories (weakest link determines readiness).

---
