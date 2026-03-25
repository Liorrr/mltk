# Model Card Generator

Auto-generate model documentation from mltk test results. Follows the Google Model Cards format.

**Module:** `mltk.report.model_card`

**CLI:** `mltk model-card results.json --model-name "FaceID v2"`

---

## Quick Start

```bash
# 1. Run tests with JSON export
pytest --mltk-report --mltk-export-json results.json

# 2. Generate model card
mltk model-card results.json --model-name "Face Recognition v2.1" --output model-card.md
```

## Generated Sections

| Section | Auto-filled from |
|---------|-----------------|
| Model Details | `--model-name`, `--model-version` args |
| Intended Use | Risk level from compliance report |
| Metrics | `assert_metric` results (accuracy, F1, AUC, etc.) |
| Fairness | `assert_no_bias` results (demographic parity, equalized odds) |
| Subgroup Performance | `assert_slice_performance` results |
| Calibration | `assert_calibration` results |
| Robustness | `assert_robust` results |
| Data Quality | Data assertion summary |
| Limitations | Failed tests and gaps |

## Python API

```python
from mltk.report.model_card import generate_model_card

generate_model_card(
    results_path="results.json",
    model_name="Face Recognition v2.1",
    model_version="2.1.0",
    output_path="model-card.md",
)
```

---
