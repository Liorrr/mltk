# report

## Overview
Report generation from test results: self-contained HTML (Plotly + Jinja2), JUnit XML, model cards, bias reports, scoring and summaries. Templates live in `report/templates` (Jinja2 `.j2` files, no code).

## Interface
- `generator.py`: `generate_report`
- `bias_report.py`: `generate_bias_report`
- `junit.py`: `format_result_to_junit`, `export_junit_xml`
- `model_card.py`: `generate_model_card`
- `score.py`: `compute_ml_test_score` (28-test production-readiness rubric)
- `summarizer.py`: `summarize_test_history`
- `visual_diff.py`: `generate_diff_report`
- `_helpers.py`: shared private helpers (also used by compliance generators)

## Invariants
- HTML reports are self-contained.
- Generators are driven only by supplied results.

## Side Effects
- Generators write output files when given an output path.

## Verification
- Run: `python -m pytest tests/test_report -q`
