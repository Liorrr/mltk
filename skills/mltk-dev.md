---
description: >
  mltk developer persona — TDD workflows, fixing test failures,
  generating test suites from scan findings, and extending mltk.
---

# mltk Developer Skill

## Role Summary

A developer using mltk writes ML tests (assertions) that validate models, data, and pipelines.
The workflow mirrors pytest TDD but operates on ML artifacts: datasets, model weights, training runs, and LLM outputs.
Developers also extend mltk with new assertions and scanners when standard coverage falls short.

---

## TDD Workflow for ML

Red-green-refactor adapted for ML:

1. **Write** — pick an assertion that encodes expected behavior
2. **Run (red)** — run it against the current artifact; expect it to fail
3. **Fix** — fix the model, data, or pipeline
4. **Run (green)** — confirm the assertion now passes
5. **Refactor** — tighten thresholds, add columns arg, improve message

Example cycle:
```python
from __future__ import annotations

from mltk.data.quality import assert_no_nulls
from mltk.scan.drift import assert_no_drift

# Step 1: write
assert_no_nulls(train_df, columns=["age", "income"])  # red — nulls exist
assert_no_drift(train_df, serve_df, threshold=0.05)   # red — p-shift in serve set

# Step 3: fix data pipeline upstream, regenerate serve_df

# Step 4: re-run — both pass
# Step 5: refactor — add severity=Severity.WARNING for drift, tighten nulls columns
```

---

## Test Suite Patterns

### Standalone (no pytest)
```python
from __future__ import annotations

from mltk.core.suite import MltkSuite
from mltk.data.quality import assert_no_nulls, assert_schema
from mltk.scan.drift import assert_no_drift

suite = MltkSuite("my_model_tests")
suite.add("schema",       lambda: assert_schema(df, schema))
suite.add("data_quality", lambda: assert_no_nulls(df))
suite.add("drift",        lambda: assert_no_drift(train_df, serve_df))
result = suite.run()
print(result.summary())
```

### pytest plugin
```python
from __future__ import annotations

import pytest
from mltk.data.quality import assert_no_nulls
from mltk.scan.drift import assert_no_drift

@pytest.mark.ml_data
def test_no_nulls(training_data):
    assert_no_nulls(training_data)

@pytest.mark.ml_drift
def test_no_drift(train_df, serve_df):
    assert_no_drift(train_df, serve_df, threshold=0.05)
```

Run: `python -m pytest tests/ -x -q -m ml_data`

---

## Fixing Common Failures

| Assertion | Failure means | Fix |
|-----------|--------------|-----|
| `assert_no_nulls` | Missing values in columns | Impute with median/mode or filter rows |
| `assert_schema` | Column type mismatch | Cast columns upstream in pipeline |
| `assert_no_drift` | Feature distribution shift | Audit data pipeline; retrain with fresh data |
| `assert_no_concept_drift` | Label relationship changed | Trigger full retraining cycle |
| `assert_no_overfitting` | Train/val gap too large | Add regularization, dropout, or more data |
| `assert_no_bias` | Disparity across groups | Rebalance training data; add fairness constraints |
| `assert_calibration` | Probabilities are miscalibrated | Apply Platt scaling or isotonic regression |
| `assert_no_hallucination` | LLM fabricates facts | Improve retrieval; add grounding guardrails |
| `assert_latency` | p99 exceeds budget | Quantize model; enable batch inference; use ONNX |
| `assert_throughput` | RPS below requirement | Scale horizontally; reduce model size |
| `assert_red_team_resilient` | Prompt injection succeeds | Harden system prompt; add input sanitization |

---

## Generating Tests from Scan Findings

```bash
# Step 1: run scan
mltk scan --path ./data --model model.pkl
```

```python
from __future__ import annotations

from mltk.scan import ScanConfig, ScanEngine
from mltk.scan.finding import ScanFinding

config = ScanConfig(data_path="./data", model_path="model.pkl")
engine = ScanEngine(config)
report = engine.run()

# Step 2: convert findings to assertion calls
for finding in report.findings:
    fn_name = finding.assertion_fn      # e.g. "assert_no_nulls"
    args    = finding.assertion_args    # e.g. {"columns": ["age"]}
    # Step 3: write as a test
    print(f"{fn_name}(df, **{args})")
```

Each `ScanFinding` (defined in `src/mltk/scan/finding.py`) carries `assertion_fn`, `assertion_args`,
and `fix_suggestion` — use all three to write a test and its fix comment.

---

## Extending mltk

### New assertion
Decorator + `assert_true` helper do the heavy lifting:

```python
from __future__ import annotations

from mltk.core.assertion import assert_true, timed_assertion
from mltk.core.result import Severity, TestResult

@timed_assertion
def assert_no_label_noise(labels, threshold: float = 0.02) -> TestResult:
    """Ensure label noise is below threshold."""
    noise_rate = _compute_noise_rate(labels)
    return assert_true(
        condition=noise_rate < threshold,
        name="data.no_label_noise",
        message=f"Label noise {noise_rate:.3f} vs threshold {threshold}",
        severity=Severity.CRITICAL,
        value=noise_rate,
        threshold=threshold,
    )
```

Wire it: export from `src/mltk/{domain}/__init__.py` and add to `src/mltk/__init__.py`.

### New scanner
Subclass `Scanner` in `src/mltk/scan/`:

```python
from __future__ import annotations

from mltk.scan.base import Scanner
from mltk.scan.finding import ScanFinding

class LabelNoiseScanner(Scanner):
    name = "label_noise"

    def scan(self, config) -> list[ScanFinding]:
        # ... compute findings ...
        return findings
```

Full patterns: `skills/mltk-templates.md`

---

## Key Assertions by Use Case

| Use Case | Assertions |
|----------|-----------|
| Data quality | `assert_no_nulls`, `assert_schema`, `assert_data_quality`, `assert_range` |
| Model performance | `assert_metric`, `assert_no_regression`, `assert_slice_performance` |
| Drift | `assert_no_drift`, `assert_no_concept_drift`, `assert_no_degradation` |
| LLM eval | `assert_faithfulness`, `assert_no_toxicity`, `assert_llm_judge_score` |
| Safety / red team | `assert_no_system_prompt_leakage`, `assert_red_team_resilient` |
| Inference | `assert_latency`, `assert_throughput`, `assert_api_contract` |
| Fairness | `assert_no_bias`, `assert_equal_opportunity` |
| Calibration | `assert_calibration`, `assert_ece` |

Full signatures: `docs/reference/full-api-index.md`

---

## Data Contracts

YAML-driven contracts validate data shape and statistics before tests run.

```bash
mltk contract init                   # generate contract template for a dataset
mltk contract validate --path ./data # validate data against the contract
mltk contract generate-tests         # emit assertion calls from contract rules
```

Contracts live in `src/mltk/contracts/`. The generated tests can be dropped directly
into a pytest file or an `MltkSuite`.

---

## Repo Conventions

- `from __future__ import annotations` must be the first code line (after module docstring)
- Tests mirror src layout: `tests/test_data/`, `tests/test_scan/`, etc.
- Lint: `ruff check src/ tests/` — auto-fix with `ruff check --fix`
- Run tests: `python -m pytest tests/ -x -q`
- Never commit directly — provide a commit message for the user
