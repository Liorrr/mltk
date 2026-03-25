# Chat Interface

Interactive Q&A about your test results. Rule-based analysis — no LLM or external API needed.

**Module:** `mltk.chat`

**CLI:** `mltk chat --results-json results.json`

---

## Quick Start

```bash
# 1. Export test results
pytest --mltk-export-json results.json

# 2. Start chat
mltk chat --results-json results.json
```

## Example Session

```
mltk chat — ask questions about your test results
Type 'help' for available commands, 'quit' to exit

mltk> what failed?
3 tests failed:
  - data.drift: PSI 0.35 > 0.1 threshold
  - model.bias: Demographic parity gap 0.15 > 0.10
  - inference.latency: P95 120ms > 50ms threshold

mltk> why did bias fail?
model.bias: Demographic parity gap 0.15 > 0.10
  Method: demographic_parity
  Threshold: 0.10

mltk> recommend
Missing test categories:
  - No calibration tests -> Add assert_calibration()
  - No adversarial tests -> Add assert_robust()

mltk> summary
Total: 50 | Passed: 47 | Failed: 3 | Score: 94.0%

mltk> slowest
5 slowest tests:
  1. inference.throughput — 2340.5ms
  2. model.metric — 156.2ms
  3. data.drift — 89.1ms

mltk> quit
Bye!
```

## Supported Questions

| Question | What it does |
|----------|-------------|
| `what failed?` / `failures` | List all failed tests |
| `why did X fail?` / `why X` | Show failure details for test X |
| `summary` / `status` | Pass/fail counts + score |
| `recommend` / `what should I test?` | Suggest missing test categories |
| `slowest` / `performance` | Show slowest tests by duration |
| `drift` | Show drift-related results |
| `bias` / `fairness` | Show bias-related results |
| `help` | List available commands |

## Python API

```python
from mltk.chat import ChatEngine

engine = ChatEngine("results.json")
print(engine.ask("what failed?"))
print(engine.ask("recommend"))
```

---
