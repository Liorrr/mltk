# Presentation Demo Script

**Event:** April 19 presentation
**Duration:** ~12 minutes (6 beats)
**Setup:** Python 3.10+, `pip install mltk[cli]`,
terminal visible to audience.

---

## Quick Start

```bash
pip install mltk[cli]
python demo/run_demo.py
```

The demo is **fully automated**. No typing during
the presentation. No external APIs, no model
downloads. Just run the script and talk over the
output.

---

## Beat 1: "The Problem" (~30 sec)

### What it does

Creates a sample DataFrame with hidden issues
(PII, nulls, subgroup bias). Prints the data and
shows Group B has a lower average score -- but
nobody tested for it.

### Talking points

- ML fails silently -- a model scores 92% overall
  but collapses on a subgroup
- Traditional pytest catches code bugs. Nothing
  catches ML bugs.
- "pytest catches code bugs. mltk catches ML bugs."

### Expected output

```
Sample ML dataset loaded.
Looks fine, right?

      name             email  age  score group
John Smith  john@example.com 25.0   0.92     A
  Jane Doe  jane@example.com  NaN   0.55     B
...

  Group A: avg score = 0.85
  Group B: avg score = 0.73
```

### Transition

> "Nobody tested for this. mltk catches it.
> Let me show you how."

---

## Beat 2: "mltk Data Scan" (~2 min)

### What it does

Runs three data quality assertions on the same
DataFrame:

1. `assert_schema` -- verifies column types PASS
2. `assert_no_nulls` -- catches the null age FAIL
3. `assert_no_pii(method="regex")` -- catches
   5 emails FAIL

Then explains that names require NER/hybrid mode.

### Talking points

- Schema validation is the first line of defense
- One null in production can crash a pipeline
- Regex catches structured PII (emails, SSNs, API
  keys) with zero dependencies
- For person names, upgrade to `method="hybrid"`
  (one parameter change)

### Expected output

```
  PASS  Schema valid
  FAIL  1 null(s) in columns: ['age']
  FAIL  5 PII match(es) in columns: ['email']
```

### Transition

> "Three assertions, three findings. But data
> quality is just the start. What about the model
> itself?"

---

## Beat 3: "Behavioral Consistency" (~2 min)

### What it does

1. Defines a mock LLM with a fragile spot (one
   paraphrase triggers a completely different
   answer)
2. Runs `assert_paraphrase_invariance` -- FAIL
   (invariance 0.50 < 0.80, 3 of 6 pairs fail)
3. Runs `assert_output_stability` with a model
   that has run-to-run jitter -- FAIL (0.60 < 0.90)

### Talking points

- Research shows 10% accuracy swings from
  paraphrasing alone (NAACL 2025)
- No other testing tool ships behavioral
  consistency as pytest assertions
- Per-input breakdown shows exactly which
  paraphrase breaks the model
- Output stability catches non-determinism
  (temperature > 0 variance, race conditions)

### Expected output

```
  FAIL  Paraphrase invariance too low (token_f1):
        0.5000 < 0.8 (3/6 pairs)
    What caused WW2?            -> Treaty of...
    What led to the second...   -> It started in...

    worst pair: inputs [0, 2], score=0.00

  FAIL  Output stability too low (token_f1):
        0.6000 < 0.9
```

### Transition

> "No competitor has this as a pytest assertion.
> You write `assert_paraphrase_invariance`, push
> to CI, and every commit is tested. Now let's
> generate test data automatically."

---

## Beat 4: "Synthetic QA + RAG Testing" (~3 min)

### What it does

1. Uses `SyntheticQAGenerator` in template mode
   (zero dependencies) to generate 5 QA pairs
   from text chunks
2. Prints the generated questions and answers
3. Runs `assert_faithfulness(method="lexical")`
   on each pair -- all PASS

### Talking points

- Template mode is deterministic and CI-safe --
  no API calls, no costs
- For production, swap in `llm_fn=my_model` for
  higher quality generation
- Faithfulness checks that answers are grounded
  in the source context
- Lexical method is zero-dep; upgrade to
  `method="nli"` for semantic checking

### Expected output

```
Generated 5 QA pairs:

  Q1: What does Unsupervised learning?
  A1: Unsupervised learning discovers hidden...

  Q2: What does Transfer learning?
  A2: Transfer learning allows pre-trained...
  ...

  PASS  Q1 faithfulness: 1.00
  PASS  Q2 faithfulness: 1.00
  ...
```

### Transition

> "Automated test data generation plus RAG
> testing in one pipeline. Now let's attack the
> model."

---

## Beat 5: "Red Team Security Scan" (~3 min)

### What it does

1. Defines a mock chatbot that refuses attack
   patterns (keyword-based safety)
2. Runs `assert_red_team_resilient` across 3
   attack categories (26 payloads) -- PASS
3. Shows per-category resilience breakdown
4. Runs `assert_encoding_mutation_resilience`
   with 3 techniques (30 mutations) -- PASS

### Talking points

- 7 attack categories mapped to OWASP LLM Top 10
- 50+ built-in educational payloads, no external
  attack tools needed
- Encoding mutations test a critical blind spot:
  models trained to refuse plaintext may comply
  with Base64/ROT13/leetspeak
- The mock chatbot resists everything -- a real
  model probably won't

### Expected output

```
  PASS  Red team resilience: 1.0000 >= 0.8
        (26/26 attacks resisted)

  Per-category breakdown:
    [OK] prompt_injection    resilience: 100%
    [OK] jailbreak           resilience: 100%
    [OK] data_extraction     resilience: 100%

  PASS  Encoding mutation resilience: 1.0000
        (30/30 mutations resisted)
```

### Transition

> "56 security checks in under a second. All
> built-in, all pytest-native. Let's wrap up."

---

## Beat 6: "The Full Picture" (~1 min)

### What it does

Prints a summary: issues found, tests passed/
failed, security checks, total time.

### Expected output

```
  Summary
  ----------------------------------------
  Issues found:      4
  Tests passed:      8 / 12
  Tests failed:      4 / 12
  Security checks:   56
  ----------------------------------------
  Total demo time:   0.1s

  224 assertions, one pip install, native pytest.

  pip install mltk[cli]
  pytest for ML.
```

### Closing talking points

- 224+ assertions across the full ML lifecycle
- 7 behavioral consistency assertions (first-mover)
- Multi-method evaluation
  (lexical -> NLI -> LLM-as-Judge)
- 8 compliance frameworks
  (EU AI Act, FDA, SR 11-7, HIPAA, OWASP LLM,
  NIST AI RMF, ISO 42001, SOC 2)
- Rust-accelerated performance
- One `pip install`, replaces 5 tools

---

## Presenter Notes

### Before the demo

1. `pip install mltk[cli]` in a clean venv
2. Run `python demo/run_demo.py` once to verify
3. Set terminal font size to 18pt+
4. Resize terminal: ~80 columns, ~40 rows
5. Have `mltk doctor` output ready as backup

### If something breaks

- Each beat is independent -- skip to the next
- The script catches all errors and prints a
  traceback; read the error and explain what
  the assertion would have done
- If Python itself fails: show the source code
  in `demo/run_demo.py` and walk through it

### Fallback: run individual beats

```python
# In a Python REPL:
from demo.run_demo import beat_1
beat_1()  # Each beat is self-contained
```

### Timing guide

| Beat | Duration | Cumulative |
|------|----------|------------|
| 1. The Problem | 0:30 | 0:30 |
| 2. Data Scan | 2:00 | 2:30 |
| 3. Behavioral | 2:00 | 4:30 |
| 4. Synthetic QA | 3:00 | 7:30 |
| 5. Red Team | 3:00 | 10:30 |
| 6. Full Picture | 1:00 | 11:30 |
| Buffer | 0:30 | 12:00 |
