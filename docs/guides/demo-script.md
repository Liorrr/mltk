# Presentation Demo Script

**Event:** April 19 presentation
**Duration:** ~14 minutes (5 beats)
**Setup:** Python 3.10+, `pip install mltk[cli,report]`,
terminal + editor visible to audience.

---

## Beat 1: "The Problem" (30 sec)

### Talking points

- ML fails silently -- a model scores 92% overall
  but collapses to 55% on a subgroup
- Traditional pytest catches code bugs. Nothing
  catches ML bugs.
- "pytest catches code bugs. mltk catches ML bugs."

### Live code

```python
# A model that looks great... until you slice it.
import numpy as np

overall_accuracy = 0.92
print(f"Overall accuracy: {overall_accuracy:.0%}")
# Looks great, right?

# But break it down by age group:
age_groups = {
    "18-35": 0.95,
    "36-55": 0.93,
    "56+":   0.55,  # <-- silent failure
}
for group, acc in age_groups.items():
    flag = " ** FAILING **" if acc < 0.70 else ""
    print(f"  {group}: {acc:.0%}{flag}")
```

### Expected output

```
Overall accuracy: 92%
  18-35: 95%
  36-55: 93%
  56+: 55% ** FAILING **
```

### Transition

> "Nobody caught that 55%. There was no test for it.
> That is the problem mltk solves. Let me show you
> how."

---

## Beat 2: "Method Dispatch" (3 min)

### Talking points

- mltk has a `method` parameter on LLM assertions
- Same data, different methods, dramatically
  different detection
- `lexical` is fast but blind to contradictions
- `nli` catches factual errors that token overlap
  misses
- One parameter change upgrades your detection

### Live code

```python
from mltk.domains.llm.safety import (
    assert_no_hallucination,
)

claims = [
    "Python was created by Guido van Rossum.",
    "Python was first released in 1991.",
    "Python is a compiled language.",  # wrong!
]
sources = [
    "Python is an interpreted programming language "
    "created by Guido van Rossum, first released "
    "in 1991.",
]

# --- Lexical: token overlap ---
r1 = assert_no_hallucination(
    claims, sources, method="lexical",
)
print(f"lexical  -> passed={r1.passed}")
print(f"  coverage: {r1.details['avg_coverage']:.2f}")
# PASSES -- most tokens match!

# --- NLI: catches the contradiction ---
r2 = assert_no_hallucination(
    claims, sources, method="nli",
)
print(f"nli      -> passed={r2.passed}")
print(f"  coverage: {r2.details['avg_coverage']:.2f}")
# FAILS -- "compiled" contradicts "interpreted"
```

### Expected output

```
lexical  -> passed=True
  coverage: 0.72
nli      -> passed=False
  coverage: 0.58
```

### Decision flowchart (show on screen)

```
What matters most?
|
+-- Speed (CI/CD, large batch)
|   +-- Need synonym handling?
|       +-- No  --> lexical
|       +-- Yes --> embedding
|
+-- Accuracy (correctness matters)
|   +-- Contradictions are dangerous?
|       +-- Yes --> nli
|       +-- No  --> embedding
|
+-- Subjective quality (creative, open-ended)
|   --> llm (bring your own judge)
|
+-- Not sure?
    --> embedding (best default)
```

### Transition

> "Method dispatch lets you trade speed for depth.
> But what about testing model *behavior* -- not
> just individual outputs? That is where we are the
> only tool on the market."

---

## Beat 3: "Behavioral Consistency" (5 min)

### Talking points

- Research shows 10% accuracy swings from
  paraphrasing alone (NAACL 2025)
- No other testing tool ships behavioral
  consistency as pytest assertions
- 7 assertions covering invariance, stability,
  equivalence, directionality, retrieval
- Show per-input breakdown -- aggregate scores
  hide the fragile inputs

### Live code -- paraphrase invariance

```python
from mltk.domains.llm.behavioral import (
    assert_paraphrase_invariance,
)

def my_model(prompt: str) -> str:
    """Simulated model with a fragile spot."""
    p = prompt.lower()
    if "cause" in p and "ww2" in p:
        return "Treaty of Versailles, economic crisis"
    if "world war" in p:
        return "Treaty of Versailles, economic crisis"
    if "second" in p and "war" in p:
        return "It started in 1939"  # inconsistent!
    return "Treaty of Versailles, economic crisis"

result = assert_paraphrase_invariance(
    model_fn=my_model,
    paraphrases=[
        "What caused WW2?",
        "Summarize the causes of World War 2",
        "Why did the second world war happen?",
        "Explain the origins of WW2",
    ],
    equivalence_method="token_f1",
    min_invariance=0.8,
)

print(f"passed: {result.passed}")
rate = result.details["invariance_rate"]
print(f"invariance_rate: {rate:.2f}")

# Show per-input outputs
for item in result.details["per_input_outputs"]:
    inp = item["input"][:45]
    out = item["output"][:50]
    print(f"  {inp:<45} -> {out}")

# Show worst pair
wp = result.details["worst_pair"]
ws = result.details["worst_score"]
print(f"\nworst pair: inputs {wp}, score={ws:.2f}")
```

### Expected output

```
passed: False
invariance_rate: 0.67

  What caused WW2?                              -> Treaty of Versailles, economic crisis
  Summarize the causes of World War 2           -> Treaty of Versailles, economic crisis
  Why did the second world war happen?           -> It started in 1939
  Explain the origins of WW2                     -> Treaty of Versailles, economic crisis

worst pair: inputs [1, 2], score=0.12
```

### Live code -- format invariance

```python
from mltk.domains.llm.behavioral import (
    assert_format_invariance,
)

def case_sensitive_model(prompt: str) -> str:
    """Model that breaks on uppercase."""
    if prompt.isupper():
        return "I don't understand the question."
    return "Photosynthesis converts light to energy."

result = assert_format_invariance(
    model_fn=case_sensitive_model,
    input_text="What is photosynthesis?",
    equivalence_method="token_f1",
    min_invariance=0.8,
)

print(f"passed: {result.passed}")
rate = result.details["invariance_rate"]
print(f"invariance_rate: {rate:.2f}")

for t in result.details["transform_results"]:
    name = t["transform"]
    score = t["score"]
    eq = "eq" if t["equivalent"] else "DIFF"
    print(f"  {name:<20} score={score:.2f} [{eq}]")
```

### Expected output

```
passed: False
invariance_rate: 0.80

  lowercase            score=1.00 [eq]
  uppercase            score=0.05 [DIFF]
  strip_whitespace     score=1.00 [eq]
  no_punctuation       score=0.95 [eq]
  double_space         score=0.95 [eq]
```

### Transition

> "No competitor has this as a pytest assertion.
> You write `assert_paraphrase_invariance`, push to
> CI, and every commit is tested for behavioral
> fragility. Now -- what if you do not use pytest?"

---

## Beat 4: "MltkSuite -- Works Everywhere" (2 min)

### Talking points

- Not everyone uses pytest -- notebooks, scripts,
  pipelines
- `MltkSuite` is a composable runner: add
  assertions, call `.run()`, export results
- Same assertions, same `TestResult` objects
- Export to HTML report with one line

### Live code

```python
from mltk.core import MltkSuite
from mltk.data import (
    assert_no_nulls,
    assert_range,
    assert_schema,
)
import pandas as pd
import numpy as np

# Create sample data
df = pd.DataFrame({
    "user_id": [1, 2, 3, 4, 5],
    "score": [0.9, 0.85, 0.7, None, 0.95],
    "label": [1, 0, 1, 1, 0],
})

# Build a suite
suite = MltkSuite("Data Quality Check")

suite.add(
    assert_schema,
    df,
    {
        "user_id": "int64",
        "score": "float64",
        "label": "int64",
    },
)
suite.add(assert_no_nulls, df)
suite.add(assert_range, df["score"], 0.0, 1.0)

# Run everything
suite.run()
print(suite.summary())

# Export to HTML
suite.to_html("demo-report.html")
print("\nReport saved to demo-report.html")
```

### Expected output

```
MltkSuite: Data Quality Check
  [PASS] assert_schema (2ms)
  [FAIL] assert_no_nulls: 1 null in column 'score'
  [PASS] assert_range (1ms)

2/3 passed, 1 failed
```

### Transition

> "Same assertions everywhere. Now let me show you
> the fastest path from zero to full model coverage."

---

## Beat 5: "mltk scan -- Zero to Coverage" (3 min)

### Talking points

- One command finds every blind spot in your model
- 8 built-in scanners: Slice, Bias, Calibration,
  Robustness, Leakage, Data, Drift, Overfit
- Each finding includes severity + reproduction
  recipe
- Generates a pytest file you can commit
- "From blind spot to test coverage in 30 seconds"

### Live code

```python
from mltk.scan import scan
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    RandomForestClassifier,
)
from sklearn.model_selection import train_test_split

# Build a deliberately biased model
np.random.seed(42)
n = 2000
df = pd.DataFrame({
    "age": np.random.randint(18, 80, n),
    "income": np.random.normal(50000, 15000, n),
    "gender": np.random.choice(
        ["M", "F"], n
    ),
    "score": np.random.normal(0.5, 0.2, n),
})

# Label correlates with age (biased)
df["label"] = (
    (df["age"] < 40).astype(int)
    & (df["score"] > 0.4).astype(int)
)

X = df[["age", "income", "score"]]
y = df["label"].values

X_train, X_test, y_train, y_test = (
    train_test_split(X, y, test_size=0.3, random_state=42)
)

model = RandomForestClassifier(
    n_estimators=50, random_state=42
)
model.fit(X_train, y_train)

# Scan the model
report = scan(
    model.predict,
    X_test,
    y_test,
    sensitive_columns=["age"],
)

# See findings
print(report.summary())

# Generate pytest file
report.to_test_file("tests/test_scan_findings.py")
print("\nGenerated: tests/test_scan_findings.py")
print("Run: pytest tests/test_scan_findings.py -v")
```

### Expected output

```
+-- mltk scan ------------------------------------+
| Model: classifier | 600 samples                 |
| Features: 3 numeric                             |
| Scanners: 7/8 run (OverfitScanner skipped)      |
| Duration: 4.2s                                   |
+--------------------------------------------------+

  X CRITICAL  Accuracy drops to 0.58 for age > 55
              (overall: 0.91)       [SliceScanner]
  X CRITICAL  Demographic parity violation on age
              (ratio: 0.62)          [BiasScanner]
  ! WARNING   Model uncalibrated (ECE: 0.14)
                              [CalibrationScanner]
  ! WARNING   Predictions unstable under 1% noise
              (5% flip rate)   [RobustnessScanner]
  i INFO      2 features with > 0.6 correlation
                                 [LeakageScanner]

Summary: 2 critical, 2 warnings, 1 info
-> Run: pytest tests/test_scan_findings.py
```

### Transition

> "Five findings. Severity levels. A pytest file
> you can commit right now. From blind spot to test
> coverage in 30 seconds."

---

## Closing Slide

**mltk -- pytest for ML**

- :test_tube: 207+ assertions across the full ML lifecycle
- :brain: 7 behavioral consistency assertions (first-mover)
- :mag: Multi-method evaluation
  (lexical :arrow_right: NLI :arrow_right: LLM-as-Judge)
- :shield: 8 compliance frameworks
  (EU AI Act, FDA, SR 11-7, HIPAA, OWASP LLM,
  NIST AI RMF, ISO 42001, SOC 2)
- :zap: Rust-accelerated performance
- :package: One `pip install`, replaces 5 tools

```bash
pip install mltk[cli,report]
```

---

## Presenter Notes

### Before the demo

1. `pip install mltk[cli,report,embedding]` in a
   clean venv
2. Pre-download NLI model:
   ```bash
   python -c "from sentence_transformers import \
     CrossEncoder; \
     CrossEncoder('cross-encoder/nli-deberta-v3-base')"
   ```
3. Test every code block runs without errors
4. Have `mltk doctor` output ready as backup
5. Set terminal font size to 18pt+

### If something breaks

- Skip to the next beat -- each is self-contained
- `MltkSuite` demo (Beat 4) has zero external deps
  and always works
- Beat 1 is pure Python, no imports needed

### Timing guide

| Beat | Duration | Cumulative |
|------|----------|------------|
| 1. The Problem | 0:30 | 0:30 |
| 2. Method Dispatch | 3:00 | 3:30 |
| 3. Behavioral Consistency | 5:00 | 8:30 |
| 4. MltkSuite | 2:00 | 10:30 |
| 5. mltk scan | 3:00 | 13:30 |
| Closing | 0:30 | 14:00 |
