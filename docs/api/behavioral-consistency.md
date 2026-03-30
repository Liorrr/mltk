# Behavioral Consistency

An LLM that answers "What caused WW2?" correctly but fails
on "Summarize the causes of WW2" is fragile. It memorized
surface patterns, not the underlying knowledge. A 10%
accuracy swing from paraphrasing alone is common -- research
(NAACL 2025) measured this across production models.

Behavioral consistency testing catches these fragile models
before they reach users. Ask the same question multiple
ways. If the answers diverge, the model is unreliable.

**No other testing tool ships these as pytest assertions.**

**Module:** `mltk.domains.llm.behavioral`

**ML Lifecycle Stage:** Post-training evaluation / CI gate

**Bugs caught:**

- Models that memorize phrasing instead of learning concepts
- Outputs that change when you switch case or add punctuation
- Non-deterministic responses across identical runs
- Hidden sensitivity to prompt formatting

---

## Why Behavioral Consistency?

Consider a customer support chatbot. A user asks:

> "How do I reset my password?"

The model answers correctly. But another user asks:

> "I forgot my password, how can I change it?"

Same intent, different words -- and the model gives a
completely different (wrong) answer. This is a **behavioral
inconsistency**. The model learned to pattern-match the
first phrasing but did not generalize.

This is not hypothetical. Research shows:

- **10% accuracy swings** from paraphrasing inputs alone
  (SCORE benchmark, NAACL 2025)
- **CheckList** (ACL 2020) demonstrated that models passing
  standard benchmarks fail systematically on simple
  linguistic perturbations
- **ProSA** showed commercial models vary 5-15% across
  semantically equivalent prompts

Traditional evaluation misses this entirely. You test one
phrasing per question, get 92% accuracy, and ship. Then
users find the 8% of phrasings where the model breaks.

Behavioral consistency testing fixes this by testing
**multiple phrasings per intent** and measuring whether
the model responds consistently.

---

## Three Assertions

### `assert_paraphrase_invariance`

Tests whether a model produces equivalent outputs when
the same question is asked in different words.

```
"What caused WW2?"
"Summarize the causes of World War 2"    --> same answer?
"Why did the second world war happen?"
```

#### Signature

```python
from mltk.domains.llm.behavioral import (
    assert_paraphrase_invariance,
)

result = assert_paraphrase_invariance(
    model_fn=my_llm.generate,
    paraphrases=[
        "What caused WW2?",
        "Summarize the causes of World War 2",
        "Why did the second world war happen?",
    ],
    equivalence_method="embedding",
    min_invariance=0.8,
)
```

#### What it does

1. Calls `model_fn` once per paraphrase
2. Compares every output pair using the chosen
   `equivalence_method`
3. Computes the mean pairwise equivalence score
4. Passes if the score meets `min_invariance`

#### Parameters

| Name | Type | Default | What |
|------|------|---------|------|
| `model_fn` | `Callable[[str], str]` | *(required)* | Function that takes a prompt, returns text |
| `paraphrases` | `list[str]` | *(required)* | 2+ semantically equivalent inputs |
| `equivalence_method` | `str` | `"embedding"` | How to compare outputs (see Methods) |
| `min_invariance` | `float` | `0.8` | Minimum mean pairwise score to pass |
| `judge_fn` | `Callable` | `None` | Required when `equivalence_method="judge"` |
| `embedding_model` | `str` | `None` | Custom sentence-transformer model |
| `nli_model` | `str` | `None` | Custom NLI cross-encoder model |

#### Returns

`TestResult` with:

- `passed`: whether `min_invariance` was met
- `score`: mean pairwise equivalence score
- `details`: individual pairwise scores, all outputs

---

### `assert_format_invariance`

Tests whether a model produces equivalent outputs when
the input formatting changes but the content stays the
same.

```
"what is photosynthesis"
"What Is Photosynthesis"        --> same answer?
"WHAT IS PHOTOSYNTHESIS?"
"  what is photosynthesis  "
```

#### Signature

```python
from mltk.domains.llm.behavioral import (
    assert_format_invariance,
)

result = assert_format_invariance(
    model_fn=my_llm.generate,
    base_input="What is photosynthesis?",
    transforms=["lowercase", "uppercase", "strip"],
    equivalence_method="embedding",
    min_invariance=0.85,
)
```

#### What it does

1. Applies each transform to `base_input` to create
   formatting variants
2. Calls `model_fn` on the original and each variant
3. Compares all output pairs using `equivalence_method`
4. Passes if mean pairwise score meets `min_invariance`

#### Built-in transforms

| Transform | Example |
|-----------|---------|
| `"lowercase"` | `"what is ww2?"` |
| `"uppercase"` | `"WHAT IS WW2?"` |
| `"title_case"` | `"What Is Ww2?"` |
| `"strip"` | removes leading/trailing whitespace |
| `"no_punctuation"` | `"What is WW2"` |
| `"extra_spaces"` | `"What  is  WW2 ?"` |

You can also pass a custom callable as a transform.

#### Parameters

| Name | Type | Default | What |
|------|------|---------|------|
| `model_fn` | `Callable[[str], str]` | *(required)* | Function that takes a prompt, returns text |
| `base_input` | `str` | *(required)* | The original input text |
| `transforms` | `list` | *(required)* | Transform names or callables |
| `equivalence_method` | `str` | `"embedding"` | How to compare outputs (see Methods) |
| `min_invariance` | `float` | `0.85` | Minimum mean pairwise score to pass |
| `judge_fn` | `Callable` | `None` | Required when `equivalence_method="judge"` |
| `embedding_model` | `str` | `None` | Custom sentence-transformer model |
| `nli_model` | `str` | `None` | Custom NLI cross-encoder model |

#### Returns

`TestResult` with:

- `passed`: whether `min_invariance` was met
- `score`: mean pairwise equivalence score
- `details`: per-transform scores, all outputs

---

### `assert_output_stability`

Tests whether a model produces consistent outputs when
given the exact same input multiple times. Catches
non-deterministic behavior from temperature, sampling,
or infrastructure issues.

```
"Explain gravity" x5 --> all 5 answers consistent?
```

#### Signature

```python
from mltk.domains.llm.behavioral import (
    assert_output_stability,
)

result = assert_output_stability(
    model_fn=my_llm.generate,
    input_text="Explain gravity in one sentence.",
    n_runs=5,
    equivalence_method="embedding",
    min_stability=0.9,
)
```

#### What it does

1. Calls `model_fn` with the same `input_text` `n_runs`
   times
2. Compares all output pairs using `equivalence_method`
3. Computes mean pairwise equivalence as the stability
   score
4. Passes if the score meets `min_stability`

#### Parameters

| Name | Type | Default | What |
|------|------|---------|------|
| `model_fn` | `Callable[[str], str]` | *(required)* | Function that takes a prompt, returns text |
| `input_text` | `str` | *(required)* | The input to repeat |
| `n_runs` | `int` | `5` | How many times to call the model |
| `equivalence_method` | `str` | `"embedding"` | How to compare outputs (see Methods) |
| `min_stability` | `float` | `0.9` | Minimum mean pairwise score to pass |
| `judge_fn` | `Callable` | `None` | Required when `equivalence_method="judge"` |
| `embedding_model` | `str` | `None` | Custom sentence-transformer model |
| `nli_model` | `str` | `None` | Custom NLI cross-encoder model |

#### Returns

`TestResult` with:

- `passed`: whether `min_stability` was met
- `score`: mean pairwise equivalence score
- `details`: all `n_runs` outputs, pairwise matrix

---

## Equivalence Methods

All three assertions share the same `equivalence_method`
parameter. This controls how two model outputs are compared
to determine if they are "the same."

For a deep dive on each method, see the
[Method Dispatch guide](../guides/method-dispatch.md).

| Method | What it does | Speed | Catches |
|--------|-------------|-------|---------|
| `"token_f1"` | Token-level F1 overlap | :zap: instant | Keyword matches |
| `"embedding"` | Cosine similarity via sentence-transformers | ~14ms/pair | Synonyms, paraphrases |
| `"entailment"` | Bidirectional NLI entailment | ~500ms/pair | Contradictions |
| `"judge"` | Your custom LLM-as-Judge function | ~2-5s/pair | Nuance, subjective quality |
| `"auto"` | Best available (entailment > embedding > token_f1) | varies | Auto-selects |
| `"label_match"` | Exact string match after normalization | :zap: instant | Classification tasks |

### `token_f1`

Tokenizes both outputs, computes F1 overlap. No
dependencies. Fast. Misses synonyms entirely.

:point_right: Use for classification or short factual
answers where exact wording matters.

### `embedding`

Encodes both outputs with a sentence-transformer and
computes cosine similarity. Default method. Good balance
of speed and semantic understanding.

:point_right: Use as the general-purpose default.

### `entailment`

Runs bidirectional NLI: checks that output A entails
output B AND output B entails output A. If both directions
score high, the outputs are semantically equivalent. This
catches contradictions that embedding similarity misses.

:point_right: Use for factual or safety-critical tasks
where contradictions are dangerous.

### `judge`

Calls your custom function with both outputs. You define
what "equivalent" means. Most flexible but slowest.

```python
def my_judge(output_a: str, output_b: str) -> float:
    """Return 0-1 equivalence score."""
    # Call your LLM, apply your rubric
    ...

result = assert_paraphrase_invariance(
    model_fn=my_llm.generate,
    paraphrases=paraphrases,
    equivalence_method="judge",
    judge_fn=my_judge,
)
```

:point_right: Use for subjective or domain-specific
equivalence (e.g., medical accuracy, legal compliance).

### `auto`

Selects the best available method based on installed
packages. Tries `entailment` first, falls back to
`embedding`, then `token_f1`. No configuration needed.

:point_right: Use when you want the best available
method without thinking about it.

### `label_match`

Normalizes both outputs (lowercase, strip whitespace,
remove punctuation) and checks for exact match. Designed
for classification tasks where the model outputs a label.

:point_right: Use when testing classification models
(e.g., sentiment: "positive" vs "POSITIVE").

---

## Examples

### Paraphrase invariance with embedding

```python
from mltk.domains.llm.behavioral import (
    assert_paraphrase_invariance,
)

result = assert_paraphrase_invariance(
    model_fn=my_llm.generate,
    paraphrases=[
        "What do you know about WW2?",
        "Summarize to me WW2",
        "What happened in world war 2?",
    ],
    equivalence_method="embedding",
    min_invariance=0.8,
)

print(result.score)    # 0.87
print(result.passed)   # True
```

### Format invariance with entailment

```python
from mltk.domains.llm.behavioral import (
    assert_format_invariance,
)

result = assert_format_invariance(
    model_fn=my_llm.generate,
    base_input="What is photosynthesis?",
    transforms=[
        "lowercase",
        "uppercase",
        "no_punctuation",
    ],
    equivalence_method="entailment",
    min_invariance=0.85,
)
```

### Output stability for determinism testing

```python
from mltk.domains.llm.behavioral import (
    assert_output_stability,
)

result = assert_output_stability(
    model_fn=my_llm.generate,
    input_text="What is the capital of France?",
    n_runs=5,
    equivalence_method="embedding",
    min_stability=0.9,
)

if not result.passed:
    print("Model is non-deterministic!")
    print(f"Stability: {result.score:.2f}")
```

### Custom format transform

```python
def add_typo(text: str) -> str:
    """Simulate a user typo."""
    words = text.split()
    if len(words) > 2:
        words[1] = words[1][::-1]  # reverse word
    return " ".join(words)


result = assert_format_invariance(
    model_fn=my_llm.generate,
    base_input="What is machine learning?",
    transforms=["lowercase", "uppercase", add_typo],
    equivalence_method="embedding",
    min_invariance=0.75,
)
```

### In a pytest test suite

```python
import pytest
from mltk.domains.llm.behavioral import (
    assert_format_invariance,
    assert_output_stability,
    assert_paraphrase_invariance,
)


@pytest.fixture
def model():
    """Load your model once per module."""
    import ollama
    def generate(prompt: str) -> str:
        resp = ollama.generate(
            model="llama3.2",
            prompt=prompt,
        )
        return resp["response"]
    return generate


def test_paraphrase_consistency(model):
    assert_paraphrase_invariance(
        model_fn=model,
        paraphrases=[
            "Explain quantum computing",
            "What is quantum computing?",
            "Describe how quantum computers work",
        ],
        min_invariance=0.75,
    )


def test_case_insensitivity(model):
    assert_format_invariance(
        model_fn=model,
        base_input="What is DNA?",
        transforms=["lowercase", "uppercase"],
        min_invariance=0.85,
    )


def test_deterministic_output(model):
    assert_output_stability(
        model_fn=model,
        input_text="What is 2 + 2?",
        n_runs=3,
        min_stability=0.95,
    )
```

---

## When to Use Which Assertion

```
What are you testing?
|
+-- Same question, different wording?
|   --> assert_paraphrase_invariance
|
+-- Same question, different formatting?
|   (case, punctuation, whitespace)
|   --> assert_format_invariance
|
+-- Same question, same input, multiple runs?
|   (testing non-determinism)
|   --> assert_output_stability
|
+-- All three? Run all three in your test suite.
    Different failure modes, complementary coverage.
```

**Practical guidance:**

- **Pre-deployment gate**: run paraphrase invariance with
  your top 20 critical questions. If the model fails on
  paraphrases of "How do I cancel my subscription?", do
  not ship it.

- **After temperature changes**: run output stability.
  Raising temperature from 0 to 0.7 should still produce
  consistent factual answers.

- **After prompt template changes**: run format invariance.
  New templates might introduce sensitivity to casing or
  whitespace.

- **CI/CD**: all three, with `equivalence_method="auto"`.
  Fast enough for every commit.

---

## Threshold Guidance

Different `equivalence_method` values produce scores on
different scales. A "0.8" in embedding is not the same as
a "0.8" in entailment.

### `token_f1` thresholds

| Score | Interpretation |
|-------|----------------|
| >= 0.60 | Strong keyword overlap |
| 0.35 - 0.60 | Moderate overlap |
| < 0.35 | Weak -- outputs likely differ |

:warning: High token overlap does not guarantee semantic
equivalence. "The war started in 1939" and "The war did
not start in 1939" have nearly identical tokens.

### `embedding` thresholds

| Score | Interpretation |
|-------|----------------|
| >= 0.85 | Semantically equivalent (strict) |
| 0.75 - 0.85 | Likely equivalent (recommended range) |
| 0.60 - 0.75 | Related but may differ in substance |
| < 0.60 | Different meanings |

:point_right: For paraphrase invariance, start with 0.80.
For format invariance, use 0.85 (outputs should be nearly
identical since only formatting changed).

### `entailment` thresholds

| Score | Interpretation |
|-------|----------------|
| >= 0.75 | Mutual entailment -- outputs agree |
| 0.50 - 0.75 | Partial agreement |
| < 0.50 | Outputs may contradict |

:point_right: Entailment is stricter than embedding.
A 0.70 entailment score is roughly equivalent to a 0.85
embedding score in practice.

### `label_match` thresholds

Binary: 1.0 if labels match, 0.0 if not. Set
`min_invariance=1.0` to require perfect consistency.

### `judge` thresholds

Depends entirely on your judge function. Define what
0.0 and 1.0 mean in your rubric, then set the threshold
accordingly.

---

## Coming in S70

Three more behavioral assertions are planned:

### Retrieval Consistency

Test whether a RAG system returns the same answer
regardless of how the retrieval query is phrased.
Combines paraphrase invariance with retrieval pipeline
testing.

### Semantic Equivalence (NLI Bidirectional)

A dedicated assertion that checks bidirectional NLI
entailment between two texts. Stricter than embedding
similarity -- catches contradictions and partial truths.

### Directional Expectations (CheckList DIR)

Test that a specific input change causes a predictable
output change. Based on CheckList's DIR (Directional
Expectation) test type. Example: adding "not" to a
sentiment input should flip the predicted sentiment.

### ParaphraseGenerator

Automatic paraphrase generation so you do not need to
write paraphrases by hand. Give it one input, get 5+
semantically equivalent variants.

---

## Research Background

Behavioral consistency testing builds on three research
threads:

### CheckList (ACL 2020)

Introduced the idea of testing NLP models with
**Minimum Functionality Tests (MFTs)**, **Invariance
Tests (INV)**, and **Directional Expectation Tests
(DIR)**. Found that models passing standard benchmarks
fail systematically on simple perturbations. mltk's
`assert_paraphrase_invariance` implements INV testing.
`assert_format_invariance` extends it to formatting.

> Ribeiro et al., "Beyond Accuracy: Behavioral Testing
> of NLP Models with CheckList", ACL 2020.

### ProSA (Prompt Sensitivity Analysis)

Measured how much model outputs change when prompts are
paraphrased. Found 5-15% accuracy variation across
semantically equivalent prompts in commercial models.
Motivated the need for systematic invariance testing
rather than single-phrasing evaluation.

### SCORE (NAACL 2025)

Large-scale benchmark showing ~10% accuracy swings from
paraphrasing across production LLMs. Demonstrated that
behavioral inconsistency is not an edge case but a
systematic problem. Reinforced that single-phrasing
benchmarks overstate model reliability.

---

## FAQ

### Do I need to write my own paraphrases?

For now, yes. S70 will add a `ParaphraseGenerator` that
creates variants automatically. Until then, write 3-5
paraphrases per critical question. Focus on your most
important user queries.

### Which equivalence method should I start with?

Start with `"embedding"`. It is the best general-purpose
default -- catches synonyms and paraphrases without the
overhead of NLI or a judge function. Upgrade to
`"entailment"` for safety-critical applications.

### Do I need a GPU?

No. All methods work on CPU. Embedding comparison takes
~14ms per pair. Entailment takes ~500ms per pair. Both
are fast enough for CI/CD with typical test sizes
(10-50 pairs).

### What if my model is non-deterministic by design?

Use `assert_output_stability` with a lower threshold.
A creative writing model at temperature=0.7 might only
achieve 0.6 stability -- that is expected. The point is
to set a floor and catch regressions. If stability drops
from 0.6 to 0.3 after a change, something broke.

### Can I test classification models?

Yes. Use `equivalence_method="label_match"` with
`min_invariance=1.0`. This checks that the model
predicts the same label regardless of phrasing or
formatting. For example, a sentiment classifier should
return "positive" for both "I love this!" and
"i love this".

### How many paraphrases should I test?

3-5 per intent is a good starting point. More
paraphrases increase coverage but also increase test
runtime (each paraphrase requires one model call).
For CI/CD, 3 paraphrases per critical question gives
good coverage without excessive runtime.
