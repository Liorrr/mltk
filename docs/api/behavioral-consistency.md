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
- RAG pipelines that return different documents for equivalent queries
- Models that fail to shift output when the prompt explicitly demands a change

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
- **MENLI** (TACL 2023) found NLI-based metrics are 15-30%
  more robust than BERTScore against adversarial attacks

Traditional evaluation misses this entirely. You test one
phrasing per question, get 92% accuracy, and ship. Then
users find the 8% of phrasings where the model breaks.

Behavioral consistency testing fixes this by testing
**multiple phrasings per intent** and measuring whether
the model responds consistently.

---

## Design Decisions

### Why `token_f1` as the default equivalence method?

Research (MENLI, TACL 2023) showed NLI bidirectional
entailment is the most robust metric. But it requires
`sentence-transformers` (~400 MB). We chose `token_f1`
as the default because:

1. **Zero dependencies.** Works on any machine, any CI
   runner, any Docker image. No GPU, no model download.
2. **Deterministic.** Same inputs always produce the same
   score. No model loading variance.
3. **Fast.** Instant computation. No 14ms-per-pair
   embedding overhead, no 500ms-per-pair NLI overhead.
4. **Sufficient for most CI gates.** Token F1 catches
   gross behavioral changes. Teams that need semantic
   depth upgrade to `"embedding"` or `"entailment"`.

Users who install `sentence-transformers` unlock
`"embedding"`, `"entailment"`, and `"auto"` methods.
The tiered approach means no user is blocked by
dependency weight.

### Why user-provided paraphrases first?

CheckList (Ribeiro et al., ACL 2020) found that
human-curated test cases catch more bugs than
auto-generated ones. `ParaphraseGenerator` is a
convenience utility, not a replacement for
domain-specific paraphrases.

### Why report per-input details, not just aggregate?

ProSA (EMNLP 2024) proved that aggregate scores mask
per-instance sensitivity. Some inputs are stable, others
wildly inconsistent. The `TestResult.details` includes
per-pair breakdown so you can find the fragile inputs.

### Why auto-detect classifier output?

`assert_paraphrase_invariance` auto-detects when all
outputs look like short labels (no whitespace, under
50 chars) and switches to `label_match`. This follows
CheckList's INV pattern: classifiers need exact label
match, not semantic similarity.

---

## Assertions

### `assert_paraphrase_invariance`

Tests whether a model produces equivalent outputs when
the same question is asked in different words.

```
"What caused WW2?"
"Summarize the causes of World War 2"    --> same answer?
"Why did the second world war happen?"
```

**Research basis:** CheckList INV tests (ACL 2020) +
SCORE benchmark showing 10% accuracy swings from
paraphrasing (NAACL 2025). MENLI (TACL 2023) informs
the NLI bidirectional method.

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
    equivalence_method="token_f1",
    min_invariance=0.8,
)
```

#### What it does

1. Calls `model_fn` once per paraphrase
2. Compares every output pair using the chosen
   `equivalence_method`
3. Computes the fraction of equivalent pairs
   (invariance rate)
4. Passes if the invariance rate meets `min_invariance`
5. Auto-detects classifier output and switches to
   `label_match` when all outputs are short labels

#### Parameters

| Name | Type | Default | What |
|------|------|---------|------|
| `model_fn` | `Callable[[str], Any]` | *(required)* | Function that takes a prompt, returns output |
| `paraphrases` | `list[str]` | *(required)* | 2+ semantically equivalent inputs |
| `equivalence_method` | `str` | `"token_f1"` | How to compare outputs (see Methods) |
| `min_invariance` | `float` | `0.8` | Minimum fraction of equivalent pairs to pass (0-1) |
| `similarity_threshold` | `float \| None` | `None` | Override per-method default threshold |
| `embedding_model` | `str` | `"all-mpnet-base-v2"` | Sentence-transformer model for `"embedding"` and `"auto"` |
| `nli_model` | `str` | `"cross-encoder/nli-deberta-v3-base"` | Cross-encoder model for `"entailment"` and `"auto"` |
| `judge_fn` | `Callable[[str, str], float] \| None` | `None` | Scorer for `equivalence_method="judge"` |

#### Returns

`TestResult` with:

- `passed`: whether `min_invariance` was met
- `invariance_rate`: fraction of equivalent pairs
- `pair_scores`: list of per-pair `{pair, score, equivalent}`
- `worst_pair`: indices of the lowest-scoring pair
- `worst_score`: score of the worst pair
- `per_input_outputs`: each paraphrase and its output

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

**Research basis:** Prompt Underspecification (Feb 2026)
showed minor formatting changes (case, spacing) change
model rankings. CheckList INV (ACL 2020) covers
label-preserving perturbations.

#### Signature

```python
from mltk.domains.llm.behavioral import (
    assert_format_invariance,
)

result = assert_format_invariance(
    model_fn=my_llm.generate,
    input_text="What is photosynthesis?",
    equivalence_method="token_f1",
    min_invariance=0.9,
)
```

#### What it does

1. Applies each transform to `input_text` to create
   formatting variants
2. Calls `model_fn` on the original and each variant
3. Compares each variant's output to the original using
   `equivalence_method`
4. Passes if the fraction of equivalent transforms meets
   `min_invariance`

#### Built-in transforms (used when `transforms=None`)

When you do not pass `transforms`, these five built-in
transforms are applied automatically:

| Name | What it does | Example input | Example output |
|------|-------------|---------------|----------------|
| `"lowercase"` | Converts to lowercase | `"What is WW2?"` | `"what is ww2?"` |
| `"uppercase"` | Converts to uppercase | `"What is WW2?"` | `"WHAT IS WW2?"` |
| `"strip_whitespace"` | Strips leading/trailing whitespace | `"  What is WW2?  "` | `"What is WW2?"` |
| `"no_punctuation"` | Removes all punctuation | `"What is WW2?"` | `"What is WW2"` |
| `"double_space"` | Replaces whitespace runs with double spaces | `"What is WW2?"` | `"What  is  WW2?"` |

Internally, each built-in transform is a `(name, callable)`
tuple. When you pass custom transforms, they are wrapped as
`("custom_0", fn)`, `("custom_1", fn)`, etc.

#### Custom transforms

Pass a list of `Callable[[str], str]` functions:

```python
result = assert_format_invariance(
    model_fn=my_llm.generate,
    input_text="What is machine learning?",
    transforms=[
        lambda t: t.lower(),
        lambda t: t.upper(),
        lambda t: t.replace("?", ""),
    ],
    min_invariance=0.9,
)
```

#### Parameters

| Name | Type | Default | What |
|------|------|---------|------|
| `model_fn` | `Callable[[str], Any]` | *(required)* | Function that takes a prompt, returns output |
| `input_text` | `str` | *(required)* | The original input text |
| `transforms` | `list[Callable[[str], str]] \| None` | `None` | Transform functions. `None` uses 5 built-in defaults |
| `equivalence_method` | `str` | `"token_f1"` | How to compare outputs (see Methods) |
| `min_invariance` | `float` | `0.9` | Minimum fraction of equivalent transforms to pass (0-1) |
| `similarity_threshold` | `float \| None` | `None` | Override per-method default threshold |
| `embedding_model` | `str` | `"all-mpnet-base-v2"` | Sentence-transformer model for `"embedding"` |

Note: `assert_format_invariance` does not accept `judge_fn`
or `nli_model` parameters. It supports `"token_f1"`,
`"embedding"`, `"entailment"`, and `"label_match"` methods.

#### Returns

`TestResult` with:

- `passed`: whether `min_invariance` was met
- `invariance_rate`: fraction of equivalent transforms
- `transform_results`: list of per-transform `{transform, input, output, score, equivalent}`
- `original_input`: the input before transforms
- `original_output`: model output for the original input

---

### `assert_output_stability`

Tests whether a model produces consistent outputs when
given the exact same input multiple times. Catches
non-deterministic behavior from temperature, sampling,
or infrastructure issues.

```
"Explain gravity" x5 --> all 5 answers consistent?
```

**Research basis:** SelfCheckGPT (EMNLP 2023) uses
sampling consistency as a hallucination signal. ProSA
(EMNLP 2024) showed per-instance sensitivity varies
widely.

#### Signature

```python
from mltk.domains.llm.behavioral import (
    assert_output_stability,
)

result = assert_output_stability(
    model_fn=my_llm.generate,
    inputs=["Explain gravity in one sentence."],
    n_runs=5,
    equivalence_method="token_f1",
    min_stability=0.9,
)
```

#### What it does

1. For each input in `inputs`, calls `model_fn` `n_runs`
   times
2. Compares all output pairs for each input using
   `equivalence_method`
3. Computes per-input stability (fraction of pairs above
   threshold) and averages across all inputs
4. Passes if the average stability meets `min_stability`

#### Parameters

| Name | Type | Default | What |
|------|------|---------|------|
| `model_fn` | `Callable[[str], Any]` | *(required)* | Function that takes a prompt, returns output |
| `inputs` | `list[str]` | *(required)* | List of input prompts to test stability on |
| `n_runs` | `int` | `5` | How many times to call the model per input (>= 2) |
| `equivalence_method` | `str` | `"token_f1"` | How to compare outputs |
| `min_stability` | `float` | `0.9` | Minimum average stability to pass (0-1) |
| `similarity_threshold` | `float \| None` | `None` | Override default (0.8 for score-based methods) |
| `embedding_model` | `str` | `"all-mpnet-base-v2"` | Sentence-transformer model for `"embedding"` |

Note: `assert_output_stability` does not accept `judge_fn`
or `nli_model` parameters. It supports `"token_f1"`,
`"embedding"`, and `"label_match"` methods only.

#### Returns

`TestResult` with:

- `passed`: whether `min_stability` was met
- `avg_stability`: mean stability across all inputs
- `per_input_stability`: list of per-input `{input, stability, n_runs, n_unique_outputs}`
- `worst_input`: the input with lowest stability
- `method`: the equivalence method used

---

### `assert_semantic_equivalence`

Tests whether two texts are semantically equivalent.
Uses bidirectional NLI (default) to detect both
equivalence and contradiction -- something cosine
similarity cannot do.

**Research basis:** MENLI (TACL 2023) showed NLI-based
metrics are 15-30% more robust than BERTScore against
adversarial attacks. Bidirectional entailment directly
captures logical equivalence.

#### Signature

```python
from mltk.domains.llm.behavioral import (
    assert_semantic_equivalence,
)

result = assert_semantic_equivalence(
    text_a="The cat sat on the mat.",
    text_b="A feline was resting on the rug.",
    method="nli",
    min_score=0.7,
)
```

#### What it does

1. Normalizes both texts (Unicode normalization)
2. Compares using the chosen `method`:
   - **`"nli"`**: Bidirectional NLI entailment. Checks
     forward (A entails B) and backward (B entails A).
     Score is `min(forward_entailment, backward_entailment)`.
     Detects contradictions explicitly.
   - **`"embedding"`**: Cosine similarity via
     sentence-transformers.
   - **`"token_f1"`**: Zero-dependency token overlap.
3. Passes if the score meets `min_score` (and no
   contradiction for NLI)

#### Parameters

| Name | Type | Default | What |
|------|------|---------|------|
| `text_a` | `str` | *(required)* | First text |
| `text_b` | `str` | *(required)* | Second text |
| `method` | `str` | `"nli"` | One of `"nli"`, `"embedding"`, `"token_f1"` |
| `min_score` | `float` | `0.7` | Minimum score to pass |
| `nli_model` | `str` | `"cross-encoder/nli-deberta-v3-base"` | Cross-encoder model for `"nli"` |
| `embedding_model` | `str` | `"all-mpnet-base-v2"` | Sentence-transformer model for `"embedding"` |

#### Returns

`TestResult` with:

- `passed`: whether `min_score` was met (and no contradiction for NLI)
- `score`: the equivalence score
- `forward_entailment` / `backward_entailment`: per-direction NLI probabilities (NLI only)
- `contradiction`: whether NLI detected a contradiction (NLI only)
- `equivalent`: whether bidirectional entailment was confirmed (NLI only)

---

### `assert_directional_expectation`

Tests that a specific input perturbation causes a
predictable output change. Implements the CheckList DIR
pattern: the complement of invariance testing.

**Research basis:** CheckList DIR (Ribeiro et al., ACL 2020)
defined directional expectation tests. LLMORPH
(ASE 2025) found 18% average failure rate across LLMs
on metamorphic relations.

#### Signature

```python
from mltk.domains.llm.behavioral import (
    assert_directional_expectation,
)

result = assert_directional_expectation(
    model_fn=my_llm.generate,
    input_text="Explain gravity",
    perturbation=lambda t: t + " in 100 words",
    direction_fn=lambda orig, pert: len(pert) < len(orig),
    perturbation_name="add length constraint",
)
```

#### What it does

1. Calls `model_fn` on the original `input_text`
2. Applies `perturbation` to `input_text`
3. Calls `model_fn` on the perturbed input
4. Calls `direction_fn(original_output, perturbed_output)`
5. Passes if `direction_fn` returns `True`

#### Parameters

| Name | Type | Default | What |
|------|------|---------|------|
| `model_fn` | `Callable[[str], str]` | *(required)* | Model under test |
| `input_text` | `str` | *(required)* | The original input prompt |
| `perturbation` | `Callable[[str], str]` | *(required)* | Transforms input in a meaningful way |
| `direction_fn` | `Callable[[str, str], bool]` | *(required)* | `(original_out, perturbed_out) -> bool` |
| `perturbation_name` | `str \| None` | `None` | Human-readable label for the perturbation |

#### Returns

`TestResult` with:

- `passed`: whether the direction was met
- `perturbation_name`: the label
- `original_output`: model output on the original input
- `perturbed_input`: the input after perturbation
- `perturbed_output`: model output on the perturbed input
- `direction_met`: same as `passed`

#### Examples

Length constraint:

```python
assert_directional_expectation(
    my_model,
    "Explain gravity",
    perturbation=lambda t: t + " in 100 words",
    direction_fn=lambda o, p: len(p) < len(o),
    perturbation_name="add length constraint",
)
```

Sentiment shift:

```python
assert_directional_expectation(
    my_model,
    "Review this product",
    perturbation=lambda t: "Give a negative review: " + t,
    direction_fn=lambda o, p: "bad" in p.lower(),
    perturbation_name="negative sentiment",
)
```

---

### `assert_retrieval_consistency`

Tests whether a RAG retrieval pipeline returns
consistent documents for semantically equivalent
queries. Uses Jaccard similarity on document ID
sets -- the standard set-overlap metric.

**Research basis:** The five-assertion family identified
in our consolidated research brief. RAG pipelines are
especially vulnerable to paraphrase sensitivity because
retrieval and generation compound inconsistency.

#### Signature

```python
from mltk.domains.llm.behavioral import (
    assert_retrieval_consistency,
)

result = assert_retrieval_consistency(
    retriever_fn=my_retriever.search,
    paraphrases=[
        "What is machine learning?",
        "Explain machine learning",
        "Describe ML",
    ],
    min_overlap=0.7,
)
```

#### What it does

1. Calls `retriever_fn` on each paraphrase to get
   document ID lists
2. Computes Jaccard similarity for every query pair:
   `|A & B| / |A | B|`
3. Averages Jaccard across all pairs
4. Passes if the average meets `min_overlap`

#### Parameters

| Name | Type | Default | What |
|------|------|---------|------|
| `retriever_fn` | `Callable[[str], list[str]]` | *(required)* | Returns document IDs for a query |
| `paraphrases` | `list[str]` | *(required)* | 2+ semantically equivalent queries |
| `min_overlap` | `float` | `0.7` | Minimum mean Jaccard similarity to pass (0-1) |

#### Returns

`TestResult` with:

- `passed`: whether `min_overlap` was met
- `avg_overlap`: mean Jaccard similarity across all pairs
- `per_pair`: list of per-pair `{query_a, query_b, jaccard, docs_a, docs_b, intersection, union}`
- `worst_pair`: the pair with lowest Jaccard score
- `n_queries`: number of paraphrases tested
- `n_pairs`: number of query pairs compared

---

## ParaphraseGenerator

Utility for producing semantically equivalent rephrasings
of a text. Two modes: deterministic templates (zero-dep,
CI-friendly) and LLM-backed generation (higher quality).

**Research basis:** CheckList (ACL 2020) found
human-curated test cases catch more bugs than
auto-generated ones. Templates are a starting point;
add domain-specific manual paraphrases for production.

#### Usage

```python
from mltk.domains.llm.behavioral import (
    ParaphraseGenerator,
)

gen = ParaphraseGenerator()

# Template-based (zero-dep, deterministic)
variants = gen.generate_template(
    "What is machine learning?", n=5,
)
# Returns: ["Explain machine learning",
#           "Describe machine learning", ...]

# LLM-based (highest quality)
variants = gen.generate_llm(
    "What is machine learning?",
    llm_fn=my_llm.generate,
    n=5,
)

# Unified interface
variants = gen.generate(
    "What is machine learning?",
    n=5,
    method="template",  # or "llm"
)
```

#### Methods

**`generate_template(text, n=5)`** -- Zero-dependency,
deterministic paraphrases using:

- Question reformulation ("What is X?" to "Explain X",
  "Describe X", "Tell me about X", "Can you explain X")
- Filler insertion ("So, ...", "Well, ...", "Basically, ...")
- Clause reordering ("A because B" to "Because B, A")
- Contraction toggling ("don't" to "do not" and back)

**`generate_llm(text, llm_fn, n=5)`** -- Sends a
rephrasing prompt to your LLM function. Parses one
paraphrase per line from the response.

**`generate(text, n=5, method="template", llm_fn=None)`**
-- Unified dispatch. `method="template"` or `method="llm"`.

#### Parameters

**`generate_template`**:

| Name | Type | Default | What |
|------|------|---------|------|
| `text` | `str` | *(required)* | Input text to paraphrase |
| `n` | `int` | `5` | Maximum paraphrases to return |

**`generate_llm`**:

| Name | Type | Default | What |
|------|------|---------|------|
| `text` | `str` | *(required)* | Input text to paraphrase |
| `llm_fn` | `Callable[[str], str]` | *(required)* | LLM callable that takes a prompt |
| `n` | `int` | `5` | Number of paraphrases to request |

**`generate`**:

| Name | Type | Default | What |
|------|------|---------|------|
| `text` | `str` | *(required)* | Input text to paraphrase |
| `n` | `int` | `5` | Maximum paraphrases to return |
| `method` | `str` | `"template"` | `"template"` or `"llm"` |
| `llm_fn` | `Callable[[str], str] \| None` | `None` | Required when `method="llm"` |

#### Combining with assertions

```python
gen = ParaphraseGenerator()
paraphrases = gen.generate_template(
    "What is quantum computing?", n=5,
)

result = assert_paraphrase_invariance(
    model_fn=my_llm.generate,
    paraphrases=[
        "What is quantum computing?",
        *paraphrases,
    ],
)
```

---

## Equivalence Methods

The invariance and stability assertions share
`equivalence_method` parameters. This controls how two
model outputs are compared to determine if they are
"the same."

For a deep dive on each method, see the
[Method Dispatch guide](../guides/method-dispatch.md).

**Method availability by assertion:**

| Method | `paraphrase_invariance` | `format_invariance` | `output_stability` | `semantic_equivalence` |
|--------|:-:|:-:|:-:|:-:|
| `"token_f1"` | yes | yes | yes | yes |
| `"embedding"` | yes | yes | yes | yes |
| `"entailment"` | yes | yes | -- | via `"nli"` |
| `"judge"` | yes | -- | -- | -- |
| `"auto"` | yes | -- | -- | -- |
| `"label_match"` | yes | yes | yes | -- |
| `"nli"` | -- | -- | -- | yes |

| Method | What it does | Speed | Catches |
|--------|-------------|-------|---------|
| `"token_f1"` | Token-level F1 overlap | instant | Keyword matches |
| `"embedding"` | Cosine similarity via sentence-transformers | ~14ms/pair | Synonyms, paraphrases |
| `"entailment"` | Bidirectional NLI entailment | ~500ms/pair | Contradictions |
| `"judge"` | Your custom LLM-as-Judge function | ~2-5s/pair | Nuance, subjective quality |
| `"auto"` | Cosine first, NLI for ambiguous zone | varies | Auto-selects |
| `"label_match"` | Exact string match after normalization | instant | Classification tasks |

### `token_f1`

Tokenizes both outputs, computes F1 overlap. No
dependencies. Fast. Misses synonyms entirely.

Use for classification or short factual answers where
exact wording matters.

### `embedding`

Encodes both outputs with a sentence-transformer and
computes cosine similarity. Good balance of speed and
semantic understanding.

Use as a general-purpose upgrade from `token_f1`.

### `entailment`

Runs bidirectional NLI: checks that output A entails
output B AND output B entails output A. If both directions
score high, the outputs are semantically equivalent. This
catches contradictions that embedding similarity misses.

Use for factual or safety-critical tasks where
contradictions are dangerous.

### `judge`

Calls your custom function with both outputs. You define
what "equivalent" means. Most flexible but slowest.
Only available in `assert_paraphrase_invariance`.

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

Use for subjective or domain-specific equivalence
(e.g., medical accuracy, legal compliance).

### `auto`

Cosine similarity first. If score >= 0.90, equivalent.
If score < 0.50, different. In the ambiguous zone
(0.50-0.90), falls back to NLI bidirectional entailment.
Only available in `assert_paraphrase_invariance`.

Use when you want the best available method without
thinking about it.

### `label_match`

Compares outputs as exact strings. Score is 1.0 if
they match, 0.0 if not. Designed for classification
tasks where the model outputs a label.

Use when testing classification models (e.g., sentiment:
"positive" vs "POSITIVE").

---

## Examples

### Paraphrase invariance (zero-dep default)

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
)

print(result.details["invariance_rate"])  # 0.87
print(result.passed)                      # True
```

### Format invariance with defaults

```python
from mltk.domains.llm.behavioral import (
    assert_format_invariance,
)

# Uses all 5 built-in transforms automatically
result = assert_format_invariance(
    model_fn=my_llm.generate,
    input_text="What is photosynthesis?",
)
```

### Format invariance with custom transforms

```python
def add_typo(text: str) -> str:
    """Simulate a user typo."""
    words = text.split()
    if len(words) > 2:
        words[1] = words[1][::-1]  # reverse word
    return " ".join(words)


result = assert_format_invariance(
    model_fn=my_llm.generate,
    input_text="What is machine learning?",
    transforms=[
        lambda t: t.lower(),
        lambda t: t.upper(),
        add_typo,
    ],
    min_invariance=0.75,
)
```

### Output stability for determinism testing

```python
from mltk.domains.llm.behavioral import (
    assert_output_stability,
)

result = assert_output_stability(
    model_fn=my_llm.generate,
    inputs=["What is the capital of France?"],
    n_runs=5,
    equivalence_method="label_match",
    min_stability=0.9,
)

if not result.passed:
    print("Model is non-deterministic!")
    details = result.details
    print(f"Stability: {details['avg_stability']:.2f}")
```

### Semantic equivalence with NLI

```python
from mltk.domains.llm.behavioral import (
    assert_semantic_equivalence,
)

result = assert_semantic_equivalence(
    text_a="The cat sat on the mat.",
    text_b="A feline was resting on the rug.",
    method="nli",
    min_score=0.7,
)
# result.details includes forward_entailment,
# backward_entailment, and contradiction flag
```

### Directional expectation

```python
from mltk.domains.llm.behavioral import (
    assert_directional_expectation,
)

result = assert_directional_expectation(
    model_fn=my_llm.generate,
    input_text="Explain gravity",
    perturbation=lambda t: t + " in one sentence",
    direction_fn=lambda orig, pert: len(pert) < len(orig),
    perturbation_name="brevity constraint",
)
```

### Retrieval consistency for RAG

```python
from mltk.domains.llm.behavioral import (
    assert_retrieval_consistency,
)

result = assert_retrieval_consistency(
    retriever_fn=my_rag.retrieve,
    paraphrases=[
        "What is machine learning?",
        "Explain ML",
        "Describe machine learning",
    ],
    min_overlap=0.7,
)
```

### ParaphraseGenerator + invariance test

```python
from mltk.domains.llm.behavioral import (
    ParaphraseGenerator,
    assert_paraphrase_invariance,
)

gen = ParaphraseGenerator()
variants = gen.generate_template(
    "What is quantum computing?", n=3,
)

result = assert_paraphrase_invariance(
    model_fn=my_llm.generate,
    paraphrases=[
        "What is quantum computing?",
        *variants,
    ],
)
```

### In a pytest test suite

```python
import pytest
from mltk.domains.llm.behavioral import (
    assert_format_invariance,
    assert_output_stability,
    assert_paraphrase_invariance,
    assert_semantic_equivalence,
    assert_directional_expectation,
    assert_retrieval_consistency,
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
        input_text="What is DNA?",
        min_invariance=0.85,
    )


def test_deterministic_output(model):
    assert_output_stability(
        model_fn=model,
        inputs=["What is 2 + 2?"],
        n_runs=3,
        min_stability=0.95,
    )


def test_semantic_equivalence():
    assert_semantic_equivalence(
        text_a="Water freezes at 0 degrees Celsius.",
        text_b="The freezing point of water is 0C.",
        method="token_f1",
        min_score=0.3,
    )


def test_directional_expectation(model):
    assert_directional_expectation(
        model_fn=model,
        input_text="Explain gravity",
        perturbation=lambda t: t + " in one word",
        direction_fn=lambda o, p: len(p) < len(o),
        perturbation_name="brevity",
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
+-- Two specific texts -- are they equivalent?
|   --> assert_semantic_equivalence
|
+-- A change SHOULD change the output?
|   (opposite of invariance)
|   --> assert_directional_expectation
|
+-- RAG: same question, different wording?
|   (testing retrieval, not generation)
|   --> assert_retrieval_consistency
|
+-- Need paraphrases but don't want to write them?
|   --> ParaphraseGenerator
|
+-- All of the above? Run them all in your test suite.
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

- **RAG pipeline testing**: run retrieval consistency to
  verify that your retriever does not return different
  documents for equivalent queries.

- **Safety-critical outputs**: run semantic equivalence
  with `method="nli"` to catch contradictions that
  cosine similarity misses.

- **CI/CD**: paraphrase + format + stability, with
  `equivalence_method="token_f1"`. Zero dependencies,
  fast enough for every commit.

---

## Threshold Guidance

Different `equivalence_method` values produce scores on
different scales. A "0.8" in embedding is not the same as
a "0.8" in entailment.

### Per-method defaults

When `similarity_threshold` is `None`, each method uses
its own default (from the invariance module):

| Method | Default threshold |
|--------|:-:|
| `token_f1` | `0.50` |
| `embedding` | `0.80` |
| `judge` | `0.70` |
| `label_match` | `1.0` (exact match) |

For `assert_output_stability`, the default threshold for
score-based methods (`token_f1` and `embedding`) is `0.8`.

### `token_f1` thresholds

| Score | Interpretation |
|-------|----------------|
| >= 0.60 | Strong keyword overlap |
| 0.35 - 0.60 | Moderate overlap |
| < 0.35 | Weak -- outputs likely differ |

Warning: High token overlap does not guarantee semantic
equivalence. "The war started in 1939" and "The war did
not start in 1939" have nearly identical tokens.

### `embedding` thresholds

| Score | Interpretation |
|-------|----------------|
| >= 0.85 | Semantically equivalent (strict) |
| 0.75 - 0.85 | Likely equivalent (recommended range) |
| 0.60 - 0.75 | Related but may differ in substance |
| < 0.60 | Different meanings |

For paraphrase invariance, start with 0.80. For format
invariance, use 0.85 (outputs should be nearly identical
since only formatting changed).

### `entailment` thresholds

| Score | Interpretation |
|-------|----------------|
| >= 0.75 | Mutual entailment -- outputs agree |
| 0.50 - 0.75 | Partial agreement |
| < 0.50 | Outputs may contradict |

Entailment is stricter than embedding. A 0.70 entailment
score is roughly equivalent to a 0.85 embedding score in
practice.

### `label_match` thresholds

Binary: 1.0 if labels match, 0.0 if not. Set
`min_invariance=1.0` to require perfect consistency.

### `judge` thresholds

Depends entirely on your judge function. Define what
0.0 and 1.0 mean in your rubric, then set the threshold
accordingly.

---

## Research Background

Behavioral consistency testing builds on these research
threads:

### CheckList (ACL 2020)

Introduced the idea of testing NLP models with
**Minimum Functionality Tests (MFTs)**, **Invariance
Tests (INV)**, and **Directional Expectation Tests
(DIR)**. Found that models passing standard benchmarks
fail systematically on simple perturbations. mltk's
`assert_paraphrase_invariance` implements INV testing.
`assert_format_invariance` extends it to formatting.
`assert_directional_expectation` implements DIR testing.

> Ribeiro et al., "Beyond Accuracy: Behavioral Testing
> of NLP Models with CheckList", ACL 2020.

### MENLI (TACL 2023)

Showed NLI-based metrics are 15-30% more robust than
BERTScore against adversarial attacks. Bidirectional
entailment ("A entails B AND B entails A") directly
captures logical equivalence. Informs
`assert_semantic_equivalence` and the `"entailment"`
equivalence method.

> Chen et al., "MENLI: Robust Evaluation Metrics from
> Natural Language Inference", TACL 2023.

### SelfCheckGPT (EMNLP 2023)

Uses sampling consistency as a hallucination signal.
If a model's repeated outputs are inconsistent, it is
likely hallucinating. Informs `assert_output_stability`.

> Manakul et al., "SelfCheckGPT: Zero-Resource
> Black-Box Hallucination Detection", EMNLP 2023.

### ProSA (EMNLP 2024)

Measured how much model outputs change when prompts are
paraphrased. Found 5-15% accuracy variation across
semantically equivalent prompts in commercial models.
Critically showed that aggregate scores hide per-instance
sensitivity. Motivated per-pair reporting in TestResult.

### SCORE (NAACL 2025)

Large-scale benchmark showing ~10% accuracy swings from
paraphrasing across production LLMs. Demonstrated that
behavioral inconsistency is not an edge case but a
systematic problem. Reinforced that single-phrasing
benchmarks overstate model reliability.

### LLMORPH (ASE 2025)

Defined 36 metamorphic relations for LLM testing. Found
18% average failure rate across LLMs. Informs the
directional expectation pattern.

### Prompt Underspecification (Feb 2026)

Showed that minor formatting changes (case, spacing)
change model rankings on benchmarks. Directly motivates
`assert_format_invariance`.

---

## FAQ

### Do I need to write my own paraphrases?

Not necessarily. `ParaphraseGenerator` creates variants
automatically using templates (zero-dep) or an LLM.
For production tests, supplement generated paraphrases
with domain-specific manual ones -- CheckList research
shows human-curated test cases catch more bugs.

### Which equivalence method should I start with?

Start with `"token_f1"` (the default). It is zero-dep,
instant, and sufficient for most CI gates. Upgrade to
`"embedding"` when you need synonym/paraphrase detection.
Use `"entailment"` for safety-critical applications
where contradictions matter.

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
formatting. `assert_paraphrase_invariance` also
auto-detects classifier-style output (short, single-token
labels) and switches to `label_match` automatically.

### How many paraphrases should I test?

3-5 per intent is a good starting point. More
paraphrases increase coverage but also increase test
runtime (each paraphrase requires one model call).
For CI/CD, 3 paraphrases per critical question gives
good coverage without excessive runtime.

### What is the difference between `assert_semantic_equivalence` and `assert_paraphrase_invariance`?

`assert_semantic_equivalence` compares two given texts
directly -- no model involved. Use it to verify that
two specific strings say the same thing.

`assert_paraphrase_invariance` runs a model on multiple
inputs and checks that the model's outputs are equivalent.
It tests model behavior, not text similarity.
