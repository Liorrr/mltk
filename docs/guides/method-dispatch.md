# Method Dispatch — Multi-Method Evaluation

Every LLM assertion in mltk answers a question: *"Is this
output correct?"* The hard part is deciding what "correct"
means -- and how deeply you want to check.

Token overlap is fast and dependency-free, but it misses
synonyms and paraphrases. Embedding similarity catches
semantic equivalence but cannot spot contradictions. NLI
entailment detects contradictions but needs a model download.
LLM-as-Judge handles nuance better than any metric, but it
is slow and requires an external model.

Each method is a tradeoff. mltk gives you a `method`
parameter on its core LLM assertions so you can pick the
right one for your use case.

---

## Why Multi-Method?

Consider two answers to *"What is the capital of France?"*:

- **Answer A:** "Paris is the capital of France."
- **Answer B:** "The French Republic's seat of government is Paris."

A token overlap check sees different words and scores low.
An embedding model sees the same meaning and scores high.
Both are correct -- the difference is what the metric
measures.

Now consider a contradiction:

- **Source:** "The Eiffel Tower was built in 1889."
- **Claim:** "The Eiffel Tower was built in 1920."

Token overlap gives a *high* score here -- most tokens
match. Embedding similarity also scores high because the
sentences are structurally similar. Only NLI catches the
contradiction because it understands entailment logic.

No single method covers everything. That is why mltk
supports four.

---

## The Four Methods

### `lexical` — Token Overlap

Splits both texts into lowercase token sets and computes the
overlap ratio. Fast, deterministic, no dependencies.

| Property | Value |
|----------|-------|
| Speed | Instant (~0.01ms) |
| Dependencies | None (built-in) |
| Catches synonyms | :x: No |
| Catches paraphrases | :x: No |
| Catches contradictions | :x: No |
| Deterministic | :white_check_mark: Yes |

:point_right: **Best for:** Quick sanity checks, CI/CD gates
where speed matters more than nuance, checking that specific
keywords appear in output.

```python
from mltk.domains.llm.rag import assert_faithfulness

result = assert_faithfulness(
    answer="The Eiffel Tower is in Paris.",
    context="The Eiffel Tower is located in Paris, France.",
    method="lexical",
    min_score=0.5,
)
```

### `embedding` — Cosine Similarity

Encodes both texts with a sentence-transformer model and
computes cosine similarity. Good balance of speed and
accuracy for semantic equivalence.

| Property | Value |
|----------|-------|
| Speed | ~14ms per pair (CPU, after model load) |
| Dependencies | `sentence-transformers` |
| Catches synonyms | :white_check_mark: Yes |
| Catches paraphrases | :white_check_mark: Yes |
| Catches contradictions | :x: No |
| Deterministic | :white_check_mark: Yes (same model + input) |

:point_right: **Best for:** Detecting semantic equivalence
when wording varies. The default upgrade from `lexical` when
you need accuracy without NLI overhead.

Default model: `all-mpnet-base-v2` (110M params, STS 87-88%).

```python
result = assert_faithfulness(
    answer="The French capital is Paris.",
    context="Paris is the capital city of France.",
    method="embedding",
    min_score=0.80,
)
```

### `nli` — Natural Language Inference

Uses a cross-encoder NLI model to compute the probability
that the source text *entails* the claim. The strongest
single metric for factual grounding -- it can detect
contradictions that embedding similarity misses entirely.

| Property | Value |
|----------|-------|
| Speed | ~500ms per pair (CPU) |
| Dependencies | `sentence-transformers` (same package as embedding) |
| Catches synonyms | :white_check_mark: Yes |
| Catches paraphrases | :white_check_mark: Yes |
| Catches contradictions | :white_check_mark: **Yes** |
| Deterministic | :white_check_mark: Yes (same model + input) |

:point_right: **Best for:** Factual accuracy checks,
hallucination detection, any case where contradictions are
dangerous (medical, legal, financial).

Default model: `cross-encoder/nli-deberta-v3-base`.

The NLI model returns three probabilities per pair:
`entailment`, `contradiction`, and `neutral`. mltk uses the
entailment probability as the score.

```python
result = assert_faithfulness(
    answer="The tower was completed in 1889.",
    context="Construction of the Eiffel Tower finished in 1889.",
    method="nli",
    min_score=0.70,
)
```

### `llm` — LLM-as-Judge

You provide a callable that takes two strings and returns a
0-1 score. mltk calls it for each evaluation pair. Most
accurate for subjective quality, creative tasks, and complex
reasoning -- but slowest and requires an external model.

| Property | Value |
|----------|-------|
| Speed | ~2-5s per pair (API-dependent) |
| Dependencies | None from mltk (you provide the function) |
| Catches synonyms | :white_check_mark: Yes |
| Catches paraphrases | :white_check_mark: Yes |
| Catches contradictions | :white_check_mark: Yes |
| Deterministic | Depends on your model (temperature=0 helps) |

:point_right: **Best for:** Subjective quality evaluation,
tasks where no metric captures the full picture, rubric-based
grading.

```python
import openai

def my_judge(claim: str, source: str) -> float:
    """Ask an LLM to score how well the claim is supported."""
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": (
                f"Score 0-1 how well this claim is supported "
                f"by the source.\n\n"
                f"Source: {source}\n"
                f"Claim: {claim}\n\n"
                f"Return ONLY a decimal number."
            ),
        }],
        temperature=0,
    )
    return float(response.choices[0].message.content.strip())


result = assert_faithfulness(
    answer="Paris has many historic landmarks.",
    context="Paris is known for the Eiffel Tower and Louvre.",
    method="llm",
    judge_fn=my_judge,
    min_score=0.7,
)
```

---

## Decision Flowchart

Use this to pick a method for your situation:

```
What matters most?
|
+-- Speed (CI/CD, large batch)
|   |
|   +-- Need synonym handling?
|       +-- No  --> lexical
|       +-- Yes --> embedding
|
+-- Accuracy (correctness matters)
|   |
|   +-- Contradictions are dangerous?
|       +-- Yes --> nli
|       +-- No  --> embedding
|
+-- Subjective quality (creative, open-ended)
|   --> llm
|
+-- Not sure?
    --> embedding (best general-purpose default)
```

**Rules of thumb:**

- **CI/CD pipelines** where tests run on every commit:
  `lexical` or `embedding`. Keep the feedback loop fast.
- **Pre-release quality gates** where accuracy matters more
  than speed: `nli`.
- **Evaluating creative or subjective output** (summaries,
  explanations, recommendations): `llm`.
- **You just want something better than token overlap** but
  do not want to think hard about it: `embedding`.

---

## Which Assertions Support Method Dispatch?

Not every assertion has a `method` parameter. Here is the
current support matrix:

| Assertion | `lexical` | `embedding` | `nli` | `llm` | Module |
|-----------|:---------:|:-----------:|:-----:|:-----:|--------|
| `assert_no_hallucination` | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | `domains.llm.safety` |
| `assert_faithfulness` | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | `domains.llm.rag` |
| `assert_semantic_similarity` | :white_check_mark: (`token`) | :white_check_mark: | -- | -- | `domains.llm.similarity` |
| `assert_context_relevancy` | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | `domains.llm.rag` |
| `assert_answer_relevancy` | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | `domains.llm.rag` |

:point_right: `assert_semantic_similarity` uses the method
name `"token"` instead of `"lexical"` for historical
reasons. Both compute token set overlap.

---

## Installation

Each method has different dependency requirements:

### `lexical` — Nothing to install

Built into mltk. Works out of the box.

```bash
pip install mltk
```

### `embedding` — sentence-transformers

```bash
pip install mltk[embedding]
# or directly:
pip install sentence-transformers
```

This downloads the `all-mpnet-base-v2` model (~90MB) on
first use. Subsequent runs use the cached model.

### `nli` — sentence-transformers (same package)

```bash
pip install mltk[embedding]
# or directly:
pip install sentence-transformers
```

:point_right: The `embedding` and `nli` methods share the
same Python package (`sentence-transformers`). Installing
one gets you both. The difference is the *model*:
`all-mpnet-base-v2` for embeddings,
`cross-encoder/nli-deberta-v3-base` for NLI. Each model
is downloaded on first use.

### `llm` — No extra dependencies

You provide the judge function. mltk does not install any
LLM client library -- bring your own (`openai`, `anthropic`,
`ollama`, a local model, or any callable).

```python
# Ollama example (free, local, no API key):
import requests

def ollama_judge(claim: str, source: str) -> float:
    resp = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3.2",
            "prompt": (
                f"Score 0.0-1.0 how well this claim is "
                f"supported by the source. Return ONLY "
                f"the number.\n\n"
                f"Source: {source}\nClaim: {claim}"
            ),
            "stream": False,
        },
    )
    return float(resp.json()["response"].strip())
```

---

## Examples

### Hallucination Detection

Test that LLM-generated claims are grounded in source
documents.

```python
from mltk.domains.llm.safety import assert_no_hallucination

claims = [
    "Python was created by Guido van Rossum.",
    "Python was first released in 1991.",
    "Python is a compiled language.",  # hallucination
]
sources = [
    "Python is an interpreted programming language created "
    "by Guido van Rossum, first released in 1991.",
]

# Lexical (fast, zero-dep)
assert_no_hallucination(claims, sources, method="lexical")

# Embedding (catches paraphrases)
assert_no_hallucination(claims, sources, method="embedding")

# NLI (catches contradictions like "compiled" vs "interpreted")
assert_no_hallucination(claims, sources, method="nli")

# LLM-as-Judge (most accurate)
assert_no_hallucination(
    claims, sources,
    method="llm",
    judge_fn=my_judge,
)
```

### Faithfulness (RAG Grounding)

Test that a RAG answer is grounded in the retrieved context.

```python
from mltk.domains.llm.rag import assert_faithfulness

answer = "The company was founded in 2020 and is based in Tel Aviv."
context = [
    "Founded in 2020, the startup operates from Tel Aviv.",
    "The team has grown to 50 employees.",
]

# Lexical
assert_faithfulness(answer, context, method="lexical", min_score=0.5)

# Embedding — higher threshold because cosine scores are higher
assert_faithfulness(answer, context, method="embedding", min_score=0.80)

# NLI — entailment probability threshold
assert_faithfulness(answer, context, method="nli", min_score=0.70)
```

### Semantic Similarity

Compare reference outputs to model-generated outputs.

```python
from mltk.domains.llm.similarity import assert_semantic_similarity

refs = [
    "Machine learning is a subset of AI.",
    "The cat sat on the mat.",
]
hyps = [
    "ML is a branch of artificial intelligence.",
    "A cat was sitting on a mat.",
]

# Token overlap
assert_semantic_similarity(refs, hyps, method="token", min_score=0.3)

# Embedding cosine (catches synonyms like "subset" / "branch")
assert_semantic_similarity(refs, hyps, method="embedding", min_score=0.80)
```

### Custom Embedding Model

Every embedding-based assertion accepts a `embedding_model`
parameter (or uses `all-mpnet-base-v2` by default). You can
swap in a larger model for better accuracy:

```python
assert_faithfulness(
    answer, context,
    method="embedding",
    embedding_model="all-mpnet-base-v2",  # larger, more accurate
    min_score=0.75,
)
```

### Custom NLI Model

Similarly, the `nli_model` parameter lets you swap the
cross-encoder:

```python
assert_no_hallucination(
    claims, sources,
    method="nli",
    nli_model="cross-encoder/nli-deberta-v3-large",
    min_coverage=0.70,
)
```

---

## Speed vs Accuracy

Approximate characteristics per evaluation pair. Actual
numbers depend on hardware, text length, and model caching.

| Method | First-call latency | Subsequent calls | Accuracy profile | Dependencies |
|--------|--------------------|------------------|------------------|--------------|
| `lexical` | ~0.01ms | ~0.01ms | Exact token match only. Misses synonyms and paraphrases. | None |
| `embedding` | ~2-5s (model load) | ~14ms | Catches semantic equivalence. Blind to contradictions. | `sentence-transformers` |
| `nli` | ~3-8s (model load) | ~500ms | Catches equivalence AND contradictions. Best factual metric. | `sentence-transformers` |
| `llm` | Varies | ~2-5s | Highest accuracy for nuanced tasks. Depends on judge quality. | User-provided |

:point_right: The `embedding` and `nli` models are cached
after the first load in a session. Subsequent calls within
the same process are fast. Across processes, the models are
cached on disk by `sentence-transformers` in
`~/.cache/huggingface/`.

### Batch performance

For large evaluation sets (100+ pairs), `embedding` is
significantly faster than `nli` because embedding models
encode all texts in a single batch, while cross-encoder NLI
evaluates each pair individually.

| Pairs | `lexical` | `embedding` | `nli` | `llm` |
|-------|-----------|-------------|-------|-------|
| 10 | <1ms | ~50ms | ~5s | ~30s |
| 100 | <1ms | ~200ms | ~50s | ~5min |
| 1000 | ~5ms | ~1.5s | ~8min | ~1hr |

---

## Threshold Guidance

Different methods produce scores on different scales with
different meanings. A "0.7" in one method is not the same
as a "0.7" in another.

### Lexical (token overlap ratio)

The score is the fraction of tokens in text A that also
appear in text B. Common words inflate the score.

| Score | Interpretation |
|-------|----------------|
| >= 0.50 | Strong keyword overlap |
| 0.30 - 0.50 | Moderate overlap, likely related |
| < 0.30 | Weak overlap, possibly unrelated |

:warning: High lexical overlap does NOT mean factual
consistency. "The tower was built in 1889" and "The tower
was not built in 1889" have nearly identical token sets.

### Embedding (cosine similarity)

Cosine similarity from sentence-transformers ranges from
approximately -1 to 1, but in practice scores cluster between
0.3 and 1.0 for English text.

| Score | Interpretation |
|-------|----------------|
| >= 0.80 | Semantically equivalent (recommended threshold) |
| 0.65 - 0.80 | Related but not equivalent |
| < 0.65 | Different topics or meanings |

:point_right: The 0.80 threshold was calibrated during mltk
development based on ML Engineer review of paraphrase
invariance research. The research literature reports optimal
thresholds ranging from 0.33 to 0.87 depending on model and
task -- so adjust for your domain.

### NLI (entailment probability)

The NLI model outputs probabilities for three classes:
`entailment`, `contradiction`, and `neutral`. mltk uses the
`entailment` probability as the score.

| Score | Interpretation |
|-------|----------------|
| >= 0.70 | Source supports the claim |
| 0.40 - 0.70 | Ambiguous -- neither clearly supported nor contradicted |
| < 0.40 | Likely not supported (check `contradiction` probability) |

:point_right: For hallucination detection, consider also
checking the `contradiction` probability in the
`TestResult.details`. A high contradiction score is a
stronger signal than a low entailment score.

### LLM-as-Judge

Scores depend entirely on your judge function's rubric.
Define what 0 and 1 mean in your prompt:

```
Score 0.0 = The claim directly contradicts the source.
Score 0.5 = The claim is partially supported.
Score 1.0 = The claim is fully supported by the source.
```

:point_right: For consistent results, set `temperature=0`
in your LLM judge and include explicit scoring criteria in
the prompt.

---

## Combining Methods

For maximum confidence, you can run multiple methods and
compare. If lexical says "supported" but NLI says
"contradiction", you have found a case where surface
similarity hides a factual error.

```python
from mltk.domains.llm.safety import assert_no_hallucination

claims = ["The vaccine was approved in 2021."]
sources = ["The vaccine received emergency authorization in 2020."]

# Run all three
lexical = assert_no_hallucination(claims, sources, method="lexical")
embedding = assert_no_hallucination(claims, sources, method="embedding")
nli = assert_no_hallucination(claims, sources, method="nli")

print(f"Lexical:   {lexical.details['avg_coverage']:.2f}")
print(f"Embedding: {embedding.details['avg_coverage']:.2f}")
print(f"NLI:       {nli.details['avg_coverage']:.2f}")
```

When lexical and embedding agree but NLI disagrees, the NLI
result is almost always more trustworthy for factual claims.

---

## Architecture Notes

### How method dispatch works internally

Every assertion with method dispatch follows the same
pattern:

1. Validate the `method` parameter against a supported set
2. Handle edge cases (empty inputs) before dispatching
3. Branch into the scoring logic for the chosen method
4. Lazy-import the backend only when needed (`embedding` and
   `nli` methods import from `_backends.py` at call time)
5. Return a `TestResult` with the method name in `details`

This design means `import mltk` never loads
sentence-transformers or any heavy model -- the dependency
is only pulled in when you actually call a method that needs
it.

### Model caching

The `_backends.py` module caches loaded models using
`@lru_cache(maxsize=4)`. This means:

- Up to 4 different sentence-transformer models can be held
  in memory simultaneously
- The same model is never loaded twice in a single process
- Cross-process, models are cached on disk by huggingface
  (`~/.cache/huggingface/`)

### Unicode normalization

When `method="lexical"` is used in `assert_no_hallucination`,
all input text is NFKC-normalized and stripped of zero-width
characters before comparison. This defends against homoglyph
and invisible character attacks that could bypass token
overlap checks.

---

## FAQ

### Which method should I start with?

Start with `embedding`. It is the best general-purpose
upgrade from pure token overlap -- catches synonyms and
paraphrases with minimal overhead. Move to `nli` only if you
need contradiction detection.

### Do I need a GPU?

No. All methods work on CPU. The `embedding` method with
`all-mpnet-base-v2` is fast enough for CI/CD on CPU. The
`nli` method is slower on CPU (~500ms per pair) but still
practical for test suites under 100 pairs.

### Can I use a different embedding model?

Yes. Pass `embedding_model="your-model-name"` to any
assertion that supports `method="embedding"`. Any model
compatible with the `sentence-transformers` library works.

### What happens if I use `method="embedding"` without installing sentence-transformers?

You get a clear `ImportError` with installation instructions:

```
ImportError: sentence-transformers is required for
embedding-based methods. Install with:
pip install mltk[embedding]
```

### Can I use the same `judge_fn` across multiple assertions?

Yes. The judge function signature is always
`(text_a: str, text_b: str) -> float`. Write it once and
pass it to `assert_no_hallucination`, `assert_faithfulness`,
or any future assertion that supports `method="llm"`.

### How do I test my judge function?

Call it directly with known inputs and verify the scores make
sense before plugging it into assertions:

```python
# Should score high (supported)
assert my_judge("Paris is in France", "France's capital is Paris") > 0.7

# Should score low (contradicted)
assert my_judge("Paris is in Germany", "France's capital is Paris") < 0.3
```
