# Synthetic QA Generator

Every mltk RAG assertion -- `assert_faithfulness`,
`assert_answer_relevancy`, `assert_no_hallucination` --
needs a question-answer pair to evaluate against. Writing
those pairs by hand is slow, biased toward the cases you
already know about, and does not scale past a few dozen
test cases.

Every major LLM evaluation framework ships a generator
to solve this: RAGAS has `TestsetGenerator`, DeepEval has
`Synthesizer`, Giskard has RAGET. All three follow the
same pipeline -- chunk documents, group related chunks,
prompt an LLM, filter by quality -- but all three require
a framework-specific LLM wrapper (LangChain class,
SDK subclass, or Giskard client) and make an LLM call
for every test case, including in CI.

mltk's `SyntheticQAGenerator` closes this gap with two
design decisions that no competitor matches:

1. **Zero-dependency template mode.** A CI pipeline at
   3 AM should not fail because an LLM API is unavailable.
   Template mode generates deterministic, reproducible QA
   pairs from structural patterns with no network calls,
   no API keys, and no extra packages. Use it for smoke
   tests and regression fixtures.

2. **Any `Callable[[str], str]` as the LLM.** No wrapper
   class. No subclassing. If you can call a function that
   takes a prompt and returns a string, you can use LLM
   mode. Any local model via Ollama, any hosted REST API,
   or a mock lambda -- all identical from the generator's
   perspective.

The output is `QAPair` dataclasses that feed directly into
existing mltk assertions. No format conversion, no
framework switching, no re-importing from a different
library.

**Module:** `mltk.domains.llm.synthetic`

**ML Lifecycle Stage:** Test data generation / RAG
evaluation / CI gate

**Bugs caught:**

- RAG systems that retrieve the right document but
  generate unfaithful answers
- Retrievers that miss relevant chunks for factual
  questions
- LLMs that hallucinate when given multi-hop questions
  requiring synthesis across two passages
- RAG pipelines that fail to say "I don't know" for
  out-of-scope questions
- Regression in answer quality after model or prompt
  changes (template mode catches this in CI with no cost)

---

## Two Modes

| Mode | When to use | Dependencies | Speed |
|------|-------------|-------------|-------|
| **Template** | CI/CD, smoke tests, deterministic fixtures | None (stdlib only) | Instant |
| **LLM** | Quality evals, diverse coverage, release testing | Any `str -> str` callable | ~1s/pair |

Template mode is the default. Pass `llm_fn` to switch to
LLM mode.

Template mode generates questions deterministically from
structural patterns. The questions are predictable and
lower diversity than LLM-generated questions, but they
are guaranteed to be answerable (the answer is
constructed from the same chunk), reproducible across
runs (with `seed`), and cost nothing to generate.
They are the right tool for regression tests, CI smoke
tests, and generating a baseline dataset before you have
LLM access.

LLM mode sends a prompt to your callable for each
question type and parses a JSON response. Quality
filtering scores each pair on self-containment and
answerability, discarding pairs that fail a threshold.
The diversity and naturalness of LLM-generated questions
is significantly higher than template mode, making it the
right tool for release evaluation and coverage testing.

---

## Quick Start

```python
from mltk.domains.llm.synthetic import SyntheticQAGenerator

# Template mode -- zero deps, instant, CI-safe
gen = SyntheticQAGenerator()
pairs = gen.generate_from_text(document, n=10)

# LLM mode -- any callable, higher quality
gen = SyntheticQAGenerator(llm_fn=my_llm)
pairs = gen.generate_from_text(document, n=20)

# Pre-split chunks (skip the internal splitter)
gen = SyntheticQAGenerator(llm_fn=my_llm)
pairs = gen.generate_from_chunks(my_chunks, n=15)

# Single pair from one chunk
pair = gen.generate_one(chunk, question_type=QuestionType.FACTUAL)

# Use directly with mltk assertions
for pair in pairs:
    assert_faithfulness(pair.answer, pair.context)
    assert_answer_relevancy(
        pair.question, pair.answer, pair.context,
    )
```

### LLM Callable Examples

The `llm_fn` parameter accepts any `Callable[[str], str]`.
All of the following work identically:

```python
# Chat-completion REST endpoint (works with any
# hosted or local API using the chat/completions format)
import httpx

def my_llm(prompt: str) -> str:
    r = httpx.post(
        "https://api.example.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {API_KEY}"},
        json={
            "model": "your-model",
            "messages": [{"role": "user", "content": prompt}],
        },
    )
    return r.json()["choices"][0]["message"]["content"]

# Ollama (local -- matches mltk benchmark standard)
import httpx

def my_llm(prompt: str) -> str:
    r = httpx.post(
        "http://localhost:11434/api/generate",
        json={"model": "llama3.2", "prompt": prompt,
              "stream": False},
    )
    return r.json()["response"]

# Generic SDK pattern (adapt to any SDK with a
# chat-completion interface)
def my_llm(prompt: str) -> str:
    response = sdk_client.chat(
        model="your-model",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.text  # adjust per SDK

# Mock (for tests -- no API call, zero cost)
my_llm = lambda p: '{"question": "What is X?", "answer": "X is Y."}'
```

---

## Question Types

Five question types cover the essential test surface for
RAG systems. Each type targets a different failure mode.

| # | Type | Description | What it tests | Example |
|---|------|-------------|---------------|---------|
| 1 | `FACTUAL` | Direct question whose answer is explicitly stated in one chunk | Retrieval accuracy, faithfulness | "What year was the API stable?" |
| 2 | `REASONING` | Requires inference beyond literal retrieval | Generator reasoning, hallucination resistance | "Why does the model degrade on inputs longer than 512 tokens?" |
| 3 | `MULTI_HOP` | Requires combining information from 2+ context chunks | Multi-chunk retrieval, context stitching | "Which feature introduced in v2.1 addresses the issue from the limitations section?" |
| 4 | `COUNTERFACTUAL` | Hypothetical framing ("What if X were not true?") | Robustness, hallucination under false premises | "If the model were trained on 2020 data only, how would recall differ?" |
| 5 | `OUT_OF_SCOPE` | Cannot be answered from the provided documents | Refusal behavior, routing accuracy | "What is the default admin password?" |

**FACTUAL** questions are the most common type and the
safest to use in template mode. Every template-mode
question is effectively factual: the answer is extracted
directly from a structural pattern in the chunk
(first sentence, numerical value, named entity).

**REASONING** questions require inference. In template
mode they are approximated by paraphrasing the factual
pattern to add "why" or "how" framing. LLM mode produces
genuinely inferential questions.

**MULTI_HOP** questions use two chunks. In template mode,
the anchor chunk and its sequential neighbor are combined.
In LLM mode, the two chunks are passed as Context A and
Context B in the prompt.

**COUNTERFACTUAL** questions test robustness. A RAG system
should produce a grounded, hypothetical answer rather than
hallucinating false facts. In template mode, the question
template is "What would change if {extracted noun} were
not present?" LLM mode produces richer hypotheticals.

**OUT_OF_SCOPE** is unique to mltk among open-source
generators. RAGAS and DeepEval do not generate
out-of-scope questions in their base libraries. Giskard's
Hub product does but the OSS library is weaker here.
mltk generates OOS questions in Sprint 1, making it the
earliest-available first-mover on this type. The reference
answer for OOS questions is always "This information is
not available in the provided context."

**v2 additions** (added in S77):
`CONVERSATIONAL` (multi-turn dialogue sequences) and
`DISTRACTING` (misleading element from a different
chunk). See the [v2 Features](#v2-features) section.

---

## API Reference

### QuestionType

```python
from mltk.domains.llm.synthetic import QuestionType

class QuestionType(str, Enum):
    FACTUAL = "factual"
    REASONING = "reasoning"
    MULTI_HOP = "multi_hop"
    COUNTERFACTUAL = "counterfactual"
    OUT_OF_SCOPE = "out_of_scope"
    CONVERSATIONAL = "conversational"   # v2 (S77)
    DISTRACTING = "distracting"         # v2 (S77)
```

`QuestionType` inherits from `str`, so values serialize
directly to JSON and compare equal to their string
equivalents: `QuestionType.FACTUAL == "factual"` is
`True`.

---

### QAPair

```python
from mltk.domains.llm.synthetic import QAPair

@dataclass
class QAPair:
    question: str
    answer: str
    context: str | list[str]
    question_type: QuestionType = QuestionType.FACTUAL
    metadata: dict[str, Any] = field(default_factory=dict)
```

| Field | Type | Description |
|-------|------|-------------|
| `question` | `str` | The generated question |
| `answer` | `str` | Reference (ground truth) answer |
| `context` | `str \| list[str]` | Source chunk(s) used to generate the pair. For multi-hop, a list of two chunks. |
| `question_type` | `QuestionType` | Question category enum |
| `metadata` | `dict` | Optional extras: source document path, chunk index, quality scores, API call count |

**`to_dict()`** serializes to a plain dict suitable for
JSON export, pandas, or JSONL writing. If `context` is a
list, chunks are joined with a space.

```python
pair = QAPair(
    question="What is the chunk size default?",
    answer="512 words.",
    context="The default chunk_size is 512 words.",
    question_type=QuestionType.FACTUAL,
    metadata={"chunk_index": 0},
)

d = pair.to_dict()
# {
#   "question": "What is the chunk size default?",
#   "answer": "512 words.",
#   "context": "The default chunk_size is 512 words.",
#   "question_type": "factual",
#   "chunk_index": 0,
# }
```

---

### SyntheticQAGenerator

```python
from mltk.domains.llm.synthetic import SyntheticQAGenerator

class SyntheticQAGenerator:
    def __init__(
        self,
        llm_fn: Callable[[str], str] | None = None,
        question_types: list[QuestionType] | None = None,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        min_chunk_words: int = 30,
        quality_filter: bool = True,
        quality_threshold: float = 0.6,
        max_retries: int = 1,
        seed: int | None = None,
    ) -> None: ...
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `llm_fn` | `Callable[[str], str] \| None` | `None` | LLM callable. `None` activates template mode. |
| `question_types` | `list[QuestionType] \| None` | All 5 types | Question types to generate. Restrict to subset for targeted coverage. |
| `chunk_size` | `int` | `512` | Word count per chunk when splitting raw text. |
| `chunk_overlap` | `int` | `50` | Overlapping words between consecutive chunks. Preserves sentence boundaries. |
| `min_chunk_words` | `int` | `30` | Minimum words a chunk must have to be used. Skips headers and fragments. |
| `quality_filter` | `bool` | `True` | When `True` and `llm_fn` is set, score each pair and discard below-threshold results. No-op in template mode. |
| `quality_threshold` | `float` | `0.6` | Minimum score (0–1) for both self-containment and answerability. |
| `max_retries` | `int` | `1` | LLM retries per pair on quality failure. |
| `seed` | `int \| None` | `None` | Random seed for reproducible chunk selection. Set in CI for deterministic output. |

---

#### generate_from_text

```python
def generate_from_text(
    self,
    text: str,
    n: int = 10,
    question_types: list[QuestionType] | None = None,
) -> list[QAPair]:
```

Split `text` into chunks using the configured `chunk_size`
and `chunk_overlap`, then generate `n` QA pairs. The
`question_types` argument overrides the instance-level
setting for this call only.

The internal splitter is a zero-dependency word-count
splitter (see `split_text` below). For token-level
splitting, pass pre-split chunks to `generate_from_chunks`
instead.

Returns `list[QAPair]`. May return fewer than `n` pairs
if the text is too short, too many chunks fail the
minimum-words check, or quality filtering discards
generated pairs.

```python
gen = SyntheticQAGenerator(seed=42)
pairs = gen.generate_from_text(
    text=my_document,
    n=20,
    question_types=[QuestionType.FACTUAL, QuestionType.REASONING],
)
```

---

#### generate_from_chunks

```python
def generate_from_chunks(
    self,
    chunks: list[str],
    n: int = 10,
    question_types: list[QuestionType] | None = None,
) -> list[QAPair]:
```

Generate `n` QA pairs from pre-split chunks. Use this
when you have already chunked documents using LlamaIndex,
LangChain, or a custom tokenizer and want consistent
chunking with your retrieval pipeline.

Chunks shorter than `min_chunk_words` are silently
skipped.

```python
# Use the same chunks as your retriever for exact coverage
gen = SyntheticQAGenerator(llm_fn=my_llm)
pairs = gen.generate_from_chunks(
    chunks=retriever.get_all_chunks(),
    n=50,
)
```

---

#### generate_one

```python
def generate_one(
    self,
    chunk: str,
    question_type: QuestionType = QuestionType.FACTUAL,
    context_chunks: list[str] | None = None,
) -> QAPair | None:
```

Generate a single QA pair from one chunk. Returns `None`
if the chunk fails the minimum-words check or quality
filtering discards the result after `max_retries`.

For `MULTI_HOP`, pass a second chunk in `context_chunks`.
The first element of the list is used as Context B; the
`chunk` argument is Context A.

```python
pair = gen.generate_one(
    chunk=chunks[0],
    question_type=QuestionType.MULTI_HOP,
    context_chunks=[chunks[1]],
)
if pair is not None:
    assert_faithfulness(pair.answer, pair.context)
```

---

#### to_dataframe

```python
def to_dataframe(self, pairs: list[QAPair]):
    """Convert QAPairs to a pandas DataFrame.

    Requires pandas (pip install pandas).
    Calls pair.to_dict() for each pair.
    """
```

Convenience method for export to CSV, parquet, or
further pandas analysis. Raises `ImportError` if
`pandas` is not installed.

---

#### to_jsonl

```python
def to_jsonl(
    self,
    pairs: list[QAPair],
    path: str,
) -> None:
    """Write QAPairs to a JSONL file (one JSON object
    per line). No extra dependencies required.
    """
```

```python
gen = SyntheticQAGenerator(llm_fn=my_llm)
pairs = gen.generate_from_text(document, n=100)
gen.to_jsonl(pairs, "test_fixtures/qa_pairs.jsonl")
```

---

### split_text

```python
from mltk.domains.llm.synthetic import split_text

def split_text(
    text: str,
    chunk_size: int = 512,
    overlap: int = 50,
) -> list[str]:
```

Zero-dependency word-count text splitter. Splits `text`
into chunks of approximately `chunk_size` words with
`overlap` words of overlap between consecutive chunks.

Use this helper directly when you need chunked output
outside of `SyntheticQAGenerator`, or to inspect how the
generator will split a document before generating.

```python
from mltk.domains.llm.synthetic import split_text

chunks = split_text(document, chunk_size=256, overlap=25)
print(f"{len(chunks)} chunks")
```

**Note:** This is a word-count splitter, not a token
splitter. At chunk_size=512, a chunk is approximately
350–450 tokens for standard English prose (0.7–0.9
tokens per word ratio). For models with a 512-token
context window, use `chunk_size=350` as a conservative
setting.

RAGAS uses 1000 tokens and DeepEval uses 1024 tokens by
default, but those defaults are tuned for retrieval, not
question generation. Smaller chunks produce more focused,
answerable questions. 512 words (default) is recommended
for question generation.

**Citation:** Rationale and comparison in
`docs/research/synthetic-data-gen-research.md`,
Section 4: Document Processing Design.

---

### QualityFilter

The quality filter runs in two stages. Stage 1 checks
chunk quality before generation. Stage 2 checks golden
quality after generation (LLM mode only).

**Stage 1 — chunk quality (template and LLM mode):**

- Length: chunk must have ≥ `min_chunk_words` words.
  Rejects headers, footnotes, and list fragments.
- Non-repetition: adjacent word repetition ratio must
  be < 0.3. Rejects auto-generated boilerplate with
  repeated phrasing.

**Stage 2 — golden quality (LLM mode only):**

After generating a QA pair, the filter sends a scoring
prompt to `llm_fn` asking for two scores:

```
Rate this question-answer pair (score 0.0 to 1.0 each):

1. Self-containment: Can the question be understood
   without reading the source document?
2. Answerability: Is the answer clearly and completely
   derivable from the given context?

Question: {question}
Context: {context}
Answer: {answer}

Return JSON:
{"self_containment": float, "answerability": float}
```

Both scores must be ≥ `quality_threshold` (default 0.6).
If either fails, the generator retries the generation
prompt once (`max_retries=1`). If the second attempt
also fails, the pair is discarded. The discarded count
is logged at `DEBUG` level.

In template mode, Stage 2 is skipped. Template questions
are guaranteed to be answerable by construction (the
answer is derived from the same chunk pattern), and
self-containment is enforced by template design (each
question includes the topic noun from the chunk).

**Design note:** DeepEval uses a two-stage quality filter
with up to 3 retries and a 0.5 threshold. mltk uses 1
retry and a 0.6 threshold -- stricter on quality, fewer
LLM calls per pair. This is the right tradeoff for
CI-adjacent usage where API cost matters.

**Citation:** DeepEval Synthesizer two-stage quality
filter (deepeval.com/docs/synthesizer-generate-from-docs,
2025 stable).

---

## Integration with mltk Assertions

`QAPair` feeds directly into mltk's existing assertion
library. This is mltk's key differentiator: generation
and evaluation are in the same library, with the same
API style, no framework switching.

### Full RAG pipeline test

```python
from mltk.domains.llm.synthetic import (
    SyntheticQAGenerator,
    QuestionType,
)
from mltk.domains.llm.rag import (
    assert_faithfulness,
    assert_answer_relevancy,
)
from mltk.domains.llm.safety import assert_no_hallucination

gen = SyntheticQAGenerator(llm_fn=my_llm)
pairs = gen.generate_from_text(my_kb_text, n=20)


def test_rag_pipeline_with_synthetic_data():
    for pair in pairs:
        # Evaluate the RAG system under test
        retrieved_context, rag_answer = (
            my_rag_pipeline(pair.question)
        )

        # Faithfulness: answer grounded in retrieved context
        assert_faithfulness(
            rag_answer,
            retrieved_context,
            min_score=0.7,
        )

        # Relevancy: answer addresses the question
        assert_answer_relevancy(
            pair.question,
            rag_answer,
            retrieved_context,
            min_score=0.6,
        )

        # Hallucination: answer consistent with reference
        assert_no_hallucination(rag_answer, pair.answer)
```

### Out-of-scope refusal test

```python
from mltk.domains.llm.synthetic import (
    SyntheticQAGenerator,
    QuestionType,
)

gen = SyntheticQAGenerator(
    llm_fn=my_llm,
    question_types=[QuestionType.OUT_OF_SCOPE],
)
oos_pairs = gen.generate_from_text(my_kb_text, n=10)


def test_rag_refuses_out_of_scope():
    for pair in oos_pairs:
        _, answer = my_rag_pipeline(pair.question)
        # The RAG should refuse, not hallucinate
        assert_no_hallucination(answer, pair.context)
```

### Template mode in CI (no LLM, zero cost)

```python
import pytest
from mltk.domains.llm.synthetic import SyntheticQAGenerator

# Fixtures generated once, deterministic with seed
@pytest.fixture(scope="session")
def qa_fixtures():
    gen = SyntheticQAGenerator(seed=42)
    return gen.generate_from_text(REFERENCE_DOC, n=30)


def test_factual_regression(qa_fixtures):
    for pair in qa_fixtures:
        answer = my_rag_pipeline(pair.question)[1]
        # No LLM needed for this assertion either
        assert pair.answer.lower() in answer.lower(), (
            f"Expected '{pair.answer}' in response"
        )
```

---

## v2 Features

S77 adds three new generation methods and two new
`QuestionType` values. All three methods require
`llm_fn` -- there is no template-mode equivalent for
multi-hop chains, conversational sequences, or
distraction pairs.

### generate_multi_hop

Generate a question that requires combining information
from two distinct context chunks. The answer cannot be
derived from either chunk alone.

```python
def generate_multi_hop(
    self,
    chunk_a: str,
    chunk_b: str,
) -> QAPair | None:
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `chunk_a` | `str` | First context chunk (Context A). |
| `chunk_b` | `str` | Second context chunk (Context B). |

Returns a `QAPair` with `question_type=MULTI_HOP` and
`context=[chunk_a, chunk_b]`. Returns `None` if either
chunk fails the minimum-words check or quality filtering
discards the result after `max_retries`.

Use this method when you need explicit control over
which two chunks are combined. For automatic multi-hop
generation across a document, use `generate_from_text`
with `question_types=[QuestionType.MULTI_HOP]` -- the
generator pairs sequential neighbors internally.

```python
gen = SyntheticQAGenerator(llm_fn=my_llm)

pair = gen.generate_multi_hop(
    chunk_a=chunks[0],   # "v2.1 introduced caching."
    chunk_b=chunks[7],   # "The known limitation is
                         #  high memory on cold starts."
)
# question: "Which feature introduced in v2.1
#            addresses the cold-start limitation?"
# context: [chunks[0], chunks[7]]
# question_type: QuestionType.MULTI_HOP

if pair is not None:
    assert_faithfulness(pair.answer, pair.context)
```

**Why it matters:** Multi-hop questions are the most
reliable predictor of retrieval pipeline quality. A
RAG system that retrieves the right single chunk but
fails to synthesize across two chunks will answer
multi-hop questions incorrectly. RAGAS and DeepEval
both support multi-hop generation but require LangChain
or a subclassed LLM wrapper. mltk requires a callable.

**Citation:** Mavi et al. (arXiv 2022, 2204.09140) --
multi-hop question answering benchmark survey.
Yang et al. 2018 (HotpotQA) -- established multi-hop
as the primary benchmark for cross-document reasoning.

---

### generate_conversational

Generate a sequence of questions that form a natural
dialogue about the source document. Each turn in the
sequence depends on the previous answer, simulating a
real user exploring a knowledge base.

```python
def generate_conversational(
    self,
    chunk: str,
    turns: int = 3,
) -> list[QAPair]:
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `chunk` | `str` | required | Source chunk for the conversation. |
| `turns` | `int` | `3` | Number of question-answer turns to generate. |

Returns `list[QAPair]` with `question_type=CONVERSATIONAL`
for each pair. The `metadata` field of each pair
contains `{"turn": int, "conversation_id": str}` so
the full dialogue sequence can be reconstructed.

Returns an empty list if the chunk fails minimum-words
or all turns are discarded by quality filtering.

```python
gen = SyntheticQAGenerator(llm_fn=my_llm)

turns = gen.generate_conversational(
    chunk=support_article,
    turns=4,
)
# turn 0: "What does this API endpoint do?"
# turn 1: "What parameters does it accept?"
# turn 2: "What happens if the required param is
#           missing?"
# turn 3: "Is there a rate limit on this endpoint?"

for pair in turns:
    context, answer = my_rag_pipeline(pair.question)
    assert_faithfulness(answer, context)
```

**Why it matters:** Production RAG systems are not
evaluated on isolated questions -- users follow up,
refine, and explore. Conversational goldens test
whether the RAG pipeline degrades across a session.
DeepEval added conversational support in 2025; RAGAS
does not support it in the base library. mltk's
implementation requires no framework wrapper.

**Citation:** Adlakha et al. 2022 (QReCC) --
conversational question answering benchmark. Conv-MIX
dataset (2023) -- established multi-turn evaluation as
the standard for deployed RAG systems.

---

### generate_distracting

Generate a question where a plausible but incorrect
answer can be constructed from a *different* chunk --
a distractor chunk. Tests whether the RAG system
retrieves the right chunk rather than the superficially
similar wrong one.

```python
def generate_distracting(
    self,
    chunk: str,
    distractor_chunk: str,
) -> QAPair | None:
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `chunk` | `str` | The correct source chunk. The `answer` is derived from this chunk. |
| `distractor_chunk` | `str` | A chunk on a related topic. The question is phrased so the distractor looks relevant. |

Returns a `QAPair` with `question_type=DISTRACTING` and
`context=chunk` (the correct context only). The
`metadata` field contains `{"distractor": distractor_chunk}`
so test code can verify which chunk the retriever
selected.

Returns `None` if either chunk fails minimum-words or
quality filtering discards the pair.

```python
gen = SyntheticQAGenerator(llm_fn=my_llm)

# chunk: "The /auth endpoint requires Bearer tokens."
# distractor: "The /health endpoint has no auth."
pair = gen.generate_distracting(
    chunk=auth_chunk,
    distractor_chunk=health_chunk,
)
# question: "Does the API require authentication?"
# answer: "Yes -- Bearer token required."
# context: auth_chunk  (the correct source)
# metadata["distractor"]: health_chunk

if pair is not None:
    retrieved, answer = my_rag_pipeline(pair.question)
    # The retriever should return auth_chunk, not
    # health_chunk
    assert_faithfulness(answer, retrieved)
    assert_answer_relevancy(
        pair.question, answer, retrieved,
    )
```

**Why it matters:** Distractor questions expose
retrieval failures that factual questions miss. A
retriever that scores chunks by superficial term
overlap will prefer the distractor when it shares
more keywords with the question. No other open-source
generator produces distractor pairs in the base
library -- this is a first-mover capability for mltk.

**Citation:** Giskard RAGET (2024) -- notes distractor
chunks as a known retrieval failure mode but does not
generate them in the OSS library. Jia & Liang 2017
(adversarial SQuAD) -- established distractor insertion
as the canonical probe for retrieval robustness.

---

### New QuestionType Values

| Value | String | Added | Description |
|-------|--------|-------|-------------|
| `QuestionType.CONVERSATIONAL` | `"conversational"` | S77 | Turn in a multi-turn dialogue sequence. Used by `generate_conversational`. |
| `QuestionType.DISTRACTING` | `"distracting"` | S77 | Question where a plausible wrong answer exists in a distractor chunk. Used by `generate_distracting`. |

Both values are valid in `question_types` filters:

```python
# Generate only conversational and distracting pairs
gen = SyntheticQAGenerator(
    llm_fn=my_llm,
    question_types=[
        QuestionType.CONVERSATIONAL,
        QuestionType.DISTRACTING,
    ],
)
```

Note: `generate_from_text` and `generate_from_chunks`
dispatch to `generate_conversational` and
`generate_distracting` internally when these types
appear in `question_types`. For `DISTRACTING`, the
generator uses the chunk's sequential neighbor as the
distractor; for explicit distractor control, call
`generate_distracting` directly.

---

## Competitive Comparison

| Feature | mltk | RAGAS | DeepEval | Giskard |
|---------|:----:|:-----:|:--------:|:-------:|
| Zero-dep template mode | **Yes** | No | No | No |
| LLM coupling | `Callable[[str], str]` | `LangchainLLMWrapper` | `DeepEvalBaseLLM` subclass | Giskard LLM client |
| Out-of-scope questions (OSS) | **Yes** | No | No | Weak |
| pytest-native output | **Yes** | No | No | No |
| Direct assertion integration | **Yes** | No | No | No |
| Quality filter | Two-stage, 1 retry | Implicit | Two-stage, 3 retries | Implicit |
| Default chunk size | 512 words | 1000 tokens | 1024 tokens | Unspecified |
| Multi-hop questions | Yes | Yes | Yes (evolution) | Yes |
| Counterfactual questions | Yes | No | Yes (evolution) | No |
| Conversational goldens | **Yes (S77)** | No (v0.2.x) | Yes (2025) | Yes |
| Distracting questions (OSS) | **Yes (S77)** | No | No | No |

**Zero-dep template mode** is mltk's strongest
differentiator. All three competitors require an LLM API
call to generate anything. A CI pipeline that cannot
afford API calls or cannot reach external services gets
nothing from RAGAS, DeepEval, or Giskard. mltk generates
deterministic, reproducible QA pairs with no network
access.

**LLM coupling** is the second differentiator. RAGAS
requires subclassing `BaseRagasLLM` or wrapping a
LangChain LLM. DeepEval requires subclassing
`DeepEvalBaseLLM` with async methods. Giskard requires
its own client interface. mltk requires a function.
If you can write `lambda p: model(p)`, you can use
mltk's LLM mode with any backend -- including local
models via Ollama, which aligns with mltk's benchmark
standard of never requiring paid APIs.

**Out-of-scope questions** test a failure mode that the
other generators ignore: what happens when the user asks
something the knowledge base cannot answer? A RAG system
should refuse or route, not hallucinate. mltk generates
OOS questions in the base library. Giskard's Hub product
generates them in the commercial tier; the OSS library
treatment is weaker.

**pytest-native output** means the `list[QAPair]` returned
by `generate_from_text` can be used directly in a pytest
fixture, parametrize decorator, or test body. No
conversion, no additional dependencies.

**Citation:** RAGAS TestsetGenerator (docs.ragas.io/en/
stable/getstarted/rag_testset_generation/). DeepEval
Synthesizer (deepeval.com/docs/synthesizer-introduction).
Giskard RAGET (docs.giskard.ai/en/latest/open_source/
testset_generation/). Competitor analysis in
`docs/research/synthetic-data-gen-research.md`,
Section 1.

---

## Configuration Reference

All parameters are set at construction time and apply
to every call on the instance. Per-call overrides for
`question_types` and `n` are available on
`generate_from_text` and `generate_from_chunks`.

| Parameter | Default | Notes |
|-----------|---------|-------|
| `chunk_size` | `512` | Words per chunk. Lower = more focused questions. |
| `chunk_overlap` | `50` | Overlapping words. Preserves sentence boundaries. |
| `min_chunk_words` | `30` | Chunk quality gate. Rejects fragments. |
| `quality_filter` | `True` | Enable Stage 2 scoring (LLM mode only). |
| `quality_threshold` | `0.6` | Min score for self-containment + answerability. |
| `max_retries` | `1` | Retries per pair on quality failure. |
| `seed` | `None` | Set for deterministic output in CI. |
| `question_types` | All 5 | Restrict to subset for targeted coverage. |

### Recommended profiles

```python
# CI smoke test -- fast, deterministic, zero deps
ci_gen = SyntheticQAGenerator(
    seed=42,
    question_types=[QuestionType.FACTUAL],
    min_chunk_words=50,
)

# Release evaluation -- high quality, diverse, all types
release_gen = SyntheticQAGenerator(
    llm_fn=my_llm,
    quality_threshold=0.7,
    max_retries=1,
    seed=None,
)

# Refusal/routing testing -- OOS only
oos_gen = SyntheticQAGenerator(
    llm_fn=my_llm,
    question_types=[QuestionType.OUT_OF_SCOPE],
)

# Multi-hop coverage -- factual + multi-hop only
multihop_gen = SyntheticQAGenerator(
    llm_fn=my_llm,
    question_types=[
        QuestionType.FACTUAL,
        QuestionType.MULTI_HOP,
    ],
    chunk_overlap=100,  # wider overlap for multi-hop
)
```

---

## CLI

```bash
# Template mode -- zero dep, deterministic
mltk generate --input docs/manual.txt --n 20 \
  --output pairs.jsonl

# LLM mode via local Ollama
mltk generate --input docs/manual.txt --n 50 \
  --llm ollama:llama3.2

# Specific question types
mltk generate --input docs/manual.txt --n 30 \
  --types factual,reasoning,out_of_scope

# From a directory of documents
mltk generate --input-dir docs/ --n 100 \
  --types factual,reasoning

# Round-trip: generate then evaluate
mltk generate --input docs/manual.txt \
  | mltk eval --assertion faithfulness
```

The `--llm` flag accepts `ollama:<model>` for local
models, or set `MLTK_LLM_ENDPOINT` with a custom
REST endpoint and `--llm endpoint:<model>` for any
hosted API. Template mode is the default when `--llm`
is omitted.

---

## Module Structure

```
src/mltk/domains/llm/
└── synthetic/
    ├── __init__.py      # SyntheticQAGenerator, QAPair,
    │                    # QuestionType, split_text
    ├── generator.py     # SyntheticQAGenerator class
    ├── _templates.py    # Prompt templates per
    │                    # QuestionType (5 types,
    │                    # 3 fallback templates each)
    ├── _splitter.py     # split_text() zero-dep
    │                    # word-count splitter
    └── _quality.py      # QualityFilter -- chunk stage
                         # + golden stage (LLM scorer)
```

---

## Research Citations

**RAGAS TestsetGenerator** (v0.2.x, current stable):
Knowledge-graph-based architecture. Nodes = document
chunks + extracted entities + keyphrases. Edges =
semantic similarity links. Three synthesizer types:
SingleHopSpecific, MultiHopAbstract, MultiHopSpecific.
Requires `LangchainLLMWrapper` and an embedding model.
docs.ragas.io/en/stable/getstarted/rag_testset_generation

**RAGAS v0.1.21 testset generation concepts:**
Earlier evolution paradigm: Simple, Reasoning,
Conditioning, Multi-context types. Explicit critique and
revision steps.
docs.ragas.io/en/v0.1.21/concepts/testset_generation.html

**DeepEval Synthesizer** (2025 stable):
Chunk → embed → cosine-similar group → LLM generate →
two-stage quality filter → optional evolution. Seven
evolution types: REASONING, MULTICONTEXT, CONCRETIZING,
CONSTRAINED, COMPARATIVE, HYPOTHETICAL, IN_BREADTH.
Two-stage quality filter: context clarity/depth/
structure/relevance, then golden self-containment/
clarity. deepeval.com/docs/synthesizer-introduction

**Giskard RAGET** (OSS, 2024):
Topic-clustering knowledge base → LLM question
generation → perturbation. Seven question types
including `oos_questions` (unique: tests refusal). OOS
questions test out-of-scope routing in the RAG pipeline.
docs.giskard.ai/en/latest/open_source/testset_generation

**Automatic Dataset Generation for Knowledge Intensive
QA** (arXiv 2025, 2505.14212): Systematic evaluation
of LLM-generated QA datasets for knowledge-intensive
tasks. Establishes that smaller generation units
(focused contexts) produce higher-answerability
questions.

**Synthetic Data Generation Using LLMs: Advances in
Text and Code** (arXiv 2025, 2503.14023): Survey of
LLM-based synthetic data generation methods. Section 3
covers QA pair generation patterns and quality filtering
approaches.

**Tiny QA Benchmark++** (arXiv 2025, 2505.12058):
Ultra-lightweight synthetic multilingual dataset
generation. Demonstrates that template-based generation
can achieve surprisingly high answerability rates
(89%+) when templates are grounded in the source
chunk -- supporting mltk's template mode design.

**Confident AI: The Definitive Guide to Synthetic Data
Generation Using LLMs** (blog, 2025):
confident-ai.com/blog/the-definitive-guide-to-synthetic-
data-generation-using-llms

**Full competitor analysis and rejected alternatives:**
`docs/research/synthetic-data-gen-research.md`,
Sections 1–3 and Section 9.
