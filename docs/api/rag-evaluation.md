# RAG, Agentic & Text Quality Evaluation

Lightweight assertions for Retrieval-Augmented Generation (RAG) pipelines, agentic tool-use workflows, and general LLM output quality. No external model dependencies — all checks run in-process using keyword overlap and regex heuristics.

**Modules:** `mltk.domains.llm`, `mltk.training`

---

## RAG Evaluation

RAG systems combine a retriever (context fetch) and a generator (LLM). Failures can occur at either stage. mltk covers the five canonical RAGAS metrics using a keyword-overlap approximation that runs without an LLM judge.

### Metrics Overview

| Metric | What it measures | Failure signal |
|--------|-----------------|----------------|
| **Faithfulness** | Does the answer contain only claims supported by the retrieved context? | Hallucinated facts not grounded in context |
| **Context Relevancy** | Are the retrieved chunks relevant to the question? | Retriever returning off-topic passages |
| **Answer Relevancy** | Does the answer address the question asked? | LLM ignoring the question, providing generic output |
| **Context Precision** | What fraction of retrieved chunks are actually useful? | Low signal-to-noise in retrieval (too many irrelevant chunks) |
| **Context Recall** | Does the retrieved context cover the expected answer? | Retriever missing key passages |

### assert_faithfulness

Verify the answer does not introduce claims absent from the retrieved context.

```python
from mltk.domains.llm import assert_faithfulness

answer = "The Eiffel Tower is 330 metres tall and located in Paris."
context = [
    "The Eiffel Tower stands 330 metres high.",
    "It is situated in Paris, France.",
]
assert_faithfulness(answer, context, min_score=0.5)
```

#### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `answer` | `str` | *(required)* | LLM-generated answer |
| `context` | `list[str]` | *(required)* | Retrieved context chunks |
| `min_score` | `float` | `0.5` | Minimum keyword-overlap ratio required |

---

### assert_context_relevancy

Verify retrieved context chunks are relevant to the question.

```python
from mltk.domains.llm import assert_context_relevancy

assert_context_relevancy(
    question="What is the capital of France?",
    context=["Paris is the capital of France.", "France is in Western Europe."],
    min_score=0.3,
)
```

#### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `question` | `str` | *(required)* | The user question |
| `context` | `list[str]` | *(required)* | Retrieved context chunks |
| `min_score` | `float` | `0.3` | Minimum overlap ratio per chunk |

---

### assert_answer_relevancy

Verify the answer actually addresses the question.

```python
from mltk.domains.llm import assert_answer_relevancy

assert_answer_relevancy(
    question="What is the speed of light?",
    answer="The speed of light is approximately 299,792,458 metres per second.",
    min_score=0.3,
)
```

#### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `question` | `str` | *(required)* | The user question |
| `answer` | `str` | *(required)* | LLM-generated answer |
| `min_score` | `float` | `0.3` | Minimum keyword overlap between question and answer |

---

### assert_context_precision

Verify the fraction of retrieved chunks that are useful (high precision = less noise).

```python
from mltk.domains.llm import assert_context_precision

assert_context_precision(
    question="Who invented the telephone?",
    context=[
        "Alexander Graham Bell is credited with inventing the telephone.",
        "Bell was born in Edinburgh, Scotland.",
        "The history of pasta in Italy dates to the 13th century.",  # irrelevant
    ],
    min_precision=0.6,
)
```

#### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `question` | `str` | *(required)* | The user question |
| `context` | `list[str]` | *(required)* | Retrieved context chunks |
| `min_precision` | `float` | `0.5` | Minimum fraction of chunks deemed relevant |

---

### assert_context_recall

Verify the retrieved context covers the expected answer (high recall = nothing important missed).

```python
from mltk.domains.llm import assert_context_recall

assert_context_recall(
    expected_answer="Alexander Graham Bell invented the telephone in 1876.",
    context=[
        "Bell patented the telephone on March 7, 1876.",
        "He demonstrated it at the Centennial Exposition.",
    ],
    min_recall=0.4,
)
```

#### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `expected_answer` | `str` | *(required)* | Ground-truth or reference answer |
| `context` | `list[str]` | *(required)* | Retrieved context chunks |
| `min_recall` | `float` | `0.4` | Minimum keyword recall ratio |

---

## Agentic Evaluation

Evaluate LLM agents that select and invoke tools. These assertions check planning quality and tool-use correctness without requiring live tool execution.

### assert_task_completion

Verify that an agent's final output satisfies the original task description.

```python
from mltk.domains.llm import assert_task_completion

assert_task_completion(
    task="Summarise the key points of the document.",
    output="The document covers three main topics: climate change, renewable energy, and policy.",
    min_score=0.3,
)
```

#### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `task` | `str` | *(required)* | Original task or user instruction |
| `output` | `str` | *(required)* | Agent's final output |
| `min_score` | `float` | `0.3` | Minimum keyword overlap between task and output |

---

### assert_tool_selection

Verify the agent chose the correct tool for a given step.

```python
from mltk.domains.llm import assert_tool_selection

assert_tool_selection(
    step_description="Search the web for recent news about AI regulation.",
    selected_tool="web_search",
    expected_tool="web_search",
)
```

#### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `step_description` | `str` | *(required)* | Natural-language description of the step |
| `selected_tool` | `str` | *(required)* | Tool name chosen by the agent |
| `expected_tool` | `str` | *(required)* | Correct tool name |

---

### assert_tool_call_correctness

Verify that tool call arguments match expected arguments within a tolerance.

```python
from mltk.domains.llm import assert_tool_call_correctness

assert_tool_call_correctness(
    tool_name="calculator",
    actual_args={"a": 10, "b": 5, "op": "add"},
    expected_args={"a": 10, "b": 5, "op": "add"},
)
```

#### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `tool_name` | `str` | *(required)* | Name of the tool being called |
| `actual_args` | `dict` | *(required)* | Arguments the agent passed |
| `expected_args` | `dict` | *(required)* | Correct/expected arguments |

---

## Text Quality

General-purpose assertions for LLM output quality that do not require a judge model.

### assert_text_length

Verify word count is within expected bounds.

```python
from mltk.domains.llm import assert_text_length

# Enforce a 50–200 word response
assert_text_length(response, min_words=50, max_words=200)

# Enforce only a minimum (no upper cap)
assert_text_length(response, min_words=10)
```

#### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `text` | `str` | *(required)* | Text to evaluate |
| `min_words` | `int \| None` | `None` | Minimum word count (inclusive) |
| `max_words` | `int \| None` | `None` | Maximum word count (inclusive) |

!!! note
    At least one of `min_words` or `max_words` must be provided.

#### Returns

`TestResult` with details:

- `word_count` — actual word count
- `min_words`, `max_words` — bounds used

---

### assert_output_format

Verify the output matches a regex pattern. Useful for enforcing structured output contracts.

```python
from mltk.domains.llm import assert_output_format

# Check output is a JSON object
assert_output_format(response, pattern=r"^\{.*\}$", description="JSON object")

# Check output starts with a specific prefix
assert_output_format(response, pattern=r"^ANSWER:", description="answer prefix")

# Check output is a valid ISO date
assert_output_format(response, pattern=r"^\d{4}-\d{2}-\d{2}$", description="ISO date")
```

#### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `text` | `str` | *(required)* | Text to evaluate |
| `pattern` | `str` | *(required)* | Regex pattern (applied with `re.search`, `re.DOTALL`) |
| `description` | `str \| None` | `None` | Human-readable label shown in failure messages |

#### Returns

`TestResult` with details:

- `pattern` — the regex used
- `description` — label for the pattern
- `text_preview` — first 100 chars of input (for diagnostics)

---

### assert_readability

Verify text is readable at or below a target grade level using the Flesch-Kincaid formula.

```python
from mltk.domains.llm import assert_readability

# Customer-facing copy: readable at 8th-grade level
assert_readability(response, max_grade_level=8.0)

# Technical documentation: allow up to 14th grade
assert_readability(response, max_grade_level=14.0)
```

**Grade level reference:**

| Grade | Audience |
|-------|----------|
| ≤ 6 | Middle school, mass-market |
| 8 | 8th grade, general consumer |
| 12 | High school senior |
| 14 | College undergrad |
| 16+ | Graduate / academic |

#### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `text` | `str` | *(required)* | Text to evaluate |
| `max_grade_level` | `float` | `12.0` | Maximum allowed Flesch-Kincaid grade level |

#### Returns

`TestResult` with details:

- `grade_level` — computed FK grade level (rounded to 2 d.p.)
- `total_words`, `total_sentences`, `total_syllables` — intermediate counts
- `max_grade_level` — threshold used

!!! tip "FK Grade Level Formula"
    ```
    FK = 0.39 × (words / sentences) + 11.8 × (syllables / words) − 15.59
    ```
    Syllable count uses a vowel-group heuristic — fast and dependency-free.

---

## Training-Serving Skew

Catches feature engineering differences between the training code path and the serving (inference) code path — the silent cause of production accuracy drops.

**Module:** `mltk.training`

### assert_no_training_serving_skew

```python
from mltk.training import assert_no_training_serving_skew
import numpy as np

# Run the same raw input through both pipelines
raw_sample = {"age": 32, "income": 75000, "credit_score": 720}
train_features = training_pipeline.transform(raw_sample)
serve_features = serving_pipeline.transform(raw_sample)

assert_no_training_serving_skew(train_features, serve_features, tolerance=0.01)
```

#### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `train_output` | `np.ndarray \| list` | *(required)* | Feature vector from the training pipeline |
| `serve_output` | `np.ndarray \| list` | *(required)* | Feature vector from the serving pipeline |
| `tolerance` | `float` | `0.01` | Maximum allowed element-wise absolute difference |

#### Returns

`TestResult` with details:

- `max_diff` — largest element-wise absolute difference
- `mean_diff` — mean absolute difference across all elements
- `num_skewed` — count of elements exceeding tolerance
- `num_elements` — total element count compared
- `tolerance` — threshold used

#### Common causes of skew

| Root cause | Symptom |
|------------|---------|
| Different imputation constants | `max_diff` is systematic across all samples |
| Different normalisation (mean/std) | `mean_diff` is non-zero, `max_diff` is large |
| Missing feature in serving pipeline | Shape mismatch — assertion fails immediately |
| Float32 vs Float64 | `max_diff` is tiny but non-zero — use `tolerance=1e-4` |
| Off-by-one in rolling window | `max_diff` appears only at boundaries |

!!! warning "Severity: CRITICAL"
    Training-serving skew fails with `Severity.CRITICAL`. It is the most common
    cause of unexplained production accuracy drops and must be caught before every deployment.

---

## Full RAG Evaluation Pipeline

A production RAG system has three stages that can fail independently: **retrieval**, **generation**, and **end-to-end quality**. Testing only one stage gives a false sense of coverage. mltk provides assertions for every layer so you can pinpoint exactly where failures originate.

### Pipeline Layers

| Layer | What fails | mltk assertions | Docs |
|-------|-----------|-----------------|------|
| **Retriever ranking** | Wrong documents retrieved, poor ordering | `assert_ndcg`, `assert_mrr`, `assert_recall_at_k`, `assert_map_at_k` | [Retrieval Metrics](retrieval-metrics.md) |
| **Generator faithfulness** | Hallucinated facts not grounded in context | `assert_faithfulness`, `assert_summary_faithfulness` | [RAG Evaluation](#rag-evaluation), [Summarization Metrics](summarization-metrics.md) |
| **Generator quality** | Output is correct but unhelpful, incoherent, or poorly written | `assert_llm_judge_score`, `assert_llm_judge_pairwise` | [LLM-as-Judge](llm-judge.md) |
| **End-to-end composite** | Aggregate RAG quality below acceptable threshold | `assert_ragas_score` | [RAG Evaluation](#rag-evaluation) |
| **Summarization fidelity** | Summary misses key info, is too verbose, or fabricates claims | `assert_summary_faithfulness`, `assert_summary_coverage`, `assert_summary_conciseness` | [Summarization Metrics](summarization-metrics.md) |

### Why test every layer?

A retriever that returns perfect documents can still produce bad answers if the generator hallucinates. A generator that is perfectly faithful can still produce poor answers if the retriever missed key documents. End-to-end metrics can mask which component is the root cause. Layer-by-layer testing isolates the failure:

```
Retriever broken + Generator OK    -> assert_ndcg FAILS, assert_faithfulness PASSES
Retriever OK     + Generator broken -> assert_ndcg PASSES, assert_faithfulness FAILS
Both broken                         -> both FAIL, but assert_ndcg pinpoints retriever
```

### Complete pytest example

This example tests all layers of a RAG pipeline in a single test module. Each test function targets one layer, so CI failures immediately tell you which component regressed.

```python
"""tests/test_rag_pipeline.py — Full RAG evaluation across all layers."""
import pytest
from mltk.domains.llm import (
    # Retriever ranking
    assert_ndcg,
    assert_mrr,
    assert_recall_at_k,
    assert_map_at_k,
    # Generator faithfulness (keyword-overlap, no LLM needed)
    assert_faithfulness,
    assert_context_relevancy,
    assert_answer_relevancy,
    # End-to-end composite
    assert_ragas_score,
    # LLM-as-Judge (subjective quality)
    assert_llm_judge_score,
    # Summarization faithfulness
    assert_summary_faithfulness,
    assert_summary_coverage,
)


# ---- Fixtures ----

@pytest.fixture
def rag_sample():
    """A single RAG evaluation sample with all required fields."""
    return {
        "question": "What year was the Eiffel Tower completed?",
        "context": [
            "The Eiffel Tower was completed in 1889 for the World's Fair.",
            "Gustave Eiffel's engineering company designed the structure.",
            "The tower stands 330 metres tall in Paris, France.",
        ],
        "answer": "The Eiffel Tower was completed in 1889.",
        "expected_answer": "The Eiffel Tower was completed in 1889 for the World's Fair in Paris.",
    }


@pytest.fixture
def retrieval_sample():
    """Retrieval ranking evaluation data."""
    return {
        # relevance_labels[i] = relevance grade for document at rank i
        # 2 = highly relevant, 1 = partially relevant, 0 = irrelevant
        "relevance_labels": [[2, 1, 0, 0, 1]],
        "relevant_ids": [{"doc_1", "doc_2", "doc_5"}],
        "retrieved_ids": [["doc_1", "doc_2", "doc_3", "doc_4", "doc_5"]],
    }


@pytest.fixture
def judge_fn():
    """LLM judge function. Replace with your preferred provider."""
    def _judge(prompt: str, response: str, criterion: str) -> float:
        # Example: call OpenAI, Anthropic, or a local Ollama model.
        # For CI, you can mock this or use a lightweight local model.
        import openai
        client = openai.OpenAI()
        result = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": (
                    f"Rate the response on '{criterion}' from 1 to 5. "
                    "Reply with ONLY a number."
                )},
                {"role": "user", "content": f"Prompt: {prompt}\nResponse: {response}"},
            ],
        )
        return float(result.choices[0].message.content.strip())
    return _judge


# ---- Layer 1: Retriever Ranking ----

class TestRetrieverRanking:
    """Verify the retrieval component returns and ranks relevant documents."""

    def test_ndcg_at_5(self, retrieval_sample):
        """PASS: nDCG@5 above threshold — retriever ranks relevant docs higher."""
        assert_ndcg(
            relevance_labels=retrieval_sample["relevance_labels"],
            k=5,
            min_score=0.5,
        )

    def test_mrr(self, retrieval_sample):
        """PASS: MRR above threshold — first relevant doc appears early."""
        assert_mrr(
            relevance_labels=retrieval_sample["relevance_labels"],
            min_score=0.5,
        )

    def test_recall_at_5(self, retrieval_sample):
        """PASS: Recall@5 above threshold — most relevant docs are retrieved."""
        assert_recall_at_k(
            relevant_ids=retrieval_sample["relevant_ids"],
            retrieved_ids=retrieval_sample["retrieved_ids"],
            k=5,
            min_recall=0.6,
        )

    def test_map_at_5(self, retrieval_sample):
        """PASS: MAP@5 above threshold — relevant docs are ranked precisely."""
        assert_map_at_k(
            relevant_ids=retrieval_sample["relevant_ids"],
            retrieved_ids=retrieval_sample["retrieved_ids"],
            k=5,
            min_map=0.4,
        )


# ---- Layer 2: Generator Faithfulness ----

class TestGeneratorFaithfulness:
    """Verify the generator does not hallucinate beyond retrieved context."""

    def test_faithfulness(self, rag_sample):
        """PASS: Answer is grounded in retrieved context."""
        assert_faithfulness(
            answer=rag_sample["answer"],
            context=rag_sample["context"],
            min_score=0.5,
        )

    def test_context_relevancy(self, rag_sample):
        """PASS: Retrieved chunks are relevant to the question."""
        assert_context_relevancy(
            question=rag_sample["question"],
            context=rag_sample["context"],
            min_score=0.3,
        )

    def test_answer_relevancy(self, rag_sample):
        """PASS: Answer addresses the question asked."""
        assert_answer_relevancy(
            question=rag_sample["question"],
            answer=rag_sample["answer"],
            min_score=0.3,
        )

    def test_summary_faithfulness(self, rag_sample):
        """PASS: If the answer is a summary, verify no fabricated claims."""
        source_text = " ".join(rag_sample["context"])
        assert_summary_faithfulness(
            source=source_text,
            summary=rag_sample["answer"],
            min_score=0.5,
        )

    def test_summary_coverage(self, rag_sample):
        """PASS: Summary captures key information from context."""
        source_text = " ".join(rag_sample["context"])
        assert_summary_coverage(
            source=source_text,
            summary=rag_sample["answer"],
            min_coverage=0.3,
        )


# ---- Layer 3: Generator Quality (LLM Judge) ----

@pytest.mark.llm_judge
class TestGeneratorQuality:
    """Subjective quality evaluation via LLM-as-Judge.

    These tests require an LLM API call. Mark with @pytest.mark.llm_judge
    so they can be skipped in offline CI (pytest -m 'not llm_judge').
    """

    def test_helpfulness(self, judge_fn, rag_sample):
        """PASS: Judge rates the answer as helpful (>= 3.5 / 5)."""
        assert_llm_judge_score(
            judge_fn=judge_fn,
            prompt=rag_sample["question"],
            response=rag_sample["answer"],
            criterion="helpfulness",
            min_score=3.5,
            scale_max=5.0,
        )

    def test_coherence(self, judge_fn, rag_sample):
        """PASS: Judge rates the answer as coherent (>= 3.0 / 5)."""
        assert_llm_judge_score(
            judge_fn=judge_fn,
            prompt=rag_sample["question"],
            response=rag_sample["answer"],
            criterion="coherence",
            min_score=3.0,
            scale_max=5.0,
        )


# ---- Layer 4: End-to-End Composite ----

class TestEndToEnd:
    """Composite RAG score — catches systemic quality drops."""

    def test_ragas_composite(self, rag_sample):
        """PASS: RAGAS composite score above threshold."""
        assert_ragas_score(
            answer=rag_sample["answer"],
            question=rag_sample["question"],
            context=rag_sample["context"],
            min_score=0.4,
        )
```

### Running the layers independently

Use pytest markers to run specific layers in different CI stages:

```bash
# Fast CI gate — no LLM calls, runs in milliseconds
pytest tests/test_rag_pipeline.py -m "not llm_judge" -q

# Nightly — include LLM judge evaluation
pytest tests/test_rag_pipeline.py -q

# Retriever-only (after changing embedding model or index)
pytest tests/test_rag_pipeline.py -k "TestRetrieverRanking" -q

# Faithfulness-only (after changing prompt template or context window)
pytest tests/test_rag_pipeline.py -k "TestGeneratorFaithfulness" -q
```

### Debugging pipeline failures

When the composite `assert_ragas_score` fails, the layer-specific tests tell you where to look:

| Symptom | Failing layer | Root cause |
|---------|--------------|------------|
| Low RAGAS score, low nDCG/MRR | Retriever | Embedding model degraded, index stale, chunking strategy wrong |
| Low RAGAS score, good nDCG, low faithfulness | Generator | LLM hallucinating despite good context — check prompt template |
| Low RAGAS score, good nDCG, good faithfulness, low judge score | Generator quality | LLM is factual but unhelpful — check system prompt, temperature |
| Good RAGAS score, low summary coverage | Summarization | LLM omitting key details — check max_tokens, prompt instructions |
