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
