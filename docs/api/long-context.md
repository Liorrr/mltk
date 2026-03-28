# Long-Context LLM Evaluation

Test whether LLMs actually *use* their full context window, not just advertise it.

**Module:** `mltk.domains.llm.long_context`

---

## Why Test Long Context?

A model with a 128K-token context window does not necessarily *use* 128K tokens of context. Benchmark marketing says "128K context" but production behavior often tells a different story: the model reads the first few thousand tokens, skims the last few hundred, and ignores everything in between.

This matters because RAG pipelines, document Q&A systems, and multi-step agent workflows routinely stuff 10K-100K tokens of context into a single prompt. If the model only attends to the edges of that context, retrieved documents in the middle are wasted bandwidth. The answer degrades silently -- no error, no warning, just a worse response that blames the model's "knowledge cutoff" when the answer was right there in the prompt.

The three assertions in this module test different failure modes:

| Failure Mode | Assertion | What It Catches |
|-------------|-----------|-----------------|
| Needle retrieval | `assert_needle_in_haystack` | Model cannot find a specific fact at certain positions in a long document |
| Context waste | `assert_context_utilization` | Model ignores most of the provided facts |
| Lost in the middle | `assert_no_lost_in_middle` | Model attends to start/end but ignores the middle of context |

### The "Lost in the Middle" Phenomenon

Liu et al. (2023) documented a consistent pattern across LLMs: when relevant information is placed in the *middle* of a long context, model performance drops significantly compared to when the same information is at the beginning or end. This is not a rare edge case -- it affects GPT-4, Claude, Llama, and virtually every transformer-based model tested.

The practical impact: if your RAG pipeline retrieves 20 documents and the most relevant one happens to land at position 10 of 20, the model may functionally ignore it. Reranking helps, but testing confirms whether your specific model and prompt structure are vulnerable.

---

## assert_needle_in_haystack

Insert a known fact (the "needle") at different positions in a long document (the "haystack") and verify the model can retrieve it. This is the standard needle-in-a-haystack evaluation used by Anthropic, OpenAI, and Google to validate context window claims.

```python
from mltk.domains.llm.long_context import assert_needle_in_haystack

def my_model(prompt: str) -> str:
    # Call your LLM here
    return llm.generate(prompt)

# A long document to embed the needle in
haystack = open("large_document.txt").read()  # 50K+ chars

result = assert_needle_in_haystack(
    model_fn=my_model,
    needle="The project deadline is March 15, 2025.",
    haystack=haystack,
    positions=[0.0, 0.25, 0.5, 0.75, 1.0],
    min_recall=0.8,
)

# Check which positions the model missed
for pos, found in result.details["per_position"].items():
    print(f"Position {pos}: {'found' if found else 'MISSED'}")
```

For each position, the needle is inserted into the haystack at that relative offset (0.0 = start, 0.5 = middle, 1.0 = end). The model is then prompted to answer a question about the needle. Success is measured by token overlap between the response and the needle text.

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `model_fn` | `Callable[[str], str]` | *(required)* | Function that takes a prompt and returns the model response |
| `needle` | `str` | *(required)* | The fact to embed and retrieve (e.g., `"The secret code is 7492"`) |
| `haystack` | `str` | *(required)* | A long document to embed the needle in |
| `positions` | `list[float] \| None` | `[0.0, 0.25, 0.5, 0.75, 1.0]` | Relative positions (0.0--1.0) to insert the needle |
| `min_recall` | `float` | `0.8` | Minimum fraction of positions where the model must find the needle |

### Result Details

| Key | Type | Description |
|-----|------|-------------|
| `recall` | `float` | Fraction of positions where the needle was found |
| `min_recall` | `float` | The threshold that was required |
| `per_position` | `dict[str, bool]` | Map of position -> whether the needle was found |
| `n_positions` | `int` | Number of positions tested |
| `needle_length` | `int` | Character length of the needle |
| `haystack_length` | `int` | Character length of the haystack |

### Interpreting Results

- **recall = 1.0**: The model finds the needle at every position. Full context window is functional.
- **recall = 0.6, misses at 0.25 and 0.5**: Classic lost-in-the-middle pattern. The model attends to edges but not the interior.
- **recall = 0.0**: The model cannot retrieve the needle at any position. The haystack may be too long, the needle too subtle, or the model's context window is not functional at this length.

---

## assert_context_utilization

Provide multiple facts in the context and verify the model uses a minimum number of them in its response. This catches models that technically accept long context but only read the first few thousand tokens.

```python
from mltk.domains.llm.long_context import assert_context_utilization

facts = [
    "The company was founded in 2019.",
    "Annual revenue reached $50M in 2024.",
    "The CEO is Jane Smith.",
    "Headquarters are in Austin, Texas.",
    "The company has 200 employees.",
    "Primary product is an analytics platform.",
    "Series B funding was $30M.",
]

result = assert_context_utilization(
    model_fn=my_model,
    facts=facts,
    question="Write a one-paragraph company overview.",
    min_facts_used=4,
)

# See which facts the model used
for i, used in enumerate(result.details["per_fact_found"]):
    status = "USED" if used else "ignored"
    print(f"Fact {i+1}: {status} -- {facts[i]}")
```

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `model_fn` | `Callable[[str], str]` | *(required)* | Function that takes a prompt and returns the model response |
| `facts` | `list[str]` | *(required)* | List of factual statements to include in the context |
| `question` | `str` | *(required)* | A question whose answer should draw on multiple facts |
| `min_facts_used` | `int` | `3` | Minimum number of facts that must appear in the response |

### Result Details

| Key | Type | Description |
|-----|------|-------------|
| `facts_used` | `int` | Number of facts found in the response |
| `min_facts_used` | `int` | The threshold that was required |
| `total_facts` | `int` | Total number of facts provided |
| `per_fact_found` | `list[bool]` | Whether each fact appeared in the response |

### Interpreting Results

- **facts_used = 7/7**: The model synthesized all provided context. Excellent utilization.
- **facts_used = 2/7, facts 1 and 2 used**: The model only read the first two facts. It is not utilizing context beyond the first few entries. Try reordering facts or using a model with better context handling.
- **facts_used = 0**: The model ignored all facts entirely. It may be relying on parametric knowledge instead of the provided context.

---

## assert_no_lost_in_middle

Test whether a model attends uniformly to the beginning, middle, and end of its context, or whether middle-positioned information is systematically ignored.

```python
from mltk.domains.llm.long_context import assert_no_lost_in_middle

facts = [
    "The speed of light is 299,792,458 m/s.",       # beginning
    "Water boils at 100 degrees Celsius.",            # middle
    "Earth orbits the Sun in 365.25 days.",           # end
]

questions = [
    "What is the speed of light?",
    "At what temperature does water boil?",
    "How long does Earth take to orbit the Sun?",
]

result = assert_no_lost_in_middle(
    model_fn=my_model,
    facts=facts,
    questions=questions,
    min_accuracy=0.7,
)

# Check per-position accuracy
for pos, acc in result.details["per_position_accuracy"].items():
    print(f"{pos}: {acc:.0%} accuracy")
# beginning: 100% accuracy
# middle:     50% accuracy   <-- lost-in-the-middle!
# end:       100% accuracy
```

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `model_fn` | `Callable[[str], str]` | *(required)* | Function that takes a prompt and returns the model response |
| `facts` | `list[str]` | *(required)* | List of facts. `facts[0]` = beginning, `facts[len//2]` = middle, `facts[-1]` = end |
| `questions` | `list[str]` | *(required)* | Parallel list of questions -- `questions[i]` asks about `facts[i]`. Must have same length as `facts`. |
| `min_accuracy` | `float` | `0.7` | Minimum fraction of questions answered correctly |

### Result Details

| Key | Type | Description |
|-----|------|-------------|
| `accuracy` | `float` | Overall fraction of questions answered correctly |
| `min_accuracy` | `float` | The threshold that was required |
| `per_position_accuracy` | `dict[str, float]` | Accuracy broken down by region (beginning, middle, end) |
| `per_position_correct` | `dict[str, bool]` | Whether each position was answered correctly |
| `n_questions` | `int` | Total number of questions tested |

### Interpreting Results

- **per_position_accuracy all near 1.0**: No lost-in-the-middle effect. The model attends uniformly.
- **beginning: 1.0, middle: 0.3, end: 0.9**: Classic lost-in-the-middle. The model strongly attends to the start, somewhat to the end, and poorly to the middle. This is the most common pattern.
- **All positions near 0.0**: The model is not using context at all. The problem is broader than position bias.

---

## Testing Strategy: Combining All Three

For a complete long-context evaluation, combine all three assertions. Each catches a different failure mode:

```python
from mltk.domains.llm.long_context import (
    assert_needle_in_haystack,
    assert_context_utilization,
    assert_no_lost_in_middle,
)

def my_model(prompt: str) -> str:
    return llm.generate(prompt)

# 1. Can the model retrieve a specific fact from a long document?
assert_needle_in_haystack(
    model_fn=my_model,
    needle="The authorization token expires at midnight UTC.",
    haystack=open("long_config_doc.txt").read(),
    min_recall=0.8,
)

# 2. Does the model use facts from across the context?
assert_context_utilization(
    model_fn=my_model,
    facts=[f"Requirement {i}: ..." for i in range(20)],
    question="Summarize the top requirements.",
    min_facts_used=10,
)

# 3. Is accuracy uniform or does it drop in the middle?
assert_no_lost_in_middle(
    model_fn=my_model,
    facts=facts_at_known_positions,
    questions=questions_per_fact,
    min_accuracy=0.7,
)
```

---

## Practical Tips

### Haystack Construction
- Use real documents, not synthetic "lorem ipsum." Models behave differently on natural vs. synthetic text.
- Aim for haystacks that are 50-80% of the model's claimed context window. A 128K model should be tested with 80K-100K token haystacks.
- Include varied content (paragraphs, lists, code blocks) to simulate realistic retrieval scenarios.

### Needle Design
- Make needles specific and unique: `"The secret code is 7492"` is better than `"The answer is yes"`.
- Avoid needles that overlap with common text patterns in the haystack.
- Test with both short needles (single fact) and longer needles (multi-sentence passages).

### Position Sampling
- The default 5 positions `[0.0, 0.25, 0.5, 0.75, 1.0]` give a reasonable coverage.
- For thorough testing, use 10+ positions: `[i / 10 for i in range(11)]`.
- Pay special attention to the 0.3-0.7 range where lost-in-the-middle effects are strongest.

### Model Function
- The `model_fn` callable should handle the full prompt (context + question) as a single string input.
- Wrap API calls in retry logic *outside* the assertion -- the assertion treats exceptions as failures.
- For chat models, format the prompt appropriately before passing to the assertion.

### Reference
- Liu, N.F., Lin, K., Hewitt, J., Paranjape, A., Bevilacqua, M., Petroni, F., & Liang, P. (2023). "Lost in the Middle: How Language Models Use Long Contexts." *arXiv:2307.03172*.
