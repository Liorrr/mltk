# LLM-as-Judge Evaluation

Use another LLM as an automated evaluator to assess subjective qualities that token overlap, keyword matching, and regex-based checks simply cannot measure.

**Module:** `mltk.domains.llm.judge`

---

## Why LLM-as-Judge?

### The subjectivity problem

Traditional NLP metrics (BLEU, ROUGE, token F1) measure surface-level overlap between generated text and a reference. They work for translation and summarization where a "correct answer" exists, but they fail for open-ended generation:

- **"Is this response helpful?"** -- Token overlap cannot tell you whether a response actually solves the user's problem.
- **"Is this answer factually correct?"** -- A paraphrased correct answer scores low on token overlap; a confidently wrong answer with matching keywords scores high.
- **"Is this coherent and well-structured?"** -- Word order, logical flow, and argument quality are invisible to bag-of-words metrics.

### The judge approach

An LLM-as-Judge evaluates model outputs the way a human would -- by reading the prompt, reading the response, and making a qualitative judgment. Research from Zheng et al. (2023) shows strong LLMs (GPT-4, Claude) achieve >80% agreement with human annotators on quality ratings, comparable to inter-annotator agreement.

mltk provides two judge patterns:

| Pattern | Use case | Reliability |
|---------|----------|-------------|
| **Absolute scoring** | Rate quality on a 1--5 scale per criterion | Good for monitoring, dashboards |
| **Pairwise comparison** | "Is A better than B?" | More reliable -- comparison is easier than calibration |

---

## Design decision: `judge_fn` as callable

mltk does **not** own the LLM call. Instead, you provide a `judge_fn` callable:

```python
# Your function -- any LLM backend works
def judge_fn(evaluation_prompt: str) -> float:
    # Call OpenAI, Anthropic, Ollama, vLLM, or anything else
    response = your_llm_client.chat(evaluation_prompt)
    return float(response)
```

**Why this design:**

- **Vendor-neutral** -- Works with OpenAI, Anthropic, Ollama, vLLM, Hugging Face, or a custom wrapper. No API keys baked into mltk.
- **User-controlled cost** -- You pick the model. Use GPT-4 for high-stakes evals, a cheap local model for CI/CD gates.
- **User-controlled latency** -- Batch calls, use async wrappers, or cache results -- mltk doesn't constrain you.
- **Testable** -- Mock `judge_fn` in tests with a simple lambda. No LLM needed for unit tests.

---

## `assert_llm_judge_score`

Assert that LLM responses meet a minimum quality score via judge evaluation.

```python
from mltk.domains.llm.judge import assert_llm_judge_score

# Mock judge for testing (returns a fixed score)
def mock_judge(prompt: str) -> float:
    return 4.2

result = assert_llm_judge_score(
    judge_fn=mock_judge,
    prompts=["What is Python?", "Explain REST APIs"],
    responses=["Python is a programming language.", "REST is an architecture style."],
    criterion="helpfulness",
    min_score=3.0,
)
# result.passed == True, result.details["avg_score"] == 4.2
```

### With a real LLM (OpenAI example)

```python
import openai

client = openai.OpenAI()

def openai_judge(evaluation_prompt: str) -> float:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": evaluation_prompt}],
        max_tokens=10,
        temperature=0.0,
    )
    return float(response.choices[0].message.content.strip())

result = assert_llm_judge_score(
    judge_fn=openai_judge,
    prompts=prompts,
    responses=model_outputs,
    criterion="correctness",
    min_score=3.5,
)
```

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `judge_fn` | `Callable[[str], float]` | *(required)* | Takes an evaluation prompt, returns a numeric score (or string containing one) |
| `prompts` | `list[str]` | *(required)* | Original prompts given to the model |
| `responses` | `list[str]` | *(required)* | Model responses to evaluate (same length as prompts) |
| `criterion` | `str` | `"helpfulness"` | Evaluation criterion -- key in `DEFAULT_CRITERIA` or custom name |
| `min_score` | `float` | `3.0` | Minimum required average score to pass |
| `max_score` | `float` | `5.0` | Upper bound of the scoring scale |
| `rubric` | `str \| None` | `None` | Custom rubric text. Overrides default for the criterion |

### Result details

| Key | Type | Description |
|-----|------|-------------|
| `avg_score` | `float` | Mean score across all items |
| `min_score` | `float` | The threshold that was required |
| `per_item_scores` | `list[dict]` | Per-item scores (and errors if any) |
| `criterion` | `str` | Which criterion was evaluated |
| `n_items` | `int` | Number of prompt/response pairs |
| `scores_below_min` | `int` | Count of items scoring below min_score |

### How rubrics work

Each criterion has a default rubric (stored in `DEFAULT_CRITERIA`) that describes what the judge should evaluate. The rubric gets formatted into a structured evaluation prompt along with the actual prompt and response.

You can override any rubric:

```python
result = assert_llm_judge_score(
    judge_fn=mock_judge,
    prompts=["Summarize this article"],
    responses=["The article discusses climate change..."],
    criterion="conciseness",
    rubric="Rate how concise the response is. A concise response uses the fewest words necessary to convey the key points without losing important information.",
    min_score=3.5,
)
```

---

## `assert_llm_judge_pairwise`

Assert that one set of responses is preferred over another by a judge LLM. More reliable than absolute scoring because comparison is cognitively easier than calibrated rating.

```python
from mltk.domains.llm.judge import assert_llm_judge_pairwise

def mock_judge(prompt: str) -> str:
    return "A"

result = assert_llm_judge_pairwise(
    judge_fn=mock_judge,
    prompts=["What is Python?"],
    responses_a=["Python is a versatile programming language used in web, data, and AI."],
    responses_b=["It's a language."],
    expected_winner="a",
    min_win_rate=0.6,
)
# result.passed == True, result.details["win_rate"] == 1.0
```

### When to use pairwise comparison

- **Model A/B testing**: Compare a fine-tuned model against the base model.
- **Prompt iteration**: Test whether a new system prompt produces better responses.
- **Fine-tuning validation**: Verify that fine-tuning improved quality without regression.
- **Provider comparison**: Compare outputs from different LLM providers on the same prompts.

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `judge_fn` | `Callable[[str], str]` | *(required)* | Takes a comparison prompt, returns `"A"`, `"B"`, or `"TIE"` |
| `prompts` | `list[str]` | *(required)* | Original prompts |
| `responses_a` | `list[str]` | *(required)* | First-candidate responses |
| `responses_b` | `list[str]` | *(required)* | Second-candidate responses |
| `expected_winner` | `str` | `"a"` | Which candidate should win -- `"a"` or `"b"` |
| `min_win_rate` | `float` | `0.6` | Minimum fraction of comparisons the expected winner must win |
| `criterion` | `str` | `"helpfulness"` | Evaluation criterion |
| `rubric` | `str \| None` | `None` | Custom rubric text |

### Result details

| Key | Type | Description |
|-----|------|-------------|
| `win_rate` | `float` | Fraction of comparisons won by expected_winner |
| `min_win_rate` | `float` | The threshold that was required |
| `wins_a` | `int` | Number of comparisons won by A |
| `wins_b` | `int` | Number of comparisons won by B |
| `ties` | `int` | Number of comparisons where neither won |
| `n_comparisons` | `int` | Total number of comparisons |

---

## `DEFAULT_CRITERIA`

Built-in evaluation rubrics for common quality dimensions:

| Criterion | What it evaluates |
|-----------|-------------------|
| `helpfulness` | Does the response directly address the question and provide actionable information? |
| `correctness` | Is the response factually accurate with no fabricated claims? |
| `coherence` | Is the response well-organized, logically structured, and easy to follow? |
| `relevance` | Does the response stay on topic and address all parts of the question? |
| `harmlessness` | Does the response avoid promoting violence, discrimination, or dangerous content? |

Use these as-is or provide custom rubrics for domain-specific criteria:

```python
# Custom criterion with custom rubric
result = assert_llm_judge_score(
    judge_fn=judge,
    prompts=prompts,
    responses=responses,
    criterion="medical_accuracy",
    rubric="Rate whether the medical information is accurate, evidence-based, and appropriately caveated with disclaimers about consulting a healthcare professional.",
    min_score=4.0,
)
```

---

## Integration with pytest

### Fixture for judge_fn

```python
import pytest
from mltk.domains.llm.judge import assert_llm_judge_score

@pytest.fixture
def judge_fn():
    """Mock judge for unit tests -- no real LLM calls."""
    def _judge(prompt: str) -> float:
        return 4.0
    return _judge

def test_model_helpfulness(judge_fn):
    result = assert_llm_judge_score(
        judge_fn=judge_fn,
        prompts=["What is machine learning?"],
        responses=["ML is a subset of AI that learns from data."],
        criterion="helpfulness",
        min_score=3.0,
    )
    assert result.passed
```

### Parametrize across criteria

```python
import pytest
from mltk.domains.llm.judge import (
    DEFAULT_CRITERIA,
    assert_llm_judge_score,
)

@pytest.fixture
def judge_fn():
    def _judge(prompt: str) -> float:
        return 4.0
    return _judge

@pytest.mark.parametrize("criterion", list(DEFAULT_CRITERIA.keys()))
def test_all_criteria(judge_fn, criterion):
    result = assert_llm_judge_score(
        judge_fn=judge_fn,
        prompts=["Explain quantum computing"],
        responses=["Quantum computing uses qubits..."],
        criterion=criterion,
        min_score=3.0,
    )
    assert result.passed
```

### Pairwise comparison in CI/CD

```python
from mltk.domains.llm.judge import assert_llm_judge_pairwise

def test_new_prompt_beats_old(judge_fn):
    """Gate: new system prompt must win 60%+ pairwise comparisons."""
    result = assert_llm_judge_pairwise(
        judge_fn=judge_fn,
        prompts=test_prompts,
        responses_a=new_prompt_outputs,
        responses_b=old_prompt_outputs,
        expected_winner="a",
        min_win_rate=0.6,
        criterion="helpfulness",
    )
    assert result.passed
```

---

## Comparison: mltk vs DeepEval

| Aspect | DeepEval | mltk |
|--------|----------|------|
| **LLM dependency** | Built-in OpenAI calls, requires API key at test time | `judge_fn` callable -- you own the LLM call |
| **Provider support** | OpenAI-first, others via config | Any provider via callable pattern |
| **Cost control** | Framework controls batching/caching | User controls everything |
| **Offline testing** | Requires real LLM or mock setup | Mock with `lambda: 4.0` -- zero infra |
| **Criteria** | Fixed metric classes (GEval, Faithfulness) | Open rubric strings -- any criterion |
| **CI/CD friendly** | Needs API key in CI secrets | Mock judge for free CI; real judge for staging |
| **Pairwise** | Not built-in (comparison not a first-class metric) | First-class `assert_llm_judge_pairwise` |

mltk's approach prioritizes **testability and vendor neutrality** over convenience. You write slightly more setup code (the `judge_fn`), but gain full control over cost, latency, and provider selection.

---

## Error handling

- If `judge_fn` **raises an exception** during scoring, the item gets score `0.0` with an error flag in `per_item_scores`. The assertion does not crash.
- If `judge_fn` **returns unparseable text** during scoring (no numeric value found), the item gets score `0.0`.
- If `judge_fn` **raises or returns gibberish** during pairwise comparison, the item counts as a **tie** (neither side benefits from judge failure).
- If `prompts` and `responses` have **different lengths**, the assertion fails immediately with a descriptive error message.

---

## Related

- [LLM Judge Defaults](judge-defaults.md) -- configure a default judge once so every subjective assertion uses it automatically
- [Multimodal Evaluation](multimodal.md) -- LLM-as-Judge assertions for image-text evaluation
- [Behavioral Consistency](behavioral-consistency.md) -- uses `method="llm"` for LLM-as-Judge dispatch
- [Assertion Index](assertion-index.md) -- full list of all 224 assertions
