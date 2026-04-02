# Solver/Scorer Evaluation Pipeline

Compose evaluation pipelines that cleanly separate *how to prompt* from
*how to grade* — then run them natively inside pytest with zero extra
dependencies.

**Since:** v0.9.0

**Modules:**

- `mltk.eval._types` — `EvalSample`, `EvalState`, `Score`, `EvalResult`
- `mltk.eval.solvers` — `Solver` ABC + `GenerateSolver`,
  `ChainOfThoughtSolver`, `FewShotSolver`, `chain()`
- `mltk.eval.scorers` — `Scorer` ABC + `ExactMatchScorer`,
  `IncludesScorer`, `PatternScorer`, `LLMJudgeScorer`
- `mltk.eval.task` — `EvalTask`, `load_dataset()`

---

## Why a Solver/Scorer Pipeline?

### The accidental coupling problem

Most ML test frameworks conflate two completely different concerns inside a
single function call:

```python
# Everything mixed together — hard to vary independently
result = assert_llm_correctness(
    model_fn=my_model,
    prompts=questions,
    expected=answers,
    strategy="chain_of_thought",   # prompting + grading in one shot
)
```

When everything is entangled:

- Changing the **prompting strategy** (zero-shot → CoT) requires rewriting
  the entire test, not swapping one component.
- Measuring the **same output** through multiple lenses (exact match AND
  LLM judge AND pattern extraction) is awkward or impossible.
- **A/B testing** two prompting strategies on the same grading criteria
  requires duplicating test logic.
- **Solver unit tests** cannot run without a real model; they are untestable
  in isolation.

### The pipeline solution

The Solver/Scorer architecture separates evaluation into two independent
abstractions:

| Concern | Abstraction | Question it answers |
|---------|-------------|---------------------|
| Prompting strategy | **Solver** | *How should the model reason?* |
| Grading logic | **Scorer** | *How correct is the output?* |

Solvers transform evaluation state — they decide what prompt to send, how
to structure few-shot examples, whether to inject chain-of-thought
reasoning. Scorers receive the *finished* state and grade it — no model
knowledge required.

This decoupling produces a combinatorial evaluation matrix from a small set
of composable primitives:

```
(M solvers) × (N scorers) × (K models) = M × N × K evaluations
```

All from writing M + N + K components rather than M × N × K tests.

### What mltk adds

This architecture is inspired by UK AISI's Inspect AI framework (the
reference implementation in the field) and adapted with three properties
that no existing framework provides simultaneously:

- **pytest-native** — `task.to_test_result(model_fn)` integrates directly
  with pytest, CI gates, and `MltkSuite`. No separate CLI or server.
- **Zero dependencies** — dataset loading uses only stdlib `csv` and
  `json`. Solvers and scorers are pure Python classes.
- **Provider-agnostic** — no model is built in. Every solver and scorer
  receives a `Callable[[str], str]` — any LLM backend, local or cloud.

---

## Quick Start

Five lines from import to assertion:

```python
from mltk.eval.solvers import GenerateSolver
from mltk.eval.scorers import ExactMatchScorer
from mltk.eval.task import EvalTask, load_dataset

def my_model(prompt: str) -> str:
    return "Paris"  # your real model here

task = EvalTask(
    name="geography-qa",
    solver=GenerateSolver(),
    scorers=ExactMatchScorer(),
    dataset=load_dataset("questions.csv"),
)

result = task.run(my_model)
assert result.metrics["ExactMatchScorer/accuracy"] >= 0.8
```

For pytest integration, replace the last two lines with:

```python
def test_geography_qa():
    test_result = task.to_test_result(my_model, min_accuracy=0.8)
    assert test_result.passed
```

### What just happened?

1. `load_dataset("questions.csv")` turned your CSV into `EvalSample` objects.
2. `EvalTask` wrapped one solver, one scorer, and the dataset together.
3. `task.run(my_model)` iterated every sample through the pipeline:
   - Created an `EvalState` from the sample.
   - Sent it through `GenerateSolver`, which called `my_model` and stored
     the response in `state.output`.
   - Passed the final state to `ExactMatchScorer`, which compared
     `state.output` to `state.sample.target`.
4. `EvalResult.metrics["ExactMatchScorer/accuracy"]` is the fraction of
   samples that passed (score >= 0.5).

---

## Core Concepts

### EvalSample — the atomic unit of evaluation data

```python
from mltk.eval._types import EvalSample

sample = EvalSample(
    input="What is the capital of France?",
    target="Paris",
    metadata={"category": "geography", "difficulty": "easy"},
)
```

`EvalSample` is **immutable once created** — solvers read from it but
never modify it. The `target` field is optional; some scorers (like
`LLMJudgeScorer`) do not require a reference answer. Metadata flows
through the pipeline and is available to both solvers and scorers via
`state.sample.metadata`.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `input` | `str` | Yes | The prompt or question |
| `target` | `str \| None` | No | Expected answer for grading |
| `metadata` | `dict` | No | Arbitrary per-sample data |

### EvalState — mutable pipeline context

```python
from mltk.eval._types import EvalState

state = EvalState(sample=sample)
# After GenerateSolver runs:
# state.output = "The capital of France is Paris."
# state.messages = [{"role": "assistant", "content": "..."}]
```

`EvalState` is the data bus that flows through the solver pipeline.
Each solver receives the state from the previous one, transforms it, and
returns it (mutated in place or as a new object — both patterns are valid).
Scorers receive the **final state** read-only and never modify it.

| Field | Type | Description |
|-------|------|-------------|
| `sample` | `EvalSample` | Original sample (read-only by convention) |
| `output` | `str` | Model/solver output — the text being graded |
| `messages` | `list[dict]` | Conversation history for multi-turn evals |
| `metadata` | `dict` | Solver-added context (CoT reasoning, etc.) |
| `completed` | `bool` | If `True`, remaining solvers are skipped |

The `completed` flag is the pipeline's short-circuit mechanism. A caching
solver that finds a stored result can set `state.completed = True`,
preventing all downstream solvers from running — avoiding redundant model
calls.

### Solver — the prompting strategy

A `Solver` answers: *given this evaluation state, what should the model
be asked, and how?*

Solvers are composable. They form a sequential pipeline where each solver
receives the state produced by the previous one. Crucially, **solvers
never call the model directly** — they receive a `generate` callable that
abstracts the LLM backend. This makes every solver independently testable
with a simple lambda stub.

```python
# A solver works with any generate function — real or mock
def mock_model(prompt: str) -> str:
    return "4"

solver = GenerateSolver()
result = solver.solve(state, mock_model)  # no LLM needed in tests
```

### Scorer — the grading logic

A `Scorer` answers: *given the model's output, how correct or good is it?*

Scorers are independent — multiple scorers can run on the same final state
in parallel (conceptually), each producing its own `Score`. This enables
**multi-dimensional evaluation**: one scorer checks factual correctness
with exact match, another checks quality with an LLM judge.

### EvalTask — the composition point

`EvalTask` is the object you construct and run. It holds:
- The solver (or solver pipeline) that defines the prompting strategy
- One or more scorers that define grading
- The dataset of samples to evaluate

```python
task = EvalTask(
    name="my-eval",
    solver=[ChainOfThoughtSolver(), GenerateSolver()],  # pipeline
    scorers=[ExactMatchScorer(), LLMJudgeScorer(my_judge)],  # multi
    dataset=samples,
)
```

### Score — a single scoring result

```python
from mltk.eval._types import Score

score = Score(
    value=0.8,          # normalized to [0.0, 1.0]
    answer="Paris",     # what the model actually said (extracted)
    explanation="Exact match after case normalization",
    metadata={"raw_score": 4.0},
)
```

`value` is always normalized to `[0.0, 1.0]`. Values >= 0.5 are
considered passing for accuracy calculations. The `answer` field captures
what the scorer extracted from the output (useful for debugging partial
credit). The `explanation` field is human-readable rationale — it shows up
in failure logs.

### EvalResult — aggregated metrics

`EvalResult` is the top-level output from `task.run()`. It contains
per-sample scores from every scorer plus aggregated metrics.

Metric keys follow the `"{ScorerName}/{metric}"` convention:

```python
result = task.run(model_fn)

# Aggregated metrics
result.metrics["ExactMatchScorer/accuracy"]   # fraction passing
result.metrics["ExactMatchScorer/mean"]        # mean score value
result.metrics["LLMJudge/correctness/mean"]   # mean judge score

# Raw per-sample access
result.scores["ExactMatchScorer"]   # list[Score], one per sample
result.samples[0].output            # what the model said on sample 0

# Convenience
result.passed           # True if all metrics >= 0.5
result.total_samples    # number of samples evaluated
result.duration_ms      # total wall time in milliseconds
```

!!! note "Why both accuracy and mean?"
    `accuracy` counts binary pass/fail (score >= 0.5) and is useful when
    scorers return 0.0 or 1.0 (exact match, pattern extraction). `mean`
    is the arithmetic mean and is useful for continuous-valued scorers
    like `LLMJudgeScorer`. Both are always computed so you can use
    whichever is appropriate for your scorer type.

---

## Solvers

### Solver ABC

Implement `Solver` to create a custom solver:

```python
from mltk.eval.solvers import Solver
from mltk.eval._types import EvalState
from collections.abc import Callable

class MyCustomSolver(Solver):
    """Appends a system prompt before generating."""

    def __init__(self, system_prompt: str) -> None:
        self.system_prompt = system_prompt

    def solve(
        self,
        state: EvalState,
        generate: Callable[[str], str],
    ) -> EvalState:
        if state.completed:
            return state  # always check completed first

        prompt = f"{self.system_prompt}\n\n{state.sample.input}"
        state.output = generate(prompt)
        state.messages.append(
            {"role": "assistant", "content": state.output}
        )
        return state

    @property
    def name(self) -> str:
        return "MyCustomSolver"
```

**Rules for custom solvers:**

1. Always check `state.completed` at the top and return early if `True`.
2. Never call the model directly — always use `generate(prompt)`.
3. Store the model's response in `state.output`.
4. Append to `state.messages` if you want conversation history preserved.
5. Return the same `state` object (mutated) or a new `EvalState` with the
   same sample.

### GenerateSolver — the simplest solver

`GenerateSolver` sends the sample input to the model and stores the
response. It is the terminal node in any solver pipeline — the step that
actually calls the LLM.

```python
from mltk.eval.solvers import GenerateSolver
from mltk.eval._types import EvalSample, EvalState

solver = GenerateSolver()

state = EvalState(sample=EvalSample(input="What is 2+2?"))
result = solver.solve(state, lambda prompt: "4")
assert result.output == "4"
```

If `state.messages` is non-empty, `GenerateSolver` uses the last message's
content as the prompt (enabling downstream use in a pipeline where an
upstream solver has already built the final prompt). Otherwise it falls
back to `state.sample.input`.

**When to use:** Always. Every pipeline ends with `GenerateSolver` (or a
solver that calls `generate` itself). If you use `chain()` with other
solvers, `GenerateSolver` is the final step.

### ChainOfThoughtSolver — step-by-step reasoning

`ChainOfThoughtSolver` prepends an instruction that asks the model to
reason step-by-step before answering. This consistently improves accuracy
on math, logic, and multi-hop reasoning tasks.

```python
from mltk.eval.solvers import ChainOfThoughtSolver

# Default template: "Think step by step before answering. ..."
solver = ChainOfThoughtSolver()

# Custom template
solver = ChainOfThoughtSolver(
    template=(
        "You are a math expert. Work through the problem "
        "systematically, showing each calculation step. "
        "State your final answer on the last line as: "
        "'Answer: <value>'"
    )
)
```

The default template is:

```
Think step by step before answering. Show your reasoning,
then provide your final answer on the last line.
```

`ChainOfThoughtSolver` injects a system message, builds a combined prompt
from all existing messages plus the sample input, calls `generate`, and
appends the user/assistant exchange to `state.messages`.

!!! tip "Pair with PatternScorer"
    When using `ChainOfThoughtSolver`, the model's output contains both
    reasoning and the final answer. Use `PatternScorer` to extract the
    final answer before comparing to the target. See the
    [Math Reasoning example](#math-reasoning-cot--patternscorer) below.

**When to use:** Multi-step problems — arithmetic, algebra, logic puzzles,
multi-hop QA. CoT reliably improves accuracy by 10–30% on reasoning tasks.
Not useful for simple factual recall (it adds latency with no benefit).

### FewShotSolver — learn from examples

`FewShotSolver` prepends demonstration examples before the actual
question, following the standard few-shot prompting pattern.

```python
from mltk.eval.solvers import FewShotSolver

solver = FewShotSolver(
    examples=[
        ("What is 2+2?", "4"),
        ("What is 3×7?", "21"),
        ("What is 10÷2?", "5"),
    ],
)
```

The default template is `"Q: {input}\nA: {output}"`. Override it:

```python
solver = FewShotSolver(
    examples=[
        ("Translate to French: Hello", "Bonjour"),
        ("Translate to French: Thank you", "Merci"),
    ],
    template="English: {input}\nFrench: {output}",
)
```

The solver formats each example, appends the test question as an
incomplete example (with `{output}` empty), and calls `generate` on the
combined prompt.

!!! warning "Example quality matters"
    Few-shot examples should be representative of the evaluation dataset.
    Unrepresentative examples (wrong domain, wrong format) can *hurt*
    accuracy compared to zero-shot. Always verify that your examples
    match the target distribution.

**When to use:** When the model needs to see the expected output format
before answering, or when zero-shot performance is inconsistent. Especially
useful for structured extraction tasks (named entities, key-value pairs)
where the format is strict.

### `chain()` — pipeline composition

`chain()` composes multiple solvers into a single pipeline solver. Solvers
run in sequence; each receives the state produced by the previous one. The
pipeline short-circuits if any solver sets `state.completed = True`.

```python
from mltk.eval.solvers import (
    ChainOfThoughtSolver,
    FewShotSolver,
    GenerateSolver,
    chain,
)

# Chain solvers left to right
pipeline = chain(
    FewShotSolver(examples=[("2+2", "4"), ("3+3", "6")]),
    ChainOfThoughtSolver(),
    GenerateSolver(),
)

print(pipeline.name)
# Pipeline(FewShotSolver -> ChainOfThoughtSolver -> GenerateSolver)
```

You can also pass a list of solvers directly to `EvalTask` — it handles
chaining automatically:

```python
task = EvalTask(
    name="math",
    solver=[ChainOfThoughtSolver(), GenerateSolver()],  # auto-chained
    scorers=ExactMatchScorer(),
    dataset=samples,
)
```

!!! note "chain() vs. list in EvalTask"
    `chain(A, B, C)` and `solver=[A, B, C]` in `EvalTask` are equivalent.
    Use `chain()` explicitly when you want to reuse the pipeline across
    multiple tasks, or when building composable pipeline components.

### Solver selection guide

| Scenario | Recommended solver(s) |
|----------|----------------------|
| Simple factual QA | `GenerateSolver()` |
| Math / logic / multi-step reasoning | `chain(ChainOfThoughtSolver(), GenerateSolver())` |
| Structured extraction (format-sensitive) | `chain(FewShotSolver(examples), GenerateSolver())` |
| Math + few-shot + reasoning | `chain(FewShotSolver(ex), ChainOfThoughtSolver(), GenerateSolver())` |
| Custom prompting strategy | Subclass `Solver`, add to chain |

---

## Scorers

### Scorer ABC

Implement `Scorer` to create a custom scorer:

```python
from mltk.eval.scorers import Scorer
from mltk.eval._types import EvalState, Score

class SentimentScorer(Scorer):
    """Score 1.0 if output matches expected sentiment label."""

    def score(self, state: EvalState) -> Score:
        if state.sample.target is None:
            return Score(
                value=0.0,
                explanation="No target sentiment provided",
            )

        output_lower = state.output.lower()
        if "positive" in output_lower:
            label = "positive"
        elif "negative" in output_lower:
            label = "negative"
        else:
            label = "neutral"

        matched = label == state.sample.target.lower()
        return Score(
            value=1.0 if matched else 0.0,
            answer=label,
            explanation=(
                f"Extracted '{label}', "
                f"expected '{state.sample.target}'"
            ),
        )

    @property
    def name(self) -> str:
        return "SentimentScorer"
```

**Rules for custom scorers:**

1. Read `state.output` for the model's response.
2. Read `state.sample.target` for the expected answer (check for `None`).
3. Return `Score(value=...)` with `value` in `[0.0, 1.0]`.
4. Populate `answer` with what you extracted (aids debugging).
5. Populate `explanation` with human-readable rationale.
6. Never call the model. Scorers are pure computation — no I/O except the
   judge pattern in `LLMJudgeScorer`.

### ExactMatchScorer — string equality

`ExactMatchScorer` returns 1.0 if the model output exactly matches the
target, 0.0 otherwise. Supports case normalization and whitespace
normalization.

```python
from mltk.eval.scorers import ExactMatchScorer

# Default: case-insensitive, whitespace-normalized
scorer = ExactMatchScorer()

# Strict: exact byte-level match
scorer = ExactMatchScorer(
    ignore_case=False,
    strip_whitespace=False,
)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ignore_case` | `True` | Lowercase both sides before comparing |
| `strip_whitespace` | `True` | Strip and collapse internal whitespace |

**When to use:** When there is a single correct textual answer and the
model is expected to reproduce it verbatim (single words, short phrases,
multiple-choice letter answers). Not suitable for free-text generation
where paraphrasing is acceptable.

### IncludesScorer — substring and regex containment

`IncludesScorer` returns 1.0 if the target appears anywhere in the
model output. Useful when the model's response may contain extra text
(explanation, context) around the correct answer.

```python
from mltk.eval.scorers import IncludesScorer

# Substring mode (default)
scorer = IncludesScorer()  # target anywhere in output
scorer = IncludesScorer(ignore_case=False)  # case-sensitive

# Regex mode
scorer = IncludesScorer(regex=True)
# state.sample.target is now treated as a regex pattern
# e.g., target=r"\b\d{4}\b" matches any 4-digit year
```

**When to use:** When the model is expected to produce the answer as part
of a longer explanation (e.g., "The capital is Paris, which..."). Also
useful for code generation tests where a specific function name or keyword
must appear in the output. Use `regex=True` for pattern-based containment
(numeric formats, specific syntax).

### PatternScorer — regex extraction + comparison

`PatternScorer` extracts an answer from the model output using a regex
capture group, then compares the extracted value to the target.

```python
from mltk.eval.scorers import PatternScorer

# Default pattern: matches "Answer: <value>" or "Final answer = <value>"
scorer = PatternScorer()

# Custom extraction pattern
scorer = PatternScorer(pattern=r"Result:\s*(\d+(?:\.\d+)?)")

# Extract JSON field
scorer = PatternScorer(
    pattern=r'"answer"\s*:\s*"([^"]+)"',
    ignore_case=True,
)
```

The default pattern is:

```
(?i)(?:answer|final answer)\s*[:=]\s*(.+)
```

It matches lines like `Answer: Paris`, `Final Answer: 42`,
`answer = yes`.

If no target is set, `PatternScorer` returns 1.0 when the pattern
matches (extraction succeeded) and 0.0 when it does not — useful for
validating output format without checking correctness.

**When to use:** Essential with `ChainOfThoughtSolver` where the model
embeds its final answer inside reasoning text. Also useful for structured
output validation (JSON fields, formatted numbers) and cases where the
model uses a consistent answer format.

!!! tip "Designing your answer format"
    If you control the prompt template, instruct the model explicitly:
    `"State your final answer as 'Answer: <value>'"`.
    The default `PatternScorer` pattern matches this format exactly, and
    the `ChainOfThoughtSolver` default template already includes this
    instruction.

### LLMJudgeScorer — LLM-as-judge

`LLMJudgeScorer` delegates grading to an LLM judge. This is the only
built-in scorer that makes an additional model call — and even then,
**mltk does not own that call**. You provide the `judge_fn`.

```python
from mltk.eval.scorers import LLMJudgeScorer

def my_judge(evaluation_prompt: str) -> float:
    # Use any LLM — OpenAI, Anthropic, Ollama, etc.
    response = your_client.chat(evaluation_prompt)
    return float(response)  # return raw score (e.g., 0–5)

# Defaults: criterion="correctness", max_score=5.0
scorer = LLMJudgeScorer(judge_fn=my_judge)

# Custom criterion and rubric
scorer = LLMJudgeScorer(
    judge_fn=my_judge,
    criterion="faithfulness",
    rubric=(
        "Rate whether the response is factually grounded in the "
        "provided source material. Penalize hallucinations and "
        "unsupported claims."
    ),
    max_score=10.0,
)
```

The scorer constructs this prompt and sends it to `judge_fn`:

```
Evaluate the following response on {criterion}.

Rubric: {rubric}

Question: {state.sample.input}
Response: {state.output}
Expected answer: {state.sample.target}

Score (0-{max_score}):
```

The raw score is normalized to `[0.0, 1.0]` by dividing by `max_score`.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `judge_fn` | required | `Callable[[str], float \| str]` |
| `criterion` | `"correctness"` | What to evaluate |
| `rubric` | `None` | Optional detailed scoring guidance |
| `max_score` | `5.0` | Maximum raw score (used for normalization) |

!!! warning "Judge calibration"
    `max_score` must match what your judge model actually returns. If your
    judge returns 0–10 but `max_score=5.0`, all scores above 5 will be
    clipped to 1.0. Always verify the score range your judge uses before
    running evaluation.

!!! note "Judge errors are caught"
    If `judge_fn` raises an exception, the scorer returns
    `Score(value=0.0)` with the error in `metadata["error"]`. A single
    bad judge call does not crash the evaluation.

See [llm-judge.md](llm-judge.md) for the full `judge_fn` pattern,
provider examples (OpenAI, Anthropic, Ollama), and mock patterns for unit
testing.

**When to use:** Subjective quality dimensions that deterministic scorers
cannot measure — helpfulness, coherence, faithfulness to source material,
style adherence. Always pair with a deterministic scorer when possible:
`LLMJudgeScorer` is expensive and slower than `ExactMatchScorer`.

### Multi-scorer evaluation

Pass a list of scorers to `EvalTask` to run multiple grading methods on
the same output:

```python
task = EvalTask(
    name="qa-comprehensive",
    solver=chain(ChainOfThoughtSolver(), GenerateSolver()),
    scorers=[
        ExactMatchScorer(),
        PatternScorer(),
        LLMJudgeScorer(judge_fn=my_judge, criterion="correctness"),
    ],
    dataset=samples,
)

result = task.run(model_fn)

# All three metrics are available independently
print(result.metrics["ExactMatchScorer/accuracy"])
print(result.metrics["PatternScorer/accuracy"])
print(result.metrics["LLMJudge/correctness/mean"])
```

Each scorer runs independently on the same final `EvalState`. A failure
in one scorer does not affect others.

**Why multi-scorer?** This is the "scorer triangulation" pattern:

- `ExactMatchScorer` is fast and cheap — catches obvious correct answers.
- `PatternScorer` catches cases where the answer is correct but embedded
  in reasoning text.
- `LLMJudgeScorer` catches partially-correct answers that string matching
  misses.

Running all three together gives you a full picture of model quality in a
single `task.run()` call.

---

## EvalTask

### Construction

```python
from mltk.eval.task import EvalTask, load_dataset
from mltk.eval.solvers import GenerateSolver
from mltk.eval.scorers import ExactMatchScorer

task = EvalTask(
    name="qa-eval",             # appears in metrics and test names
    solver=GenerateSolver(),    # single solver or list
    scorers=ExactMatchScorer(), # single scorer or list
    dataset=load_dataset("questions.csv"),
)
```

`EvalTask` validates inputs eagerly at construction time. You get
`ValueError` immediately if `solver`, `scorers`, or `dataset` is empty —
not midway through a long evaluation run.

### Execution: `task.run(model_fn)`

```python
result = task.run(model_fn)
```

`run()` iterates every sample in sequence:

1. Creates `EvalState(sample=sample)`.
2. Runs each solver in the pipeline. If any solver sets
   `state.completed = True`, the pipeline short-circuits.
3. Runs every scorer on the final state. Scorer exceptions are caught and
   recorded as `Score(value=0.0, explanation="Scorer error: ...")`.
4. Collects all scores.

After all samples:

5. Computes two aggregate metrics per scorer:
   - `{name}/accuracy` — fraction of scores >= 0.5
   - `{name}/mean` — arithmetic mean of score values

```python
result.metrics  # {"ExactMatchScorer/accuracy": 0.92, ...}
result.scores   # {"ExactMatchScorer": [Score(...), Score(...)]}
result.samples  # [EvalState(...), ...]
result.duration_ms    # e.g., 1243.7
result.total_samples  # e.g., 50
result.passed         # True if all metrics >= 0.5
```

### pytest integration: `task.to_test_result(model_fn)`

```python
def test_model_quality():
    result = task.to_test_result(
        model_fn=my_model,
        min_accuracy=0.8,   # default: 0.8
    )
    assert result.passed
```

`to_test_result()` runs the evaluation and converts `EvalResult` into a
mltk `TestResult`. The assertion name is `eval.task.{task_name}`. Severity
is always `CRITICAL` — failures raise `MltkAssertionError` in pytest, the
same way all other mltk assertions work.

!!! tip "Use to_test_result in MltkSuite"
    `to_test_result()` returns a `TestResult`, making it composable with
    `MltkSuite`:

    ```python
    suite = MltkSuite("model-evaluation")
    suite.add(task.to_test_result, model_fn, min_accuracy=0.85)
    result = suite.run()
    ```

    See [suite-api.md](suite-api.md) for the full `MltkSuite` reference.

---

## Dataset Loading

### `load_dataset()` — CSV and JSON

```python
from mltk.eval.task import load_dataset

# CSV with default column names (input, target)
samples = load_dataset("data.csv")

# CSV with custom column names
samples = load_dataset(
    "trivia.csv",
    input_column="question",
    target_column="answer",
)

# JSON file
samples = load_dataset("data.json")
```

`load_dataset` uses only Python stdlib (`csv`, `json`) — no pandas
required. It automatically maps any columns not named `input_column` or
`target_column` to `EvalSample.metadata`.

### CSV format

```csv
input,target,category,difficulty
"What is the capital of France?","Paris","geography","easy"
"Solve: 15% of 200","30","math","medium"
"What year was Python created?","1991","history","easy"
```

The `input` column is required. The `target` column is optional — rows
without a target value produce `EvalSample(input=..., target=None)`.

### JSON format

```json
[
    {
        "input": "What is the capital of France?",
        "target": "Paris",
        "category": "geography"
    },
    {
        "input": "Solve: 15% of 200",
        "target": "30",
        "category": "math"
    }
]
```

The JSON file must be a top-level array of objects. Each object must
contain the `input_column` key.

### Manual construction

For tests and quick experiments, construct samples directly:

```python
from mltk.eval._types import EvalSample

samples = [
    EvalSample("What is 2+2?", "4"),
    EvalSample("What is 3×7?", "21", metadata={"type": "multiplication"}),
    EvalSample("Capital of Japan?", "Tokyo"),
]
```

---

## Advanced Patterns

### Custom solver: inject system context

```python
from mltk.eval.solvers import Solver
from mltk.eval._types import EvalState
from collections.abc import Callable

class SystemPromptSolver(Solver):
    """Inject a system prompt before generating."""

    def __init__(self, system_prompt: str) -> None:
        self.system_prompt = system_prompt

    def solve(
        self,
        state: EvalState,
        generate: Callable[[str], str],
    ) -> EvalState:
        if state.completed:
            return state

        full_prompt = (
            f"[System]: {self.system_prompt}\n\n"
            f"[User]: {state.sample.input}"
        )
        state.output = generate(full_prompt)
        state.messages.append(
            {"role": "assistant", "content": state.output}
        )
        return state

    @property
    def name(self) -> str:
        return "SystemPromptSolver"
```

### Custom scorer: embedding similarity

```python
from mltk.eval.scorers import Scorer
from mltk.eval._types import EvalState, Score

class EmbeddingSimilarityScorer(Scorer):
    """Score based on cosine similarity between output and target."""

    def __init__(self, embed_fn) -> None:
        self.embed_fn = embed_fn  # Callable[[str], list[float]]

    def score(self, state: EvalState) -> Score:
        if state.sample.target is None:
            return Score(value=0.0, explanation="No target")

        emb_output = self.embed_fn(state.output)
        emb_target = self.embed_fn(state.sample.target)

        # cosine similarity
        dot = sum(a * b for a, b in zip(emb_output, emb_target))
        norm_a = sum(x**2 for x in emb_output) ** 0.5
        norm_b = sum(x**2 for x in emb_target) ** 0.5
        similarity = dot / (norm_a * norm_b + 1e-8)

        return Score(
            value=max(0.0, min(1.0, similarity)),
            explanation=f"Cosine similarity: {similarity:.4f}",
        )

    @property
    def name(self) -> str:
        return "EmbeddingSimilarity"
```

### Solver pipeline: CoT + few-shot combined

```python
from mltk.eval.solvers import (
    ChainOfThoughtSolver,
    FewShotSolver,
    GenerateSolver,
    chain,
)

# Complex pipeline: examples → reasoning → answer
pipeline = chain(
    FewShotSolver(
        examples=[
            ("What is 15% of 200?", "30"),
            ("What is 25% of 80?", "20"),
        ],
    ),
    ChainOfThoughtSolver(
        template=(
            "Solve step by step. "
            "State your final answer as: 'Answer: <value>'"
        )
    ),
    GenerateSolver(),
)

# Pair with PatternScorer to extract the final answer
task = EvalTask(
    name="percent-math",
    solver=pipeline,
    scorers=PatternScorer(),
    dataset=samples,
)
```

### Multi-scorer: triangulate quality

```python
# Run three independent lenses on the same output
task = EvalTask(
    name="open-qa",
    solver=chain(ChainOfThoughtSolver(), GenerateSolver()),
    scorers=[
        ExactMatchScorer(),          # strict: exact string match
        IncludesScorer(),            # lenient: correct answer anywhere
        LLMJudgeScorer(              # qualitative: judge rates quality
            judge_fn=my_judge,
            criterion="correctness",
        ),
    ],
    dataset=samples,
)

result = task.run(model_fn)

# Compare the three signals
exact = result.metrics["ExactMatchScorer/accuracy"]
includes = result.metrics["IncludesScorer/accuracy"]
judge = result.metrics["LLMJudge/correctness/mean"]

print(f"Strict: {exact:.1%}, Lenient: {includes:.1%}, Judge: {judge:.2f}")
```

### A/B testing: compare prompting strategies

```python
from mltk.eval.solvers import GenerateSolver, ChainOfThoughtSolver, chain

samples = load_dataset("math_problems.csv")
scorer = ExactMatchScorer()

# Strategy A: zero-shot
task_zero_shot = EvalTask(
    name="zero-shot",
    solver=GenerateSolver(),
    scorers=scorer,
    dataset=samples,
)

# Strategy B: chain-of-thought
task_cot = EvalTask(
    name="chain-of-thought",
    solver=chain(ChainOfThoughtSolver(), GenerateSolver()),
    scorers=PatternScorer(),   # extract answer from reasoning
    dataset=samples,
)

result_a = task_zero_shot.run(model_fn)
result_b = task_cot.run(model_fn)

accuracy_a = result_a.metrics["ExactMatchScorer/accuracy"]
accuracy_b = result_b.metrics["PatternScorer/accuracy"]

print(f"Zero-shot:       {accuracy_a:.1%}")
print(f"Chain-of-thought: {accuracy_b:.1%}")
print(f"CoT improvement: +{(accuracy_b - accuracy_a):.1%}")
```

---

## Integration with mltk

### MltkSuite integration

`EvalTask.to_test_result()` returns a `TestResult`, making it fully
composable with `MltkSuite`:

```python
from mltk.core.suite import MltkSuite
from mltk.eval.task import EvalTask, load_dataset
from mltk.eval.solvers import GenerateSolver
from mltk.eval.scorers import ExactMatchScorer, LLMJudgeScorer

suite = MltkSuite("model-qa-suite")

# Eval pipeline task
qa_task = EvalTask(
    name="factual-qa",
    solver=GenerateSolver(),
    scorers=[ExactMatchScorer(), LLMJudgeScorer(my_judge)],
    dataset=load_dataset("qa_dataset.csv"),
)
suite.add(qa_task.to_test_result, my_model, min_accuracy=0.85)

# Mix with standard mltk assertions
from mltk.domains.llm.latency import assert_latency
suite.add(assert_latency, my_model, prompts, max_p95_ms=500)

result = suite.run()
print(result.passed)       # True/False
print(result.pass_rate)    # 0.0–1.0
result.to_junit("report.xml")
```

### pytest plugin

In your pytest test file:

```python
# test_model_eval.py
import pytest
from mltk.eval.task import EvalTask, load_dataset
from mltk.eval.solvers import chain, ChainOfThoughtSolver, GenerateSolver
from mltk.eval.scorers import ExactMatchScorer, PatternScorer

@pytest.fixture
def model_fn():
    # Return your real model callable here
    return lambda prompt: "stub"

def test_factual_qa(model_fn):
    task = EvalTask(
        name="factual-qa",
        solver=GenerateSolver(),
        scorers=ExactMatchScorer(),
        dataset=load_dataset("tests/data/qa.csv"),
    )
    result = task.to_test_result(model_fn, min_accuracy=0.85)
    assert result.passed, result.details

def test_math_reasoning(model_fn):
    task = EvalTask(
        name="math-cot",
        solver=chain(ChainOfThoughtSolver(), GenerateSolver()),
        scorers=PatternScorer(),
        dataset=load_dataset("tests/data/math.csv"),
    )
    result = task.to_test_result(model_fn, min_accuracy=0.80)
    assert result.passed, result.details
```

Run with:

```bash
pytest test_model_eval.py -v
```

---

## Examples

### Basic QA evaluation

Evaluate a factual Q&A model with exact match scoring:

```python
from mltk.eval.task import EvalTask, load_dataset
from mltk.eval.solvers import GenerateSolver
from mltk.eval.scorers import ExactMatchScorer

def my_model(prompt: str) -> str:
    # your model here
    ...

samples = load_dataset(
    "trivia.csv",
    input_column="question",
    target_column="answer",
)

task = EvalTask(
    name="trivia-qa",
    solver=GenerateSolver(),
    scorers=ExactMatchScorer(),
    dataset=samples,
)

result = task.run(my_model)
print(f"Accuracy: {result.metrics['ExactMatchScorer/accuracy']:.1%}")
print(f"Evaluated: {result.total_samples} samples")
print(f"Duration: {result.duration_ms:.0f}ms")
```

### Math reasoning (CoT + PatternScorer)

Chain-of-thought reasoning for math problems with answer extraction:

```python
from mltk.eval.task import EvalTask
from mltk.eval.solvers import (
    chain,
    ChainOfThoughtSolver,
    FewShotSolver,
    GenerateSolver,
)
from mltk.eval.scorers import PatternScorer
from mltk.eval._types import EvalSample

samples = [
    EvalSample("What is 15% of 200?", "30"),
    EvalSample("A train travels 120km in 1.5 hours. Speed?", "80"),
    EvalSample("Solve: x + 7 = 15. What is x?", "8"),
]

task = EvalTask(
    name="math-reasoning",
    solver=chain(
        FewShotSolver(
            examples=[("What is 2+2?", "4"), ("What is 3×4?", "12")],
        ),
        ChainOfThoughtSolver(
            template=(
                "Work through the problem step by step. "
                "State your final answer as: 'Answer: <number>'"
            )
        ),
        GenerateSolver(),
    ),
    scorers=PatternScorer(pattern=r"Answer:\s*(\d+(?:\.\d+)?)"),
    dataset=samples,
)

result = task.run(my_model)
assert result.metrics["PatternScorer/accuracy"] >= 0.8
```

### Summarization (LLMJudgeScorer)

Evaluate summarization quality with an LLM judge:

```python
from mltk.eval.task import EvalTask, load_dataset
from mltk.eval.solvers import GenerateSolver
from mltk.eval.scorers import LLMJudgeScorer

def faithfulness_judge(prompt: str) -> float:
    """Return score 0–5 for faithfulness to source."""
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=5,
    )
    return float(response.choices[0].message.content.strip())

summarization_samples = load_dataset(
    "summaries.csv",
    input_column="article",
    target_column="reference_summary",
)

task = EvalTask(
    name="summarization",
    solver=GenerateSolver(),
    scorers=[
        LLMJudgeScorer(
            judge_fn=faithfulness_judge,
            criterion="faithfulness",
            rubric=(
                "Rate how faithfully the summary represents "
                "only facts from the original article, "
                "without hallucination or addition."
            ),
            max_score=5.0,
        ),
        LLMJudgeScorer(
            judge_fn=faithfulness_judge,
            criterion="conciseness",
            rubric=(
                "Rate how concise the summary is. "
                "Penalize unnecessary repetition."
            ),
            max_score=5.0,
        ),
    ],
    dataset=summarization_samples,
)

result = task.run(summarization_model)
print(result.metrics["LLMJudge/faithfulness/mean"])
print(result.metrics["LLMJudge/conciseness/mean"])
```

### Code generation (IncludesScorer)

Verify that generated code contains required elements:

```python
from mltk.eval.task import EvalTask
from mltk.eval.solvers import GenerateSolver
from mltk.eval.scorers import IncludesScorer
from mltk.eval._types import EvalSample

# Target is the function/keyword that must appear in the output
samples = [
    EvalSample(
        "Write a Python function to reverse a list",
        "def ",
    ),
    EvalSample(
        "Implement binary search in Python",
        "while ",  # must use a loop, not recursion
    ),
    EvalSample(
        "Write a Python context manager",
        "__enter__",
    ),
]

task = EvalTask(
    name="codegen-syntax",
    solver=GenerateSolver(),
    scorers=IncludesScorer(ignore_case=False),
    dataset=samples,
)

result = task.run(code_model)
assert result.metrics["IncludesScorer/accuracy"] == 1.0
```

---

## Comparison with Competitors

mltk's eval pipeline is inspired by and compared against the major
frameworks in the field:

| Feature | Inspect AI | Braintrust | LangSmith | HELM | DeepEval | **mltk** |
|---------|-----------|-----------|-----------|------|---------|--------|
| Solver/scorer separation | Yes (async) | Partial | Partial | Yes | No | **Yes** |
| Pytest-native integration | No | No | No | No | No | **Yes** |
| Zero external dependencies | No | No | No | No | No | **Yes** |
| Provider-agnostic model call | Yes | Yes | Yes | Yes | Yes | **Yes** |
| Multi-scorer per task | Yes | Yes | Yes | Yes | Yes | **Yes** |
| Built-in LLM-as-judge | Yes | Yes | Yes | No | Yes | **Yes** |
| Pipeline short-circuiting | Yes | No | No | No | No | **Yes** |
| CSV/JSON dataset loading | Yes | Yes | Yes | Yes | Yes | **Yes** |
| Cloud platform required | No | Optional | Optional | No | Optional | **No** |
| Separate CLI/server | Yes | Yes | Yes | Yes | Optional | **No** |

### What mltk uniquely adds

**pytest-native** — No other framework integrates with pytest as a first
class citizen. `task.to_test_result()` produces a `TestResult` that
participates in `MltkSuite`, generates JUnit XML, and raises
`MltkAssertionError` exactly like all other mltk assertions. Eval results
live in the same output as data quality tests and model performance tests.

**Zero dependencies** — Inspect AI requires `inspect_ai` (13+ transitive
dependencies including `rich`, `click`, `pydantic`, async runtimes).
Braintrust and LangSmith require cloud accounts. mltk's eval pipeline uses
only Python stdlib — it works anywhere Python 3.10+ works.

**Full ML lifecycle** — Other eval frameworks are evaluation-only. mltk
combines the eval pipeline with data quality testing (`assert_no_drift`,
`assert_schema`), model performance testing (`assert_metric`), inference
testing (`assert_latency`), and training testing (`assert_no_leakage`) in
a single tool. One `pytest` command runs the entire ML lifecycle.

**Multi-method dispatch** — mltk's core `assert_*` pattern and the eval
pipeline share the same `TestResult` contract. A single `MltkSuite` can
mix behavioral assertions, eval pipeline results, and structural model
checks — all producing comparable, aggregatable metrics.

---

## Research Citations

The mltk eval pipeline synthesizes ideas from three research lineages:

**Inspect AI (UK AISI, 2024)**
The primary architectural inspiration. Inspect's Solver/Scorer separation
— where solvers define prompting strategy and scorers define grading — is
the core design pattern adapted for mltk. Key differences: mltk is
synchronous (no async overhead for typical batch sizes), pytest-native
(no separate `inspect eval` CLI), and zero-dependency.
Source: `github.com/UKGovernmentBEIS/inspect_ai`
Research brief: `docs/research/inspect-ai-solver-scorer.md`

**Braintrust (2023–2024)**
The cleanest three-primitive separation (`data`, `task`, `scores`) in the
commercial field. mltk adopts Braintrust's insight that the scorer
interface should be unified across code-based, LLM-judge, and custom
scorers — all sharing the same `(state) -> Score` contract.
Source: `braintrustdata.com`
Research brief: `docs/research/eval-pipeline-competitors.md`

**HELM (Stanford CRFM, Liang et al. 2022)**
The reference for multi-metric evaluation architecture. HELM's strict
separation between Adapter (prompt construction) and Metric (scoring) —
using `ScenarioState` as a shared data bus — directly informs mltk's
`EvalState` design. HELM also established the practice of running 7+
metric dimensions simultaneously on the same evaluation run.
Source: `github.com/stanford-crfm/helm`
Research brief: `docs/research/academic-eval-frameworks.md`

**lm-eval-harness (EleutherAI, Gao et al. 2021)**
The framework powering the Hugging Face Open LLM Leaderboard. Its
`filter_list` concept — named post-processing pipelines that transform
raw outputs before scoring — informs `ChainOfThoughtSolver`'s design of
separating reasoning injection from answer extraction.
Source: `github.com/EleutherAI/lm-evaluation-harness`
Research brief: `docs/research/academic-eval-frameworks.md`

**Zheng et al. (2023)** — "Judging LLM-as-a-Judge with MT-Bench and
Chatbot Arena." Established that strong LLM judges achieve >80% agreement
with human annotators, justifying `LLMJudgeScorer` as a production-grade
grading method rather than a research prototype.

---

## See Also

- [llm-judge.md](llm-judge.md) — `judge_fn` pattern, provider examples,
  mock patterns for unit testing `LLMJudgeScorer`
- [suite-api.md](suite-api.md) — `MltkSuite` integration, JUnit export,
  programmatic pass/fail
- [behavioral-consistency.md](behavioral-consistency.md) — Behavioral
  assertions that complement eval pipeline grading
- [pytest-plugin.md](pytest-plugin.md) — pytest plugin configuration and
  `--mltk-yaml` flag for YAML-driven eval
