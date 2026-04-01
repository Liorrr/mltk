# LLM Judge Defaults

Configure a default judge once and every subjective assertion uses it automatically --
no more passing `judge_fn` to every call.

**Module:** `mltk.domains.llm.judge_defaults`

---

## Why Default Judges?

### The repetition problem

Every subjective mltk assertion (faithfulness, coherence, toxicity via LLM, etc.)
accepts a `judge_fn` callable. That design gives you full control: pick any
provider, any model, any cost point. But in a real test suite with 20+ assertions,
the same `judge_fn` setup appears 20+ times:

```python
# Without defaults -- repeated in every test file
def openai_judge(prompt: str) -> float:
    return float(
        client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        ).choices[0].message.content.strip()
    )

assert_faithfulness(answers, contexts, judge_fn=openai_judge)
assert_coherence(texts, judge_fn=openai_judge)
assert_llm_judge_score(
    judge_fn=openai_judge, prompts=p, responses=r, criterion="helpfulness"
)
# ...17 more assertions all repeating the same judge_fn
```

`configure_default_judge` solves this by registering a module-level default that
every assertion falls back to when `judge_fn` is not provided:

```python
# Once at the top of conftest.py or test setup
from mltk.domains.llm.judge_defaults import configure_default_judge
configure_default_judge(openai_judge)

# Now every assertion finds the judge automatically
assert_faithfulness(answers, contexts)          # uses openai_judge
assert_coherence(texts)                         # uses openai_judge
assert_llm_judge_score(prompts=p, responses=r)  # uses openai_judge
```

---

## `configure_default_judge`

Register a callable as the module-level default judge. Called once per session.

```python
from mltk.domains.llm.judge_defaults import configure_default_judge

configure_default_judge(
    judge_fn: Callable[[str], str] | None,
) -> None
```

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `judge_fn` | `Callable[[str], str] \| None` | *(required)* | Callable that accepts an evaluation prompt string and returns the judge response string. Pass `None` to clear the default and revert to requiring explicit `judge_fn` arguments. |

### Effect

After calling `configure_default_judge(fn)`:

- Any assertion that accepts `judge_fn` will use `fn` when the argument is omitted.
- The default is stored at module level in `mltk.domains.llm.judge_defaults._default_judge`.
- Thread-safe: protected by a module-level lock.
- Calling `configure_default_judge` again replaces the previous default.
- Pass `None` to clear the default and revert to requiring explicit `judge_fn` arguments.

### Example: OpenAI judge

```python
import openai
from mltk.domains.llm.judge_defaults import configure_default_judge

client = openai.OpenAI()

def openai_judge(prompt: str) -> float:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=10,
        temperature=0.0,
    )
    return float(response.choices[0].message.content.strip())

configure_default_judge(openai_judge)
```

### Example: Ollama judge (local, no API key)

```python
import urllib.request
import json
from mltk.domains.llm.judge_defaults import configure_default_judge

def ollama_judge(prompt: str) -> float:
    """Call a local Ollama model. Requires `ollama serve` running."""
    payload = json.dumps({
        "model": "llama3.2",
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }).encode()
    req = urllib.request.Request(
        "http://localhost:11434/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as resp:
        body = json.loads(resp.read())
    text = body["message"]["content"].strip()
    # Extract numeric score from response
    import re
    match = re.search(r"\b([0-5](?:\.\d+)?)\b", text)
    return float(match.group(1)) if match else 0.0

configure_default_judge(ollama_judge)
```

### Example: Mock judge for unit tests

```python
from mltk.domains.llm.judge_defaults import configure_default_judge

# Fast, deterministic, zero cost -- ideal for unit tests
configure_default_judge(lambda prompt: "4.0")
```

---

## Clearing the Default Judge

To remove the currently registered default judge, pass `None`:

```python
from mltk.domains.llm.judge_defaults import configure_default_judge

configure_default_judge(None)
```

After clearing, assertions that use `resolve_judge` with `method="auto"` will
fall back to the `fallback_method` (typically `"lexical"`).

---

## `get_default_judge`

Retrieve the currently registered default judge callable (or `None`).

```python
from mltk.domains.llm.judge_defaults import get_default_judge

fn = get_default_judge()  # Returns the callable or None
```

---

## Auto-Fallback Chain

When an assertion needs a judge and no `judge_fn` was provided, mltk applies this
fallback chain in order:

```
1. Explicit judge_fn argument      -- Always used if provided
2. Default judge (module-level)    -- Used if configure_default_judge was called
3. Embedding similarity            -- Used if sentence-transformers is installed
4. Lexical overlap (token F1)      -- Always available, no dependencies
```

The fallback chain applies per assertion. Some assertions (e.g., `assert_faithfulness`)
can meaningfully use embedding or lexical methods and always have a fallback path.
Others (e.g., `assert_llm_judge_score`) require a judge and raise `ValueError` if no
judge is available via the first two steps.

### Fallback behavior by assertion

| Assertion | Without any judge | With default judge |
|-----------|------------------|--------------------|
| `assert_faithfulness` | Lexical NLI fallback | Judge handles it |
| `assert_coherence` | Embedding cosine fallback | Judge handles it |
| `assert_no_toxicity` | Regex patterns (built-in) | Judge handles it |
| `assert_semantic_similarity` | Token F1 | Judge handles it |
| `assert_llm_judge_score` | `ValueError` -- judge required | Default judge used |
| `assert_llm_judge_pairwise` | `ValueError` -- judge required | Default judge used |

### Disabling fallbacks

To force all assertions to use a judge (and fail explicitly if none is configured),
set the `require_judge` flag per call:

```python
# Forces a judge -- falls back to nothing, raises if no judge configured
assert_faithfulness(
    answers,
    contexts,
    require_judge=True,
)
```

---

## Integration with Existing Assertions

### `assert_faithfulness`

Checks whether generated answers are grounded in the provided context.
With a default judge, the LLM evaluates semantic entailment instead of using
keyword overlap.

```python
from mltk.domains.llm.judge_defaults import configure_default_judge
from mltk.domains.llm import assert_faithfulness

configure_default_judge(lambda prompt: "4.2")

result = assert_faithfulness(
    answers=["The report confirms Q3 revenue grew 12%."],
    contexts=["Q3 revenue increased by 12% year-over-year."],
    threshold=0.8,
    # judge_fn omitted -- default judge used automatically
)
assert result.passed
print(result.details["method"])  # "judge"
```

### `assert_no_toxicity` with LLM judge

The built-in regex detection is fast but narrow. A judge catches nuanced cases
(sarcasm, coded language, context-dependent offensiveness):

```python
from mltk.domains.llm.judge_defaults import configure_default_judge
from mltk.domains.llm import assert_no_toxicity

configure_default_judge(openai_judge)

result = assert_no_toxicity(
    texts=["Yeah, totally brilliant move there."],
    max_toxic_pct=0.0,
    use_judge=True,  # Opt-in to judge -- regex not always sufficient
)
```

### `assert_coherence`

Evaluates whether the output has clear logical structure and readable flow.
Without a judge this falls back to embedding coherence proxies. With a judge:

```python
from mltk.domains.llm import assert_coherence

result = assert_coherence(
    texts=model_outputs,
    threshold=0.75,
    # Default judge used if configured
)
print(result.details["method"])  # "judge" or "embedding"
```

---

## pytest Integration

### conftest.py pattern

Set the default judge once in `conftest.py`. Every test file in the session
inherits it automatically.

```python
# conftest.py
import pytest
from mltk.domains.llm.judge_defaults import configure_default_judge


def pytest_configure(config):
    """Register default judge before any tests run."""
    configure_default_judge(lambda prompt: "4.0")


def pytest_unconfigure(config):
    """Clean up after test session."""
    configure_default_judge(None)
```

### Swapping judges per test class

```python
import pytest
from mltk.domains.llm.judge_defaults import configure_default_judge

class TestWithMockJudge:
    @pytest.fixture(autouse=True)
    def use_mock_judge(self):
        configure_default_judge(lambda p: "4.0")
        yield
        configure_default_judge(None)

    def test_faithfulness_passes(self):
        from mltk.domains.llm import assert_faithfulness
        result = assert_faithfulness(
            answers=["Paris is the capital of France."],
            contexts=["France's capital is Paris."],
        )
        assert result.passed


class TestWithRealJudge:
    @pytest.fixture(autouse=True)
    def use_ollama_judge(self, ollama_judge_fn):
        configure_default_judge(ollama_judge_fn)
        yield
        configure_default_judge(None)
```

### Parametrize judge backends

```python
import pytest
from mltk.domains.llm.judge_defaults import configure_default_judge
from mltk.domains.llm import assert_faithfulness

def make_mock_judge(score: float):
    return lambda prompt: score

@pytest.mark.parametrize("score,should_pass", [
    (4.5, True),
    (2.0, False),
])
def test_faithfulness_thresholds(score, should_pass):
    configure_default_judge(make_mock_judge(score))
    result = assert_faithfulness(
        answers=["The sky is blue."],
        contexts=["Scientific studies confirm the sky appears blue."],
        threshold=0.75,
    )
    assert result.passed == should_pass
```

---

## Environment-Based Judge Selection

For teams that run different judges in CI vs production, use an environment
variable to select the backend:

```python
# conftest.py
import os
from mltk.domains.llm.judge_defaults import configure_default_judge

def _build_judge():
    backend = os.environ.get("MLTK_JUDGE_BACKEND", "mock")

    if backend == "openai":
        import openai
        client = openai.OpenAI()
        def _fn(prompt: str) -> float:
            r = client.chat.completions.create(
                model=os.environ.get("MLTK_JUDGE_MODEL", "gpt-4o-mini"),
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0.0,
            )
            return float(r.choices[0].message.content.strip())
        return _fn

    if backend == "ollama":
        import urllib.request, json, re
        model = os.environ.get("MLTK_JUDGE_MODEL", "llama3.2")
        def _fn(prompt: str) -> float:
            payload = json.dumps({
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
            }).encode()
            req = urllib.request.Request(
                "http://localhost:11434/api/chat",
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req) as resp:
                body = json.loads(resp.read())
            text = body["message"]["content"].strip()
            match = re.search(r"\b([0-5](?:\.\d+)?)\b", text)
            return float(match.group(1)) if match else 0.0
        return _fn

    # default: mock judge for local/CI runs with no LLM access
    return lambda prompt: 4.0


def pytest_configure(config):
    configure_default_judge(_build_judge())
```

Run with:

```bash
# CI gate (fast, free, deterministic)
pytest tests/llm/

# Staging gate (real Ollama judge, no API key needed)
MLTK_JUDGE_BACKEND=ollama pytest tests/llm/

# Production pre-deploy validation (highest accuracy)
MLTK_JUDGE_BACKEND=openai MLTK_JUDGE_MODEL=gpt-4o pytest tests/llm/
```

---

## Error Handling

| Scenario | Behavior |
|----------|----------|
| `judge_fn` omitted, no default configured, fallback available | Fallback method (embedding / lexical) is used |
| `judge_fn` omitted, no default, no fallback (`require_judge=True`) | `ValueError` with message explaining how to configure |
| `judge_fn` raises exception during scoring | Item scores `0.0`; error flag in `per_item_scores`; assertion does not crash |
| `judge_fn` returns non-numeric text | Score `0.0` extracted from first number found; if none, `0.0` |
| `configure_default_judge` called with `None` | Clears the default; assertions fall back to `fallback_method` |
