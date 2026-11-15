# LLM Judge Defaults

Configure a default judge once for `assert_with_judge`, so that assertion does not
need a `judge_fn` on every call. Other subjective assertions do not implicitly read
this setting.

**Module:** `mltk.domains.llm.judge_defaults`

---

## Why Default Judges?

### The repetition problem

`assert_with_judge` accepts a `judge_fn` callable. That design gives you full control: pick any
provider, any model, any cost point. But in a real test suite with 20+ assertions,
the same `judge_fn` setup appears 20+ times:

```python
from mltk.domains.llm.judge import assert_llm_judge_score
from mltk.domains.llm.rag import assert_faithfulness

# Without defaults -- repeated in every test file
def judge(prompt: str) -> float:
    return 4.0


def faithfulness_judge(answer: str, context: str) -> float:
    return 0.9


answer = "The answer is grounded."
context = "The context supports the answer."
prompts = ["question"]
responses = ["answer"]
assert_faithfulness(answer, context, method="llm", judge_fn=faithfulness_judge)
assert_llm_judge_score(
    judge_fn=judge, prompts=prompts, responses=responses, criterion="helpfulness"
)
# ...17 more assertions all repeating the same judge_fn
```

`configure_default_judge` registers a module-level default used by
`assert_with_judge` when `judge_fn` is not provided:

```python
# Once at the top of conftest.py or test setup
from mltk.domains.llm.judge_defaults import configure_default_judge

openai_judge = lambda prompt: "0.8"
answer = "The answer is grounded."
context = "The context supports the answer."
configure_default_judge(openai_judge)

# The convenience assertion finds the judge automatically
from mltk.domains.llm.judge_defaults import assert_with_judge
assert_with_judge("faithfulness", answer, context)  # uses openai_judge

# RAG, safety, and assert_llm_judge_score still require their own judge_fn.
```

---

## `configure_default_judge`

Register a callable as the module-level default judge. Called once per session.

```python
from collections.abc import Callable
from mltk.domains.llm.judge_defaults import configure_default_judge

def configure_default_judge(judge_fn: Callable[[str], str] | None) -> None: ...
```

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `judge_fn` | `Callable[[str], str] \| None` | *(required)* | Callable that accepts an evaluation prompt string and returns the judge response string. Pass `None` to clear the default; `assert_with_judge` then uses its `fallback_method`, while other assertions never read this default. |

### Effect

After calling `configure_default_judge(fn)`:

- `assert_with_judge` uses `fn` when its `judge_fn` argument is omitted.
- The default is stored at module level in `mltk.domains.llm.judge_defaults._default_judge`.
- Thread-safe: protected by a module-level lock.
- Calling `configure_default_judge` again replaces the previous default.
- Pass `None` to clear the default; `assert_with_judge` then uses its `fallback_method`, while other assertions never read this default.

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

def ollama_judge(prompt: str) -> str:
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
    return body["message"]["content"]

configure_default_judge(ollama_judge)
```

### Example: Mock judge for unit tests

```python
from mltk.domains.llm.judge_defaults import configure_default_judge

# Fast, deterministic, zero cost -- ideal for unit tests
configure_default_judge(lambda prompt: "0.8")
```

---

## Clearing the Default Judge

To remove the currently registered default judge, pass `None`:

```python
from mltk.domains.llm.judge_defaults import configure_default_judge

configure_default_judge(None)
```

After clearing, `assert_with_judge` uses its `fallback_method` (lexical overlap
by default). `resolve_judge` is a helper for custom assertions; no built-in
assertion calls it.

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
2. Default judge (module-level)    -- Used by assert_with_judge if configured
3. Lexical token coverage          -- Used by assert_with_judge when no judge exists
```

The fallback chain applies to `assert_with_judge`. Its lexical score is the share
of tokens in the first text that also appear in the reference text, not an F1 score.
RAG and safety assertions do not
consume this module-level default, and `assert_llm_judge_score` requires its own
judge function.

### Fallback behavior by assertion

| Assertion | Without any judge | With default judge |
|-----------|------------------|--------------------|
| `assert_with_judge` | Lexical token coverage | Default judge used |
| RAG and safety assertions | Their configured fallback or `judge_fn` requirement | Default ignored |
| `assert_llm_judge_score` | `judge_fn` is a required argument | Default ignored |
| `assert_llm_judge_pairwise` | `judge_fn` is a required argument | Default ignored |

### Requiring a judge

`assert_faithfulness(answer, context, ..., method="llm", judge_fn=None)` fails
with `method='llm' requires a judge_fn callable`; it does not fall back or read
the configured default judge. Its judge callback receives `(answer_or_text,
context)` and returns a float. `assert_with_judge` has no flag that disables its
lexical fallback; set `fallback_method` to something other than `"lexical"` to
fail when no judge is available.

```python
from mltk.domains.llm.rag import assert_faithfulness

def my_judge(answer: str, context: str) -> float:
    return 0.9

answer = "The sky is blue."
context = "Scientific studies confirm the sky appears blue."
assert_faithfulness(
    answer,
    context,
    min_score=0.7,
    method="llm",
    judge_fn=my_judge,
)
```

---

## Integration with Existing Assertions

### Using `assert_with_judge` for faithfulness

Checks whether generated answers are grounded in the provided context using the
configured default judge.

```python
from mltk.domains.llm.judge_defaults import (
    assert_with_judge,
    configure_default_judge,
)

configure_default_judge(lambda prompt: "0.8")

result = assert_with_judge(
    "faithfulness",
    "The report confirms Q3 revenue grew 12%.",
    "Q3 revenue increased by 12% year-over-year.",
    min_score=0.8,
)
assert result.passed
print(result.details["method"])  # "llm"
```

### Other assertions

The built-in regex detection is fast but narrow. For nuanced toxicity detection,
use `method="classifier"`, which runs a local toxicity classifier and requires
downloading its model. `configure_default_judge` does not affect toxicity.

```python
from mltk.domains.llm import assert_no_toxicity

result = assert_no_toxicity(
    texts=["Yeah, totally brilliant move there."],
    max_toxic_pct=0.0,
    method="classifier",
)
```

`configure_default_judge` is not consumed by the RAG and safety assertions or by
`assert_llm_judge_score`; pass their own `judge_fn` where supported.

---

## pytest Integration

### conftest.py pattern

Set the default judge once in `conftest.py` for tests that call
`assert_with_judge`.

```python
# conftest.py
import pytest
from mltk.domains.llm.judge_defaults import configure_default_judge


def pytest_configure(config):
    """Register default judge before any tests run."""
    configure_default_judge(lambda prompt: "0.8")


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
        configure_default_judge(lambda p: "0.8")
        yield
        configure_default_judge(None)

    def test_with_judge_passes(self):
        from mltk.domains.llm.judge_defaults import assert_with_judge
        result = assert_with_judge(
            "faithfulness",
            "Paris is the capital of France.",
            "France's capital is Paris.",
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
from mltk.core.assertion import MltkAssertionError
import pytest
from mltk.domains.llm.judge_defaults import assert_with_judge, configure_default_judge

def make_mock_judge(score: float):
    return lambda prompt: score

@pytest.mark.parametrize("score,should_pass", [
    (0.9, True),
    (0.2, False),
])
def test_judge_thresholds(score, should_pass):
    configure_default_judge(make_mock_judge(score))
    if should_pass:
        result = assert_with_judge(
            "faithfulness",
            "The sky is blue.",
            "Scientific studies confirm the sky appears blue.",
            min_score=0.75,
        )
        assert result.passed
    else:
        with pytest.raises(MltkAssertionError):
            assert_with_judge(
                "faithfulness",
                "The sky is blue.",
                "Scientific studies confirm the sky appears blue.",
                min_score=0.75,
            )
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
        import urllib.request, json
        model = os.environ.get("MLTK_JUDGE_MODEL", "llama3.2")
        def _fn(prompt: str) -> str:
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
            return body["message"]["content"]
        return _fn

    # default: mock judge for local/CI runs with no LLM access
    return lambda prompt: "0.8"


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
| `judge_fn` omitted, no default configured, fallback available | `assert_with_judge` uses lexical token coverage (`fallback_method="lexical"`, the only built-in fallback) |
| `judge_fn` omitted, no default, `fallback_method` is not `"lexical"` | The assertion fails with `No judge available and fallback method '<name>' requires one.` |
| `judge_fn` raises exception during scoring | It is a judge error governed by `on_error`: `fail_if_all` fails when all items error, `strict` fails on any error, and `zero` zero-fills |
| Default judge (set with `configure_default_judge`) returns text | `assert_with_judge` uses the strict parser on a 0–1 scale: JSON `"score"` is authoritative, fenced/prefixed JSON is accepted, and `N/max` requires denominator `1`. `legacy_parse=True` restores first-number scraping. Unparseable or out-of-range scores are errors governed by `on_error`; scores outside `[0, 1]` are never clamped. |
| `configure_default_judge` called with `None` | Clears the default; `assert_with_judge` falls back to its `fallback_method`. Other assertions never read the default. |
