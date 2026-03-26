# LLM Evaluation

Lightweight LLM/GenAI evaluation assertions — no external model dependencies. Covers semantic similarity, toxicity detection, hallucination checking, and LLM-specific latency metrics (TTFT/ITL).

**Module:** `mltk.domains.llm`

For RAG-specific assertions (faithfulness, context relevancy, answer relevancy, context precision, context recall) and agentic evaluation (tool-call accuracy, step-count bounds), see the full reference in [RAG, Agentic & Text Quality Evaluation](rag-evaluation.md).

---

## Similarity

### assert_semantic_similarity

Assert semantic similarity between reference and generated texts meets a minimum threshold. Supports token-level F1 (default, no dependencies) or embedding cosine similarity.

```python
from mltk.domains.llm import assert_semantic_similarity

references = ["The cat sat on the mat."]
hypotheses = ["A cat was sitting on a mat."]
assert_semantic_similarity(references, hypotheses, min_score=0.5, method="token_f1")
```

#### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `references` | `list[str]` | *(required)* | Reference texts |
| `hypotheses` | `list[str]` | *(required)* | Model-generated texts |
| `min_score` | `float` | `0.5` | Minimum required similarity score (0-1) |
| `method` | `str` | `"token_f1"` | Comparison method: `"token_f1"` or `"cosine"` |

---

## Safety

### assert_no_toxicity

Assert that the fraction of toxic texts stays below a threshold. Uses regex/keyword-based detection (no external model).

```python
from mltk.domains.llm import assert_no_toxicity

texts = ["Great product!", "This is terrible garbage."]
assert_no_toxicity(texts, max_toxic_pct=0.1)
```

#### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `texts` | `list[str]` | *(required)* | Texts to check |
| `max_toxic_pct` | `float` | `0.1` | Maximum fraction of texts allowed to be toxic (0-1) |
| `patterns` | `list[str] \| None` | `None` | Custom regex patterns for toxicity. None = built-in patterns |

### assert_no_hallucination

Assert that model claims are grounded in provided source material. Uses keyword overlap factuality checking (no LLM judge required).

```python
from mltk.domains.llm import assert_no_hallucination

claims = ["Paris is the capital of France."]
sources = ["France is a country in Europe. Its capital city is Paris."]
assert_no_hallucination(claims, sources, method="keyword_overlap")
```

#### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `claims` | `list[str]` | *(required)* | Claims or statements to verify |
| `sources` | `list[str]` | *(required)* | Source/context documents for grounding |
| `method` | `str` | `"keyword_overlap"` | Verification method |

---

## LLM Latency

### assert_ttft

Assert Time to First Token is within an acceptable latency bound. Calls the provided function and measures time until the first yielded/returned token.

```python
from mltk.domains.llm import assert_ttft

assert_ttft(my_streaming_fn, "prompt text", max_ms=500)
```

#### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `func` | `Callable` | *(required)* | Streaming function to measure |
| `*args` | | | Arguments to pass to `func` |
| `max_ms` | `float` | `500` | Maximum allowed TTFT in milliseconds |

### assert_itl

Assert Inter-Token Latency stays within bounds. Measures the average delay between consecutive tokens from a streaming function.

```python
from mltk.domains.llm import assert_itl

assert_itl(my_streaming_fn, "prompt text", max_ms=100)
```

#### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `func` | `Callable` | *(required)* | Streaming function to measure |
| `*args` | | | Arguments to pass to `func` |
| `max_ms` | `float` | `100` | Maximum allowed average inter-token latency in milliseconds |

---
