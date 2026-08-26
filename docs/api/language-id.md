# Language Identification

Gate generated or ingested text on a detected ISO 639-1 language.

**Module:** `mltk.domains.nlp.language`

**Install:** `pip install mlspec[langdetect]`

---

### assert_language

Uses `langdetect` with a fixed seed so CI is deterministic. Empty or
whitespace-only text fails closed (does not score as a language).

```python
from mltk.domains.nlp import assert_language

assert_language(
    "The cat sat on the mat and watched the birds outside.",
    expected="en",
    min_prob=0.5,
)
```

#### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `text` | `str` | *(required)* | Text to classify |
| `expected` | `str` | *(required)* | ISO 639-1 code (`en`, `fr`, …) |
| `min_prob` | `float \| None` | `None` | Optional minimum probability for `expected` |
| `severity` | `Severity` | `CRITICAL` | CRITICAL raises on failure |

#### Returns

`TestResult` named `nlp.language` with `detected`, `expected`, and
`probability` in `details`.
