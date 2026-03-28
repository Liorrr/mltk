# Summarization Evaluation Metrics

Lightweight assertions for evaluating LLM-generated summaries. No external model dependencies -- all checks run in-process using token overlap and character length ratios.

**Module:** `mltk.domains.llm.summarization`

---

## Why Summarization Metrics?

A good summary must satisfy three properties simultaneously:

| Property | Question it answers | Failure signal |
|----------|-------------------|----------------|
| **Coverage** | Did the summary retain the source's key content? | Important information was dropped |
| **Compression** | Is the summary actually shorter than the source? | "Summary" is barely shorter (or absurdly terse) |
| **Faithfulness** | Does the summary stay grounded in the source? | Summary hallucinated content not in the source |

These three metrics form a **triangle**: optimizing one can hurt another. A summary that copies the source verbatim has perfect coverage and faithfulness but zero compression. A one-word summary has great compression but terrible coverage. A summary full of hallucinated claims may read fluently but has low faithfulness.

Testing all three together gives you confidence that the summary is actually useful.

---

## Metrics

### assert_summary_coverage

Measures what fraction of the source's vocabulary the summary preserves.

**Formula:**

```
coverage = |source_tokens & summary_tokens| / |source_tokens|
```

**When to use:** When you need to verify the summary did not drop important content. This is the **recall** side of summarization quality.

```python
from mltk.domains.llm.summarization import assert_summary_coverage

source = (
    "Machine learning uses data to train predictive models. "
    "These models generalize patterns from training examples."
)
summary = "Machine learning trains models using data."

result = assert_summary_coverage(source, summary, min_coverage=0.3)
# result.details["coverage"] -> 0.5+ (shares "machine", "learning", "data", "models", "train"...)
```

#### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `source` | `str` | *(required)* | Original text being summarized |
| `summary` | `str` | *(required)* | Generated summary to evaluate |
| `min_coverage` | `float` | `0.3` | Minimum required coverage ratio (0.0--1.0) |

#### Result Details

| Key | Type | Description |
|-----|------|-------------|
| `coverage` | `float` | Computed coverage ratio |
| `min_coverage` | `float` | Threshold that was required |
| `source_tokens` | `int` | Number of unique tokens in the source |
| `summary_tokens` | `int` | Number of unique tokens in the summary |
| `common_tokens` | `int` | Number of tokens shared between source and summary |

---

### assert_summary_compression

Measures whether the summary is appropriately shorter than the source.

**Formula:**

```
compression_ratio = len(summary) / len(source)
```

**When to use:** As a structural sanity check. A "summary" that is 95% the length of the source is not summarizing. A "summary" that is 1% of the source probably lost critical information.

```python
from mltk.domains.llm.summarization import assert_summary_compression

source = "A detailed article about climate change... " * 50
summary = "Climate change affects global temperatures and ecosystems."

result = assert_summary_compression(
    source, summary,
    min_ratio=0.01,  # at least 1% of source
    max_ratio=0.5,   # at most 50% of source
)
```

#### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `source` | `str` | *(required)* | Original text being summarized |
| `summary` | `str` | *(required)* | Generated summary to evaluate |
| `min_ratio` | `float` | `0.1` | Minimum compression ratio (summary must be at least this fraction) |
| `max_ratio` | `float` | `0.5` | Maximum compression ratio (summary must be at most this fraction) |

#### Result Details

| Key | Type | Description |
|-----|------|-------------|
| `compression_ratio` | `float` | Computed ratio `len(summary) / len(source)` |
| `min_ratio` | `float` | Lower bound that was required |
| `max_ratio` | `float` | Upper bound that was required |
| `source_length` | `int` | Character length of the source |
| `summary_length` | `int` | Character length of the summary |

---

### assert_summary_faithfulness

Measures what fraction of the summary's content comes from the source (vs. hallucinated).

**Formula:**

```
faithfulness = |summary_tokens & source_tokens| / |summary_tokens|
```

**When to use:** When you need to detect hallucinated content in summaries. This is the **precision** side of summarization quality. A summary can cover key topics but ADD information not present in the source -- faithfulness catches that.

```python
from mltk.domains.llm.summarization import assert_summary_faithfulness

source = "Python is a high-level programming language known for readability."
summary = "Python is a programming language."  # all words from source

result = assert_summary_faithfulness(source, summary, min_faithfulness=0.5)
# result.details["faithfulness"] -> 1.0 (no novel tokens)
```

#### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `source` | `str` | *(required)* | Original text being summarized |
| `summary` | `str` | *(required)* | Generated summary to evaluate |
| `min_faithfulness` | `float` | `0.5` | Minimum required faithfulness ratio (0.0--1.0) |

#### Result Details

| Key | Type | Description |
|-----|------|-------------|
| `faithfulness` | `float` | Computed faithfulness ratio |
| `min_faithfulness` | `float` | Threshold that was required |
| `summary_tokens` | `int` | Number of unique tokens in the summary |
| `source_tokens` | `int` | Number of unique tokens in the source |
| `novel_tokens` | `int` | Tokens in the summary that are NOT in the source |

---

## When to Use Each Metric

| Goal | Metric | Analogy |
|------|--------|---------|
| "Did the summary capture the source?" | `assert_summary_coverage` | Recall |
| "Did the summary stay faithful?" | `assert_summary_faithfulness` | Precision |
| "Is it actually a summary?" | `assert_summary_compression` | Efficiency |

Use all three together for a complete evaluation:

```python
from mltk.domains.llm.summarization import (
    assert_summary_compression,
    assert_summary_coverage,
    assert_summary_faithfulness,
)

source = "..."  # original document
summary = "..."  # LLM-generated summary

# 1. Did it keep the important parts?
assert_summary_coverage(source, summary, min_coverage=0.3)

# 2. Is it actually shorter?
assert_summary_compression(source, summary, min_ratio=0.1, max_ratio=0.5)

# 3. Did it stay grounded in the source?
assert_summary_faithfulness(source, summary, min_faithfulness=0.5)
```

---

## Comparison with ROUGE

ROUGE (Recall-Oriented Understudy for Gisting Evaluation) is the traditional summarization metric. Here is how mltk's metrics relate:

| Aspect | ROUGE | mltk summarization |
|--------|-------|--------------------|
| **Approach** | N-gram overlap (ROUGE-1, ROUGE-2, ROUGE-L) | Token-set overlap (unigram) |
| **Dependencies** | Requires `rouge-score` or similar library | Pure Python, zero dependencies |
| **Speed** | Fast | Faster (set operations only) |
| **Reference needed?** | Yes (gold reference summary) | No (compares against source directly) |
| **Precision/Recall** | ROUGE-P, ROUGE-R, ROUGE-F | Coverage (recall), Faithfulness (precision) |
| **Compression** | Not measured | Built-in `assert_summary_compression` |
| **N-gram order** | Captures bigram/trigram patterns | Unigram only (simpler, less sensitive to phrasing) |

**When to prefer mltk metrics:** When you want fast, dependency-free smoke tests that compare the summary directly against the source document (no gold reference needed). Ideal for CI/CD pipelines and automated testing.

**When to prefer ROUGE:** When you have gold-reference summaries and need n-gram-level evaluation that is sensitive to word order and phrasing.
