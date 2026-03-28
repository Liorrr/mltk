# Retrieval Metrics

Assertions for evaluating the ranking quality of search engines, recommendation systems, and RAG retrievers. Pure Python -- no external dependencies.

**Module:** `mltk.domains.llm.retrieval`

---

## Why Retrieval Metrics

RAG has two halves: a **retriever** that selects and ranks documents, and a **generator** (LLM) that produces an answer from those documents. mltk already provided five generator-side assertions (faithfulness, context precision, context recall, context relevancy, answer relevancy) to catch hallucination and grounding failures. But if the retriever feeds the wrong documents to the LLM, even a perfect generator produces garbage.

These four retrieval assertions close the gap. They evaluate the retriever in isolation, before the LLM ever sees the results, answering the question: **did we fetch the right documents, in the right order?**

---

## The Two Halves of RAG Testing

```
    User Query
        |
        v
  +-------------------+          +---------------------------+
  |     RETRIEVER      |          |        GENERATOR          |
  |   (search/embed)   |          |          (LLM)            |
  +---------+---------+          +-----------+---------------+
            |                                |
      Ranked Documents                 Generated Answer
            |                                |
  +---------v----------+          +----------v-----------------+
  |  RETRIEVER METRICS  |          |    GENERATOR METRICS       |
  |                     |          |                            |
  |  assert_ndcg        |          |  assert_faithfulness       |
  |  assert_mrr         |          |  assert_context_precision  |
  |  assert_recall_at_k |          |  assert_context_recall     |
  |  assert_map_at_k    |          |  assert_context_relevancy  |
  |                     |          |  assert_answer_relevancy   |
  +---------------------+          +----------------------------+
```

**Retriever metrics** answer: "Did we fetch the right documents, ranked correctly?"

**Generator metrics** answer: "Did the LLM use those documents faithfully and relevantly?"

Test both halves. A RAG system is only as strong as its weakest half.

---

## assert_ndcg

Assert that mean Normalized Discounted Cumulative Gain at K meets a minimum threshold.

nDCG is the standard metric for evaluating ranked retrieval when you have **graded relevance labels** (e.g., 0 = irrelevant, 1 = somewhat relevant, 2 = relevant, 3 = highly relevant). Documents ranked higher contribute more to the score than documents ranked lower.

### Formula

For a single query:

```
DCG@k  = sum_{i=1}^{k} (2^rel_i - 1) / log2(i + 1)
IDCG@k = DCG@k computed on the ideal (perfectly sorted) ranking
nDCG@k = DCG@k / IDCG@k
```

The final score is the **mean nDCG@k across all queries**. A score of 1.0 means perfect ranking for every query.

**Why the logarithmic discount?** Users scan results top-to-bottom. A relevant document at position 1 is far more valuable than one at position 10. The `log2(i + 1)` denominator encodes this: position 1 gets a discount of 1.0, position 2 gets 0.63, position 10 gets 0.29.

**Why the exponential gain?** The `2^rel - 1` numerator separates relevance grades exponentially. A grade-3 document contributes 7.0 while a grade-1 document contributes only 1.0. This reflects the reality that a "highly relevant" document is not just slightly better than "somewhat relevant" -- it is dramatically better.

### When to use

- You have **graded relevance labels** (not just binary relevant/irrelevant).
- You care about **position** -- a relevant document at rank 1 should count more than at rank 10.
- Standard metric for search engines, recommendation systems, and RAG retriever evaluation.

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `y_true` | `list[list[int]]` | *(required)* | Relevance labels per query. `y_true[i]` is a list of integer grades (e.g., `[3, 2, 0, 1]`) for the documents of query *i*. |
| `y_scores` | `list[list[float]]` | *(required)* | Model-predicted scores per query. `y_scores[i]` must have the same length as `y_true[i]`. Higher score = higher predicted rank. |
| `k` | `int` | `10` | Cutoff position. Only the top *k* documents (sorted by `y_scores`) contribute to the score. |
| `min_ndcg` | `float` | `0.8` | Minimum acceptable mean nDCG@k. The assertion fails if the computed score is below this threshold. |

### Returns

`TestResult` with details:

- `ndcg` -- computed mean nDCG@k
- `min_ndcg` -- threshold used
- `k` -- cutoff position
- `num_queries` -- number of queries evaluated
- `per_query_ndcg` -- list of nDCG@k per query (for debugging)

### Example

```python
from mltk.domains.llm.retrieval import assert_ndcg

# Two queries, each with 4 documents
# Relevance: 0=irrelevant, 1=marginal, 2=relevant, 3=highly relevant
y_true = [
    [3, 2, 0, 1],   # query 1: doc0 is best, doc1 good, doc3 marginal, doc2 irrelevant
    [1, 0, 0, 1],   # query 2: doc0 and doc3 are marginally relevant
]

# Predicted scores (higher = model thinks more relevant)
y_scores = [
    [0.9, 0.8, 0.2, 0.5],   # model ranks: doc0 > doc1 > doc3 > doc2 (good!)
    [0.8, 0.1, 0.3, 0.7],   # model ranks: doc0 > doc3 > doc2 > doc1 (reasonable)
]

# Assert mean nDCG@4 is at least 0.5
result = assert_ndcg(y_true, y_scores, k=4, min_ndcg=0.5)
```

### Worked calculation (Query 1)

Given `y_true = [3, 2, 0, 1]` and `y_scores = [0.9, 0.8, 0.2, 0.5]`:

1. Sort by score descending: ranking is doc0(rel=3), doc1(rel=2), doc3(rel=1), doc2(rel=0)
2. DCG@4:
    - Position 1: (2^3 - 1) / log2(2) = 7.0 / 1.0 = 7.000
    - Position 2: (2^2 - 1) / log2(3) = 3.0 / 1.585 = 1.893
    - Position 3: (2^1 - 1) / log2(4) = 1.0 / 2.0 = 0.500
    - Position 4: (2^0 - 1) / log2(5) = 0.0 / 2.322 = 0.000
    - DCG@4 = 9.393
3. Ideal ranking: [3, 2, 1, 0] (sorted descending) -> IDCG@4 = 9.393
4. nDCG@4 = 9.393 / 9.393 = **1.0** (perfect ranking for this query)

---

## assert_mrr

Assert that Mean Reciprocal Rank meets a minimum threshold.

MRR looks at the rank of the **first relevant result** for each query and takes the reciprocal (1/rank). It is the standard metric when you care about **how quickly the user finds one good result** -- think search autocomplete, FAQ lookup, or single-document retrieval.

### Formula

```
MRR = (1 / |Q|) * sum_{i=1}^{|Q|} 1 / rank_i
```

Where `rank_i` is the position (1-indexed) of the first relevant result for query *i*. If no result is relevant, that query contributes 0.

**Score interpretation:**

| MRR value | Meaning |
|-----------|---------|
| 1.0 | The first result is always relevant (perfect) |
| 0.5 | On average, the first relevant result is at position 2 |
| 0.33 | On average, the first relevant result is at position 3 |
| 0.0 | No query has any relevant results |

### When to use

- You have **binary relevance** (relevant or not, no grading).
- You only care about the **first** relevant result -- "does the top result answer the question?"
- Common in: FAQ search, passage retrieval for RAG, autocomplete ranking.

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `queries_results` | `list[list[bool]]` | *(required)* | Boolean relevance per result per query. `queries_results[i]` is a list of booleans indicating whether each retrieved result is relevant for query *i*, in ranked order. |
| `min_mrr` | `float` | `0.5` | Minimum acceptable MRR. The assertion fails if the computed MRR is below this threshold. |

### Returns

`TestResult` with details:

- `mrr` -- computed MRR score
- `min_mrr` -- threshold used
- `num_queries` -- number of queries evaluated
- `per_query_rr` -- list of reciprocal ranks per query

### Example

```python
from mltk.domains.llm.retrieval import assert_mrr

# Three queries, each with ranked retrieval results
queries_results = [
    [False, True, False],   # first relevant at rank 2 -> RR = 1/2 = 0.5
    [True, False, False],   # first relevant at rank 1 -> RR = 1/1 = 1.0
    [False, False, True],   # first relevant at rank 3 -> RR = 1/3 = 0.333
]

# MRR = (0.5 + 1.0 + 0.333) / 3 = 0.611
result = assert_mrr(queries_results, min_mrr=0.5)
```

### Practical tip

If your retriever returns document IDs, convert to boolean lists before calling `assert_mrr`:

```python
relevant_ids = {"doc_7", "doc_12"}
retrieved_ids = ["doc_3", "doc_7", "doc_15", "doc_12"]

# Convert to boolean relevance list
relevance = [doc_id in relevant_ids for doc_id in retrieved_ids]
# [False, True, False, True]

# Wrap in a list (one query)
result = assert_mrr([relevance], min_mrr=0.5)
```

---

## assert_recall_at_k

Assert that mean Recall@K meets a minimum threshold.

Recall@K measures the fraction of all relevant documents that appear in the top K retrieved results. Low recall means the retriever is **missing important documents** -- even if the ones it finds are good.

### Formula

For a single query:

```
Recall@k = |relevant intersect retrieved[:k]| / |relevant|
```

The final score is the **mean Recall@K across all queries**.

**Score interpretation:**

| Recall@K value | Meaning |
|----------------|---------|
| 1.0 | Every relevant document appears in the top K |
| 0.5 | Half the relevant documents are missing from the top K |
| 0.0 | None of the relevant documents were retrieved |

### When to use

- **Coverage matters more than ranking.** You need to find *all* relevant documents, not just put the best one first.
- Common in: legal discovery, medical literature review, RAG systems where missing a key passage causes hallucination.
- Pair with `assert_ndcg` or `assert_map_at_k` if you also care about rank ordering.

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `relevant` | `list[set]` | *(required)* | Relevant document IDs per query. `relevant[i]` is a set of ground-truth relevant IDs for query *i*. |
| `retrieved` | `list[list]` | *(required)* | Retrieved document IDs per query, in ranked order. `retrieved[i]` is an ordered list of IDs returned by the retriever for query *i*. |
| `k` | `int` | `10` | Cutoff position. Only the first *k* retrieved documents are considered. |
| `min_recall` | `float` | `0.8` | Minimum acceptable mean Recall@K. The assertion fails if the computed score is below this threshold. |

### Returns

`TestResult` with details:

- `recall` -- computed mean Recall@K
- `min_recall` -- threshold used
- `k` -- cutoff position
- `num_queries` -- number of queries evaluated
- `per_query_recall` -- list of Recall@K per query

### Example

```python
from mltk.domains.llm.retrieval import assert_recall_at_k

# Query 1: 3 relevant docs, retriever found 2 of them in top 3
# Query 2: 2 relevant docs, retriever found 1 of them in top 3
relevant = [
    {"d1", "d2", "d3"},       # query 1 ground truth
    {"d4", "d5"},              # query 2 ground truth
]

retrieved = [
    ["d1", "d3", "d6", "d2"],  # query 1 results (d1, d3 relevant in top 3)
    ["d5", "d7", "d8"],        # query 2 results (d5 relevant in top 3)
]

# Recall@3: query1 = 2/3 = 0.667, query2 = 1/2 = 0.5
# Mean = (0.667 + 0.5) / 2 = 0.583
result = assert_recall_at_k(relevant, retrieved, k=3, min_recall=0.5)
```

### The K tradeoff

| K value | Effect |
|---------|--------|
| Small K (3-5) | Strict -- tests if the retriever surfaces the right docs quickly. Useful for user-facing search where nobody scrolls past page 1. |
| Medium K (10-20) | Balanced -- standard for RAG systems where the LLM receives ~10 context chunks. |
| Large K (50-100) | Lenient -- tests total retrieval pool. Useful when downstream reranking will filter further. |

---

## assert_map_at_k

Assert that Mean Average Precision at K meets a minimum threshold.

MAP@K combines precision and ranking. For each query, it computes precision at every position where a relevant document is found, then averages those values. This rewards systems that rank relevant documents **higher** -- not just retrieve them somewhere in the list.

### Formula

For a single query:

```
AP@k = (1 / |relevant|) * sum_{j=1}^{k} Precision@j * rel(j)
```

Where `Precision@j` is the number of relevant documents in positions 1..j divided by j, and `rel(j) = 1` if the j-th document is relevant, 0 otherwise.

The final score is the **mean AP@K across all queries** (hence MAP).

### When to use

- You care about both **coverage** (like recall) and **ranking quality** (like nDCG), but have only **binary relevance** labels.
- MAP@K is stricter than Recall@K: two systems with identical recall will have different MAP scores if one ranks relevant docs higher.
- Common in: document retrieval benchmarks (TREC, MS MARCO), RAG evaluations, search quality monitoring.

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `relevant` | `list[set]` | *(required)* | Relevant document IDs per query. `relevant[i]` is a set of ground-truth relevant IDs for query *i*. |
| `retrieved` | `list[list]` | *(required)* | Retrieved document IDs per query, in ranked order. `retrieved[i]` is an ordered list of IDs for query *i*. |
| `k` | `int` | `10` | Cutoff position. Only the first *k* retrieved documents are considered. |
| `min_map` | `float` | `0.5` | Minimum acceptable MAP@K. The assertion fails if the computed score is below this threshold. |

### Returns

`TestResult` with details:

- `map_score` -- computed MAP@K
- `min_map` -- threshold used
- `k` -- cutoff position
- `num_queries` -- number of queries evaluated
- `per_query_ap` -- list of AP@K per query

### Example

```python
from mltk.domains.llm.retrieval import assert_map_at_k

relevant = [
    {"d1", "d3"},   # query 1: two relevant docs
    {"d2"},          # query 2: one relevant doc
]

retrieved = [
    ["d1", "d2", "d3"],   # query 1: d1 at rank 1, d3 at rank 3
    ["d1", "d2", "d3"],   # query 2: d2 at rank 2
]

# Query 1 AP@3:
#   rank 1: d1 is relevant -> P@1 = 1/1 = 1.0
#   rank 2: d2 is NOT relevant -> skip
#   rank 3: d3 is relevant -> P@3 = 2/3 = 0.667
#   AP = (1.0 + 0.667) / 2 = 0.833

# Query 2 AP@3:
#   rank 1: d1 is NOT relevant -> skip
#   rank 2: d2 is relevant -> P@2 = 1/2 = 0.5
#   AP = 0.5 / 1 = 0.5

# MAP@3 = (0.833 + 0.5) / 2 = 0.667
result = assert_map_at_k(relevant, retrieved, k=3, min_map=0.5)
```

### Worked calculation (Query 1)

Step through each position in the retrieved list `["d1", "d2", "d3"]` with `relevant = {"d1", "d3"}`:

| Position (j) | Document | Relevant? | Hits so far | Precision@j | Contributes? |
|--------------|----------|-----------|-------------|-------------|--------------|
| 1 | d1 | Yes | 1 | 1/1 = 1.000 | Yes (1.000) |
| 2 | d2 | No | 1 | 1/2 = 0.500 | No |
| 3 | d3 | Yes | 2 | 2/3 = 0.667 | Yes (0.667) |

AP@3 = (1.000 + 0.667) / 2 relevant docs = **0.833**

---

## Choosing the Right Metric

| Metric | Relevance type | Cares about rank? | Best for |
|--------|---------------|-------------------|----------|
| **nDCG@K** | Graded (0, 1, 2, 3) | Yes (position-weighted) | Search engines, recommendations, any system with graded relevance labels |
| **MRR** | Binary (yes/no) | Only first hit | FAQ search, single-answer retrieval, autocomplete, "did we get it right on the first try?" |
| **Recall@K** | Binary (yes/no) | No (just coverage) | Legal discovery, medical search, RAG (where missing a key passage causes hallucination) |
| **MAP@K** | Binary (yes/no) | Yes (precision at each hit) | Document retrieval benchmarks, balanced ranking + coverage evaluation |

### Decision flowchart

1. Do you have graded relevance labels (0/1/2/3)? --> Use **nDCG@K**.
2. Do you only care about the first relevant result? --> Use **MRR**.
3. Do you need to find ALL relevant documents? --> Use **Recall@K**.
4. Do you need both coverage AND good ranking? --> Use **MAP@K**.

For RAG systems, the recommended combination is **Recall@K** (ensure nothing critical is missed) plus **MRR** or **MAP@K** (ensure the best documents are ranked first).

---

## Integration with RAG Assertions

Retrieval metrics test the retriever in isolation. Combine them with the existing generator metrics for end-to-end RAG quality:

```python
import pytest
from mltk.domains.llm.retrieval import (
    assert_ndcg,
    assert_mrr,
    assert_recall_at_k,
    assert_map_at_k,
)
from mltk.domains.llm.rag import (
    assert_faithfulness,
    assert_context_relevancy,
    assert_answer_relevancy,
    assert_context_precision,
    assert_context_recall,
)


# --- Retriever tests (run BEFORE generator tests) ---

class TestRetriever:
    """Evaluate retrieval ranking quality in isolation."""

    def test_ndcg_graded_relevance(self, retriever, eval_queries):
        """Verify ranked results match graded relevance labels."""
        y_true = [q["relevance_grades"] for q in eval_queries]
        y_scores = [retriever.score(q["query"]) for q in eval_queries]
        assert_ndcg(y_true, y_scores, k=10, min_ndcg=0.75)

    def test_mrr_first_hit(self, retriever, eval_queries):
        """Verify the first relevant result appears quickly."""
        results = []
        for q in eval_queries:
            docs = retriever.search(q["query"], top_k=10)
            results.append([doc.id in q["relevant_ids"] for doc in docs])
        assert_mrr(results, min_mrr=0.6)

    def test_recall_coverage(self, retriever, eval_queries):
        """Verify all relevant passages are retrieved."""
        relevant = [q["relevant_ids"] for q in eval_queries]
        retrieved = [
            [doc.id for doc in retriever.search(q["query"], top_k=20)]
            for q in eval_queries
        ]
        assert_recall_at_k(relevant, retrieved, k=20, min_recall=0.8)

    def test_map_ranking_quality(self, retriever, eval_queries):
        """Verify relevant documents are ranked high."""
        relevant = [q["relevant_ids"] for q in eval_queries]
        retrieved = [
            [doc.id for doc in retriever.search(q["query"], top_k=10)]
            for q in eval_queries
        ]
        assert_map_at_k(relevant, retrieved, k=10, min_map=0.5)


# --- Generator tests (run AFTER retriever is validated) ---

class TestGenerator:
    """Evaluate LLM answer quality given retrieved context."""

    def test_faithfulness(self, rag_pipeline, eval_queries):
        """Verify answers are grounded in context (no hallucination)."""
        for q in eval_queries:
            result = rag_pipeline.run(q["query"])
            assert_faithfulness(result.answer, result.context_chunks, min_score=0.5)

    def test_context_relevancy(self, rag_pipeline, eval_queries):
        """Verify retrieved context is relevant to the question."""
        for q in eval_queries:
            result = rag_pipeline.run(q["query"])
            assert_context_relevancy(
                q["query"], result.context_chunks, min_score=0.3
            )

    def test_answer_relevancy(self, rag_pipeline, eval_queries):
        """Verify answers address the question asked."""
        for q in eval_queries:
            result = rag_pipeline.run(q["query"])
            assert_answer_relevancy(q["query"], result.answer, min_score=0.3)

    def test_context_precision(self, rag_pipeline, eval_queries):
        """Verify low noise in retrieved context."""
        for q in eval_queries:
            result = rag_pipeline.run(q["query"])
            assert_context_precision(
                q["query"], result.context_chunks, min_precision=0.6
            )

    def test_context_recall(self, rag_pipeline, eval_queries):
        """Verify retrieved context covers the expected answer."""
        for q in eval_queries:
            result = rag_pipeline.run(q["query"])
            assert_context_recall(
                q["expected_answer"], result.context_chunks, min_recall=0.4
            )
```

This gives you a complete RAG test suite: 4 retriever assertions to verify you fetched the right documents, then 5 generator assertions to verify the LLM used them correctly. Run retriever tests first -- if the retriever is broken, generator tests will fail for the wrong reasons.

---

## Import Quick Reference

```python
from mltk.domains.llm.retrieval import (
    assert_ndcg,
    assert_mrr,
    assert_recall_at_k,
    assert_map_at_k,
)
```

All four assertions return a `TestResult` and can be used with the mltk pytest plugin (`--mltk-report`), YAML test definitions, and the server platform.
