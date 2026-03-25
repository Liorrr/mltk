# Advanced Drift Detection

Additional drift methods and embedding drift for text/image/multimodal models.

---

## New Drift Methods

### Jensen-Shannon (`method="js"`)
Symmetric, bounded [0,1]. Evidently's default for categorical features on large datasets.

### Wasserstein (`method="wasserstein"`)
Proportional to mean shift magnitude. Evidently's default for numeric features when n>1000.

### Auto-Select (`method="auto"`)
Automatically selects best method: Wasserstein for numeric n>1000, JS for categorical, KS otherwise.

---

## Embedding Drift

```python
from mltk.data import assert_no_embedding_drift

assert_no_embedding_drift(ref_embeddings, cur_embeddings, method="cosine", threshold=0.1)
```

Methods: `cosine` (centroid distance), `euclidean`, `mmd` (Maximum Mean Discrepancy).

---
