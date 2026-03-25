# Embedding Drift Detection Research

**Date:** 2026-03-25
**Scope:** Extend `assert_no_drift` to support embedding vectors (text, image, multimodal)
**Status:** Research complete, proposal ready for implementation

---

## 1. Problem Statement

The current `assert_no_drift` in `mltk/data/drift.py` supports four methods for
scalar/tabular features: KS test, PSI, KL divergence, and chi-squared. All operate
on `pd.Series` (one-dimensional data).

Modern ML systems increasingly rely on embeddings -- dense vector representations
from models like BERT (768-dim), OpenAI text-embedding-3 (1536-dim), CLIP (512-dim),
ResNet (2048-dim), and multimodal encoders. These embeddings drive search, retrieval,
classification, and generative pipelines. When the distribution of embeddings drifts
(new topics, image domain shift, model version change, upstream encoder update), the
downstream system degrades silently.

There is no standard way to test embedding drift today in mltk. This research
evaluates five methods and proposes an implementation plan.

---

## 2. Methods Evaluated

### 2.1 Centroid Distance (Cosine / Euclidean)

**How it works:**
Compute the mean vector (centroid) of the reference embeddings and the mean vector of
the current embeddings. Measure the distance between them using cosine distance or
Euclidean (L2) distance.

```
centroid_ref = mean(reference_embeddings, axis=0)   # shape: (D,)
centroid_cur = mean(current_embeddings, axis=0)      # shape: (D,)
cosine_dist  = 1 - dot(centroid_ref, centroid_cur) / (norm(ref) * norm(cur))
euclid_dist  = norm(centroid_ref - centroid_cur)
```

**Metric produced:** A single scalar distance value.

**Thresholds:**
- Cosine distance ranges from 0 (identical) to 2 (opposite). In practice, drift
  starts being detectable at values as low as 0.001-0.01, making this non-intuitive
  to tune. Evidently notes thresholds can be "as low as 0.001 for a change you
  already want to detect."
- Euclidean distance has no fixed upper bound; threshold depends on the embedding
  model's scale. Normalization to unit vectors before comparison helps.
- Recommended default: cosine_distance > 0.015 for normalized embeddings.

**Pros:**
- Simplest to implement and understand (3 lines of numpy).
- Fastest: O(N*D) to compute centroids, O(D) to compare.
- Works well for detecting bulk shifts (entire distribution moves).
- No additional dependencies.

**Cons:**
- Insensitive to changes in spread/shape (variance drift). Two distributions
  with the same mean but different variance look identical.
- Averages out local structure -- a cluster splitting in two may produce the same
  centroid.
- Threshold is hard to set without domain-specific calibration.

**When sufficient:**
- Quick smoke test or monitoring dashboard signal.
- When you expect drift to manifest as a bulk topic/domain shift.
- As a fast first-pass before running a more expensive method.

**Key sources:** Evidently blog, Arize documentation, Aparna Dhinakaran (TDS).


### 2.2 Maximum Mean Discrepancy (MMD)

**How it works:**
MMD is a kernel-based distance between two probability distributions. It maps
samples from both distributions into a Reproducing Kernel Hilbert Space (RKHS) and
compares their mean embeddings in that space. The key insight: unlike KL divergence,
MMD never estimates densities, which makes it reliable in high dimensions.

```
MMD^2(P, Q) = E[k(x,x')] + E[k(y,y')] - 2*E[k(x,y)]
```

where k is a kernel function (typically Gaussian RBF):
```
k(x, y) = exp(-||x - y||^2 / (2 * sigma^2))
```

Statistical significance is assessed via permutation test: pool all samples, randomly
split into two groups N times (typically 100-1000 permutations), compute MMD^2 for
each split, and compare the real MMD^2 against this null distribution to get a p-value.

**Kernel bandwidth (sigma):** By default, use the "median heuristic" -- set sigma to
the median pairwise distance between reference samples. Alibi-detect also supports
passing multiple bandwidth values and averaging the kernel evaluation.

**Metric produced:** MMD^2 statistic + p-value from permutation test.

**Thresholds:**
- p-value < 0.05 indicates drift (standard statistical testing).
- The raw MMD^2 value is not directly interpretable across different datasets.

**Pros:**
- Statistically principled with formal hypothesis testing (p-value).
- Does not degrade in high dimensions (no density estimation).
- Captures distributional changes that centroid distance misses (variance,
  multi-modal shifts, tail behavior).
- Well-studied theoretically (Gretton et al., 2012).

**Cons:**
- Computational cost: O(N^2 * D) for the kernel matrix, multiplied by the number
  of permutations. For N=10000 and D=768, this is expensive.
- The permutation test adds wall-clock time.
- The raw MMD^2 score is non-interpretable; you rely on the p-value.
- Requires understanding of kernel methods for advanced tuning.

**Practical guidance:**
- alibi-detect recommends subsampling if N > ~5000 to keep runtime manageable.
- Use `n_permutations=100` for fast checks, `n_permutations=1000` for rigorous tests.
- Candidate for Rust acceleration (kernel matrix computation is parallelizable).

**Key sources:** alibi-detect docs, Gretton et al. (JMLR 2012), APXML course.


### 2.3 PCA + Univariate Drift (Share of Drifted Components)

**How it works:**
1. Fit PCA on the reference embeddings to reduce dimensionality (e.g., 768 -> 50 or
   to explain 95% of variance).
2. Project both reference and current embeddings onto the principal components.
3. For each component, run a univariate drift test (KS, Wasserstein, or PSI).
4. Count how many components show significant drift.
5. If the share of drifted components exceeds a threshold (e.g., 20%), declare drift.

```
pca = PCA(n_components=50).fit(reference_embeddings)
ref_reduced = pca.transform(reference_embeddings)    # (N_ref, 50)
cur_reduced = pca.transform(current_embeddings)      # (N_cur, 50)

drifted = 0
for i in range(50):
    p = ks_test(ref_reduced[:, i], cur_reduced[:, i])
    if p < 0.05:
        drifted += 1

share = drifted / 50  # e.g., 0.32 = 32% of components drifted
drift_detected = share > 0.2
```

**Metric produced:** Share of drifted components (0.0 to 1.0), plus per-component
p-values for drill-down.

**Thresholds:**
- Component-level: p-value < 0.05 (KS) or Wasserstein > 0.1 (Evidently default).
- Aggregate: share > 0.2 (20% of components drifted) = drift detected.
- Evidently uses Wasserstein distance with threshold 0.1 as the default
  component-level test.

**Important caveat:** PCA does NOT preserve cosine angles between vectors. Cosine
distance is inconsistent after PCA. Use Euclidean-based tests (KS, Wasserstein)
on PCA-projected components. Evidently's experiments confirm this.

**Pros:**
- Most interpretable: you can see WHICH components drifted and how much.
- Reuses existing univariate drift tests (KS, PSI) from mltk.
- Handles high-dimensional embeddings efficiently via dimensionality reduction.
- Allows Bonferroni or FDR correction for multiple testing.

**Cons:**
- PCA is fit on reference data -- if current data has new structure orthogonal to
  reference PCs, it may be missed.
- Additional dependency on scikit-learn for PCA.
- Two-stage threshold tuning (component-level + aggregate share).
- Information loss from dimensionality reduction.

**Key sources:** Evidently blog + Evidently docs, DriftLens paper (arXiv:2309.10000).


### 2.4 Domain Classifier (Model-Based)

**How it works:**
Train a binary classifier to distinguish between reference embeddings (label=0) and
current embeddings (label=1). If the classifier can reliably separate them, the
distributions differ. The drift score is the ROC AUC of the classifier.

```
X = concat(reference_embeddings, current_embeddings)  # (N_ref + N_cur, D)
y = [0]*N_ref + [1]*N_cur

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25)
clf = LogisticRegression(max_iter=1000)  # or RandomForest, or LightGBM
clf.fit(X_train, y_train)
roc_auc = roc_auc_score(y_val, clf.predict_proba(X_val)[:, 1])

drift_detected = roc_auc > 0.55  # random = 0.50
```

**Metric produced:** ROC AUC score (0.0-1.0). Random classifier = 0.5.

**Thresholds:**
- Evidently default for N > 1000: ROC AUC > 0.55 = drift.
- Evidently default for N <= 1000: compare against 95th percentile of random
  classifier ROC AUC (computed via 1000 random permutations).
- ROC AUC > 0.60 = moderate drift, > 0.70 = significant drift.

**Pros:**
- Evidently recommends this as the "sensible default" -- most consistent performance
  across different datasets and embedding models.
- ROC AUC is familiar and interpretable to ML practitioners.
- Works directly in the original embedding space (no PCA required, though PCA or
  UMAP can speed it up).
- Flexible: can use any classifier (logistic regression, random forest, gradient
  boosting).
- Captures complex distributional differences beyond mean/variance shifts.

**Cons:**
- Requires scikit-learn dependency.
- Training a classifier is more expensive than computing a distance.
- Results can vary with classifier choice and hyperparameters.
- Not a formal statistical test (no p-value), though the permutation baseline
  provides statistical grounding.

**Key sources:** Evidently blog + docs, NannyML blog, Deepchecks docs.


### 2.5 Per-Dimension Wasserstein Distance

**How it works:**
Compute the Wasserstein-1 distance (Earth Mover's Distance) between each dimension
of the reference and current embeddings independently. Aggregate via mean or
fraction exceeding a threshold.

This is a simpler variant of method 2.3 (PCA + univariate) but operates directly on
raw embedding dimensions instead of PCA components.

**Metric produced:** Mean Wasserstein distance across dimensions, or share of
dimensions with Wasserstein > threshold.

**Thresholds:**
- Per-dimension Wasserstein > 0.1 = drifted (Evidently default).
- Aggregate share > 0.2 = drift detected.

**Pros:**
- No dimensionality reduction needed -- operates on raw embedding dimensions.
- Wasserstein distance has physical interpretation (minimal "work" to transform one
  distribution into another).

**Cons:**
- O(N * D * N*log(N)) -- computing Wasserstein for each of D dimensions is expensive
  for large D (768+).
- Raw embedding dimensions may not be independently meaningful (unlike PCA components).
- Many simultaneous tests create multiple-testing risk.

**Key sources:** Evidently docs.

---

## 3. Text vs. Image vs. Multimodal Drift

### 3.1 Text Embedding Drift

**Common causes:** New topics, vocabulary shift, language change, sentiment shift,
new classes/intents, spam injection.

**Typical models:** BERT (768d), all-MiniLM-L6-v2 (384d), OpenAI text-embedding-3
(1536d or 3072d), Sentence-BERT, E5, BGE.

**Considerations:**
- Text embeddings are typically L2-normalized, so cosine distance is natural.
- Semantic drift (new topics) shows up clearly in centroid shift.
- Syntactic/stylistic drift (same topics, different phrasing) may need the domain
  classifier or MMD to detect.
- Embedding model updates silently change the embedding space -- always version-lock
  the encoder when setting baselines.

### 3.2 Image Embedding Drift

**Common causes:** Lighting changes, camera angle, new object types not in training,
image quality degradation (blur, noise), domain shift (indoor vs outdoor).

**Typical models:** ResNet-50 (2048d), EfficientNet (1280d), ViT (768d), DINO (768d),
CLIP visual encoder (512d).

**Considerations:**
- Image embeddings are often NOT L2-normalized by default -- normalize before
  computing cosine distance.
- Drift often manifests as new visual categories appearing (out-of-distribution
  objects), which centroid distance may miss.
- The domain classifier tends to outperform centroid distance for image drift
  because visual domain shifts are complex and multi-faceted.
- Arize research shows cosine distance was "more sensitive and dramatic when drift
  was increased" compared to Euclidean for image embeddings.

### 3.3 Multimodal Embedding Drift (CLIP, etc.)

**Common causes:** Modality imbalance shift, alignment degradation, one modality
drifting while the other stays stable.

**Typical models:** CLIP (512d joint space), ALIGN, Florence, LLaVA embeddings.

**Considerations:**
- Multimodal embeddings live in a shared space but have a documented "modality gap"
  -- image and text embeddings cluster in different regions even when semantically
  paired. Monitor each modality's centroid separately AND the cross-modal alignment.
- CLIP normalizes to the unit sphere, making cosine distance the natural metric.
- Test both: (a) drift within a single modality, (b) drift in the cross-modal
  alignment (average cosine similarity between paired image-text embeddings).
- New research (2026) on "Centroid Uniformity Loss" addresses closing the modality
  gap, but drift in this gap itself is a signal worth monitoring.

### 3.4 Summary Table: Modality Guidance

| Aspect              | Text                    | Image                     | Multimodal              |
|---------------------|-------------------------|---------------------------|-------------------------|
| Default method      | Domain classifier       | Domain classifier         | Centroid + classifier   |
| Fast check          | Cosine centroid          | Cosine centroid            | Per-modality centroid   |
| Normalization       | Usually pre-normalized  | Normalize before testing   | Usually pre-normalized  |
| Key drift signal    | Topic/intent shift      | New visual categories      | Modality gap change     |
| Typical dimensions  | 384-3072                | 512-2048                   | 512-1024                |

---

## 4. Handling High-Dimensional Embeddings

### 4.1 The Curse of Dimensionality

At 768-3072 dimensions, distance metrics become less discriminative (all pairwise
distances concentrate). This affects MMD and centroid distance more than the domain
classifier, which learns a discriminative boundary.

### 4.2 Dimensionality Reduction Strategies

| Strategy              | When to use                             | Preserve cosine? |
|-----------------------|-----------------------------------------|-------------------|
| PCA (retain 95% var)  | Before univariate tests                 | No (use Euclidean)|
| Random projection     | When PCA is too slow (N << D)           | Approximately     |
| UMAP (2D/3D)          | Visualization only, not for testing     | No                |
| None (raw dimensions) | Domain classifier, centroid distance    | Yes               |

**Practical defaults:**
- For N > 1000 and D > 200: PCA to min(50, D) components for univariate tests.
- For domain classifier: PCA optional (helps speed, not required for quality).
- For centroid distance: operate on raw dimensions (centroid is already a single
  vector, so D doesn't matter for the final comparison).
- For MMD: PCA to ~100 components recommended when D > 500 and N > 5000, to keep
  the kernel matrix computation tractable.

### 4.3 Sample Size Requirements

| Method              | Minimum N (ref+cur)  | Recommended N | Notes                        |
|---------------------|----------------------|---------------|------------------------------|
| Centroid distance   | 30                   | 500+          | More samples = stabler mean  |
| MMD                 | 100                  | 500-5000      | O(N^2) cost limits scaling   |
| PCA + univariate    | 200                  | 1000+         | Need enough for PCA fit      |
| Domain classifier   | 200                  | 1000+         | Need train+val split         |

---

## 5. Proposal: Extend `assert_no_drift` with Embedding Methods

### 5.1 Design Principles

1. **Same entry point:** `assert_no_drift` gains new methods, not a new function name.
   Users switch from `method="ks"` to `method="embedding_cosine"` naturally.
2. **NumPy arrays as input:** Embeddings are `np.ndarray` of shape `(N, D)`, not
   `pd.Series`. We add an overload that accepts 2D arrays.
3. **Sensible defaults:** `method="embedding"` uses the domain classifier (Evidently's
   recommended default). Quick aliases for fast checks.
4. **Progressive dependency:** Only numpy for centroid distance. scikit-learn for PCA,
   domain classifier. scipy for MMD permutation test.
5. **Rust acceleration path:** Cosine centroid and MMD kernel matrix are hot paths
   that can be accelerated in the Rust extension later.

### 5.2 New Methods

| Method string            | Algorithm                  | Dependencies      | Speed    |
|--------------------------|----------------------------|--------------------|----------|
| `"embedding_cosine"`     | Cosine centroid distance   | numpy              | Fast     |
| `"embedding_euclidean"`  | Euclidean centroid distance | numpy              | Fast     |
| `"embedding_mmd"`        | MMD with RBF kernel        | numpy (+scipy opt) | Slow     |
| `"embedding_pca"`        | PCA + KS on components     | scikit-learn       | Medium   |
| `"embedding_classifier"` | Domain classifier (ROC AUC)| scikit-learn       | Medium   |
| `"embedding"`            | Alias for classifier       | scikit-learn       | Medium   |

### 5.3 API Design

```python
# --- New function signature (separate from scalar assert_no_drift) ---

@timed_assertion
def assert_no_embedding_drift(
    reference: np.ndarray,          # shape (N_ref, D)
    current: np.ndarray,            # shape (N_cur, D)
    method: str = "classifier",     # default = domain classifier
    threshold: float | None = None, # method-specific default if None
    *,
    # Method-specific parameters
    n_components: int | None = None,      # PCA components (default: min(50, D))
    n_permutations: int = 100,            # MMD permutation test iterations
    classifier: str = "logistic",         # "logistic", "random_forest"
    normalize: bool = True,               # L2-normalize embeddings before testing
    component_method: str = "ks",         # univariate test for PCA method
    component_threshold: float = 0.05,    # per-component threshold for PCA method
) -> TestResult:
    """Assert no significant drift in embedding distributions.

    Args:
        reference: Reference embeddings, shape (N_ref, D).
        current: Current embeddings, shape (N_cur, D).
        method: Detection method:
            - "cosine": Cosine distance between centroids.
            - "euclidean": Euclidean distance between centroids.
            - "mmd": Maximum Mean Discrepancy with RBF kernel.
            - "pca": PCA + per-component univariate tests.
            - "classifier": Domain classifier ROC AUC (default, recommended).
        threshold: Custom threshold. If None, uses method-specific default.
        n_components: Number of PCA components. Default: min(50, D).
        n_permutations: Permutation count for MMD test. Default: 100.
        classifier: Classifier type for domain classifier method.
        normalize: Whether to L2-normalize embeddings. Default: True.
        component_method: Univariate test for PCA method. Default: "ks".
        component_threshold: Per-component drift threshold. Default: 0.05.

    Returns:
        TestResult with drift statistics.

    Example:
        >>> assert_no_embedding_drift(train_emb, prod_emb)  # uses classifier
        >>> assert_no_embedding_drift(train_emb, prod_emb, method="cosine")
        >>> assert_no_embedding_drift(train_emb, prod_emb, method="mmd", n_permutations=500)
    """
```

### 5.4 Default Thresholds

```python
_EMBEDDING_THRESHOLDS: dict[str, float] = {
    "cosine": 0.015,       # cosine distance > 0.015 = drift
    "euclidean": 0.1,      # euclidean distance > 0.1 (after normalization)
    "mmd": 0.05,           # p-value < 0.05 = drift (permutation test)
    "pca": 0.2,            # share of drifted components > 20%
    "classifier": 0.55,    # ROC AUC > 0.55 = drift (Evidently default)
}
```

### 5.5 Implementation Plan (per method)

#### Method 1: Cosine Centroid (`"cosine"`)
```python
def _embedding_cosine(ref: np.ndarray, cur: np.ndarray, threshold: float) -> TestResult:
    ref_centroid = ref.mean(axis=0)
    cur_centroid = cur.mean(axis=0)
    cos_dist = 1.0 - np.dot(ref_centroid, cur_centroid) / (
        np.linalg.norm(ref_centroid) * np.linalg.norm(cur_centroid)
    )
    passed = cos_dist < threshold
    # return TestResult...
```

#### Method 2: Euclidean Centroid (`"euclidean"`)
```python
def _embedding_euclidean(ref: np.ndarray, cur: np.ndarray, threshold: float) -> TestResult:
    ref_centroid = ref.mean(axis=0)
    cur_centroid = cur.mean(axis=0)
    dist = float(np.linalg.norm(ref_centroid - cur_centroid))
    passed = dist < threshold
    # return TestResult...
```

#### Method 3: MMD (`"mmd"`)
```python
def _embedding_mmd(ref: np.ndarray, cur: np.ndarray, threshold: float,
                   n_permutations: int = 100) -> TestResult:
    # Median heuristic for kernel bandwidth
    from scipy.spatial.distance import pdist
    dists = pdist(ref[:500], metric="sqeuclidean")  # subsample for speed
    sigma2 = float(np.median(dists))

    def rbf_mmd2(x, y):
        xx = np.exp(-cdist(x, x, "sqeuclidean") / (2 * sigma2))
        yy = np.exp(-cdist(y, y, "sqeuclidean") / (2 * sigma2))
        xy = np.exp(-cdist(x, y, "sqeuclidean") / (2 * sigma2))
        return xx.mean() + yy.mean() - 2 * xy.mean()

    mmd2_observed = rbf_mmd2(ref, cur)

    # Permutation test
    pooled = np.vstack([ref, cur])
    n_ref = len(ref)
    count_ge = 0
    for _ in range(n_permutations):
        perm = np.random.permutation(len(pooled))
        perm_ref = pooled[perm[:n_ref]]
        perm_cur = pooled[perm[n_ref:]]
        if rbf_mmd2(perm_ref, perm_cur) >= mmd2_observed:
            count_ge += 1
    p_value = (count_ge + 1) / (n_permutations + 1)

    passed = p_value > threshold
    # return TestResult with mmd2, p_value...
```

#### Method 4: PCA + Univariate (`"pca"`)
```python
def _embedding_pca(ref: np.ndarray, cur: np.ndarray, threshold: float,
                   n_components: int, component_method: str,
                   component_threshold: float) -> TestResult:
    from sklearn.decomposition import PCA
    pca = PCA(n_components=n_components).fit(ref)
    ref_proj = pca.transform(ref)
    cur_proj = pca.transform(cur)

    drifted = 0
    component_results = []
    for i in range(n_components):
        result = assert_no_drift(
            pd.Series(ref_proj[:, i]),
            pd.Series(cur_proj[:, i]),
            method=component_method,
            threshold=component_threshold,
        )
        if not result.passed:
            drifted += 1
        component_results.append(result)

    share = drifted / n_components
    passed = share <= threshold
    # return TestResult with share, drifted_count, variance_explained...
```

#### Method 5: Domain Classifier (`"classifier"`)
```python
def _embedding_classifier(ref: np.ndarray, cur: np.ndarray, threshold: float,
                          classifier: str = "logistic") -> TestResult:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score

    X = np.vstack([ref, cur])
    y = np.array([0]*len(ref) + [1]*len(cur))

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)
    roc_auc = roc_auc_score(y_val, clf.predict_proba(X_val)[:, 1])

    passed = roc_auc <= threshold
    # return TestResult with roc_auc...
```

### 5.6 File Structure

```
src/mltk/data/
    drift.py                 # existing scalar drift (unchanged)
    embedding_drift.py       # NEW: all embedding drift methods
    __init__.py              # add assert_no_embedding_drift to exports

tests/test_data/
    test_drift.py            # existing (unchanged)
    test_embedding_drift.py  # NEW: tests for all 5 embedding methods
```

### 5.7 Dependencies

| Method     | Required package    | Optional extra   |
|------------|---------------------|------------------|
| cosine     | numpy (already dep) | --               |
| euclidean  | numpy (already dep) | --               |
| mmd        | numpy + scipy       | `mltk[scipy]`    |
| pca        | scikit-learn        | `mltk[sklearn]`  |
| classifier | scikit-learn        | `mltk[sklearn]`  |

All embedding methods gracefully degrade with ImportError and a clear install
message, matching the existing pattern in `drift.py` for scipy.

### 5.8 Rust Acceleration Candidates

These are the hot paths suitable for Rust acceleration in a future sprint:

1. **Cosine centroid distance** -- sum + normalize + dot product on `Vec<f64>`.
2. **MMD kernel matrix** -- the O(N^2 * D) pairwise RBF kernel computation.
3. **L2 normalization** -- row-wise normalization of the embedding matrix.

These can be added to `rust/src/lib.rs` as `#[pyfunction]` entries, with Python
fallbacks in `_rust.py` matching the existing KS/PSI pattern.

---

## 6. Test Plan

### 6.1 Test Fixtures (conftest.py additions)

```python
@pytest.fixture
def reference_embeddings() -> np.ndarray:
    """Reference embedding set: 500 vectors in 128-dim space."""
    rng = np.random.default_rng(42)
    return rng.normal(0, 1, (500, 128))

@pytest.fixture
def same_distribution_embeddings() -> np.ndarray:
    """Embeddings from same distribution (different seed)."""
    rng = np.random.default_rng(99)
    return rng.normal(0, 1, (500, 128))

@pytest.fixture
def drifted_embeddings() -> np.ndarray:
    """Embeddings with clear drift: shifted mean."""
    rng = np.random.default_rng(99)
    return rng.normal(2, 1.5, (500, 128))

@pytest.fixture
def high_dim_embeddings() -> tuple[np.ndarray, np.ndarray]:
    """768-dim embeddings simulating BERT-scale."""
    rng = np.random.default_rng(42)
    ref = rng.normal(0, 1, (200, 768))
    cur = rng.normal(0, 1, (200, 768))
    return ref, cur
```

### 6.2 Test Cases per Method

For each of the 5 methods:
- **No drift:** Same distribution -> `result.passed is True`.
- **Clear drift:** Mean-shifted distribution -> raises `MltkAssertionError`.
- **Custom threshold:** Verify threshold override works.
- **Edge case -- small N:** Minimum viable sample size.
- **Edge case -- high dimensionality:** 768-dim and 1536-dim inputs.
- **Edge case -- mismatched dimensions:** ref.shape[1] != cur.shape[1] -> error.
- **Edge case -- single sample:** N=1 -> clear error message.

Additional:
- **Normalization test:** Verify `normalize=True` L2-normalizes rows.
- **PCA variance explained:** Verify PCA retains >= 95% variance by default.
- **Classifier reproducibility:** Same input -> same ROC AUC (seeded).

Estimated: ~30-35 new tests.

---

## 7. Recommendations

### 7.1 Default Method

**Domain classifier** (`method="classifier"`) should be the default, consistent with
Evidently's recommendation. It provides the most consistent detection across different
embedding models and drift types, with an interpretable metric (ROC AUC).

### 7.2 Quick-Check Method

Expose `method="cosine"` as the fast path for monitoring dashboards and CI/CD checks
where speed matters more than sensitivity.

### 7.3 Threshold Guidance in Docs

Provide a clear guide:

| Situation                          | Recommended method | Why                                    |
|------------------------------------|--------------------|----------------------------------------|
| CI/CD pipeline (speed matters)     | `cosine`           | Sub-millisecond, numpy only            |
| Production monitoring              | `classifier`       | Best sensitivity, interpretable AUC    |
| Statistical rigor required         | `mmd`              | Formal hypothesis test with p-value    |
| Need to explain WHICH dims drifted | `pca`              | Per-component drill-down               |
| First time, unsure                 | `classifier`       | Evidently-recommended default          |

### 7.4 Sprint Schedule

This feature fits naturally into **Sprint 9** (Monitoring + Tabular + Full Docs)
in the backlog, or can be pulled forward as a standalone sprint. Estimated effort:

- Implementation: 1 sprint (5 methods + tests + docs)
- Rust acceleration: separate follow-up sprint

---

## 8. Competitive Landscape

| Tool          | Embedding drift methods                   | Default           |
|---------------|-------------------------------------------|--------------------|
| Evidently     | Centroid, MMD, PCA+univariate, classifier | Domain classifier  |
| Deepchecks    | Domain classifier (UMAP/PCA reduction)   | Classifier         |
| alibi-detect  | MMD, LSDD, classifier, context-aware MMD | MMD                |
| Arize/Phoenix | Euclidean centroid, cosine centroid       | Euclidean centroid |
| NannyML       | Domain classifier                        | Classifier         |
| Frouros       | MMD, various statistical tests           | Varies             |
| **mltk (proposed)** | Centroid, MMD, PCA, classifier      | Classifier         |

mltk's differentiator: the assertion-based API (`assert_no_embedding_drift`) that
integrates with pytest, producing pass/fail results rather than dashboard metrics.

---

## Sources

- [5 Methods to Detect Drift in ML Embeddings -- Evidently AI](https://www.evidentlyai.com/blog/embedding-drift-detection)
- [Monitoring Embeddings Drift -- Evidently AI Course](https://learn.evidentlyai.com/ml-observability-course/module-3-ml-monitoring-for-unstructured-data/monitoring-embeddings-drift)
- [Embeddings Drift Parameters -- Evidently Documentation](https://docs-old.evidentlyai.com/user-guide/customization/embeddings-drift-parameters)
- [Data Drift Algorithm -- Evidently Documentation](https://docs-old.evidentlyai.com/reference/data-drift-algorithm)
- [Maximum Mean Discrepancy -- alibi-detect Documentation](https://docs.seldon.io/projects/alibi-detect/en/latest/cd/methods/mmddrift.html)
- [Online MMD -- alibi-detect Documentation](https://docs.seldon.io/projects/alibi-detect/en/latest/cd/methods/onlinemmddrift.html)
- [MMD Drift on CIFAR-10 -- alibi-detect Example](https://docs.seldon.io/projects/alibi-detect/en/latest/examples/cd_mmd_cifar10.html)
- [Measuring Embedding Drift -- Aparna Dhinakaran (TDS)](https://medium.com/data-science/measuring-embedding-drift-aa9b7ddb84ae)
- [How to Measure Drift in ML Embeddings -- Elena Samuylova (TDS)](https://towardsdatascience.com/how-to-measure-drift-in-ml-embeddings-ee8adfe1e55e/)
- [Monitoring Embedding/Vector Drift Using Euclidean Distance -- Arize AI](https://arize.com/blog-course/embedding-drift-euclidean-distance/)
- [Embeddings Drift -- Deepchecks Documentation](https://docs.deepchecks.com/stable/nlp/auto_checks/train_test_validation/plot_embeddings_drift.html)
- [Multivariate Drift Detection Using Domain Classifier -- NannyML](https://www.nannyml.com/blog/data-drift-domain-classifier)
- [Drift Detection in Text Data with Document Embeddings (arXiv:2309.10000)](https://arxiv.org/abs/2309.10000)
- [Drift Detection in LLMs: A Practical Guide -- Tony Siciliani](https://medium.com/@tsiciliani/drift-detection-in-large-language-models-a-practical-guide-3f54d783792c)
- [Monitoring Drift in Embeddings and Unstructured Data -- APXML](https://apxml.com/courses/monitoring-managing-ml-models-production/chapter-2-advanced-drift-detection/embedding-drift-monitoring)
- [Frouros -- Open-source drift detection library](https://github.com/IFCA-Advanced-Computing/frouros)
- [alibi-detect -- Algorithms for drift detection (GitHub)](https://github.com/SeldonIO/alibi-detect)
- [Maximum Mean Discrepancy for Dummies -- Chen Chen](https://chchannn.github.io/posts/maximum-mean-discrepancy-for-dummies/)
- [Handling Drift in Industrial Defect Detection Through MMD-Based Methods (SciTePress 2025)](https://www.scitepress.org/Papers/2025/131709/131709.pdf)
