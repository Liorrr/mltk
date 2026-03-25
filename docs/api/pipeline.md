# Pipeline Testing

Validate ML pipeline reproducibility and end-to-end execution. Catches non-deterministic training, broken pipelines, and artifact corruption.

**Module:** `mltk.pipeline`

---

## assert_reproducible

Assert a function produces identical output across multiple runs with the same seed.

```python
from mltk.pipeline import assert_reproducible

assert_reproducible(train_model, X, y, seed=42, runs=3, tolerance=0.001)
```

## assert_checksum

Assert a model artifact matches an expected hash.

```python
from mltk.pipeline import assert_checksum

assert_checksum("model.pkl", expected_hash="sha256:abc123...")
```

## assert_pipeline

Assert an end-to-end pipeline runs without errors.

```python
from mltk.pipeline import assert_pipeline

assert_pipeline([load_data, preprocess, train, evaluate], input_data=config)
```

---
