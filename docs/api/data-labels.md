# Label Quality Testing

Labels are the ground truth that ML models learn from. Bad labels produce bad models — no amount of architecture tuning can fix training on wrong answers. Label quality testing catches class imbalance, missing classes, and insufficient samples before they corrupt model training.

**Module:** `mltk.data.labels`

**ML Lifecycle Stage:** Data Collection / Data Labeling / Pre-training

**When to use:**
- After annotation batches: verify labelers produced balanced, complete labels
- Before training: check that all expected classes exist with sufficient samples
- Dataset versioning: detect label distribution changes between versions
- Active learning: verify that new labeled data covers underrepresented classes

---

## assert_label_balance

Assert that class distribution is not too imbalanced.

```python
from mltk.data import assert_label_balance

assert_label_balance(df["label"], max_ratio=10.0)
```

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `labels` | `pd.Series` | *(required)* | Series containing class labels |
| `max_ratio` | `float` | `10.0` | Maximum allowed ratio of majority to minority class |

### Returns

`TestResult` with details:
- `class_counts` -- dict of label to count
- `majority_class` -- the most frequent label
- `minority_class` -- the least frequent label
- `ratio` -- actual majority/minority ratio
- `max_ratio` -- configured threshold

### How ratio works

```
ratio = count(most_frequent_class) / count(least_frequent_class)

Example: 900 positive, 100 negative → ratio = 9.0
With max_ratio=10.0 → PASS (9.0 < 10.0)
With max_ratio=5.0  → FAIL (9.0 > 5.0)
```

### Why it matters for ML

Imbalanced datasets are the most common data quality issue in ML. A model trained on 99% negative, 1% positive will learn to predict "negative" for everything and achieve 99% accuracy — while being completely useless. CV and NLP domains are especially prone:
- **Face recognition**: some demographics underrepresented
- **Anomaly detection**: anomalies are rare by definition
- **NLP classification**: some categories naturally less frequent

### Example

```python
import pandas as pd
import pytest
from mltk.data import assert_label_balance

@pytest.mark.ml_data
def test_binary_classification_balance():
    """Positive/negative ratio should not exceed 10:1."""
    df = pd.read_csv("data/training.csv")
    assert_label_balance(df["label"], max_ratio=10.0)

@pytest.mark.ml_data
def test_multiclass_balance():
    """No class should be 20x more frequent than another."""
    df = pd.read_csv("data/categories.csv")
    assert_label_balance(df["category"], max_ratio=20.0)
```

### Edge Cases

- **Single class**: ratio is 1.0 (passes any threshold, but you probably want `assert_label_coverage` too)
- **Empty Series**: fails with clear error
- **Null labels**: counted as a separate class (use `assert_no_nulls` first)

### Related Tests

| Test | What it validates |
|------|-------------------|
| `test_balanced_labels` | Equal class distribution passes |
| `test_imbalanced_labels` | 100:1 ratio exceeds max_ratio=10 |
| `test_multiclass_balanced` | 3+ classes all within ratio |
| `test_single_class` | Only one class present (ratio=1, passes) |

---

## assert_label_coverage

Assert that all expected label classes are present with sufficient samples.

```python
from mltk.data import assert_label_coverage

assert_label_coverage(df["label"], expected_labels={"cat", "dog", "bird"}, min_samples=10)
```

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `labels` | `pd.Series` | *(required)* | Series containing class labels |
| `expected_labels` | `set[str] \| None` | `None` | Required label classes. None = check all observed classes |
| `min_samples` | `int` | `1` | Minimum samples required per class |

### Returns

`TestResult` with details:
- `class_counts` -- dict of label to count
- `missing_labels` -- labels in expected but not in data
- `insufficient_labels` -- labels with fewer than min_samples
- `total_classes` -- number of unique classes found

### Why it matters for ML

A missing class in training data means the model will **never** predict that class correctly. If your image classifier expects 10 categories but the training batch only has 8, the model will be blind to 2 categories in production. Even if a class exists but has only 3 samples, the model can't learn meaningful patterns from it.

### Example

```python
import pandas as pd
import pytest
from mltk.data import assert_label_coverage

@pytest.mark.ml_data
def test_all_categories_represented():
    """Every product category must have at least 50 training examples."""
    df = pd.read_csv("data/products.csv")
    assert_label_coverage(
        df["category"],
        expected_labels={"electronics", "clothing", "food", "home"},
        min_samples=50,
    )

@pytest.mark.ml_data
def test_minimum_samples_per_class():
    """Each class needs at least 10 examples for reliable training."""
    df = pd.read_csv("data/training.csv")
    assert_label_coverage(df["label"], min_samples=10)
```

### Edge Cases

- **expected_labels=None**: Only checks that every observed class has >= min_samples. Won't catch missing classes.
- **Extra classes**: Classes in the data but not in expected_labels are ignored (not an error)
- **min_samples=1**: Default — only checks class existence, not quantity

### Related Tests

| Test | What it validates |
|------|-------------------|
| `test_all_labels_present` | All expected labels exist in data |
| `test_missing_label` | Expected label not in data triggers failure |
| `test_sufficient_samples` | Each class has >= min_samples |
| `test_insufficient_samples` | Class with too few samples fails |
| `test_auto_detect_labels` | No expected_labels — checks all observed classes |

---
