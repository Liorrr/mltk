# Versioned Evaluation Datasets

Attach metadata, version numbers, and integrity fingerprints to evaluation
datasets — then store and retrieve them from a local registry that tracks
the full version history and produces semantic diffs across releases.

**Since:** v0.9.0

**Modules:**

- `mltk.eval.dataset` — `DatasetCard`, `EvalDataset`, `DatasetDiff`,
  `DatasetRegistry`
- `mltk.eval.dataset` — `assert_dataset_quality`

---

## Why Versioned Evaluation Datasets?

### The silent degradation problem

Most evaluation frameworks treat datasets as static files. They have no
memory of what changed, who authored the data, or whether the current
file is the same one a previous run used.

```python
# Typical unversioned pattern — no history, no integrity check
samples = load_dataset("my_eval.csv")
result = task.run(model_fn)
# Did the file change since last week? Unknown.
# Are 30% of items duplicates? Unknown.
# Is this the same dataset the model was benchmarked against? Unknown.
```

When everything is anonymous:

- Two runs that appear to compare the same model may use datasets that
  differ by hundreds of rows — producing misleading score trends.
- Removing a category from an evaluation silently breaks per-category
  analysis; there is no diff to show what changed.
- A model can be inadvertently benchmarked against an evaluation set that
  was corrupted, extended, or balanced differently since the last run.
- Reproducibility claims require *identical* data — not "same filename."

### What versioning gives you

The `EvalDataset` / `DatasetRegistry` stack solves this with four
complementary mechanisms:

| Mechanism | What it provides |
|-----------|-----------------|
| **Semantic versioning** | A human-readable signal (`MAJOR.MINOR.PATCH`) that communicates what kind of change happened |
| **SHA-256 fingerprint** | A cryptographic proof that the content has not changed since the version was stored |
| **DatasetCard** | Machine-readable provenance — task type, license, source, author, tags |
| **DatasetDiff** | A structural and content comparison between any two stored versions |

### What mltk adds

This architecture synthesizes patterns from HuggingFace Datasets,
lm-eval-harness, and DVC — adapted into three properties no existing
evaluation tool provides simultaneously:

- **pytest-native** — `assert_dataset_quality()` integrates directly with
  pytest, CI gates, and `MltkSuite`. No separate tooling required.
- **Zero dependencies** — registry operations use only Python stdlib
  (`json`, `hashlib`, `pathlib`). No DVC, no HF Hub, no cloud.
- **Eval-lifecycle aware** — the registry is designed for evaluation
  datasets specifically: it surfaces category balance, target coverage,
  and duplicate rates as first-class quality dimensions.

---

## Quick Start

Five steps from raw data to a versioned, quality-gated dataset:

```python
from mltk.eval.dataset import (
    DatasetCard,
    EvalDataset,
    DatasetRegistry,
    assert_dataset_quality,
)
from mltk.eval._types import EvalSample

# 1. Build a card — the dataset's identity document
card = DatasetCard(
    description="Geography Q&A evaluation set",
    task="question-answering",
    source="expert-curated",
    license="apache-2.0",
    tags=["geography", "factual", "zero-shot"],
    author="ml-team",
)

# 2. Create the versioned dataset
dataset = EvalDataset(
    name="geography-qa",
    version="1.0.0",
    samples=[
        EvalSample("Capital of France?", "Paris",
                   metadata={"category": "europe"}),
        EvalSample("Capital of Japan?", "Tokyo",
                   metadata={"category": "asia"}),
    ],
    card=card,
)

# 3. Assert quality gates before storing
assert_dataset_quality(dataset, min_samples=50)

# 4. Register and save to disk
registry = DatasetRegistry("~/.mltk/datasets")
registry.save(dataset)

# 5. Load it back later — identical fingerprint guaranteed
loaded = registry.load("geography-qa", version="1.0.0")
assert loaded.fingerprint == dataset.fingerprint
```

### What just happened?

1. `DatasetCard` captured the provenance metadata — task, source, and
   license — that makes the dataset citable and auditable.
2. `EvalDataset` assigned a semantic version (`1.0.0`) and computed a
   SHA-256 fingerprint over the full sample list at construction time.
3. `assert_dataset_quality` ran four quality gates — minimum sample
   count, target coverage, duplicate rate, and category diversity.
4. `DatasetRegistry.save()` serialized the dataset (card + samples) to
   `~/.mltk/datasets/geography-qa/1.0.0/` and updated the version index.
5. `registry.load()` deserialized the stored version and re-computed the
   fingerprint; an assertion mismatch would raise `ValueError`.

---

## Data Model

### DatasetCard — provenance metadata

A `DatasetCard` is the identity document for an evaluation dataset. It
captures the non-sample information needed to understand, cite, and
assess the suitability of a dataset.

```python
from mltk.eval.dataset import DatasetCard
from datetime import date

card = DatasetCard(
    description=(
        "500-item Q&A benchmark covering world geography, "
        "suitable for zero-shot and few-shot evaluation."
    ),
    task="question-answering",
    source="expert-curated",
    license="apache-2.0",
    tags=["geography", "factual", "zero-shot", "multilingual"],
    created=date(2026, 4, 1),
    author="ml-team",
)
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `description` | `str` | Yes | Plain-language summary of the dataset's purpose |
| `task` | `str` | Yes | Task category (`question-answering`, `text-classification`, etc.) |
| `source` | `str` | Yes | How data was collected (`expert-curated`, `crowdsourced`, `machine-generated`, `found`) |
| `license` | `str` | Yes | SPDX identifier (e.g. `apache-2.0`, `mit`, `cc-by-4.0`) or `proprietary` |
| `tags` | `list[str]` | No | Arbitrary discovery tags for filtering and search |
| `created` | `date` | No | Dataset creation date; defaults to today |
| `author` | `str` | No | Team or individual who owns the dataset |

!!! note "Cards travel with datasets"
    A `DatasetCard` is embedded inside `EvalDataset` and serialized
    alongside samples when saved to the registry. Loading a dataset
    always returns the card that was stored with it — there is no way
    to have a dataset without its provenance.

**Why `source` matters:** Evaluation datasets collected from different
sources have different contamination risk profiles. Datasets generated by
the same LLM family being evaluated are high-risk for self-grading
inflation. Explicitly recording the source makes contamination audits
possible. See [Research Citations](#research-citations) for the
contamination detection literature.

---

### EvalDataset — the versioned dataset object

`EvalDataset` is the central object. It wraps a list of `EvalSample`
objects with a semantic version, a card, and a content fingerprint.

```python
from mltk.eval.dataset import EvalDataset
from mltk.eval._types import EvalSample

dataset = EvalDataset(
    name="geography-qa",
    version="1.2.0",
    samples=[
        EvalSample(
            input="What is the capital of France?",
            target="Paris",
            metadata={"category": "europe", "difficulty": "easy"},
        ),
        EvalSample(
            input="What is the capital of Japan?",
            target="Tokyo",
            metadata={"category": "asia", "difficulty": "easy"},
        ),
        # ... more samples
    ],
    card=card,
)
```

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `name` | `str` | Yes | Dataset identifier — used as the registry key |
| `version` | `str` | Yes | Semantic version string: `MAJOR.MINOR.PATCH` |
| `samples` | `list[EvalSample]` | Yes | The evaluation items |
| `card` | `DatasetCard` | Yes | Provenance and metadata |
| `fingerprint` | `str` | Auto | SHA-256 of the serialized sample list; computed at construction |

#### Computed properties

```python
# Number of samples
print(dataset.sample_count)      # 500

# Fraction of samples that have a non-None target
print(dataset.target_coverage)   # 0.98

# Unique values from metadata["category"] across all samples
print(dataset.categories)        # {"europe", "asia", "americas", ...}
```

| Property | Return type | Description |
|----------|-------------|-------------|
| `sample_count` | `int` | Length of the samples list |
| `target_coverage` | `float` | Fraction with a non-`None` target (`0.0`–`1.0`) |
| `categories` | `set[str]` | Distinct `metadata["category"]` values across all samples |

!!! tip "Categories require consistent metadata keys"
    `dataset.categories` reads from `sample.metadata["category"]` on
    each sample. Samples that lack this key are excluded from the set.
    Use a consistent metadata schema across all samples in a dataset —
    if some items use `"category"` and others use `"topic"`, the
    `categories` property will be incomplete.

#### Factory methods

Three class methods construct an `EvalDataset` from common data sources:

```python
# From a CSV file — columns: input, target, and any metadata columns
dataset = EvalDataset.from_csv(
    path="data/geography.csv",
    name="geography-qa",
    version="1.0.0",
    card=card,
    input_column="question",    # default: "input"
    target_column="answer",     # default: "target"
)

# From a JSON file — list of {"input": ..., "target": ..., ...} dicts
dataset = EvalDataset.from_json(
    path="data/geography.json",
    name="geography-qa",
    version="1.0.0",
    card=card,
)

# From an existing list of dicts
records = [
    {"input": "Capital of France?", "target": "Paris",
     "category": "europe"},
    {"input": "Capital of Japan?",  "target": "Tokyo",
     "category": "asia"},
]
dataset = EvalDataset.from_dict(
    records=records,
    name="geography-qa",
    version="1.0.0",
    card=card,
)
```

All three factories call `EvalSample` for each record and auto-compute
the fingerprint. Any column that is not `input_column` or `target_column`
is mapped into `EvalSample.metadata`.

---

### DatasetDiff — comparing two versions

`DatasetDiff` captures the structural difference between two stored
versions of the same dataset — at the row level (which samples were added
or removed) and at the schema level (which metadata keys changed).

```python
from mltk.eval.dataset import DatasetRegistry

registry = DatasetRegistry("~/.mltk/datasets")
diff = registry.diff("geography-qa", "1.0.0", "1.1.0")

print(diff.old_version)      # "1.0.0"
print(diff.new_version)      # "1.1.0"
print(len(diff.added_samples))            # 23 — new samples in v1.1.0
print(len(diff.removed_samples))          # 0  — no samples removed
print(len(diff.unchanged_samples))        # 477 — samples identical in both versions
print(diff.schema_changes)   # [] — no metadata key changes
print(diff.suggested_bump)   # "patch" — only additions, no removals
```

| Field | Type | Description |
|-------|------|-------------|
| `old_version` | `str` | Version string of the base (older) dataset |
| `new_version` | `str` | Version string of the comparison (newer) dataset |
| `added` | `int` | Number of samples in new but not in old |
| `removed` | `int` | Number of samples in old but not in new |
| `unchanged` | `int` | Number of samples identical in both versions |
| `schema_changes` | `list[str]` | Metadata keys added or removed between versions |
| `suggested_bump` | `str` | `"major"`, `"minor"`, or `"patch"` — see [Versioning Rules](#versioning-rules) |

!!! warning "Diff uses sample content hashing, not IDs"
    Samples are compared by hashing their `(input, target)` pair.
    `EvalSample` has no built-in primary key. If you reword a question
    (even slightly), the diff reports the old wording as removed and the
    new wording as added — both counts increment by 1. This is intentional:
    any change to a test item changes what is being measured.

---

### DatasetRegistry — the local version store

`DatasetRegistry` is a directory-backed store. Each dataset gets its own
subdirectory; each version is a subdirectory under that.

```
~/.mltk/datasets/
├── geography-qa/
│   ├── index.json              ← version list + fingerprints
│   ├── 1.0.0/
│   │   ├── dataset.json        ← card + samples
│   │   └── fingerprint.sha256  ← SHA-256 of dataset.json
│   └── 1.1.0/
│       ├── dataset.json
│       └── fingerprint.sha256
└── rag-eval/
    ├── index.json
    └── 0.1.0/
        ├── dataset.json
        └── fingerprint.sha256
```

```python
from mltk.eval.dataset import DatasetRegistry

registry = DatasetRegistry("~/.mltk/datasets")
```

The `registry_dir` path is created automatically if it does not exist.

---

## Registry Operations

### save — persist a dataset version

```python
registry.save(dataset)
```

`save()` will raise `ValueError` if the same `name` and
`version` combination is already stored. Versions are immutable once
written — to update a dataset, increment the version number and call
`save()` again.

```python
# Correct: increment version for any content change
updated_dataset = EvalDataset(
    name="geography-qa",
    version="1.1.0",
    samples=[...],
    card=card,
)
registry.save(updated_dataset)  # stores alongside 1.0.0
```

!!! warning "Versions are write-once"
    This is intentional. The SHA-256 fingerprint stored at save time
    is the ground truth for that version. Allowing overwrites would
    break the integrity guarantee — a `load()` call returning a
    different dataset for the same version string would silently
    invalidate every result that cited that version.

    To correct a stored dataset, use `registry.delete()` followed by
    `registry.save()`. This is a destructive operation: document the
    reason in the dataset's changelog or card description.

---

### load — retrieve a stored version

```python
# Load a specific version
dataset = registry.load("geography-qa", version="1.0.0")

# Load the latest version (highest semver)
dataset = registry.load("geography-qa")
```

`load()` re-computes the SHA-256 fingerprint of the loaded data and
compares it to the stored `fingerprint.sha256`. A mismatch raises
`ValueError`, indicating the stored file was modified outside
the registry API.

```python
# Verify fingerprint manually
dataset = registry.load("geography-qa", version="1.0.0")
assert dataset.fingerprint == "a3f2b8c91d4e5f6071..."  # pin in test
```

Pinning the expected fingerprint in a test is the strongest reproducibility
guarantee available — it asserts that the exact bytes evaluated during
the benchmark are the bytes loaded now.

---

### list — enumerate stored datasets

```python
# All dataset names in the registry
names = registry.list()   # ["geography-qa", "rag-eval", "bias-v2"]
```

---

### versions — version history for one dataset

```python
versions = registry.versions("geography-qa")
# ["1.0.0", "1.1.0", "2.0.0"]
```

Versions are returned in ascending semver order, oldest first.

---

### exists — check before loading

```python
if registry.exists("geography-qa", version="1.0.0"):
    dataset = registry.load("geography-qa", version="1.0.0")
```

Without a version argument, `exists()` returns `True` if the dataset
name is known at all (any version).

```python
registry.exists("geography-qa")           # True if any version stored
registry.exists("geography-qa", "3.0.0")  # True only if that version exists
```

---

### diff — compare two stored versions

```python
diff = registry.diff("geography-qa", "1.0.0", "1.1.0")
print(f"Added: {len(diff.added_samples)}, Removed: {len(diff.removed_samples)}")
print(f"Suggested bump: {diff.suggested_bump}")
```

See the [`DatasetDiff`](#datasetdiff-comparing-two-versions) section for
the full field reference.

---

### delete — remove a stored version

```python
# Remove a specific version
registry.delete("geography-qa", version="1.0.0")

# Remove all versions of a dataset
registry.delete("geography-qa")
```

`delete()` is irreversible. After deletion, `load()` will raise
`FileNotFoundError` for the removed version. The version is also
removed from `index.json`.

---

## Dataset Cards

A dataset card is the single place to record everything a consumer needs
to assess whether a dataset is appropriate for their evaluation task and
how to cite it. The `DatasetCard` model maps to the HuggingFace Dataset
Card standard (see [Research Citations](#research-citations)) but strips
it down to the fields relevant for evaluation-specific use.

### Minimum viable card

```python
card = DatasetCard(
    description="300-item RAG faithfulness eval, legal domain.",
    task="question-answering",
    source="expert-curated",
    license="apache-2.0",
)
```

This is the minimum required set. A dataset without a card cannot be
saved to the registry.

### Full card with all optional fields

```python
from datetime import date

card = DatasetCard(
    description=(
        "500-item geography Q&A benchmark for zero-shot and "
        "few-shot settings. Covers 6 continents, 40 countries. "
        "Difficulty-stratified: 30% easy, 50% medium, 20% hard. "
        "No known contamination with MMLU or TriviaQA training sets."
    ),
    task="question-answering",
    source="expert-curated",
    license="apache-2.0",
    tags=[
        "geography",
        "factual",
        "zero-shot",
        "few-shot",
        "difficulty-stratified",
    ],
    created=date(2026, 4, 1),
    author="ml-team",
)
```

!!! tip "Description is the most important field"
    Write the description as if the reader has never seen this dataset
    before. Include: what it measures, what it does *not* measure, the
    difficulty distribution, known limitations, and any contamination
    caveats. A thorough description prevents misuse and enables correct
    citation.

### Card serialization

Cards are serialized to JSON as part of `dataset.json` and round-trip
cleanly:

```python
import json

card_dict = card.to_dict()
restored = DatasetCard.from_dict(card_dict)
assert restored.description == card.description
```

---

## Versioning Rules

mltk uses semantic versioning (`MAJOR.MINOR.PATCH`) for datasets,
following the DSLP / Nature Scientific Data (2024) standard. The rules
map schema compatibility to version bump size:

| Version bump | When to use | Consumer impact |
|-------------|-------------|-----------------|
| **PATCH** `x.x.N` | Additional samples added; no schema change; category distribution stable | Safe — existing analysis and score comparisons are valid |
| **MINOR** `x.N.0` | New metadata keys added; new categories introduced; existing samples unchanged | Safe for existing code; new code needed to use new keys |
| **MAJOR** `N.0.0` | Samples removed or modified; metadata keys renamed or removed; category scheme changed | Breaking — all prior benchmark scores must be re-evaluated |

### When `DatasetDiff.suggested_bump` fires

The registry's `diff()` method computes a suggested version bump after
comparing two stored versions. It is a suggestion, not enforcement — you
decide the final version string. Use it as a sanity check:

```python
diff = registry.diff("geography-qa", "1.0.0", "2.0.0")

if diff.suggested_bump == "major" and not version_is_major_bump:
    raise ValueError(
        f"Removed {len(diff.removed_samples)} samples and changed schema "
        f"but version was only a minor bump. Use 2.0.0."
    )
```

### Pre-1.0.0 conventions

Before a dataset reaches production stability, use `0.x.y`:

```
v0.1.0  — Initial dataset, exploratory
v0.2.0  — Added 3 new categories (treat as MINOR)
v0.2.1  — Added 50 new samples to existing categories (PATCH)
v1.0.0  — Production release; schema frozen
v1.1.0  — Added optional "difficulty" metadata key (MINOR)
v2.0.0  — Removed "legacy_label" key; re-encoded categories (MAJOR)
```

!!! note "Pre-1.0.0: MINOR carries breaking-change weight"
    This mirrors pre-1.0.0 semver conventions in software. Minor bumps
    before v1.0.0 may contain schema changes that would be MAJOR
    after stabilization.

---

## Quality Gates

`assert_dataset_quality` is the gating function that validates an
`EvalDataset` against four quality dimensions before it is stored or
used in evaluation.

```python
from mltk.eval.dataset import assert_dataset_quality

assert_dataset_quality(
    dataset,
    min_samples=100,
    min_target_coverage=0.95,
    max_duplicate_rate=0.01,
    min_categories=3,
)
```

If any gate fails, `MltkAssertionError` is raised with a structured
message that names the failing dimension and the observed value.

### Parameter reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dataset` | `EvalDataset` | — | The dataset to validate |
| `min_samples` | `int` | `10` | Minimum number of samples required |
| `min_target_coverage` | `float` | `0.9` | Minimum fraction of samples with a non-`None` target |
| `max_duplicate_rate` | `float` | `0.0` | Maximum fraction of exact duplicate `(input, target)` pairs |
| `min_categories` | `int` | `1` | Minimum number of distinct `metadata["category"]` values |

### What each gate catches

**`min_samples`** — Prevents evaluation on trivially small datasets
where aggregate accuracy has high variance. A dataset with 10 samples
produces ±10% accuracy bands at the single-sample level, making
improvement signals unreliable. The MIT TDQM completeness dimension
(Pipino et al., 2002) defines completeness as a structural property of
the dataset, not just an annotation property.

**`min_target_coverage`** — Targets are required for all scorers except
`LLMJudgeScorer`. A dataset where 20% of items lack targets will silently
undercount failures for `ExactMatchScorer` and `IncludesScorer`, inflating
accuracy. Coverage below 90% is almost always a data pipeline bug.

**`max_duplicate_rate`** — Exact duplicate test items evaluate the same
capability twice, inflate sample counts, and bias accuracy estimates
toward over-represented questions. The quality research baseline
(dataset-cards-quality.md) sets a 0% exact duplicate target; the default
allows a 0% rate with explicit opt-in for higher tolerances.

**`min_categories`** — A single-category dataset cannot detect
category-level regressions. At least 3 distinct categories enables
per-category accuracy breakdown in `EvalResult`.

### Using quality gates in pytest

```python
import pytest
from mltk.eval.dataset import (
    EvalDataset,
    DatasetRegistry,
    assert_dataset_quality,
)

def test_dataset_quality():
    registry = DatasetRegistry("~/.mltk/datasets")
    dataset = registry.load("geography-qa", version="1.0.0")

    assert_dataset_quality(
        dataset,
        min_samples=200,
        min_target_coverage=0.99,
        max_duplicate_rate=0.0,
        min_categories=5,
    )
```

### Using quality gates in MltkSuite

```python
from mltk.core.suite import MltkSuite
from mltk.eval.dataset import (
    EvalDataset,
    DatasetRegistry,
    assert_dataset_quality,
)

registry = DatasetRegistry("~/.mltk/datasets")
dataset = registry.load("geography-qa")

suite = MltkSuite("dataset-integrity")
suite.add(
    assert_dataset_quality,
    dataset,
    min_samples=200,
    min_target_coverage=0.99,
    max_duplicate_rate=0.0,
    min_categories=5,
)
result = suite.run()
assert result.passed
```

---

## CI/CD Integration

### Version pinning in CI

Pin the exact version and fingerprint in CI to guarantee that every run
evaluates the identical dataset:

```python
# tests/conftest.py
import pytest
from mltk.eval.dataset import DatasetRegistry

REGISTRY_DIR = "~/.mltk/datasets"
DATASET_NAME = "geography-qa"
DATASET_VERSION = "1.2.0"
# Obtain via: registry.load(DATASET_NAME, DATASET_VERSION).fingerprint
EXPECTED_FINGERPRINT = "a3f2b8c91d4e5f6071829304a5b6c7d8e9f0a1b2"

@pytest.fixture(scope="session")
def pinned_dataset():
    registry = DatasetRegistry(REGISTRY_DIR)
    dataset = registry.load(DATASET_NAME, version=DATASET_VERSION)
    assert dataset.fingerprint == EXPECTED_FINGERPRINT, (
        f"Dataset fingerprint mismatch for {DATASET_NAME}@"
        f"{DATASET_VERSION}. "
        f"Expected: {EXPECTED_FINGERPRINT}, "
        f"Got: {dataset.fingerprint}. "
        "The stored dataset file may have been modified."
    )
    return dataset
```

### Pre-registration quality gate

Run quality gates before pushing a new version to the registry. This
pattern treats the registry as a deployment target — just as you would
gate a model release behind a test suite:

```python
# scripts/register_dataset.py
import sys
from mltk.eval.dataset import (
    EvalDataset,
    DatasetCard,
    DatasetRegistry,
    assert_dataset_quality,
)

def register_new_version(csv_path: str, version: str) -> None:
    card = DatasetCard(
        description="Geography Q&A benchmark",
        task="question-answering",
        source="expert-curated",
        license="apache-2.0",
    )

    dataset = EvalDataset.from_csv(
        path=csv_path,
        name="geography-qa",
        version=version,
        card=card,
    )

    # Quality gates must pass before registration
    assert_dataset_quality(
        dataset,
        min_samples=200,
        min_target_coverage=0.99,
        max_duplicate_rate=0.0,
        min_categories=5,
    )

    registry = DatasetRegistry("~/.mltk/datasets")
    registry.save(dataset)
    print(f"Registered geography-qa@{version} "
          f"({dataset.sample_count} samples, "
          f"fingerprint: {dataset.fingerprint[:12]}...)")

if __name__ == "__main__":
    register_new_version(
        csv_path=sys.argv[1],
        version=sys.argv[2],
    )
```

```bash
python scripts/register_dataset.py data/geography_v2.csv 1.1.0
# Registered geography-qa@1.1.0 (523 samples, fingerprint: a3f2b8c91d4e...)
```

### Diff check before merging

Add a diff check to catch breaking version bumps before they merge:

```yaml
# .github/workflows/dataset-check.yml
- name: Check dataset version bump
  run: |
    python - << 'EOF'
    from mltk.eval.dataset import DatasetRegistry
    registry = DatasetRegistry("~/.mltk/datasets")
    diff = registry.diff(
        "geography-qa",
        old_version="1.0.0",
        new_version="${{ env.NEW_VERSION }}"
    )
    if len(diff.removed_samples) > 0 and not diff.suggested_bump == "major":
        raise SystemExit(
            f"Removed {len(diff.removed_samples)} samples but version is not "
            f"a major bump. Use {diff.new_version} as "
            f"a MAJOR version or restore removed samples."
        )
    print(f"Diff OK: +{len(diff.added_samples)} -{len(diff.removed_samples)} "
          f"(suggested: {diff.suggested_bump})")
    EOF
```

---

## Examples

### Example 1 — Create, validate, and register

End-to-end workflow: build a dataset from a CSV, run quality gates, and
store it in the registry:

```python
from mltk.eval.dataset import (
    DatasetCard,
    EvalDataset,
    DatasetRegistry,
    assert_dataset_quality,
)

card = DatasetCard(
    description=(
        "500-item RAG faithfulness evaluation. Documents sourced "
        "from public domain legal texts. Answers verified by two "
        "independent annotators (kappa=0.83)."
    ),
    task="question-answering",
    source="expert-curated",
    license="cc-by-4.0",
    tags=["rag", "faithfulness", "legal", "annotated"],
    author="ml-team",
)

dataset = EvalDataset.from_csv(
    path="data/rag_faithfulness.csv",
    name="rag-faithfulness",
    version="1.0.0",
    card=card,
    input_column="question",
    target_column="expected_answer",
)

assert_dataset_quality(
    dataset,
    min_samples=400,
    min_target_coverage=1.0,   # every item must have a target
    max_duplicate_rate=0.0,
    min_categories=4,
)

registry = DatasetRegistry("~/.mltk/datasets")
registry.save(dataset)

print(f"Registered: {dataset.name}@{dataset.version}")
print(f"Samples:    {dataset.sample_count}")
print(f"Coverage:   {dataset.target_coverage:.1%}")
print(f"Categories: {len(dataset.categories)}")
print(f"Fingerprint: {dataset.fingerprint}")
```

---

### Example 2 — Version upgrade with diff validation

Update an existing dataset from v1.0.0 to v1.1.0 by adding a new
category. Use the diff to verify the version bump is appropriate:

```python
from mltk.eval.dataset import (
    DatasetCard,
    EvalDataset,
    DatasetRegistry,
)

registry = DatasetRegistry("~/.mltk/datasets")

# Load the previous version to read its card
old_dataset = registry.load("geography-qa", version="1.0.0")

# Build the updated version with the new card
updated_card = DatasetCard(
    description=(
        "Geography Q&A benchmark — v1.1.0 adds Oceania category "
        "(45 new items). Prior categories unchanged."
    ),
    task=old_dataset.card.task,
    source=old_dataset.card.source,
    license=old_dataset.card.license,
    tags=old_dataset.card.tags + ["oceania"],
    author=old_dataset.card.author,
)

new_dataset = EvalDataset.from_csv(
    path="data/geography_v1_1.csv",
    name="geography-qa",
    version="1.1.0",
    card=updated_card,
)
registry.save(new_dataset)

# Validate the version bump
diff = registry.diff("geography-qa", "1.0.0", "1.1.0")

assert len(diff.removed_samples) == 0, (
    f"Expected no removals in a MINOR bump; got {len(diff.removed_samples)}"
)
assert diff.suggested_bump in ("patch", "minor"), (
    f"Unexpected diff: {len(diff.added_samples)} added, {len(diff.removed_samples)} removed, "
    f"schema changes: {diff.schema_changes}"
)

print(f"v1.0.0 → v1.1.0: +{len(diff.added_samples)} samples, "
      f"suggested bump: {diff.suggested_bump}")
```

---

### Example 3 — Fingerprint-pinned evaluation

Use a pinned fingerprint in a test to guarantee that the benchmarked
dataset has not drifted since the score was recorded:

```python
import pytest
from mltk.eval.dataset import DatasetRegistry
from mltk.eval.task import EvalTask
from mltk.eval.solvers import GenerateSolver
from mltk.eval.scorers import ExactMatchScorer

# Pinned at benchmark time
GEOGRAPHY_FINGERPRINT = "a3f2b8c91d4e5f6071829304a5b6c7d8e9f0a1b2"

@pytest.fixture(scope="session")
def geography_dataset():
    registry = DatasetRegistry("~/.mltk/datasets")
    dataset = registry.load("geography-qa", version="1.0.0")
    assert dataset.fingerprint == GEOGRAPHY_FINGERPRINT, (
        "Dataset has changed since this score was recorded. "
        "Re-run the benchmark and update the pinned fingerprint."
    )
    return dataset

def test_geography_accuracy(geography_dataset, my_model):
    task = EvalTask(
        name="geography-qa",
        solver=GenerateSolver(),
        scorers=ExactMatchScorer(),
        dataset=geography_dataset.samples,
    )
    result = task.to_test_result(my_model, min_accuracy=0.85)
    assert result.passed, result.details
```

---

### Example 4 — Full lifecycle: build, register, diff, evaluate

A complete lifecycle that demonstrates all major APIs together:

```python
from mltk.eval.dataset import (
    DatasetCard,
    EvalDataset,
    DatasetRegistry,
    assert_dataset_quality,
)
from mltk.eval.task import EvalTask
from mltk.eval.solvers import GenerateSolver
from mltk.eval.scorers import ExactMatchScorer

# -- Build --
card = DatasetCard(
    description="Arithmetic reasoning evaluation set, difficulty-stratified.",
    task="question-answering",
    source="expert-curated",
    license="apache-2.0",
    tags=["math", "arithmetic", "difficulty-stratified"],
    author="ml-team",
)
dataset_v1 = EvalDataset.from_csv(
    path="data/arithmetic_v1.csv",
    name="arithmetic-eval",
    version="1.0.0",
    card=card,
)

# -- Validate --
assert_dataset_quality(
    dataset_v1,
    min_samples=300,
    min_target_coverage=1.0,
    max_duplicate_rate=0.0,
    min_categories=3,  # easy / medium / hard
)

# -- Register --
registry = DatasetRegistry("~/.mltk/datasets")
registry.save(dataset_v1)

# -- Upgrade to v1.1.0 (patch: 50 new easy samples) --
dataset_v11 = EvalDataset.from_csv(
    path="data/arithmetic_v1_1.csv",
    name="arithmetic-eval",
    version="1.1.0",
    card=card,
)
assert_dataset_quality(dataset_v11, min_samples=350)
registry.save(dataset_v11)

# -- Diff --
diff = registry.diff("arithmetic-eval", "1.0.0", "1.1.0")
assert len(diff.removed_samples) == 0
print(f"Diff: +{len(diff.added_samples)} added, {len(diff.removed_samples)} removed")
print(f"Suggested bump: {diff.suggested_bump}")

# -- Evaluate against v1.1.0 --
loaded = registry.load("arithmetic-eval", version="1.1.0")
task = EvalTask(
    name="arithmetic",
    solver=GenerateSolver(),
    scorers=ExactMatchScorer(),
    dataset=loaded.samples,
)

def my_model(prompt: str) -> str:
    return "42"  # replace with real model

result = task.run(my_model)
print(f"Accuracy: {result.metrics['ExactMatchScorer/accuracy']:.1%}")
print(f"Dataset:  {loaded.name}@{loaded.version} "
      f"({loaded.fingerprint[:12]}...)")
```

---

## Competitor Comparison

mltk's dataset registry is inspired by and compared against the major
versioning tools in the field:

| Feature | DVC | HF Datasets | lm-eval-harness | MLflow | **mltk** |
|---------|-----|-------------|----------------|--------|--------|
| Semantic versioning | Via Git tags | `revision=` (Git tags) | No (YAML-per-task) | No | **Yes (semver)** |
| Content fingerprint | MD5 per file | SHA-256 (auto on push) | No | Digest per run | **Yes (SHA-256)** |
| Dataset card / metadata | README.md (optional) | Mandatory YAML | Task YAML (partial) | No | **Yes (DatasetCard)** |
| Structural diff | No built-in | No built-in | No | No | **Yes (DatasetDiff)** |
| Quality gates | No | No | No | No | **Yes (assert_dataset_quality)** |
| Pytest-native | No | No | No | No | **Yes** |
| Zero external dependencies | No (DVC CLI + remote) | No (HF Hub) | No (HF Hub) | No (MLflow server) | **Yes** |
| Cloud account required | Optional (for remotes) | Optional | No | Optional | **No** |
| Per-category coverage check | No | No | No | No | **Yes** |
| Duplicate detection | No | No | No | No | **Yes** |

### What mltk uniquely adds

**Structural diff with suggested version bump** — No evaluation framework
provides a `DatasetDiff` that computes which samples were added and
removed and suggests the appropriate semver bump based on those changes.
DVC tracks file-level hashes but does not understand evaluation dataset
semantics (categories, targets, duplicates).

**Quality gates as assertions** — `assert_dataset_quality` raises
`MltkAssertionError` and participates in `MltkSuite` — the same contract
as `assert_no_drift`, `assert_latency`, and `assert_metric`. Dataset
quality becomes a first-class citizen of the ML test suite, not a
separate pre-processing script.

**Eval-lifecycle integration** — Other tools version data in isolation.
mltk's `EvalDataset.samples` is directly compatible with `EvalTask` and
the solver/scorer pipeline in [eval-pipeline.md](eval-pipeline.md).
The registry is the data source for evaluations, not a separate system.

**Zero dependencies** — DVC requires a CLI install, remote storage
configuration, and `.dvc` file management. HuggingFace Datasets requires
`datasets`, `huggingface-hub`, and internet access. mltk's registry uses
only `json`, `hashlib`, and `pathlib` — it works in air-gapped
environments, CI containers with no extra pip installs, and local
notebooks.

---

## Research Citations

The mltk dataset versioning system synthesizes patterns from four
research and tooling lineages:

**HuggingFace Dataset Cards (2022)**
The `DatasetCard` model maps to the HuggingFace `DatasetCardData` schema.
The `task`, `source`, `license`, `tags`, and `description` fields are
drawn from the HF Hub's YAML front-matter standard, stripped down to the
fields relevant for evaluation (no `language_creators`, `multilinguality`,
or `size_categories`, which are irrelevant for small eval sets).
Source: `huggingface.co/docs/hub/en/datasets-cards`
Research brief: `docs/research/dataset-cards-quality.md`

**Gebru et al. — Datasheets for Datasets (2021)**
The `DatasetCard.description` field is designed to answer the seven
Datasheet sections: Motivation, Composition, Collection Process,
Preprocessing/Labeling, Uses, Distribution, Maintenance. A
description that covers all seven is a complete datasheet.
DOI: `10.1145/3458723`
Research brief: `docs/research/dataset-cards-quality.md`

**DSLP / Nature Scientific Data — Semver for Datasets (2024)**
The `MAJOR.MINOR.PATCH` mapping (breaking schema change → MAJOR, new
columns/categories → MINOR, added rows → PATCH) is drawn from the Data
Science Lifecycle Process proposal and its 2024 Nature Scientific Data
formalization. `DatasetDiff.suggested_bump` implements the DSLP bump
rules directly.
Research brief: `docs/research/dataset-versioning-patterns.md`

**DVC — Content-Addressable Storage for ML Data (2020–2024)**
The SHA-256 fingerprint approach — where a single hash over the full
dataset content is the immutable identity of a version — adapts DVC's
MD5-per-file CAS model to the evaluation dataset context. mltk uses a
single hash over the serialized sample list rather than per-file hashes
because evaluation datasets are logically one unit (the sample list),
not a directory tree.
Source: `github.com/iterative/dvc`
Research brief: `docs/research/dataset-versioning-patterns.md`

**MIT TDQM — Data Quality Dimensions (Pipino, Lee, Wang 2002)**
The four quality dimensions in `assert_dataset_quality` (completeness
via `min_target_coverage`, uniqueness via `max_duplicate_rate`,
completeness via `min_samples`, and coverage via `min_categories`) map
to the MIT Total Data Quality Management framework's most applicable
dimensions for evaluation datasets.
Source: *Communications of the ACM*, 2002.
Research brief: `docs/research/dataset-cards-quality.md`

**Easy2Hard-Bench — Difficulty Distribution (NeurIPS 2024)**
The three-tier difficulty distribution recommendation (30% easy / 50%
medium / 20% hard) in `DatasetCard.description` guidelines is drawn
from Easy2Hard-Bench's standardized difficulty methodology, which
empirically validated this split as the most discriminating for current
model capability ranges.
Source: NeurIPS 2024 Datasets and Benchmarks Track.
Research brief: `docs/research/dataset-cards-quality.md`

---

## See Also

- [eval-pipeline.md](eval-pipeline.md) — `EvalTask`, `EvalSample`,
  solvers and scorers; consume registered datasets with `dataset.samples`
- [suite-api.md](suite-api.md) — `MltkSuite` integration; mix dataset
  quality assertions with model performance assertions
- [llm-judge.md](llm-judge.md) — `LLMJudgeScorer` patterns for
  evaluation datasets that lack hard ground-truth targets
- [pytest-plugin.md](pytest-plugin.md) — pytest plugin configuration
  for loading pinned dataset versions via `--mltk-yaml`
