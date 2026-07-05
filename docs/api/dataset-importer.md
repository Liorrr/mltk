# Smart Dataset Importer

Point mltk at a dataset — a local CSV/JSON/Parquet file or a HuggingFace
Hub id — and get back a normalized, column-role-mapped result ready to
become a versioned [`EvalDataset`](eval-datasets.md).

**Since:** Unreleased (S97, sprint 1 of a 3-sprint epic — see
[Roadmap](../roadmap.md#smart-dataset-importer-test-suite-mapper))

**Modules:**

- `mltk.importer` — `DatasetImporter`, `ColumnRole`, `ColumnMapping`,
  `ImportResult`, `auto_map_columns`

**Install:** `pip install mltk[importer]` (adds `datasets>=4.0` for
HuggingFace Hub loading — 4.0+ drops script-based/remote-code dataset
loading — and `pyarrow>=14` for local Parquet files). This is an
optional extra — a plain `pip install mltk` never pulls in either
dependency, and `import mltk` does not import `mltk.importer`.

---

## Why a Smart Importer?

Turning a dataset into a runnable mltk evaluation used to mean hand-writing
`EvalSample` objects and manually deciding which column is the prompt,
which is the expected answer, and which is retrieval context:

```python
# Manual — works, but someone has to read every column by hand first
samples = [
    EvalSample(input=row["question"], target=row["answer"],
               metadata={"context": row["passage"]})
    for row in raw_rows
]
```

`DatasetImporter` automates the column-role decision with deterministic
heuristics, shows you the inferred mapping before anything is built, and
lets you override any column it got wrong — then hands you a normal
[`EvalDataset`](eval-datasets.md) that plugs straight into the existing
registry, quality gates, and eval pipeline.

---

## Quick Start

```python
from mltk.importer import DatasetImporter, ColumnRole

# 1. Load and auto-map column roles
result = DatasetImporter.load("qa.csv")

# 2. Inspect the inferred mapping before trusting it
print(result.mapping.preview())
# question | input | What is the capital of France?
# answer | golden | Paris
# passage | context | France is in Western Europe.
# category | label | geography
# id | metadata | 1

# 3. Override anything the heuristics got wrong (optional)
fixed = result.mapping.override("category", ColumnRole.METADATA)

# 4. Materialize a versioned EvalDataset
dataset = result.to_eval_dataset(name="my-qa", version="0.1.0")
```

From here, `dataset` is a normal `EvalDataset` — save it with
`DatasetRegistry`, run `assert_dataset_quality`, or hand `dataset.samples`
to an `EvalTask`. See [Versioned Eval Datasets](eval-datasets.md) for that
half of the pipeline.

---

## Supported Sources

| Source | Status |
|--------|--------|
| Local CSV | Supported — `pandas.read_csv` |
| Local JSON | Supported — bare array or `{"samples": [...]}` shape |
| Local Parquet | Supported — requires `pyarrow` (`pip install mltk[importer]`) |
| HuggingFace Hub | Supported — requires `datasets` (`pip install mltk[importer]`); no network calls happen unless you actually call `.load()` with a Hub id |
| URL (remote CSV/JSON/Parquet) | **Not yet implemented** — raises `NotImplementedError`; download the file locally first |
| SQL / database source | **Not planned for the current epic** — not in the S97-S99 scope; see [Roadmap](../roadmap.md#smart-dataset-importer-test-suite-mapper) for the adapter roadmap (Kaggle, OpenML, object storage) |
| HuggingFace streaming mode (`streaming=True`) | **Not supported** — `DatasetImporter` always fully materializes the dataset (`dataset.to_list()`) before returning. Large Hub datasets that don't fit in memory are out of scope for S97 |

```python
DatasetImporter.load("data/qa.csv")               # local CSV
DatasetImporter.load("data/qa.json")               # local JSON
DatasetImporter.load("data/qa.parquet")            # local Parquet
DatasetImporter.load("squad")                      # HuggingFace Hub id
DatasetImporter.load("squad", split="validation")  # explicit split (default: "train")
```

!!! warning "No streaming, no database source, no URL fetch (yet)"
    If you need any of these, `DatasetImporter.load()` will either raise
    `NotImplementedError` (URL) or is simply not wired up (streaming,
    database). These are explicitly deferred, not silently degraded —
    there is no fallback that quietly loads a truncated or sampled
    dataset instead.

---

## Column Role Auto-Mapping

`auto_map_columns()` infers a `ColumnRole` for every column using
deterministic name + dtype heuristics — no ML, no network calls, no
guessing when a column doesn't clearly match. The same input always
produces the same mapping.

Keyword matching is **token-based, never substring-based**: a column
name is split on non-alphanumeric characters and camelCase boundaries
into word-tokens, and a keyword must match a whole token. `question_id`
does **not** match the `question` keyword (its tokens are `question`
and `id`); a column named `question` does.

| `ColumnRole` | Meaning | Maps to |
|---|---|---|
| `INPUT` | The prompt/question | `EvalSample.input` |
| `GOLDEN` | Expected answer/reference | `EvalSample.target` (first match); extra matches → `metadata["references"]` |
| `CONTEXT` | Retrieval context/passage (RAG) | `metadata["context"]` (`str` if one column, else `list[str \| None]`) |
| `LABEL` | Classification label | `metadata["label"]` **and** `metadata["category"]` (scalar if one column, else a `dict` under `metadata["label"]` only — see [Label/Category Mirroring](#labelcategory-mirroring)) |
| `METADATA` | Arbitrary passthrough | `metadata[<column name>]` |
| `IGNORE` | Explicitly excluded | Dropped |
| `UNKNOWN` | Heuristics found no confident match, or the role was already claimed by another column | Surfaced in `preview()` for manual `override()` — **never silently dropped or guessed** |

Heuristic priority (first match wins, checked in this order):

1. **Metadata-suffix rule (highest priority)** — a column whose *last*
   token is one of `id`/`idx`/`index`/`uid`/`uuid`/`timestamp`/`date`/`split`
   → `METADATA`. This runs before every other rule, so `question_id` and
   `created_date` are `METADATA`, not `INPUT`/`GOLDEN`.
2. **`INPUT`** — a token match on `input`/`prompt`/`question`/`query`,
   **string dtype required**. If more than one column qualifies, the
   first in column order becomes `INPUT` and the rest are demoted to
   `UNKNOWN` — surfaced in `preview()`/`validate()`, and still passed
   through as `metadata`, never silently dropped.
3. **`CONTEXT`** — a token match on `context`/`passage`/`document`/`chunk`/`retrieved`.
4. **`GOLDEN`** — a token match on `answer`/`target`/`expected`/`golden`/
   `reference`/`output`/`completion`/`response`, **string dtype
   required** — *except* when the column's whole name is exactly one of
   those keywords, in which case a numeric dtype is also accepted (e.g.
   a numeric `answer` column in a math-QA dataset maps to `GOLDEN`).
   `output_tokens` is a token match rather than a whole-name match, so
   it still requires string dtype and falls through to `UNKNOWN` if
   numeric.
5. **`LABEL`** — a token match on `label`/`class`/`category`, any dtype.
6. **`METADATA`** — a token match on `id`/`index`/`source`/`metadata`/`split`/
   `timestamp`/`date` anywhere in the tokens (not just the last one, which
   the higher-priority suffix rule above already covers).
7. **`text` fallback** — among still-unmapped columns, those with a
   `text` token and string dtype (e.g. `text`, `article_text`) are
   candidates; a lone candidate becomes `INPUT` only if no column
   already claimed `INPUT`. Two or more candidates all stay `UNKNOWN`.

Everything else is `UNKNOWN`.

```python
result = DatasetImporter.load("qa.csv")

problems = result.mapping.validate()
if problems:
    print(result.mapping.preview())  # show the user what needs fixing
    # e.g. "columns with UNKNOWN role need review: notes"

# Force a column's role manually — see Overrides below for the
# exclusive-override semantics these kwargs apply:
result = DatasetImporter.load(
    "qa.csv", input_column="passage", target_column="category",
)
```

### Overrides

`DatasetImporter.load(input_column=..., target_column=...)` is
**exclusive**: it guarantees the named column becomes
`EvalSample.input`/`.target` by demoting any other column currently
holding that role to `UNKNOWN`. `ColumnMapping.override(column, role,
exclusive=True)` does the same thing manually, after you've seen the
mapping. The default, `ColumnMapping.override(column, role)` (i.e.
`exclusive=False`), only reassigns the one named column and leaves
every other column's role untouched.

```python
# Exclusive: whatever column previously had role INPUT is demoted to
# UNKNOWN, so "passage" is guaranteed to become EvalSample.input.
result = DatasetImporter.load("qa.csv", input_column="passage")

# Same thing, after the fact:
fixed = result.mapping.override("passage", ColumnRole.INPUT, exclusive=True)

# Non-exclusive (default): only "category" changes role.
fixed = result.mapping.override("category", ColumnRole.METADATA)
```

### Label/Category Mirroring

A single `LABEL` column is written to **both** `metadata["label"]` and
`metadata["category"]`, so `EvalDataset.categories` and
`assert_dataset_quality(min_categories=...)` work without any extra
plumbing. When more than one column maps to `LABEL`, they collapse to
a `dict` under `metadata["label"]` only — `metadata["category"]` is not
populated in that case.

---

## Missing / Empty Cells

A missing cell (`None`, NaN, or a blank string) never becomes the
literal string `"None"`, and a non-missing value keeps its original
type — nothing gets stringified in transit.

- `EvalSample.input` is non-optional, so a missing `INPUT` cell becomes
  `""`.
- `GOLDEN`, `references`, and `context` are unchanged: a missing cell
  becomes `None` for that entry.
- `LABEL`, `METADATA`/`UNKNOWN` passthrough columns, and any
  extra-`INPUT` column demoted and passed through as metadata all
  normalize a missing cell to `None` (never NaN, never the string
  `"None"`).

---

## No Mock Data

If a column's role can't be confidently inferred, it is left `UNKNOWN` and
surfaced in `preview()`/`validate()` — mltk never invents a role or fills
in a plausible-looking value for a column it isn't sure about.

---

## What's Next

S97 covers loading, normalization, and column auto-mapping. Not yet
built (see [Roadmap](../roadmap.md#smart-dataset-importer-test-suite-mapper)):

- **S98** (second sprint of the epic) — task-type classification
  (classification / QA-RAG / summarization / generation / retrieval) and
  generation of a matching `MltkSuite` + committable pytest file;
  `mltk import <source>` CLI.
- **S99** (third sprint of the epic) — an MCP tool, golden-set binding
  with an `LLMJudgeScorer` fallback when no exact golden exists, and
  `DatasetRegistry` integration.

---

## See Also

- [Versioned Eval Datasets](eval-datasets.md) — `EvalDataset`,
  `DatasetRegistry`, `assert_dataset_quality`; the object
  `to_eval_dataset()` produces and where it's stored/validated next
- [Evaluation Pipeline](eval-pipeline.md) — `EvalTask`, solvers, and
  scorers that consume `dataset.samples`
- [LLM-as-Judge](llm-judge.md) — `LLMJudgeScorer` fallback for datasets
  without an exact golden answer
