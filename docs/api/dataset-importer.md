# Smart Dataset Importer

Point mltk at a dataset — a local CSV/JSON/Parquet file or a HuggingFace
Hub id — and get back a normalized, column-role-mapped result ready to
become a versioned [`EvalDataset`](eval-datasets.md).

**Since:** Unreleased (Sprint 1 of a 3-sprint epic — see
[Roadmap](../roadmap.md#smart-dataset-importer-test-suite-mapper))

**Modules:**

- `mltk.importer` — `DatasetImporter`, `ColumnRole`, `ColumnMapping`,
  `ImportResult`, `auto_map_columns`

**Install:** `pip install mltk[importer]` (adds `datasets>=2.0` for
HuggingFace Hub loading and `pyarrow>=14` for local Parquet files). This
is an optional extra — a plain `pip install mltk` never pulls in either
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
from mltk.importer import DatasetImporter

# 1. Load and auto-map column roles
result = DatasetImporter.load("qa.csv")

# 2. Inspect the inferred mapping before trusting it
print(result.mapping.preview())
# column   | role    | sample
# question | input   | What is the capital of France?
# answer   | golden  | Paris
# passage  | context | France is in Western Europe.
# category | label   | geography
# id       | metadata| 1

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
| SQL / database source | **Not planned for the current epic** — not in the Sprint 1-3 scope; see [Roadmap](../roadmap.md#smart-dataset-importer-test-suite-mapper) for the adapter roadmap (Kaggle, OpenML, object storage) |
| HuggingFace streaming mode (`streaming=True`) | **Not supported** — `DatasetImporter` always fully materializes the dataset (`dataset.to_list()`) before returning. Large Hub datasets that don't fit in memory are out of scope for Sprint 1 |

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

| `ColumnRole` | Meaning | Maps to |
|---|---|---|
| `INPUT` | The prompt/question | `EvalSample.input` |
| `GOLDEN` | Expected answer/reference | `EvalSample.target` (first match); extra matches → `metadata["references"]` |
| `CONTEXT` | Retrieval context/passage (RAG) | `metadata["context"]` (`str` if one column, else `list[str \| None]`) |
| `LABEL` | Classification label | `metadata["label"]` (scalar if one column, else `dict`) |
| `METADATA` | Arbitrary passthrough | `metadata[<column name>]` |
| `IGNORE` | Explicitly excluded | Dropped |
| `UNKNOWN` | Heuristics found no confident match | Surfaced in `preview()` for manual `override()` — **never silently dropped or guessed** |

Heuristic priority (first match wins, checked in this order): column
names containing `input`/`prompt`/`question`/`query` → `INPUT`; then
`context`/`passage`/`document`/`chunk`/`retrieved` → `CONTEXT`; then
`answer`/`target`/`expected`/`golden`/`reference`/`output`/`completion`/
`response` (string dtype only) → `GOLDEN`; then `label`/`class`/`category`
→ `LABEL`; then `id`/`index`/`source`/`metadata`/`split`/`timestamp`/`date`
→ `METADATA`; a lone remaining `text`-named column becomes `INPUT` only if
no other column already claimed it. Everything else is `UNKNOWN`.

```python
result = DatasetImporter.load("qa.csv")

problems = result.mapping.validate()
if problems:
    print(result.mapping.preview())  # show the user what needs fixing
    # e.g. "columns with UNKNOWN role need review: notes"

# Force a column's role manually — input_column/target_column kwargs
# apply the override at load time, before you ever see the mapping:
result = DatasetImporter.load(
    "qa.csv", input_column="passage", target_column="category",
)
```

---

## Missing / Empty Cells

A missing cell (`None`, NaN, or a blank string) never becomes the literal
string `"None"`. `EvalSample.input` is non-optional, so a missing `INPUT`
cell becomes `""`; every other role (`GOLDEN`, `references`, `context`)
becomes `None` for that entry.

---

## No Mock Data

If a column's role can't be confidently inferred, it is left `UNKNOWN` and
surfaced in `preview()`/`validate()` — mltk never invents a role or fills
in a plausible-looking value for a column it isn't sure about.

---

## What's Next

Sprint 1 covers loading, normalization, and column auto-mapping. Not yet
built (see [Roadmap](../roadmap.md#smart-dataset-importer-test-suite-mapper)):

- **Sprint 2** — task-type classification (classification / QA-RAG /
  summarization / generation / retrieval) and generation of a matching
  `MltkSuite` + committable pytest file; `mltk import <source>` CLI.
- **Sprint 3** — an MCP tool, golden-set binding with an `LLMJudgeScorer`
  fallback when no exact golden exists, and `DatasetRegistry` integration.

---

## See Also

- [Versioned Eval Datasets](eval-datasets.md) — `EvalDataset`,
  `DatasetRegistry`, `assert_dataset_quality`; the object
  `to_eval_dataset()` produces and where it's stored/validated next
- [Evaluation Pipeline](eval-pipeline.md) — `EvalTask`, solvers, and
  scorers that consume `dataset.samples`
- [LLM-as-Judge](llm-judge.md) — `LLMJudgeScorer` fallback for datasets
  without an exact golden answer
