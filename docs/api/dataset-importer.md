# Smart Dataset Importer

Point mltk at a dataset — a local CSV/JSON/Parquet file or a HuggingFace
Hub id — and get back a normalized, column-role-mapped result ready to
become a versioned [`EvalDataset`](eval-datasets.md).

**Since:** Unreleased (S97 + S98, sprints 1–2 of a 3-sprint epic — see
[Roadmap](../roadmap.md#smart-dataset-importer-test-suite-mapper))

**Modules:**

- `mltk.importer` — `DatasetImporter`, `ColumnRole`, `ColumnMapping`,
  `ImportResult`, `auto_map_columns`, `TaskType`, `classify_task`,
  `build_suite`, `compute_baseline_thresholds`, `generate_pytest`
- `mltk import <source>` — one-command CLI over the whole pipeline
  (see [The mltk import CLI](#the-mltk-import-cli))

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

# 5. Classify the task type and build a runnable in-memory suite
from mltk.importer import build_suite, classify_task, generate_pytest

task_type = classify_task(result.mapping)          # TaskType.QA_RAG
suite = build_suite(dataset, result.mapping, task_type)
print(suite.run().passed)

# 6. Or emit a committable pytest file
generate_pytest(result, task_type, output_path="test_my_qa_qa_rag.py")
```

Or do all of the above in one command:

```console
$ mltk import qa.csv
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

## Task-Type Classification

`classify_task(mapping, dataset=None)` classifies the imported dataset
into a `TaskType` from the roles present in the mapping — deterministic
and role-presence-based, like the column heuristics. First matching rule
wins:

| Rule | Roles present | `TaskType` |
|---|---|---|
| 1 | `CONTEXT` and `GOLDEN` | `QA_RAG` |
| 2 | `CONTEXT`, no `GOLDEN` | `RETRIEVAL` |
| 3 | `LABEL`, no `GOLDEN` | `CLASSIFICATION` |
| 4 | `GOLDEN` whose column name has a whole-token summary keyword (`summary`/`summaries`/`highlights`/`tldr`/`abstract`) | `SUMMARIZATION` |
| 5 | any other `GOLDEN` | `GENERATION` |
| 6 | none of the above | `GENERATION` |

Keyword matching reuses the importer's token semantics: `summary_text`
matches (`summary` is a whole token), `summarylike` does not. The
`dataset` argument is accepted but currently unused — it is reserved
for future data-peek heuristics.

The classifier drives which assertions the suite generator and pytest
emitter bind:

| Task type | Bound assertions |
|---|---|
| `QA_RAG` | `assert_faithfulness`, `assert_answer_relevancy`, `assert_context_relevancy` |
| `RETRIEVAL` | `assert_context_relevancy` |
| `CLASSIFICATION` | `assert_metric` (accuracy) |
| `SUMMARIZATION` | `assert_summary_coverage`, `assert_summary_compression`, `assert_summary_faithfulness` |
| `GENERATION` | `assert_output_format` |
| every task type | `assert_schema`, `assert_no_nulls`, `assert_dataset_quality` (data sanity) |

---

## Two-Tier Test Semantics

At import time there is no model yet, so both the in-memory suite and
the emitted pytest file follow the same honest two-tier rule:

- **Tier 1 — dataset sanity, runs immediately.** Schema, null checks on
  the `INPUT`/`GOLDEN` columns, and `assert_dataset_quality` with
  *baseline thresholds* computed from the imported data (see below).
- **Tier 2 — model quality, ready but inert.** The task-type assertions
  above are generated fully wired, but they need model predictions —
  they stay skipped (pytest) or omitted (in-memory suite) until you
  provide a predictor or judge. mltk never fabricates predictions or
  lowers thresholds to force a green run.

### Baseline thresholds

`compute_baseline_thresholds(dataset)` snapshots quality gates the
current dataset already satisfies, bucketed to readable 0.05 increments
(never overfit exact floats):

- `min_samples` — the current sample count
- `min_target_coverage` — actual coverage, floored to a 0.05 multiple
- `max_duplicate_rate` — `0.01` when there are no duplicate inputs,
  else the actual rate ceiled to a 0.05 multiple
- `min_categories` — the distinct category count, or omitted when the
  dataset has no categories

Treat these as a starting point to tighten over time — they gate
*regressions* (the dataset shrinking, losing coverage, gaining
duplicates), not absolute quality.

---

## Building a Runnable Suite

`build_suite(eval_dataset, mapping, task_type, *, judge_fn=None)`
returns an [`MltkSuite`](suite-api.md) named
`import:<dataset-name>`:

- Always adds Tier 1 `assert_dataset_quality` with the baseline
  thresholds above.
- With a `judge_fn` (`(text_a, text_b) -> float` in `[0, 1]`), adds
  judge-scored **dataset-side** checks for `QA_RAG` (is the golden
  answer faithful to the context? relevant to the question? is the
  context relevant?) and `RETRIEVAL` (context relevancy only). Samples
  missing a target or context are skipped for the checks that need
  them — never filled in.
- Without a `judge_fn`, the suite is Tier 1 only; the model-bound
  assertions live in the emitted pytest file instead.

```python
suite = build_suite(dataset, result.mapping, task_type,
                    judge_fn=my_llm_judge)
report = suite.run()          # never raises; failures are results
print(f"{report.passed_count}/{report.total} passed")
```

---

## Emitting a Committable Pytest File

`generate_pytest(import_result, task_type, *, dataset_name=None,
output_path=None)` returns (and optionally writes) a self-contained
pytest scaffold:

- **Fixtures** — `import_result` (reloads the source through
  `DatasetImporter`), `df` (pandas view for schema/null checks),
  `dataset` (the `EvalDataset`), and `predict_fn`.
- **`class TestDataSanity`** (Tier 1, runs now) — `test_schema` against
  a dtype snapshot taken at emit time, `test_no_nulls` on the
  `INPUT`/`GOLDEN` columns (omitted with an explanatory comment if the
  imported data already has missing values there), and
  `test_dataset_quality` with explicit baseline-threshold kwargs, each
  marked `# baseline from import — tighten as needed`.
- **`class TestModelQuality`** (Tier 2, skipped) — the task-type
  assertions from the table above, iterating your dataset's samples
  through `predict_fn`. Assertions that need information an import
  cannot provide (protected attributes for `assert_no_bias`, a JSON
  schema for `assert_json_schema`) are emitted as commented-out lines
  with a `requires ...` note instead of tests that could never run.
- The file is a scaffold — **edit it freely**. Generation is
  deterministic: the same input produces byte-identical output (no
  timestamps), and every emitted file is `ast.parse`-validated.

### Wiring your model

The `predict_fn` fixture un-skips all Tier-2 tests at once, two ways:

```console
$ MLTK_PREDICT_FN=myproject.model:predict pytest test_my_qa_qa_rag.py
```

or edit the fixture to return any `prompt -> prediction` callable.
Until then, Tier-2 tests report as *skipped* — a freshly emitted file
runs green out of the box:

```console
$ pytest test_tiny_qa_rag.py -q
...sss                                                    [100%]
3 passed, 3 skipped
```

---

## The mltk import CLI

`mltk import <source>` runs the whole pipeline — load, preview the
mapping, classify, build the suite, and (by default) write the pytest
scaffold to `./test_<stem>_<task_type>.py`:

```console
$ mltk import qa.csv
column | role | sample
question | input | What is the capital of France?
answer | golden | Paris
passage | context | France is a country in Western Europe. ...
category | label | geography
id | metadata | 1
task type: qa_rag
suite: import:qa (1 registered assertions)
pytest file written: test_qa_qa_rag.py
Tier-2 tests are skipped until predict_fn is wired. Set
MLTK_PREDICT_FN=module:callable or edit the fixture.
```

| Option | Effect |
|---|---|
| `--split TEXT` | Dataset split for HuggingFace sources (default `train`) |
| `--input-column TEXT` | Force-map a column as the eval input (exclusive override) |
| `--target-column TEXT` | Force-map a column as the eval target (exclusive override) |
| `--name TEXT` | Dataset name (default: sanitized source stem) |
| `--output PATH` | Where to write the pytest file (default `./test_<stem>_<task_type>.py`) |
| `--force` | Overwrite an existing output file. Without it, an existing file is never touched — it is a scaffold you may have edited |
| `--no-emit` | Preview + classify + build the suite without writing a file |

Mapping problems are printed as warnings; if no column maps to `INPUT`
(so no `EvalDataset` can be built), the command exits non-zero instead
of guessing. The command works without the `mltk[importer]` extra
installed only for `--help`; actual imports need
`pip install mltk[importer]` for HuggingFace sources.

---

## What's Next

S97 covered loading, normalization, and column auto-mapping; S98 added
task-type classification, suite generation, the pytest emitter, and the
`mltk import` CLI. Not yet built (see
[Roadmap](../roadmap.md#smart-dataset-importer-test-suite-mapper)):

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
