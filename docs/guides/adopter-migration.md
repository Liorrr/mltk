# Adopter migration and usage

This guide describes the implemented evaluation and assertion changes relative
to the v0.16.0 baseline. Review judge response formats, stored-output identities,
empty refusal datasets, and consumers of semantic-equivalence results when
adopting these changes.

## Numeric judge parsing

Numeric judge text is parsed strictly by default. Ask the judge for a standalone
number, a JSON object with a numeric `score`, or one explicit `score: N` or `N/M`
expression. Prose containing an incidental number, such as `Found 3 issues`, no
longer supplies a score.
Sentence-ending punctuation such as `Score: 4.5.` and a balanced Markdown bold
label such as `**Score:** 4.5` are accepted. Run-on numbers such as `4.5.6`,
competing score expressions, and mismatched fraction scales still error.

The shared parser removes balanced `<think>...</think>` blocks before parsing.
Use `reasoning_delimiters=(("<analysis>", "</analysis>"),)` to remove additional
reasoning blocks. Unbalanced or mismatched reasoning tags produce no score.
JSON may appear inside Markdown fences or after prose. The last score-bearing
JSON object is authoritative: an invalid final score does not fall back to an
earlier valid one. Booleans, nonfinite scores, malformed score JSON, and multiple
explicit score expressions are rejected. Malformed score JSON before a later
valid object is also rejected.

### Match the configured scale

`mltk.core.judge_parse.parse_judge_score()` returns raw units, not a normalized
fraction. For example, `4/5` returns `4.0`. With `expected_max_score=5`, `4/10`
returns `None`; it is not rescaled. A fraction denominator must be positive and
finite. Without `expected_max_score`, the shared parser does not require a
particular denominator.

The public adapters supply the expected scale:

| API | Expected fraction denominator | Score treatment |
| --- | --- | --- |
| `assert_llm_judge_score` in `mltk.domains.llm.judge` | `max_score` (default `5.0`) | Raw scores are averaged and compared with `min_score`. |
| `LLMJudgeScorer` in `mltk.eval` | `max_score` (default `5.0`) | Divides the raw score by `max_score` and clamps to `[0, 1]`. |
| `assert_with_judge` in `mltk.domains.llm.judge_defaults` | `1.0` | Uses the score on its configured 0–1 scale. |
| `assert_prompt_faithfulness`, `assert_image_coherence` in `mltk.domains.multimodal.alignment`; `assert_image_helpfulness`, `assert_vqa_accuracy` in `mltk.domains.multimodal.vlm` | `1.0` | Numeric judge branches expect the 0–1 scale. |

The following parser checks run locally without calling a model:

```python
from mltk.core.judge_parse import parse_judge_score

assert parse_judge_score('<think>3 issues</think>{"score": 4}') == 4.0
assert parse_judge_score("4/5", expected_max_score=5) == 4.0
assert parse_judge_score("4/10", expected_max_score=5) is None
assert parse_judge_score("Found 3 issues") is None
assert parse_judge_score(
    '<analysis>3 issues</analysis>score: 4',
    reasoning_delimiters=(("<analysis>", "</analysis>"),),
) == 4.0
```

### Compatibility parsing and error policy

The public numeric assertions listed above and `LLMJudgeScorer` accept
`legacy_parse=True`. The shared parser spells this option `legacy=True`.
Compatibility parsing first tries a whole-response JSON score and then the
first unsigned number in the text. It ignores reasoning delimiters and fraction
scale mismatches, and retains permissive nonfinite conversion. Prefer explicit
score responses when updating judge prompts; compatibility parsing can select
a number from reasoning instead of the intended grade.

For example, this executable check demonstrates the scale-check opt-out:

```python
from mltk.core.judge_parse import parse_judge_score

assert parse_judge_score("4/10", expected_max_score=5, legacy=True) == 4.0
```

Numeric callback values retain their sign and value rather than passing through
the legacy text-number search. `LLMJudgeScorer` keeps its numeric normalization
path and rejects nonfinite scores even with `legacy_parse=True`.

Parsing compatibility and error handling are independent. The error-policy
keyword on assertions is **`on_error`**, not `error_policy`:

The assertion default changes aggregation relative to v0.16.0: an errored judge
reply no longer contributes a zero to `assert_llm_judge_score`'s mean. For
example, scores `4` and an error now average `4`, rather than `2`. All errors
still fail by default. Use `on_error="zero"` to restore zero filling and the
corresponding threshold result. The policy names already existed on evaluation
tasks; they are now also exposed on the judge assertions.

| `on_error` | Treatment |
| --- | --- |
| `"fail_if_all"` (default) | Exclude errored items from aggregation; fail when all attempted items errored. Valid scores must still meet the threshold. |
| `"strict"` | Fail if any item errored, even when the remaining scores meet the threshold. |
| `"zero"` | Include errored items as zero scores and apply the threshold. This explicitly restores zero filling. |

A parse failure, including a fraction scale mismatch, counts as an error.
For a single-item numeric assertion, the default and strict policies fail even
at a zero threshold. Critical assertion failures raise `MltkAssertionError`.
In `assert_llm_judge_score`, default/strict error entries have `score=None` in
`details["per_item_scores"]`; inspect `errored_count` and `n_valid` as well as
`avg_score`. With `on_error="zero"`, those entries carry `0.0` and remain marked
errored.

`LLMJudgeScorer` records a failure as `Score(value=0.0, errored=True)` with error
metadata. Set `on_error` on `EvalTask.to_test_result(...)`, not on the scorer.
`EvalTask.score(...)` excludes errored scores from its aggregate metrics and
reports error counts separately. Pairwise, yes/no, OCR, and lexical fallback
contracts are unchanged by numeric parsing.

## Score stored outputs by sample identity

`EvalSample` adds optional `sample_id: str | None = None` after `metadata`, so
existing positional construction such as `EvalSample("question", "answer", {})`
still works. Live `task.run(model_fn)` does not require sample IDs.

To score stored answers, assign a stable ID to every dataset sample and call
`task.score(outputs)` with a mapping from those IDs to output strings:

- Every sample ID must be a unique, nonblank string. Missing IDs, duplicate IDs,
  and output keys absent from the dataset raise `ValueError` before scoring.
  A non-mapping `outputs` argument raises `TypeError`.
- IDs are matched exactly; do not rely on automatic trimming, numeric coercion,
  row positions, or prompt text to establish identity.
- A missing output or a non-string output value records an errored score for
  each scorer on that sample. It does not drop the sample from accounting.
  An empty string is a valid output and is scored normally.
- Results follow dataset order, regardless of mapping insertion order.

This complete example uses a small illustrative dataset and stored answers. It
executes real exact-match scoring without a model call:

```python
from mltk.eval import EvalSample, EvalTask, ExactMatchScorer, GenerateSolver

samples = [
    EvalSample("What is 2 + 2?", "4", sample_id="addition-1"),
    EvalSample("What is 3 + 3?", "6", sample_id="addition-2"),
]
outputs = {"addition-2": "6", "addition-1": "4"}
task = EvalTask(
    name="stored-addition",
    solver=GenerateSolver(),
    scorers=ExactMatchScorer(),
    dataset=samples,
)

result = task.score(outputs)
assert result.metrics["ExactMatchScorer/accuracy"] == 1.0
assert [state.sample.sample_id for state in result.samples] == [
    "addition-1", "addition-2",
]
assert result.errors["ExactMatchScorer"]["errored"] == 0

gate = task.to_test_result(
    outputs=outputs,
    min_accuracy=1.0,
    on_error="strict",
    min_valid_fraction=1.0,
)
assert gate.passed
```

The task constructor still requires a nonempty solver configuration, scorer
configuration, and dataset. Stored scoring bypasses all solvers and generation;
scorers receive the sample and stored output without synthesized messages or
solver metadata. Scorers themselves still execute: using `LLMJudgeScorer` can
therefore call your judge. The example uses `ExactMatchScorer` to remain local.

`to_test_result` requires exactly one of `model_fn` or `outputs`; supplying both
or neither raises `ValueError`. It scores the supplied outputs again, then
applies the accuracy and error gates. All scorers must meet `min_accuracy`
(default `0.8`). `min_valid_fraction`, when supplied, must be a real number in
`(0, 1]` and applies per scorer under every error policy, including `"zero"`.
Boolean values are rejected. Gate failures raise `MltkAssertionError`.

Persist `sample_id` as a top-level field in JSON or as a CSV column. By default,
`EvalDataset.from_csv`, `EvalDataset.from_json`, and `load_dataset` extract it
into `EvalSample.sample_id` rather than `metadata`. **This changes metadata and
fingerprints for files that already had a `sample_id` column/key.** Pass the
keyword-only `id_column=None` to preserve the previous metadata and input/target
fingerprint, or `id_column="record_id"` to select another identity field.

Selected identities must be strings or null. Non-string JSON IDs raise a
`ValueError` identifying the record and field; they are not coerced to strings.
Missing, null and empty IDs become `None`. Stored scoring still requires unique,
nonblank IDs. With `id_column=None`, arbitrary values remain ordinary metadata.

Dataset serialization and registry round trips preserve explicit IDs. ID-bearing
fingerprints include the IDs, so changing IDs changes the fingerprint. ID-less
scalar datasets retain their existing serialized sample shape and fingerprint.

For offline comparison of recorded runs, see [stored-run parity](../api/parity.md).

## Empty refusal datasets

`assert_refusal_consistency` in `mltk.domains.llm.safety` now defaults to
`on_empty="fail"`. Choose a policy deliberately when an empty prompt list is
possible:

| `on_empty` | Empty-input result |
| --- | --- |
| `"fail"` (default) | Critical failure raising `MltkAssertionError`; no measured refusal rate. |
| `"pass"` | Legacy pass with `refusal_rate=1.0`, `n_total=0`, and `n_refused=0`. The rate is a compatibility value, not measured safety evidence. |
| `"skip"` | Passing informational result with `details["skipped"]=True` and a reason; no measured refusal rate. This does not itself raise a pytest skip. |

Invalid policies fail before any model calls, including with nonempty input.
Nonempty refusal scoring is unchanged. This example verifies that an empty skip
does not call the model:

```python
from mltk.domains.llm.safety import assert_refusal_consistency

def unexpected_model_call(prompt: str) -> str:
    raise AssertionError("An empty dataset must not call the model")

result = assert_refusal_consistency(
    unexpected_model_call, [], on_empty="skip",
)
assert result.passed
assert result.details["skipped"] is True
assert "refusal_rate" not in result.details
```

## Distinguish unchecked contradiction from a negative result

`assert_semantic_equivalence` in `mltk.domains.llm.behavioral.semantic` now sets
`details["contradiction"]` to `None` (JSON `null`) for `method="token_f1"` and
`method="embedding"`, on both passing and failing results. Neither method checks
contradiction; their scores and threshold decisions are unchanged.

Only `method="nli"` produces a checked boolean: `False` means no contradiction
was detected, while `True` means contradiction was detected. Use `is None` and
`is False` rather than a truthiness check that treats both as the same result.
There is no compatibility option to restore unchecked `False` values. Select
NLI when contradiction evidence is required.

```python
from mltk.domains.llm.behavioral.semantic import assert_semantic_equivalence

result = assert_semantic_equivalence(
    "The door is open.", "The door is open.", method="token_f1",
)
contradiction = result.details["contradiction"]
assert contradiction is None  # Not checked by token overlap.
assert (contradiction is False) is False  # Not a checked negative.
```
