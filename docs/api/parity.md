# Stored-run parity

`mltk.parity` compares recorded results offline. It never invokes a model or a
judge. Record stable item IDs, repeated trial IDs, layer values and provenance
before comparing runs. Use `EvalTask.score()` upstream when scoring stored model
answers.

## Product parity

Product parity uses two baseline runs (A1, A2) to measure ordinary variability,
then compares a candidate (B) on stable items. Public product comparisons require
three distinct run IDs; supply independently acquired baseline repeats. Distinct
IDs prevent accidental reuse but do not certify independent acquisition. Every required layer and stratum
must have sufficient coverage, a quiet baseline, passing preconditions and
successful planted-change control evidence. A detected behavioral change fails
parity, including improvements. This gate is about behavioral preservation.

```python
from mltk.parity import (
    LayerSpec, Manifest, assert_parity, compare_product, load_run, write_report,
)

manifest = Manifest(items={"sample-001": "en", "sample-002": "he"})
layers = [LayerSpec("correct", oracle=lambda value: value, oracle_id="correct-v1")]
a1 = load_run("baseline-1.json", manifest=manifest, layers=layers)
a2 = load_run("baseline-2.json", manifest=manifest, layers=layers)
b = load_run("candidate.json", manifest=manifest, layers=layers)
report = compare_product(a1, a2, b, layers=layers, manifest=manifest)
# Without control evidence this is NOT_VERIFIED, never a passing release gate.
artifacts = write_report(report, "parity-results")
assert_parity(report, artifacts=artifacts)
```

Use a sufficiently large, representative manifest for useful sensitivity. With
20 stable items and zero observed baseline changes, the Wilson 95% upper noise
bound is approximately 0.161 and four changes are needed to exceed that bound.
A tiny sample cannot demonstrate useful detection sensitivity.

`compare_runs(base, candidate, layers=..., manifest=..., config=ParityConfig())` provides a two-run
descriptive comparison without an A/A noise estimate. It does not establish
noise-calibrated product parity. Its intervals use and record `config.confidence`
(default 0.95). Empty or wholly ineligible input is NOT_VERIFIED with reason
`NO_COMPARABLE_ITEMS`, rather than a failed identity comparison.

### Data and reduction

`Run(run_id, label, items, refs={})` contains
`Item(item_id, stratum, trials, source_id=None, metadata={})` records.
`Trial(trial_id, status="ok", values={}, error=None)` contains values keyed by
layer name. Failed trials require an explicit error reason. The independent
`Manifest` declares all expected items, their strata, and trial IDs (default
`("1", "2", "3")`). Optional `eligible_ids` freezes per-layer item eligibility.

`LayerSpec` supports `bool`, `class`, `id_set` and finite `number` values.
Classes require a declared `vocabulary`. Boolean and numeric values are
distinct. ID sets are unordered. JSON encodes them as
`{"type":"id_set","values":["a","b"]}`. Null and absent values are unscorable.
Layer roles are `verdict`, `precondition` and `diagnostic`. Diagnostics do not
gate release. An optional pure oracle returns `True`, `False` or `None`; provide
a versioned `oracle_id` with it. No callable is imported from an input file.

The default three-trial reducer requires at least two valid values and a strict
majority among valid values. Two failures with fewer than two valid values yield
FAILED. Ties are INDETERMINATE; insufficient evidence is NOT_SCORABLE. Agreement
uses scheduled trials as its denominator. For other repeat counts, set
`min_valid` and `failed_threshold` explicitly. Continuous scores use
`kind="number", reducer="mean"` and average valid trials per item.

`load_run` accepts a Run, a dictionary, or a UTF-8 JSON/JSONL path. JSON uses
`schema_id="mltk.parity.run"`, `schema_version="1.0"`, `run_id`, `label`, `refs`
and `items`. JSONL starts with the
same run fields except `items`, plus `expected_item_ids`, `expected_count` and
`expected_trial_ids`; subsequent lines contain item objects. Duplicate JSON
keys/IDs, unknown schemas, invalid values and manifest inconsistencies fail
validation. An external manifest preserves missing expected records as gaps.
Without one, an incomplete JSONL population is rejected.

### Results and policy

Product status is PASS, FAIL or NOT_VERIFIED. Item records explain unchanged
states, correctness flips, changed states without an oracle, source changes,
execution failures and Unconcluded evidence. A known changed evidence source is
excluded from A/B statistics. Unknown provenance is compared and flagged.
`source_id` identifies evidence, not a model build; put build identifiers in refs.

Observed baseline noise and unresolved cases are reported separately. Both enter
the conservative noise bound, without counting an item twice. Every stratum has
its own denominators, exclusions, confidence intervals, noise floor, minimum
detectable changes, oracle flip rates and exact McNemar result. McNemar tests
marginal correctness change; a large p-value does not establish equivalence.

Candidate execution failure on a previously oracle-passing item always blocks
release, including one failed trial hidden by two successful majority votes.
`ParityConfig(fail_on_negative_flip=True)` also blocks measured negative
flips below the detection threshold. Set that policy when comparing runs;
`assert_parity` uses the engine's final status. Its deprecated
`fail_on_negative_flip` argument, if supplied, must equal the recorded config.
Optional seeded item bootstrap and
within-stratum Benjamini–Hochberg results are configured with `bootstrap_seed`
and `bh_q`; repeated trials never count as independent samples.

Coverage is checked on both sides of each required layer/stratum. For `E`
eligible items and `e=config.established_fraction` (default 0.8), at least
`ceil(e * E)` items must establish baseline evidence and at least that many
must be comparable on the candidate side. The candidate denominator `D` counts
comparable reduced states, including layers without an oracle; it is distinct
from the oracle-pair denominator. Candidate exclusions stay excluded and counted.
Insufficient candidate coverage gives `DETECTION_NOT_DEMONSTRATED` with reason
`CANDIDATE_NOT_ESTABLISHED` and prevents PASS. Coverage and quietness thresholds
use exact rational arithmetic so decimal boundaries do not move by one item.
Independent observed failures can still make the overall status FAIL.

A layer/stratum with zero eligible items is `NOT_APPLICABLE` and is omitted from
required gates, including preconditions. This differs from an eligible cell
whose observations are all excluded: that cell still blocks verification.
A report without any required eligible data remains NOT_VERIFIED.

### Planted controls

Use `make_positive_control(base, a2=repeat, manifest=manifest, layer=layer, n_items=6,
mutation=lambda value: not value, seed=42)` for a Boolean layer. It returns a
`ControlFixture` containing a deep copy, selected item IDs, before/after values
and any explicitly injected failed-trial IDs. Supply a `LayerSpec`, not a layer
name: planting requires the reducer and eligibility rules. Only A/A-stable items
can be selected. The helper does not prove detection.

Pass that fixture to `check_positive_control(base, fixture, layers=layers,
manifest=manifest, a2=repeat)`. This runs identity and planted comparisons through
the ordinary engine and returns a `(layer, stratum) -> evidence record` map for
`compare_product(..., control=evidence)`. Generate and check controls for each
verdict layer/stratum. Preconditions separately require complete passing evidence.
The real A2 is required by both helpers; there is no zero-noise fallback to A1.
Ordinary `AA_*` exclusions do not invalidate a detected control. Evidence records
contain `passed`, `a2_run_id`, `seed`, `n_items`, and `changed_item_ids`, and are
retained in the report's rates. Product comparison rejects evidence with an A2
run ID different from the release repeat, and rejects unbound Boolean evidence.
Use the same baseline, repeat, manifest and configuration as the release
comparison. A missing/failed control prevents passing. Stored control evidence
is still a caller attestation; it must represent those checks, not an assumed
success. Run IDs alone do not authenticate recordings or configuration.

## Harness parity

```python
from mltk.parity import compare_harness

report = compare_harness(
    old_scores, new_scores,
    recordings=recording_manifest,
    layers=score_layers,
    expected_diff=[],
    margins={"quality": 0.02},
)
```

Precondition layers additionally require every scheduled trial to pass their
oracle in both runs. A false verdict fails; an unavailable oracle is not verified.
Equality alone cannot satisfy a precondition.

Both runs must reference the manifest's `recording_hash`, contain exactly the
expected item/trial keys and contain no failed trial records. Scores must be
present and non-null for each eligible layer; an ineligible layer needs no score
on that item. Record/trial/hash integrity remains global. Integrity
failure returns NOT_RUN. Scored numeric layers use paired, item-weighted TOST
within each stratum; margins are explicit positive native-unit values keyed by
layer name or `(layer, stratum)`. Missing margins or fewer than two pairs yield
NOT_VERIFIED. Equivalence requires the 90% paired t interval strictly inside
the bounds at alpha 0.05; constant differences must also be strictly inside.
Input/integrity `ValueError` and `TypeError` failures become NOT_RUN; unexpected
programming exceptions propagate so a bug cannot be reported as missing evidence.

Deterministic layers compare every trial as well as the reduced item state.
Expected differences require `item_id`, `stratum`, `layer`, `direction`
(`increase`, `decrease`, `changed`) and `reason`; optionally specify `trial_id`.
Nonnumeric changes require exact `from`/`to` values. Unlisted or wrong-direction
changes fail. A listed change that no longer occurs is stale and fails. A ledger
entry for a scored layer never removes observations from TOST. Explicit ledger
entries are audited even for diagnostic layers; stale or mismatched entries
still fail their cell and the report.

### Numerical limits

The stdlib exact McNemar implementation supports at most 1,000,000 discordant
pairs (`b+c`); larger totals raise a validation error rather than risk an
inaccurate significance result. Interior Clopper–Pearson intervals have the same
count limit, while all-success/all-failure endpoints use stable analytic formulas.
Wilson, TOST, BH and bootstrap do not inherit this count limit. An overflowing
candidate-minus-baseline score difference leaves an optional bootstrap result
NOT_VERIFIED with an explicit reason; it does not erase the ordinary parity gate.

## Artifacts and assertions

`write_report` returns `ArtifactPaths` with paths and SHA-256 hashes for:

- `rates.json`: report envelope (schema, versions, mode, status, refs and config)
  and rates, without item rows.
- `items.jsonl`: canonical item evidence; a typed header carries the envelope,
  followed by typed item rows.
- `items.csv`: a tabular view of those rows, without repeating the envelope's
  refs or versions on every row. `ArtifactPaths.items` names this CSV;
  `ArtifactPaths.items_jsonl` names the canonical row file.
- `gaps.json`: Unconcluded records for later review of missing knowledge or capability.

Reports use `schema_id="mltk.parity.report"` and `schema_version="1.0"`.
Run and report schemas have separate MAJOR.MINOR versions. Reports record
`mlspec_version` alongside `engine_version`; an unavailable installed package
version is explicitly `"unknown"`. Item rows sort by layer order, layer name,
stratum and item ID. Rates sort by layer order and stratum, with a deterministic
layer-name tie breaker. Value changes do not reorder item identities.

Parity artifacts use UTF-8, stable ordering and no generated timestamps. Identical
normalized input/configuration under the same engine and supported Python runtime
produces identical bytes. Outer pytest exports retain normal timestamps and
durations. `assert_parity` records a normal mltk result, raises on failed or
incomplete verification, and uses the engine-built `report.message`. That message
identifies failing `layer@stratum` checks, verdicts, reasons and relevant coverage,
change/noise/MDC or harness-ledger counts, followed by up to ten offending item IDs,
artifact references and a corrective hint. Before writing artifacts the message
explicitly says they have not been written.

Result details put failing checks first, then use declared layer order and
stratum. The first ten compact rates contain `layer`, `stratum`, `verdict`,
`reason`, `changed_rate`, `floor`, `mdc`, `b`, `c` and `n_compared`; a `failing`
list retains failing cell names. Offenders retain item ID, stratum, layer and
classification. Artifact paths are relative to the output directory, with
SHA-256 hashes for all four files. The assertion is discoverable through
the CLI's Parity category.

Runtime/tensor adapters and model acquisition are outside this stored-run API.
