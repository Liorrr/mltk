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

`compare_runs(base, candidate, layers=..., manifest=...)` provides a two-run
descriptive comparison without an A/A noise estimate. It does not establish
noise-calibrated product parity.

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
`schema_version="1"`, `run_id`, `label`, `refs` and `items`. JSONL starts with the
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
release, including one failed trial hidden by two successful majority votes. `ParityConfig(fail_on_negative_flip=True)` or
`assert_parity(report, fail_on_negative_flip=True)` also blocks measured negative
flips below the detection threshold. Optional seeded item bootstrap and
within-stratum Benjamini–Hochberg results are configured with `bootstrap_seed`
and `bh_q`; repeated trials never count as independent samples.

### Planted controls

Use `make_positive_control(base, layer=layer, n_items=6,
mutation=lambda value: not value, seed=42)` for a Boolean layer. It returns a
`ControlFixture` containing a deep copy, selected item IDs, before/after values
and any explicitly injected failed-trial IDs. The helper does not prove detection.

Pass that fixture to `check_positive_control(base, fixture, layers=layers,
manifest=manifest, a2=repeat)`. This runs identity and planted comparisons through
the ordinary engine and returns a `(layer, stratum) -> bool` evidence map for
`compare_product(..., control=evidence)`. Generate and check controls for each
verdict layer/stratum. Preconditions separately require complete passing evidence.
Use the same baseline, repeat, manifest and configuration
as the release comparison. A missing/failed control prevents passing. Stored
control evidence supplied directly is a caller attestation; it must represent
those checks, not an assumed success.

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
expected item/trial keys and contain no failed or null score records. Integrity
failure returns NOT_RUN. Scored numeric layers use paired, item-weighted TOST
within each stratum; margins are explicit positive native-unit values keyed by
layer name or `(layer, stratum)`. Missing margins or fewer than two pairs yield
NOT_VERIFIED. Equivalence requires the 90% paired t interval strictly inside
the bounds at alpha 0.05; constant differences must also be strictly inside.

Deterministic layers compare every trial as well as the reduced item state.
Expected differences require `item_id`, `stratum`, `layer`, `direction`
(`increase`, `decrease`, `changed`) and `reason`; optionally specify `trial_id`.
Nonnumeric changes require exact `from`/`to` values. Unlisted or wrong-direction
changes fail. A listed change that no longer occurs is stale and fails. A ledger
entry for a scored layer never removes observations from TOST.

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

- `rates.json`: complete report, configuration, provenance, item and rate evidence.
- `items.csv`: item evidence with schema/engine versions and provenance per row.
- `gaps.json`: Unconcluded records for later review of missing knowledge or capability.

Parity artifacts use UTF-8, stable ordering and no generated timestamps. Identical
normalized input/configuration under the same engine and supported Python runtime
produces identical bytes. Outer pytest exports retain normal timestamps and
durations. `assert_parity` records a normal mltk result, raises on failed or
incomplete verification, and embeds a bounded first-ten preview with artifact
paths/hashes rather than the full dataset. The assertion is discoverable through
the CLI's Parity category.

Runtime/tensor adapters and model acquisition are outside this stored-run API.
