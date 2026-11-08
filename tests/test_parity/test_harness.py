"""Synthetic stored-score fixtures for harness integrity and change accounting."""

from __future__ import annotations

import copy
import json
from dataclasses import asdict

import pytest

from mltk.parity.harness import compare_harness
from mltk.parity.models import Item, LayerSpec, Manifest, Run, Trial


def synthetic_runs(*, kind="bool", reducer="majority", values=None, strata=("en", "en")):
    """Deliberately synthetic recordings, never adopter measurements."""
    values = values if values is not None else [True, True, True]
    layer = LayerSpec("result", kind=kind, reducer=reducer,
                      vocabulary=("a", "b", "c") if kind == "class" else ())
    manifest = Manifest({f"synthetic-{i}": s for i, s in enumerate(strata)},
                        recording_hash="synthetic-recording-v1")
    items = [Item(item_id, stratum, [Trial(str(j + 1), values={"result": value})
                                   for j, value in enumerate(values)])
             for item_id, stratum in manifest.items.items()]
    base = Run("synthetic-base", "synthetic base", items,
               refs={"recording_hash": manifest.recording_hash, "build": "synthetic-build-a"})
    candidate = copy.deepcopy(base)
    candidate.run_id = "synthetic-candidate"
    candidate.refs["build"] = "synthetic-build-b"
    return base, candidate, manifest, [layer]


def compare(fixture, **kwargs):
    base, candidate, manifest, layers = fixture
    return compare_harness(base, candidate, recordings=manifest, layers=layers, **kwargs)


def entry(**kwargs):
    return {"item_id": "synthetic-0", "stratum": "en", "layer": "result",
            "direction": "changed", "reason": "Synthetic policy transition",
            "from": True, "to": False, **kwargs}


def change(candidate, values, index=0):
    for trial, value in zip(candidate.items[index].trials, values, strict=True):
        trial.values["result"] = value


def test_t6_identity_and_listed_unlisted_change():
    fixture = synthetic_runs()
    assert compare(fixture).status == "PASS"
    change(fixture[1], [False] * 3)
    assert compare(fixture).status == "FAIL"
    report = compare(fixture, expected_diff=[entry()])
    assert report.status == "PASS"
    audit = report.config["expected_diff_audit"][0]
    assert audit["status"] == "matched"
    assert audit["covers"] == ["item", "1", "2", "3"]
    assert report.refs["base"]["refs"]["build"] == "synthetic-build-a"
    assert report.refs["candidate"]["refs"]["build"] == "synthetic-build-b"


def test_t6_unparsed_item_is_not_run():
    fixture = synthetic_runs()
    fixture[1].items.pop()
    assert compare(fixture).status == "NOT_RUN"


@pytest.mark.parametrize("mutation", [
    "hash_missing", "hash_wrong", "manifest_hash_missing", "manifest_missing",
    "duplicate_item", "duplicate_trial", "missing_trial", "extra_trial", "extra_item",
    "stratum", "schema", "null", "missing_layer", "failed", "error", "empty_error",
    "wrong_type", "nonfinite", "unknown_layer", "invalid_reducer", "invalid_threshold",
])
def test_t8_integrity_fails_closed(mutation):
    fixture = synthetic_runs()
    base, candidate, manifest, layers = fixture
    trial = candidate.items[0].trials[0]
    if mutation == "hash_missing":
        candidate.refs.pop("recording_hash")
    elif mutation == "hash_wrong":
        candidate.refs["recording_hash"] = "different"
    elif mutation == "manifest_hash_missing":
        manifest.recording_hash = None
    elif mutation == "manifest_missing":
        fixture = base, candidate, None, layers
    elif mutation == "duplicate_item":
        candidate.items.append(copy.deepcopy(candidate.items[0]))
    elif mutation == "duplicate_trial":
        candidate.items[0].trials.append(copy.deepcopy(trial))
    elif mutation == "missing_trial":
        candidate.items[0].trials.pop()
    elif mutation == "extra_trial":
        candidate.items[0].trials.append(Trial("extra", values={"result": True}))
    elif mutation == "extra_item":
        extra = copy.deepcopy(candidate.items[0])
        extra.item_id = "unknown"
        candidate.items.append(extra)
    elif mutation == "stratum":
        candidate.items[0].stratum = "he"
    elif mutation == "schema":
        candidate.schema_version = "unsupported"
    elif mutation == "null":
        trial.values["result"] = None
    elif mutation == "missing_layer":
        trial.values.clear()
    elif mutation == "failed":
        trial.status, trial.error = "failed", "Synthetic scoring failure"
    elif mutation == "error":
        trial.error = "Synthetic parsing error despite ok status"
    elif mutation == "empty_error":
        trial.error = ""
    elif mutation == "wrong_type":
        trial.values["result"] = 1
    elif mutation == "nonfinite":
        trial.values["result"] = float("nan")
    elif mutation == "unknown_layer":
        trial.values["unknown"] = True
    elif mutation == "invalid_reducer":
        layers[0].reducer = "mystery"
    elif mutation == "invalid_threshold":
        layers[0].min_valid = 4
    report = compare(fixture)
    assert report.status == "NOT_RUN"
    assert report.reasons


def test_majority_cannot_hide_trial_change():
    fixture = synthetic_runs()
    change(fixture[1], [True, True, False])
    assert compare(fixture).status == "FAIL"
    assert compare(fixture, expected_diff=[entry()]).status == "FAIL"
    report = compare(fixture, expected_diff=[entry(trial_id="3")])
    assert report.status == "PASS"
    assert report.items[0]["base"]["value"] is True
    assert report.items[0]["candidate"]["value"] is True


def test_item_entry_does_not_cover_opposite_trial_transition():
    fixture = synthetic_runs(values=[True, True, False])
    change(fixture[1], [False, False, True])
    report = compare(fixture, expected_diff=[entry()])
    assert report.status == "FAIL"
    assert report.items[0]["trial_changes"][-1]["expected"] is False
    assert compare(fixture, expected_diff=[entry(), entry(trial_id="3", **{
        "from": False, "to": True,
    })]).status == "PASS"


def test_trial_entries_can_explain_reduced_change_without_item_entry():
    fixture = synthetic_runs()
    change(fixture[1], [False] * 3)
    assert compare(fixture, expected_diff=[entry(trial_id=t) for t in ("1", "2", "3")]).passed


def test_numeric_item_entry_does_not_cover_different_same_direction_transition():
    fixture = synthetic_runs(kind="number", values=[1, 1, 9])
    change(fixture[1], [2, 2, 10])
    numeric = {k: v for k, v in entry(direction="increase").items() if k not in {"from", "to"}}
    assert compare(fixture, expected_diff=[numeric]).status == "FAIL"
    assert compare(fixture, expected_diff=[numeric, {**numeric, "trial_id": "3"}]).passed


@pytest.mark.parametrize("amendment", [
    {"reason": ""}, {"reason": "   "}, {"item_id": "unknown"}, {"layer": "unknown"},
    {"stratum": "he"}, {"trial_id": "unknown"}, {"trial_id": None},
    {"direction": "unknown"}, {"from": 1}, {"extra": True},
])
def test_invalid_ledger_is_not_run(amendment):
    assert compare(synthetic_runs(), expected_diff=[entry(**amendment)]).status == "NOT_RUN"


@pytest.mark.parametrize("missing", ["from", "to", "reason", "direction"])
def test_missing_ledger_fields(missing):
    ledger_entry = entry()
    ledger_entry.pop(missing)
    assert compare(synthetic_runs(), expected_diff=[ledger_entry]).status == "NOT_RUN"


@pytest.mark.parametrize("conflicting", [False, True])
def test_duplicate_ledger_is_not_run(conflicting):
    first = entry()
    second = entry(direction="decrease") if conflicting else entry()
    assert compare(synthetic_runs(), expected_diff=[first, second]).status == "NOT_RUN"


def test_stale_ledger_fails():
    report = compare(synthetic_runs(), expected_diff=[entry()])
    assert report.status == "FAIL"
    assert report.config["expected_diff_audit"][0]["status"] == "stale"


@pytest.mark.parametrize("direction", ["increase", "decrease"])
def test_wrong_numeric_direction_fails(direction):
    fixture = synthetic_runs(kind="number", values=[2, 2, 2])
    change(fixture[1], [1 if direction == "increase" else 3] * 3)
    expected = {k: v for k, v in entry(direction=direction).items() if k not in {"from", "to"}}
    report = compare(fixture, expected_diff=[expected])
    assert report.status == "FAIL"
    assert report.config["expected_diff_audit"][0]["status"] == "mismatch"


def test_wrong_nonnumeric_transition_fails():
    fixture = synthetic_runs(kind="class", values=["a"] * 3)
    change(fixture[1], ["b"] * 3)
    assert compare(fixture, expected_diff=[entry(**{"from": "a", "to": "c"})]).status == "FAIL"
    assert compare(fixture, expected_diff=[entry(**{"from": "a", "to": "b"})]).passed


def test_t11_missing_margin_single_item_boundary_and_zero_variance():
    fixture = synthetic_runs(kind="number", reducer="mean", values=[0.0] * 3)
    assert compare(fixture).status == "NOT_VERIFIED"
    assert compare(fixture, margins={"result": 0.5}).status == "PASS"
    for i in range(2):
        change(fixture[1], [0.5] * 3, i)
    report = compare(fixture, margins={"result": 0.5})
    assert report.status == "FAIL"
    assert report.rates[0]["reason"] == "zero_variance_at_or_outside_margin"
    assert compare(fixture, margins={"result": 0.5001}).passed
    fixture = synthetic_runs(kind="number", reducer="mean", values=[0.0] * 3, strata=("en",))
    assert compare(fixture, margins={"result": 0.5}).status == "NOT_VERIFIED"


@pytest.mark.parametrize("margin", [0, -1, True, float("nan"), float("inf"), "0.1"])
def test_t11_invalid_margin_is_not_verified(margin):
    fixture = synthetic_runs(kind="number", reducer="mean", values=[0.0] * 3)
    report = compare(fixture, margins={"result": margin})
    assert report.status == "NOT_VERIFIED"
    json.dumps(asdict(report), allow_nan=False)


@pytest.mark.parametrize("alpha", [0, 0.5, True, float("nan"), "0.05"])
def test_invalid_alpha_is_not_verified(alpha):
    fixture = synthetic_runs(kind="number", reducer="mean", values=[0.0] * 3)
    assert compare(fixture, margins={"result": 1}, alpha=alpha).status == "NOT_VERIFIED"


def test_scored_pairing_is_per_item_separate_strata_and_margin_override():
    fixture = synthetic_runs(kind="number", reducer="mean", values=[0.0] * 3,
                             strata=("en", "he", "en", "he"))
    for i in (1, 3):
        change(fixture[1], [0.6, 0.9, 1.5], i)  # Synthetic item mean 1.0.
    report = compare(fixture, margins={"result": 0.5, ("result", "he"): 1.1})
    assert report.passed
    rates = {r["stratum"]: r for r in report.rates}
    assert rates["en"]["n"] == rates["he"]["n"] == 2
    assert rates["en"]["mean_difference"] == 0
    assert rates["he"]["mean_difference"] == 1
    assert compare(fixture, margins={"result": 0.5}).status == "FAIL"


def test_t11_two_pair_t_interval_matches_analytic_cauchy_reference():
    # n=2 => df=1; t_0.95 = tan(0.45*pi) = 6.313751514675041.
    # Differences [-.1, .1] have mean 0 and standard error .1.
    fixture = synthetic_runs(kind="number", reducer="mean", values=[0.0] * 3)
    change(fixture[1], [-0.1] * 3)
    change(fixture[1], [0.1] * 3, 1)
    report = compare(fixture, margins={"result": 0.7})
    assert report.passed
    assert report.rates[0]["ci_high"] == pytest.approx(0.6313751514675041, rel=1e-10)
    assert report.rates[0]["ci_low"] == pytest.approx(-0.6313751514675041, rel=1e-10)
    assert compare(fixture, margins={"result": 0.6}).status == "FAIL"


def test_score_ledger_never_removes_observations_from_tost():
    fixture = synthetic_runs(kind="number", reducer="mean", values=[0.0] * 3)
    change(fixture[1], [2.0] * 3)
    expected = entry(direction="increase", **{"from": 0.0, "to": 2.0})
    report = compare(fixture, expected_diff=[expected], margins={"result": 0.1})
    assert report.status == "FAIL"
    assert report.rates[0]["n"] == 2
    assert report.config["expected_diff_audit"][0]["status"] == "matched"


def test_finite_scores_with_overflowing_difference_cannot_pass():
    fixture = synthetic_runs(kind="number", reducer="mean", values=[-1e308] * 3)
    change(fixture[1], [1e308] * 3)
    report = compare(fixture, margins={"result": 1.0})
    assert report.status == "NOT_VERIFIED"
    assert report.rates[0]["reason"] == "nonfinite_difference"
    json.dumps(asdict(report), allow_nan=False)


def test_no_required_data_cannot_pass():
    fixture = synthetic_runs(strata=())
    assert compare(fixture).status == "NOT_VERIFIED"
    fixture = synthetic_runs()
    fixture[3][0].role = "diagnostic"
    assert compare(fixture).status == "NOT_VERIFIED"
    fixture[3][0].role = "verdict"
    fixture[2].eligible_ids = {"result": ()}
    assert compare(fixture).status == "NOT_VERIFIED"


def test_indeterminate_full_records_cannot_pass():
    fixture = synthetic_runs(kind="class", values=["a", "b", "c"])
    assert compare(fixture).status == "NOT_VERIFIED"


def test_status_priority_fail_over_not_verified_then_integrity_over_fail():
    fixture = synthetic_runs()
    fixture[3].append(LayerSpec("score", kind="number", reducer="mean"))
    for run in fixture[:2]:
        for item in run.items:
            for trial in item.trials:
                trial.values["score"] = 0.0
    change(fixture[1], [False] * 3)
    assert compare(fixture).status == "FAIL"
    fixture[1].items[0].trials[0].error = "Synthetic error"
    assert compare(fixture).status == "NOT_RUN"


def test_no_mutation_and_deterministic_reordering():
    fixture = synthetic_runs(kind="id_set", values=[frozenset({"b", "a"})] * 3)
    before = copy.deepcopy(fixture)
    report = compare(fixture)
    assert fixture == before
    assert report.passed
    for run in fixture[:2]:
        run.items.reverse()
        for item in run.items:
            item.trials.reverse()
    reordered = compare(fixture)
    assert json.dumps(asdict(report), sort_keys=True, allow_nan=False) == json.dumps(
        asdict(reordered), sort_keys=True, allow_nan=False)


def test_offline_comparison_never_uses_network(monkeypatch):
    import socket

    def forbidden(*args, **kwargs):
        pytest.fail("Offline harness attempted network/model execution")

    monkeypatch.setattr(socket, "socket", forbidden)
    assert compare(synthetic_runs()).passed


def test_invalid_input_preserves_provenance_and_ledger():
    fixture = synthetic_runs()
    fixture[1].refs["recording_hash"] = "synthetic-wrong-hash"
    report = compare(fixture, expected_diff=[entry()])
    assert report.status == "NOT_RUN"
    assert report.refs["recording_hash"] == fixture[2].recording_hash
    assert report.refs["candidate"]["refs"]["recording_hash"] == "synthetic-wrong-hash"
    assert report.config["expected_diff"] == [entry()]


@pytest.mark.parametrize("failure", ["unavailable", "exception", "malformed"])
def test_statistics_backend_failures_cannot_pass(monkeypatch, failure):
    from mltk.core import statistics

    def broken(*args, **kwargs):
        if failure == "unavailable":
            raise ImportError("Synthetic unavailable backend")
        if failure == "exception":
            raise RuntimeError("Synthetic backend failure")
        return {"status": "PASS"}

    monkeypatch.setattr(statistics, "paired_tost", broken)
    fixture = synthetic_runs(kind="number", reducer="mean", values=[0.0] * 3)
    report = compare(fixture, margins={"result": 0.1})
    assert report.status == "NOT_VERIFIED"
    assert report.rates[0]["reason"] == (
        "statistics_unavailable" if failure == "unavailable" else "statistics_error")


def test_oracle_errors_are_not_run():
    def broken_oracle(value):
        raise RuntimeError("Synthetic oracle failure")

    fixture = synthetic_runs()
    fixture[3][0].oracle = broken_oracle
    fixture[3][0].oracle_id = "synthetic-pure-oracle-v1"
    report = compare(fixture)
    assert report.status == "NOT_RUN"
    assert "Synthetic oracle failure" in report.reasons[0]


def test_ledger_and_manifest_reordering_is_deterministic():
    fixture = synthetic_runs()
    change(fixture[1], [False] * 3)
    ledger = [entry(trial_id=t) for t in ("1", "2", "3")]
    fixture[2].eligible_ids = {"result": tuple(fixture[2].items)}
    before = asdict(compare(fixture, expected_diff=ledger))
    fixture[2].trial_ids = tuple(reversed(fixture[2].trial_ids))
    fixture[2].eligible_ids["result"] = tuple(reversed(fixture[2].eligible_ids["result"]))
    after = asdict(compare(fixture, expected_diff=reversed(ledger)))
    assert before == after


def test_id_set_transitions_are_exact_and_unordered():
    fixture = synthetic_runs(kind="id_set", values=[frozenset({"a", "b"})] * 3)
    change(fixture[1], [frozenset({"b", "c"})] * 3)
    ledger = [entry(**{"from": {"b", "a"}, "to": {"c", "b"}})]
    report = compare(fixture, expected_diff=ledger)
    assert report.passed
    assert report.config["expected_diff"][0]["from"] == {
        "type": "id_set", "values": ["a", "b"],
    }
    json.dumps(asdict(report), allow_nan=False)


def test_diagnostic_change_is_descriptive_but_stale_ledger_is_always_error():
    fixture = synthetic_runs()
    fixture[3].append(LayerSpec("diagnostic", role="diagnostic"))
    for run in fixture[:2]:
        for item in run.items:
            for trial in item.trials:
                trial.values["diagnostic"] = run is fixture[0]
    report = compare(fixture)
    assert report.passed
    assert next(r for r in report.rates if r["layer"] == "diagnostic")["status"] == "FAIL"
    assert compare(fixture, expected_diff=[entry()]).status == "FAIL"


def test_general_k_uses_declared_policy_and_exact_trial_ids():
    fixture = synthetic_runs()
    fixture[2].trial_ids = ("1", "2")
    fixture[3][0].min_valid = 2
    fixture[3][0].failed_threshold = 1
    for run in fixture[:2]:
        for item in run.items:
            item.trials.pop()
    report = compare(fixture)
    assert report.passed
    assert report.config["layers"][0]["effective_min_valid"] == 2
    assert report.config["layers"][0]["effective_failed_threshold"] == 1
    fixture[3][0].min_valid = None
    assert compare(fixture).status == "NOT_RUN"


def test_invalid_policy_is_not_run_even_for_empty_population():
    fixture = synthetic_runs(strata=())
    fixture[3][0].min_valid = 100
    assert compare(fixture).status == "NOT_RUN"


def test_empty_layers_cannot_pass():
    fixture = synthetic_runs()
    fixture[3].clear()
    for run in fixture[:2]:
        for item in run.items:
            for trial in item.trials:
                trial.values.clear()
    assert compare(fixture).status == "NOT_VERIFIED"


def test_unresolved_rows_have_gap_fields_and_full_reproducibility_config():
    fixture = synthetic_runs(kind="class", values=["a", "b", "c"])
    fixture[3][0].oracle_id = "synthetic-oracle-policy-v2"
    fixture[3][0].order = 7
    fixture[3][0].majority_fraction = 0.6
    report = compare(fixture)
    assert report.status == "NOT_VERIFIED"
    assert all(row["group"] == "Unconcluded" for row in report.items)
    assert all(row["reason"] == "unresolved_item_state" for row in report.items)
    assert all(row["classification"] == "INDETERMINATE" for row in report.items)
    assert report.config["manifest"] == {
        "items": fixture[2].items, "trial_ids": ["1", "2", "3"],
        "recording_hash": "synthetic-recording-v1", "eligible_ids": {},
    }
    assert report.config["layers"] == [{
        "name": "result", "kind": "class", "role": "verdict", "order": 7,
        "oracle_id": "synthetic-oracle-policy-v2", "vocabulary": ["a", "b", "c"],
        "reducer": "majority", "min_valid": None, "failed_threshold": None,
        "effective_min_valid": 2, "effective_failed_threshold": 2,
        "majority_fraction": 0.6,
    }]
    assert report.schema_version
    assert report.engine_version


def test_vocabulary_order_is_canonical_without_mutating_layer():
    fixture = synthetic_runs(kind="class", values=["a"] * 3)
    before = asdict(compare(fixture))
    fixture[3][0].vocabulary = ("c", "a", "b")
    after = asdict(compare(fixture))
    assert before == after
    assert fixture[3][0].vocabulary == ("c", "a", "b")


@pytest.mark.parametrize(("minimum", "threshold", "effective"), [
    (None, None, (2, 2)), (3, 1, (3, 1)),
])
def test_layer_config_preserves_requested_and_effective_policy(minimum, threshold, effective):
    fixture = synthetic_runs()
    fixture[3][0].min_valid = minimum
    fixture[3][0].failed_threshold = threshold
    report = compare(fixture)
    assert report.passed
    config = report.config["layers"][0]
    assert config["min_valid"] == minimum
    assert config["failed_threshold"] == threshold
    assert (config["effective_min_valid"], config["effective_failed_threshold"]) == effective


@pytest.mark.parametrize("values", [[False] * 3, [True, True, False], [True] * 3])
def test_preconditions_require_every_trial_to_pass_even_with_identical_states(values):
    fixture = synthetic_runs(values=values)
    fixture[3][0].role = "precondition"
    fixture[3][0].oracle = lambda value: value
    fixture[3][0].oracle_id = "synthetic-identity-v1"
    report = compare(fixture)
    expected = "PASS" if all(values) else "FAIL"
    assert report.status == expected
    assert report.rates[0]["comparison_status"] == "PASS"
    assert report.rates[0]["precondition_status"] == expected
    if expected == "FAIL":
        assert report.items[0]["classification"] == "PRECONDITION_FAILED"
        assert report.items[0]["reason"] == "precondition_failed"
        assert report.items[0]["base"]["pass_all_trials"] is False
        assert report.items[0]["candidate"]["pass_all_trials"] is False


@pytest.mark.parametrize("side", [0, 1])
def test_ledger_cannot_exempt_a_failing_precondition_on_either_side(side):
    fixture = synthetic_runs()
    fixture[3][0].role = "precondition"
    fixture[3][0].oracle = lambda value: value
    fixture[3][0].oracle_id = "synthetic-identity-v1"
    change(fixture[side], [False] * 3)
    ledger = entry(**{"from": side == 1, "to": side == 0})
    report = compare(fixture, expected_diff=[ledger])
    assert report.status == "FAIL"
    assert report.config["expected_diff_audit"][0]["status"] == "matched"
    assert report.rates[0]["comparison_status"] == "PASS"
    assert report.rates[0]["reason"] == "precondition_failed"
    assert report.items[0]["precondition"]["base" if side == 0 else "candidate"] is False


@pytest.mark.parametrize("oracle_kind", ["absent", "none", "partial"])
def test_incomplete_precondition_oracle_evidence_is_an_unconcluded_gap(oracle_kind):
    fixture = synthetic_runs(values=[True, True, False])
    layer = fixture[3][0]
    layer.role = "precondition"
    if oracle_kind != "absent":
        layer.oracle_id = "synthetic-incomplete-oracle-v1"
        layer.oracle = (lambda value: None) if oracle_kind == "none" else (
            lambda value: True if value else None)
    report = compare(fixture)
    assert report.status == "NOT_VERIFIED"
    assert report.rates[0]["reason"] == "precondition_oracle_incomplete"
    assert report.items[0]["group"] == "Unconcluded"
    assert report.items[0]["classification"] == "PRECONDITION_NOT_VERIFIED"
    assert report.items[0]["reason"] == "precondition_oracle_incomplete"
    assert report.items[0]["comparison_status"] == "PASS"


@pytest.mark.parametrize("oracle_result", [True, False, None])
@pytest.mark.parametrize("margin", [None, 0.5])
def test_mean_preconditions_gate_independently_of_tost(oracle_result, margin):
    fixture = synthetic_runs(kind="number", reducer="mean", values=[1.0] * 3)
    fixture[3][0].role = "precondition"
    fixture[3][0].oracle = lambda value: oracle_result
    fixture[3][0].oracle_id = "synthetic-constant-oracle-v1"
    report = compare(fixture, margins={"result": margin})
    expected = ("FAIL" if oracle_result is False else
                "NOT_VERIFIED" if oracle_result is None or margin is None else "PASS")
    assert report.status == expected
    assert report.rates[0]["comparison_status"] == (
        "NOT_VERIFIED" if margin is None else "PASS")
    if oracle_result is False:
        assert report.rates[0]["reason"] == "precondition_failed"


def test_measured_change_failure_overrides_incomplete_precondition_oracle():
    fixture = synthetic_runs(kind="number", reducer="mean", values=[0.0] * 3)
    fixture[3][0].role = "precondition"  # No oracle: incomplete evidence.
    for index in range(2):
        change(fixture[1], [2.0] * 3, index)
    report = compare(fixture, margins={"result": 0.1})
    assert report.status == "FAIL"
    assert report.rates[0]["comparison_status"] == "FAIL"
    assert report.rates[0]["precondition_status"] == "NOT_VERIFIED"


def test_failed_precondition_dominates_incomplete_oracle_evidence():
    fixture = synthetic_runs(values=[True, True, False])
    fixture[3][0].role = "precondition"
    fixture[3][0].oracle = lambda value: None if value else False
    fixture[3][0].oracle_id = "synthetic-failure-and-incomplete-v1"
    assert compare(fixture).status == "FAIL"


@pytest.mark.parametrize("integrity_error", ["failed_trial", "missing_trial", "wrong_hash"])
def test_precondition_failure_cannot_override_not_run(integrity_error):
    fixture = synthetic_runs(values=[False] * 3)
    fixture[3][0].role = "precondition"
    fixture[3][0].oracle = lambda value: value
    fixture[3][0].oracle_id = "synthetic-identity-v1"
    if integrity_error == "failed_trial":
        trial = fixture[1].items[0].trials[0]
        trial.status, trial.error = "failed", "Synthetic failed recording"
    elif integrity_error == "missing_trial":
        fixture[1].items[0].trials.pop()
    else:
        fixture[1].refs["recording_hash"] = "synthetic-wrong-hash"
    assert compare(fixture).status == "NOT_RUN"


def test_mean_precondition_checks_trials_even_when_reduced_oracle_passes():
    fixture = synthetic_runs(kind="number", reducer="mean", values=[1.0, 1.0, -0.5])
    fixture[3][0].role = "precondition"
    fixture[3][0].oracle = lambda value: value > 0
    fixture[3][0].oracle_id = "synthetic-positive-oracle-v1"
    report = compare(fixture, margins={"result": 0.1})
    assert report.status == "FAIL"
    assert report.items[0]["base"]["oracle"] is True
    assert report.items[0]["base"]["pass_all_trials"] is False
    assert report.rates[0]["comparison_status"] == "PASS"
    assert report.rates[0]["reason"] == "precondition_failed"
