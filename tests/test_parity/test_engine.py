"""Synthetic engine fixtures with independently specified expected counts."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import asdict

import pytest

from mltk.parity.engine import compare_product, compare_runs
from mltk.parity.models import Item, LayerSpec, Manifest, ParityConfig, Run, Trial


def synthetic(n=20, strata=("en",), layers=None):
    layers = layers or [LayerSpec("v", oracle=lambda value: value, oracle_id="synthetic-v1")]
    items = [
        Item(
            f"{s}-{i:02}",
            s,
            [Trial(str(j), values={layer.name: True for layer in layers}) for j in range(1, 4)],
            source_id="synthetic-source",
        )
        for s in strata
        for i in range(n)
    ]
    base = Run("synthetic-a1", "synthetic", items, refs={"build": "synthetic-build"})
    a2, candidate = deepcopy(base), deepcopy(base)
    a2.run_id, candidate.run_id = "synthetic-a2", "synthetic-b"
    manifest = Manifest({i.item_id: i.stratum for i in items})
    controls = {
        (layer.name, s): {
            "passed": True,
            "a2_run_id": a2.run_id,
            "seed": 7,
            "n_items": 1,
            "changed_item_ids": [f"{s}-00"],
        }
        for layer in layers
        for s in strata
    }
    return base, a2, candidate, layers, manifest, controls


def flip(item, layer="v", value=False):
    for trial in item.trials:
        trial.values[layer] = value


def product(data, **kwargs):
    a1, a2, candidate, layers, manifest, controls = data
    return compare_product(
        a1, a2, candidate, layers=layers, manifest=manifest, control=controls, **kwargs
    )


def test_t1_t2_t4_identity_six_flips_and_reference_statistics():
    data = synthetic()
    report = product(data)
    rate = report.rates[0]
    assert report.status == "PASS"
    assert rate["verdict"] == "NO_CHANGE_ABOVE_MDC"
    assert rate["changed_rate"] == 0
    assert round(rate["floor"], 3) == 0.161
    assert rate["mdc"] == 4
    for item in data[2].items[:6]:
        flip(item)
    report = product(data)
    rate = report.rates[0]
    assert report.status == "FAIL"
    assert rate["verdict"] == "DETECTED"
    assert (rate["b"], rate["c"], rate["n_compared"]) == (6, 0, 20)
    assert rate["mcnemar_p"] == 0.03125
    assert rate["nfr"] == 0.3
    assert rate["nfr_interval"]["n"] == 20


def test_t4_observed_noise_two_and_quiet():
    data = synthetic()
    for item in data[1].items[:2]:
        flip(item)
    rate = product(data).rates[0]
    assert rate["noisy"] == 2
    assert rate["quiet"]
    assert round(rate["floor"], 3) == 0.301
    assert rate["n_stable"] == rate["n_compared"] == 18


def test_t5_missing_population_does_not_shrink_and_establishment_fails():
    data = synthetic(8)
    data[1].items[:] = data[1].items[:6]
    report = product(data)
    rate = report.rates[0]
    assert report.status == "NOT_VERIFIED"
    assert rate["n_eligible"] == 8
    assert rate["n_valid_aa"] == 6
    assert not rate["established"]
    assert "NOT_ESTABLISHED" in rate["reasons"]
    assert rate["n_unconcluded"] == rate["aa_unresolved"] == 2
    assert rate["noisy"] == 0
    assert rate["excluded_reasons"] == {"AA_EXCLUDED": 2}
    gaps = [r for r in report.items if r["group"] == "Unconcluded"]
    assert len(gaps) == 2
    assert all(r["classification"] == "UNCHANGED" for r in gaps)


def test_indeterminate_not_double_counted_in_conservative_noise():
    data = synthetic()
    for item in data[1].items[:2]:
        for trial, value in zip(item.trials, (True, False, None), strict=True):
            trial.values["v"] = value
    rate = product(data).rates[0]
    assert (rate["noisy"], rate["aa_unresolved"], rate["conservative_noise_count"]) == (0, 2, 2)
    assert round(rate["floor"], 3) == 0.301
    assert rate["quiet"]


def test_nonquiet_descriptive_no_change_cannot_pass():
    data = synthetic()
    for item in data[1].items[:3]:
        flip(item)
    report = product(data)
    assert report.rates[0]["verdict"] == "NO_CHANGE_ABOVE_MDC"
    assert report.rates[0]["established"]
    assert not report.rates[0]["quiet"]
    assert report.status == "NOT_VERIFIED"


@pytest.mark.parametrize("failed_count", [1, 2, 3])
def test_any_candidate_execution_regression_blocks_but_not_as_oracle_flip(failed_count):
    data = synthetic()
    for trial in data[2].items[0].trials[:failed_count]:
        trial.status, trial.error = "failed", "synthetic execution failure"
    report = product(data)
    rate, row = report.rates[0], report.items[0]
    assert report.status == "FAIL"
    assert rate["execution_regression_count"] == 1
    assert rate["from_failure_count"] == int(failed_count >= 2)
    assert rate["n_compared"] == 20 - int(failed_count >= 2)
    assert rate["b"] == 0
    assert row["classification"] == ("UNCHANGED" if failed_count == 1 else "FAILED_CAND")
    assert not rate["gate_passed"]
    pair = compare_runs(data[0], data[2], layers=data[3], manifest=data[4])
    assert pair.status == "FAIL"


def test_strict_negative_flip_optional_below_detection_floor():
    data = synthetic()
    flip(data[2].items[0])
    assert product(data).status == "PASS"
    report = product(data, config=ParityConfig(fail_on_negative_flip=True))
    assert report.status == "FAIL"
    assert "STRICT_NEGATIVE_FLIP" in report.reasons


def test_unknown_sources_and_oracles_with_known_changed_evidence_excluded():
    layer = LayerSpec(
        "v",
        oracle=lambda v: None if v == "unknown" else v == "pass",
        oracle_id="synthetic-v1",
        kind="class",
        vocabulary=("pass", "fail", "unknown"),
    )
    data = synthetic(layers=[layer])
    for run in data[:3]:
        for item in run.items:
            flip(item, value="pass")
            item.source_id = None
        flip(run.items[1], value="unknown")
    data[0].items[0].source_id = "old"
    data[2].items[0].source_id = "new"
    flip(data[2].items[0], value="fail")
    flip(data[2].items[2], value="fail")
    report = product(data)
    rate = report.rates[0]
    assert (rate["n_state_compared"], rate["n_compared"], rate["n_oracle_unknown"]) == (19, 18, 1)
    assert rate["excluded_reasons"] == {"SOURCE_CHANGED": 1}
    assert rate["b"] == 1
    assert report.items[0]["classification"] == "CHANGED_SOURCE"
    assert report.items[2]["source_unknown"]


def test_missing_controls_empty_eligibility_and_preconditions():
    data = synthetic()
    data[5].clear()
    assert product(data).status == "NOT_VERIFIED"
    assert product(data).rates[0]["reason"] == "CONTROL_MISSING"
    data[4].eligible_ids["v"] = ()
    rate = product(data).rates[0]
    assert not rate["established"]
    assert rate["n_eligible"] == 0
    assert rate["changed_rate"] is None
    layers = [
        LayerSpec("v"),
        LayerSpec("p", role="precondition", oracle=lambda v: v, oracle_id="synthetic-v1"),
    ]
    data = synthetic(layers=layers)
    data[2].items[0].trials[0].values["p"] = False
    report = product(data)
    assert report.status == "NOT_VERIFIED"
    assert all(not r["preconditions_passed"] for r in report.rates)


def test_diagnostics_never_gate_and_alone_never_pass():
    data = synthetic(layers=[LayerSpec("v"), LayerSpec("d", role="diagnostic")])
    for item in data[2].items:
        flip(item, "d")
    assert product(data).status == "PASS"
    data = synthetic(layers=[LayerSpec("d", role="diagnostic")])
    assert product(data).status == "NOT_VERIFIED"
    assert compare_runs(data[0], data[2], layers=data[3], manifest=data[4]).status == "NOT_VERIFIED"


def test_pair_and_product_different_denominators_and_reconciliation():
    data = synthetic()
    flip(data[1].items[0])
    flip(data[2].items[0])
    data[2].items[1].trials.clear()
    data[2].items[2].source_id = "changed"
    pair = compare_runs(data[0], data[2], layers=data[3], manifest=data[4]).rates[0]
    calibrated = product(data).rates[0]
    assert pair["n_compared"] == 18
    assert calibrated["n_compared"] == 17
    assert pair["b"] == 1
    assert calibrated["b"] == 0
    for rate in (pair, calibrated):
        assert rate["n_state_compared"] + rate["n_excluded"] == rate["n_eligible"]
        assert sum(rate["excluded_reasons"].values()) == rate["n_excluded"]


def test_bootstrap_item_weights_bh_strata_and_effective_metadata():
    layers = [
        LayerSpec("score", kind="number", reducer="mean"),
        LayerSpec("v", oracle=lambda v: v, oracle_id="synthetic-v1"),
    ]
    data = synthetic(strata=("en", "he"), layers=layers)
    for run in data[:3]:
        for item in run.items:
            flip(item, "score", 1.0)
    for item in data[2].items:
        if item.stratum == "he":
            flip(item, "score", 2.0)
            flip(item)
    config = ParityConfig(bootstrap_seed=7, bootstrap_resamples=99, bh_q=0.05)
    report = product(data, config=config)
    rates = {(r["layer"], r["stratum"]): r for r in report.rates}
    assert rates["score", "en"]["bootstrap"]["mean_difference"] == 0
    assert rates["score", "he"]["bootstrap"]["mean_difference"] == 1
    assert rates["score", "he"]["bootstrap"]["n"] == 20
    assert rates["v", "en"]["mcnemar_p_adjusted"] == 1
    assert rates["v", "he"]["mcnemar_p_adjusted"] < 0.05
    assert rates["v", "he"]["bh_family"] == ["score", "v"]
    assert report.refs["a1"]["run_id"] == data[0].run_id
    assert report.config["manifest"]["items"] == data[4].items
    assert report.config["layers"][0]["min_valid"] == 2
    reordered = deepcopy(data)
    for run in reordered[:3]:
        run.items.reverse()
        for item in run.items:
            item.trials.reverse()
    reordered[4].trial_ids = ("3", "2", "1")
    assert product(reordered, config=config) == report


@pytest.mark.parametrize(
    "config",
    [
        ParityConfig(confidence=1),
        ParityConfig(quiet_fraction=-1),
        ParityConfig(established_fraction=0),
        ParityConfig(bh_q=True),
        ParityConfig(bootstrap_resamples=0),
    ],
)
def test_bad_configuration_rejected(config):
    with pytest.raises(ValueError, match="Invalid|Confidence|bh_q|bootstrap_resamples"):
        product(synthetic(), config=config)


def test_upstream_order_and_balanced_churn():
    data = synthetic(
        layers=[
            LayerSpec("later", order=2, oracle=lambda v: v, oracle_id="v1"),
            LayerSpec("first", order=1, oracle=lambda v: v, oracle_id="v1"),
        ]
    )
    for run in data[:2]:
        for item in run.items[:3]:
            flip(item, "first")
    for item in data[2].items[3:6]:
        flip(item, "first")
    flip(data[2].items[0], "later")
    report = product(data)
    first = next(r for r in report.rates if r["layer"] == "first")
    assert first["b"] == first["c"] == 3
    assert first["mcnemar_p"] == 1
    assert first["verdict"] == "DETECTED"
    assert all(
        r["upstream_flipped_layer"] == "first" for r in report.items if r["item_id"] == "en-00"
    )


@pytest.mark.parametrize(("first", "second"), [(0, 1), (0, 2), (1, 2)])
def test_public_product_requires_distinct_run_ids(first, second):
    data = synthetic()
    data[second].run_id = data[first].run_id
    with pytest.raises(ValueError, match="three distinct run_id"):
        product(data)


def test_public_product_rejects_reusing_same_aa_recording():
    a1, _, candidate, layers, manifest, controls = synthetic()
    with pytest.raises(ValueError, match="three distinct run_id"):
        compare_product(a1, a1, candidate, layers=layers, manifest=manifest, control=controls)


@pytest.mark.parametrize("base_value", [-1e308, 1e308])
@pytest.mark.parametrize(
    ("changed", "status", "verdict"),
    [
        (1, "PASS", "NO_CHANGE_ABOVE_MDC"),
        (6, "FAIL", "DETECTED"),
    ],
)
def test_nonfinite_bootstrap_difference_is_unverified_without_changing_gate(
    base_value, changed, status, verdict
):
    data = synthetic(layers=[LayerSpec("score", kind="number", reducer="mean")])
    for run in data[:3]:
        for item in run.items:
            flip(item, "score", base_value)
    for item in data[2].items[:changed]:
        flip(item, "score", -base_value)
    ordinary = product(data)
    report = product(data, config=ParityConfig(bootstrap_seed=7, bootstrap_resamples=99))
    rate = report.rates[0]
    assert report.status == ordinary.status == status
    assert rate["verdict"] == ordinary.rates[0]["verdict"] == verdict
    assert rate["gate_passed"] == ordinary.rates[0]["gate_passed"]
    bootstrap = rate["bootstrap"]
    assert bootstrap["status"] == "NOT_VERIFIED"
    assert bootstrap["reason"] == "NONFINITE_DIFFERENCE"
    assert bootstrap["ci_low"] is None
    assert bootstrap["ci_high"] is None
    assert bootstrap["mean_difference"] is None
    assert (bootstrap["n"], bootstrap["seed"], bootstrap["resamples"]) == (20, 7, 99)
    assert len(rate["bootstrap_item_ids"]) == 20
    json.dumps(asdict(report), allow_nan=False)
