"""Review regressions for candidate coverage, noisy controls and empty cells."""

from __future__ import annotations

import pytest
from tests.test_parity.test_engine import flip, product, synthetic

from mltk.parity.control import check_positive_control, make_positive_control
from mltk.parity.engine import compare_product, compare_runs
from mltk.parity.models import LayerSpec, Manifest, ParityConfig


@pytest.mark.parametrize("missing", ["record", "failed", "indeterminate", "source"])
def test_candidate_coverage_cannot_pass_on_one_of_twenty(missing):
    data = synthetic(layers=[LayerSpec("v")])
    for item in data[2].items[1:]:
        if missing == "record":
            item.trials.clear()
        elif missing == "failed":
            for trial in item.trials:
                trial.status, trial.error = "failed", "synthetic failure"
        elif missing == "indeterminate":
            for trial, value in zip(item.trials, (True, False, None), strict=True):
                trial.values["v"] = value
        else:
            item.source_id = "changed-source"
            flip(item)
    if missing == "record":
        data[2].items[:] = data[2].items[:1]
    report = product(data)
    rate = report.rates[0]
    assert report.status == "NOT_VERIFIED"
    assert rate["verdict"] == "DETECTION_NOT_DEMONSTRATED"
    assert rate["reason"] == "CANDIDATE_NOT_ESTABLISHED"
    assert (rate["n_state_compared"], rate["n_compared"], rate["n_excluded"]) == (1, 0, 19)
    assert (rate["required_established"], rate["n_eligible"]) == (16, 20)
    assert sum(rate["excluded_reasons"].values()) == 19


@pytest.mark.parametrize(
    ("n", "fraction", "required"), [(20, 0.8, 16), (90, 0.7, 63), (25, 0.28, 7)]
)
@pytest.mark.parametrize("shortfall", [0, 1])
def test_candidate_establishment_exact_boundary(n, fraction, required, shortfall):
    data = synthetic(n, layers=[LayerSpec("v")])
    data[2].items[:] = data[2].items[: required - shortfall]
    report = product(data, config=ParityConfig(established_fraction=fraction))
    rate = report.rates[0]
    assert rate["required_established"] == required
    assert rate["n_state_compared"] == required - shortfall
    assert rate["n_compared"] == 0  # Coverage must not use the oracle denominator.
    assert report.status == ("NOT_VERIFIED" if shortfall else "PASS")


def test_quietness_uses_exact_rational_floor_at_point_seven_times_ninety():
    data = synthetic(90)
    for item in data[1].items[:63]:
        flip(item)
    rate = product(data, config=ParityConfig(quiet_fraction=0.7, established_fraction=0.2)).rates[0]
    assert rate["allowed_noise"] == 63
    assert rate["quiet"]


@pytest.mark.parametrize("noise", [1, 2])
def test_real_noisy_a2_control_detects_and_binds_report_evidence(noise):
    a1, a2, candidate, layers, manifest, _ = synthetic()
    noisy_ids = {item.item_id for item in a2.items[:noise]}
    for item in a2.items[:noise]:
        flip(item)
    fixture = make_positive_control(
        a1, a2=a2, manifest=manifest, layer=layers[0], n_items=6, mutation=lambda v: not v, seed=7
    )
    assert not noisy_ids.intersection(fixture.changed_item_ids)
    evidence = check_positive_control(a1, fixture, a2=a2, layers=layers, manifest=manifest)
    record = evidence["v", "en"]
    assert record == {
        "passed": True,
        "a2_run_id": a2.run_id,
        "seed": 7,
        "n_items": 6,
        "changed_item_ids": list(fixture.changed_item_ids),
    }
    report = compare_product(a1, a2, candidate, layers=layers, manifest=manifest, control=evidence)
    assert report.status == "PASS"
    assert report.rates[0]["control_evidence"] == record
    assert report.rates[0]["n_excluded"] == noise


def test_control_mismatched_repeat_and_boolean_claim_are_rejected():
    a1, a2, candidate, layers, manifest, controls = synthetic()
    controls["v", "en"]["a2_run_id"] = "another-repeat"
    with pytest.raises(ValueError, match="A2 run_id"):
        compare_product(a1, a2, candidate, layers=layers, manifest=manifest, control=controls)
    with pytest.raises(ValueError, match="control evidence"):
        compare_product(
            a1, a2, candidate, layers=layers, manifest=manifest, control={("v", "en"): True}
        )


def test_control_helpers_require_distinct_real_a2_and_target_policy():
    a1, a2, _, layers, manifest, _ = synthetic()
    with pytest.raises(TypeError, match="a2"):
        make_positive_control(
            a1, manifest=manifest, layer=layers[0], n_items=6, mutation=lambda v: not v, seed=7
        )
    with pytest.raises(ValueError, match="independent A2"):
        make_positive_control(
            a1,
            a2=a1,
            manifest=manifest,
            layer=layers[0],
            n_items=6,
            mutation=lambda v: not v,
            seed=7,
        )
    with pytest.raises(ValueError, match="LayerSpec"):
        make_positive_control(
            a1, a2=a2, manifest=manifest, layer="v", n_items=6, mutation=lambda v: not v, seed=7
        )
    fixture = make_positive_control(
        a1, a2=a2, manifest=manifest, layer=layers[0], n_items=6, mutation=lambda v: not v, seed=7
    )
    with pytest.raises(TypeError, match="a2"):
        check_positive_control(a1, fixture, layers=layers, manifest=manifest)
    with pytest.raises(ValueError, match="independent A2"):
        check_positive_control(a1, fixture, a2=a1, layers=layers, manifest=manifest)
    a2.run_id = "other-a2"
    with pytest.raises(ValueError, match="A2 run_id"):
        check_positive_control(a1, fixture, a2=a2, layers=layers, manifest=manifest)


def test_control_does_not_plant_on_missing_failed_indeterminate_or_changed_sources():
    a1, a2, _, layers, manifest, _ = synthetic()
    a2.items[0].trials.clear()
    for trial in a2.items[1].trials:
        trial.status, trial.error = "failed", "synthetic failure"
    for trial, value in zip(a2.items[2].trials, (True, False, None), strict=True):
        trial.values["v"] = value
    a2.items[3].source_id = "different"
    manifest.eligible_ids["v"] = tuple(i.item_id for i in a1.items[:18])
    fixture = make_positive_control(
        a1, a2=a2, manifest=manifest, layer=layers[0], n_items=14, mutation=lambda v: not v, seed=7
    )
    assert set(fixture.changed_item_ids) == {i.item_id for i in a1.items[4:18]}
    with pytest.raises(ValueError, match="Requested"):
        make_positive_control(
            a1,
            a2=a2,
            manifest=manifest,
            layer=layers[0],
            n_items=15,
            mutation=lambda v: not v,
            seed=7,
        )


@pytest.mark.parametrize("empty", ["manifest", "eligibility"])
def test_empty_pair_is_not_verified(empty):
    a1, _, candidate, layers, manifest, _ = synthetic()
    if empty == "manifest":
        a1.items.clear()
        candidate.items.clear()
        manifest = Manifest({})
    else:
        manifest.eligible_ids["v"] = ()
    report = compare_runs(a1, candidate, layers=layers, manifest=manifest)
    assert report.status == "NOT_VERIFIED"
    assert report.reasons == ["NO_COMPARABLE_ITEMS"]
    assert all(r["verdict"] == "NOT_APPLICABLE" and not r["required"] for r in report.rates)


def test_pair_records_and_uses_configured_confidence():
    a1, _, candidate, layers, manifest, _ = synthetic()
    ordinary = compare_runs(a1, candidate, layers=layers, manifest=manifest)
    custom = compare_runs(
        a1, candidate, layers=layers, manifest=manifest, config=ParityConfig(confidence=0.8)
    )
    assert custom.config["confidence"] == 0.8
    assert custom.rates[0]["nfr_interval"]["ci_high"] < ordinary.rates[0]["nfr_interval"]["ci_high"]


@pytest.mark.parametrize("mode", ["pair", "product"])
def test_ineligible_cells_and_preconditions_do_not_block_other_strata(mode):
    layers = [
        LayerSpec("v"),
        LayerSpec("p", role="precondition", oracle=lambda v: v, oracle_id="synthetic-v1"),
    ]
    data = synthetic(strata=("en", "he"), layers=layers)
    data[4].eligible_ids["p"] = tuple(i for i, s in data[4].items.items() if s == "en")
    data[5].pop(("p", "he"))
    report = (
        product(data)
        if mode == "product"
        else compare_runs(data[0], data[2], layers=layers, manifest=data[4])
    )
    assert report.status == "PASS"
    cell = next(r for r in report.rates if (r["layer"], r["stratum"]) == ("p", "he"))
    assert cell["verdict"] == "NOT_APPLICABLE"
    assert not cell["required"]
    # A real eligible cell with entirely missing evidence remains required.
    for run in data[:3]:
        for item in run.items:
            if item.stratum == "he":
                for trial in item.trials:
                    trial.values.pop("v")
    report = (
        product(data)
        if mode == "product"
        else compare_runs(data[0], data[2], layers=layers, manifest=data[4])
    )
    assert report.status != "PASS"
    cell = next(r for r in report.rates if (r["layer"], r["stratum"]) == ("v", "he"))
    assert cell["required"]
    assert cell["verdict"] != "NOT_APPLICABLE"


def test_all_ineligible_product_cells_remain_not_verified_without_controls():
    data = synthetic()
    data[4].eligible_ids["v"] = ()
    data[5].clear()
    report = product(data)
    assert report.status == "NOT_VERIFIED"
    assert report.reasons == ["NO_COMPARABLE_ITEMS"]
    assert report.rates[0]["verdict"] == "NOT_APPLICABLE"


@pytest.mark.parametrize("mode", ["pair", "product"])
def test_engine_status_includes_failure_and_negative_gates(mode):
    data = synthetic()
    for trial in data[2].items[0].trials:
        trial.status, trial.error = "failed", "synthetic execution failure"
    report = (
        product(data)
        if mode == "product"
        else compare_runs(data[0], data[2], layers=data[3], manifest=data[4])
    )
    assert report.rates[0]["from_failure_count"] == 1
    assert report.status == "FAIL"
    data = synthetic()
    flip(data[2].items[0])
    config = ParityConfig(fail_on_negative_flip=True)
    report = (
        product(data, config=config)
        if mode == "product"
        else compare_runs(data[0], data[2], layers=data[3], manifest=data[4], config=config)
    )
    assert report.status == "FAIL"


@pytest.mark.parametrize("shortfall", [0, 1])
def test_aa_establishment_uses_same_exact_boundary(shortfall):
    data = synthetic(90)
    data[1].items[:] = data[1].items[: 63 - shortfall]
    report = product(data, config=ParityConfig(established_fraction=0.7, quiet_fraction=0.4))
    rate = report.rates[0]
    assert rate["required_established"] == 63
    assert rate["n_valid_aa"] == 63 - shortfall
    assert rate["established"] is (not shortfall)
    assert report.status == ("NOT_VERIFIED" if shortfall else "PASS")
