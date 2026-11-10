"""Synthetic controls must demonstrate their own changes without altering inputs."""

from __future__ import annotations

from copy import deepcopy

import pytest

from mltk.parity.control import check_positive_control, make_positive_control
from mltk.parity.engine import compare_product
from mltk.parity.models import Item, LayerSpec, Manifest, Run, Trial


def synthetic():
    run = Run(
        "synthetic",
        "synthetic",
        [
            Item(f"i{i:02}", "en", [Trial(str(j), values={"v": True}) for j in range(1, 4)])
            for i in range(20)
        ],
    )
    a2 = deepcopy(run)
    a2.run_id = "synthetic-a2"
    return (
        run,
        a2,
        LayerSpec("v", oracle=lambda v: v, oracle_id="synthetic-v1"),
        Manifest({i.item_id: i.stratum for i in run.items}),
    )


def test_deep_copy_exact_seeded_selection_and_failure_counts():
    run, a2, layer, manifest = synthetic()
    original = deepcopy(run)
    fixture = make_positive_control(
        run,
        a2=a2,
        manifest=manifest,
        layer=layer,
        n_items=6,
        mutation=lambda v: not v,
        seed=7,
        failed_trials=2,
    )
    assert run == original
    assert len(fixture.changed_item_ids) == 6
    assert len(fixture.failed_trial_ids) == 2
    assert sum(t.status == "failed" for i in fixture.run.items for t in i.trials) == 2
    actual = {i.item_id for i in fixture.run.items if any(not t.values["v"] for t in i.trials)}
    assert actual == set(fixture.changed_item_ids)
    run.items.reverse()
    for item in run.items:
        item.trials.reverse()
    repeated = make_positive_control(
        run,
        a2=a2,
        manifest=manifest,
        layer=layer,
        n_items=6,
        mutation=lambda v: not v,
        seed=7,
        failed_trials=2,
    )
    assert repeated.changed_item_ids == fixture.changed_item_ids
    assert repeated.failed_trial_ids == fixture.failed_trial_ids
    assert repeated.changes == fixture.changes


@pytest.mark.parametrize(
    ("n", "failed", "mutation"),
    [
        (-1, 0, lambda v: not v),
        (21, 0, lambda v: not v),
        (1, 4, lambda v: not v),
        (1, 0, lambda v: v),
        (1, 0, lambda _: 1),
        (1, 0, lambda _: None),
    ],
)
def test_impossible_counts_noop_and_bad_mutation_rejected(n, failed, mutation):
    run, a2, layer, manifest = synthetic()
    original = deepcopy(run)
    with pytest.raises(ValueError, match="counts|Requested|Mutation"):
        make_positive_control(
            run,
            a2=a2,
            manifest=manifest,
            layer=layer,
            n_items=n,
            failed_trials=failed,
            mutation=mutation,
            seed=7,
        )
    assert run == original


def test_control_must_really_detect_and_identity_must_match():
    run, a2, layer, manifest = synthetic()
    fixture = make_positive_control(
        run, a2=a2, manifest=manifest, layer=layer, n_items=6, mutation=lambda v: not v, seed=7
    )
    assert check_positive_control(run, fixture, a2=a2, layers=[layer], manifest=manifest)[
        "v", "en"
    ]["passed"]
    weak = make_positive_control(
        run, a2=a2, manifest=manifest, layer=layer, n_items=1, mutation=lambda v: not v, seed=7
    )
    assert not check_positive_control(run, weak, a2=a2, layers=[layer], manifest=manifest)[
        "v", "en"
    ]["passed"]
    identity = deepcopy(run)
    identity.items[0].trials[0].values["v"] = False
    assert not check_positive_control(
        run, fixture, a2=a2, layers=[layer], manifest=manifest, identity=identity
    )["v", "en"]["passed"]


@pytest.mark.parametrize("mismatch", ["item", "stratum", "trial", "layer", "selected"])
def test_identity_or_fixture_target_mismatch_rejected(mismatch):
    run, a2, layer, manifest = synthetic()
    fixture = make_positive_control(
        run, a2=a2, manifest=manifest, layer=layer, n_items=6, mutation=lambda v: not v, seed=7
    )
    identity = deepcopy(run)
    if mismatch == "item":
        identity.items[0].item_id = "other"
    elif mismatch == "stratum":
        identity.items[0].stratum = "he"
    elif mismatch == "trial":
        identity.items[0].trials[0].trial_id = "unexpected"
    elif mismatch == "layer":
        fixture.layer = "other"
    else:
        fixture.changed_item_ids = ("other",)
    evidence = check_positive_control(
        run, fixture, a2=a2, layers=[layer], manifest=manifest, identity=identity
    )
    assert evidence
    assert not any(record["passed"] for record in evidence.values())


def test_failed_planted_items_cannot_fake_detection():
    run, a2, layer, manifest = synthetic()
    fixture = make_positive_control(
        run,
        a2=a2,
        manifest=manifest,
        layer=layer,
        n_items=6,
        mutation=lambda v: not v,
        seed=7,
        failed_trials=18,
    )
    assert not check_positive_control(run, fixture, a2=a2, layers=[layer], manifest=manifest)[
        "v", "en"
    ]["passed"]


def test_controls_do_not_claim_unplanted_stratum():
    run, a2, layer, manifest = synthetic()
    extra = deepcopy(run.items)
    for item in extra:
        item.item_id = "he-" + item.item_id
        item.stratum = "he"
    run.items.extend(extra)
    a2.items.extend(deepcopy(extra))
    manifest.items.update({i.item_id: "he" for i in extra})
    fixture = make_positive_control(
        run, a2=a2, manifest=manifest, layer=layer, n_items=12, mutation=lambda v: not v, seed=7
    )
    evidence = check_positive_control(run, fixture, a2=a2, layers=[layer], manifest=manifest)
    for stratum in ("en", "he"):
        planted = sum(manifest.items[i] == stratum for i in fixture.changed_item_ids)
        assert evidence["v", stratum]["passed"] == (planted >= 4)


@pytest.mark.parametrize("missing", ["a1_trial", "a1_item", "a2_trial", "a2_item"])
def test_baseline_gaps_use_engine_policy_in_control_evidence(missing):
    base, a2, layer, manifest = synthetic()
    candidate = deepcopy(base)
    candidate.run_id = "candidate"
    recording = base if missing.startswith("a1") else a2
    if missing.endswith("trial"):
        recording.items[0].trials.pop()
    else:
        recording.items.pop(0)
    fixture = make_positive_control(
        base, a2=a2, manifest=manifest, layer=layer, n_items=6, mutation=lambda v: not v, seed=42
    )
    evidence = check_positive_control(base, fixture, a2=a2, layers=[layer], manifest=manifest)
    assert evidence["v", "en"]["passed"] is True
    report = compare_product(base, a2, candidate, layers=[layer], manifest=manifest,
                             control=evidence)
    assert report.status == "PASS"
    assert report.rates[0]["n_eligible"] == 20
    assert report.rates[0]["n_state_compared"] == (19 if missing.endswith("item") else 20)


@pytest.mark.parametrize("missing_count", [3, 5])
def test_baseline_gaps_cannot_bypass_quietness_or_coverage(missing_count):
    base, a2, layer, manifest = synthetic()
    a2.items = a2.items[missing_count:]
    fixture = make_positive_control(
        base, a2=a2, manifest=manifest, layer=layer, n_items=8, mutation=lambda v: not v, seed=42
    )
    evidence = check_positive_control(base, fixture, a2=a2, layers=[layer], manifest=manifest)
    assert evidence["v", "en"]["passed"] is False
    candidate = deepcopy(base)
    candidate.run_id = "candidate"
    report = compare_product(base, a2, candidate, layers=[layer], manifest=manifest,
                             control=evidence)
    assert report.status == "NOT_VERIFIED"
    assert "NOT_QUIET" in report.rates[0]["reasons"]
    if missing_count == 5:
        assert "CANDIDATE_NOT_ESTABLISHED" in report.rates[0]["reasons"]


@pytest.mark.parametrize("target", ["identity", "planted"])
@pytest.mark.parametrize("change", ["item", "trial", "duplicate_item", "duplicate_trial",
                                    "empty_item"])
def test_control_copies_must_preserve_base_record_keys(target, change):
    base, a2, layer, manifest = synthetic()
    if change == "empty_item":
        base.items[0].trials.clear()
    fixture = make_positive_control(
        base, a2=a2, manifest=manifest, layer=layer, n_items=6, mutation=lambda v: not v, seed=42
    )
    identity = deepcopy(base)
    run = identity if target == "identity" else fixture.run
    if change in {"item", "empty_item"}:
        run.items.pop(0)
    elif change == "trial":
        run.items[0].trials.pop()
    elif change == "duplicate_item":
        run.items.append(deepcopy(run.items[0]))
    else:
        run.items[0].trials.append(deepcopy(run.items[0].trials[0]))
    evidence = check_positive_control(base, fixture, a2=a2, layers=[layer], manifest=manifest,
                                      identity=identity)
    assert evidence["v", "en"]["passed"] is False


def test_stratum_controls_cover_each_cell_with_the_release_manifest():
    base, a2, layer, manifest = synthetic()
    extra = deepcopy(base.items)
    for item in extra:
        item.item_id = "he-" + item.item_id
        item.stratum = "he"
    base.items.extend(extra)
    a2.items.extend(deepcopy(extra))
    manifest.items.update({item.item_id: item.stratum for item in extra})
    evidence = {}
    for stratum in ("en", "he"):
        fixture = make_positive_control(
            base, a2=a2, manifest=manifest, layer=layer, n_items=6,
            mutation=lambda v: not v, seed=42, stratum=stratum,
        )
        assert len(fixture.changed_item_ids) == 6
        assert {manifest.items[i] for i in fixture.changed_item_ids} == {stratum}
        checked = check_positive_control(base, fixture, a2=a2, layers=[layer], manifest=manifest)
        assert checked["v", stratum]["passed"] is True
        assert checked["v", "he" if stratum == "en" else "en"]["passed"] is False
        evidence["v", stratum] = checked["v", stratum]
        reordered = deepcopy(base)
        reordered.items.reverse()
        repeated = make_positive_control(
            reordered, a2=a2, manifest=manifest, layer=layer, n_items=6,
            mutation=lambda v: not v, seed=42, stratum=stratum,
        )
        assert repeated.changed_item_ids == fixture.changed_item_ids
    candidate = deepcopy(base)
    candidate.run_id = "candidate"
    report = compare_product(base, a2, candidate, layers=[layer], manifest=manifest,
                             control=evidence)
    assert report.status == "PASS"
    assert len(report.rates) == 2


@pytest.mark.parametrize("stratum", ["unknown", "", 17])
def test_control_rejects_invalid_stratum_filter(stratum):
    base, a2, layer, manifest = synthetic()
    with pytest.raises(ValueError, match="stratum"):
        make_positive_control(base, a2=a2, manifest=manifest, layer=layer, n_items=0,
                              mutation=lambda v: not v, seed=42, stratum=stratum)


def test_stratum_control_cannot_borrow_stable_items_from_another_cell():
    base, a2, layer, manifest = synthetic()
    for item in base.items[5:]:
        item.stratum = "he"
    for item in a2.items[5:]:
        item.stratum = "he"
    manifest.items = {item.item_id: item.stratum for item in base.items}
    with pytest.raises(ValueError, match="Requested more"):
        make_positive_control(base, a2=a2, manifest=manifest, layer=layer, n_items=6,
                              mutation=lambda v: not v, seed=42, stratum="en")
