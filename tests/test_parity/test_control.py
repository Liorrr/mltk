"""Synthetic controls must demonstrate their own changes without altering inputs."""

from __future__ import annotations

from copy import deepcopy

import pytest

from mltk.parity.control import check_positive_control, make_positive_control
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
