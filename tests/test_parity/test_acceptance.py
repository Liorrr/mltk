"""Stored synthetic product acceptance: no runtime model or network required."""

from __future__ import annotations

from copy import deepcopy

from mltk.parity.control import check_positive_control, make_positive_control
from mltk.parity.engine import compare_product, compare_runs
from mltk.parity.models import Item, LayerSpec, Manifest, ParityConfig, Run, Trial


def test_t7_independent_strata_and_real_control_evidence():
    layer = LayerSpec("v", oracle=lambda v: v, oracle_id="synthetic-v1")
    items = [
        Item(f"{s}-{i:02}", s, [Trial(str(j), values={"v": True}) for j in range(1, 4)])
        for s in ("en", "he")
        for i in range(20)
    ]
    a1 = Run("a1", "synthetic", items)
    a2, candidate = deepcopy(a1), deepcopy(a1)
    a2.run_id, candidate.run_id = "a2", "candidate"
    manifest = Manifest({i.item_id: i.stratum for i in items})
    fixture = make_positive_control(
        a1, a2=a2, manifest=manifest, layer=layer, n_items=20, mutation=lambda v: not v, seed=9
    )
    control = check_positive_control(a1, fixture, layers=[layer], manifest=manifest, a2=a2)
    assert {key: record["passed"] for key, record in control.items()} == {
        ("v", "en"): True,
        ("v", "he"): True,
    }
    config = ParityConfig(bh_q=0.05)
    before = compare_product(
        a1, a2, candidate, layers=[layer], manifest=manifest, config=config, control=control
    )
    assert before.status == "PASS"
    for item in candidate.items:
        if item.stratum == "he":
            for trial in item.trials:
                trial.values["v"] = False
    after = compare_product(
        a1, a2, candidate, layers=[layer], manifest=manifest, config=config, control=control
    )
    assert before.rates[0] == after.rates[0]
    assert after.rates[1]["verdict"] == "DETECTED"
    assert after.status == "FAIL"
    pair = compare_runs(a1, candidate, layers=[layer], manifest=manifest)
    assert pair.status == "FAIL"


def test_passing_preconditions_need_no_planted_precondition_control():
    layers = [
        LayerSpec("v", oracle=lambda v: v, oracle_id="synthetic-v1"),
        LayerSpec("p", role="precondition", oracle=lambda v: v, oracle_id="synthetic-v1"),
    ]
    base = Run(
        "a",
        "synthetic",
        [
            Item(str(i), "en", [Trial(str(j), values={"v": True, "p": True}) for j in range(1, 4)])
            for i in range(20)
        ],
    )
    manifest = Manifest({i.item_id: "en" for i in base.items})
    a2, candidate = deepcopy(base), deepcopy(base)
    a2.run_id, candidate.run_id = "synthetic-a2", "synthetic-candidate"
    fixture = make_positive_control(
        base, a2=a2, manifest=manifest, layer=layers[0], n_items=6, mutation=lambda v: not v, seed=1
    )
    control = check_positive_control(base, fixture, a2=a2, layers=layers, manifest=manifest)
    assert control["v", "en"]["passed"]
    report = compare_product(base, a2, candidate, layers=layers, manifest=manifest, control=control)
    assert report.status == "PASS"
