"""End-to-end synthetic stored parity with real controls and artifact reloads."""
from __future__ import annotations

import copy
import json

import pytest

from mltk.core.assertion import MltkAssertionError
from mltk.parity import (
    Item,
    LayerSpec,
    Manifest,
    Run,
    Trial,
    assert_parity,
    check_positive_control,
    compare_harness,
    compare_product,
    load_run,
    make_positive_control,
    run_to_dict,
    write_report,
)


def population():
    manifest = Manifest({f"item-{i:02}": "en" for i in range(20)}, recording_hash="synthetic-v1")
    layer = LayerSpec("correct", oracle=lambda value: value, oracle_id="identity-v1")
    base = Run("a1", "synthetic baseline", [
        Item(i, "en", [Trial(t, values={"correct": True}) for t in manifest.trial_ids], "source-v1")
        for i in manifest.items
    ], refs={"recording_hash": "synthetic-v1", "build": "a"})
    return manifest, layer, base


def test_full_product_controls_serialization_and_gate(tmp_path, monkeypatch):
    import socket

    def forbidden(*args, **kwargs):
        raise AssertionError("Parity must not make network calls")

    monkeypatch.setattr(socket, "socket", forbidden)
    manifest, layer, base = population()
    a2 = copy.deepcopy(base)
    a2.run_id = "a2"
    fixture = make_positive_control(base, a2=a2, manifest=manifest, layer=layer,
                                    n_items=6, mutation=lambda v: not v, seed=42)
    evidence = check_positive_control(base, fixture, a2=a2, layers=[layer], manifest=manifest)
    assert evidence["correct", "en"]["passed"] is True
    assert evidence["correct", "en"]["a2_run_id"] == "a2"
    path = tmp_path / "recording.json"
    path.write_text(json.dumps(run_to_dict(base)), encoding="utf-8")
    reloaded = load_run(path, manifest=manifest, layers=[layer])
    reloaded.run_id = "a2"
    candidate = copy.deepcopy(base)
    candidate.run_id = "b"
    report = compare_product(base, reloaded, candidate, layers=[layer], manifest=manifest,
                             control=evidence)
    assert report.status == "PASS"
    artifacts = write_report(report, tmp_path / "first")
    assert assert_parity(report, artifacts=artifacts).passed
    reordered = copy.deepcopy(base)
    reordered.items.reverse()
    for item in reordered.items:
        item.trials.reverse()
    rerun = compare_product(reordered, reloaded, candidate, layers=[layer], manifest=manifest,
                            control=evidence)
    assert write_report(rerun, tmp_path / "second").hashes == artifacts.hashes
    changed = compare_product(base, reloaded, fixture.run, layers=[layer], manifest=manifest,
                              control=evidence)
    assert changed.status == "FAIL"
    assert changed.rates[0]["mcnemar_p"] == pytest.approx(0.03125)
    assert changed.rates[0]["nfr"] == pytest.approx(0.3)
    with pytest.raises(MltkAssertionError):
        assert_parity(changed)


def test_incomplete_aa_evidence_is_exported_as_gap(tmp_path):
    manifest, layer, base = population()
    a2 = copy.deepcopy(base)
    a2.run_id = "a2"
    a2.items[0].trials.clear()
    candidate = copy.deepcopy(base)
    candidate.run_id = "b"
    report = compare_product(base, a2, candidate, layers=[layer], manifest=manifest)
    artifacts = write_report(report, tmp_path)
    gaps = json.loads(artifacts.gaps.read_text())["items"]
    assert [row["item_id"] for row in gaps] == ["item-00"]
    assert report.rates[0]["noisy"] == 0
    assert report.rates[0]["aa_unresolved"] == 1


def test_harness_trial_change_not_hidden_by_majority():
    manifest, layer, base = population()
    candidate = copy.deepcopy(base)
    candidate.items[0].trials[0].values["correct"] = False
    report = compare_harness(base, candidate, recordings=manifest, layers=[layer])
    assert report.status == "FAIL"
    ledger = [{"item_id": "item-00", "stratum": "en", "layer": "correct", "trial_id": "1",
               "direction": "changed", "from": True, "to": False, "reason": "approved fixture"}]
    expected = compare_harness(base, candidate, recordings=manifest, layers=[layer],
                               expected_diff=ledger)
    assert expected.status == "PASS"
    stale = compare_harness(base, base, recordings=manifest, layers=[layer], expected_diff=ledger)
    assert stale.status == "FAIL"


def test_cli_discovers_parity():
    from mltk.cli._discovery import discover_assertions

    discovered = discover_assertions()
    assert any(entry["name"] == "assert_parity" for entry in discovered["Parity"])
