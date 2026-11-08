"""Artifact integrity checks using explicitly synthetic recordings."""
from __future__ import annotations

import json

import pytest

from mltk.parity.loaders import load_run
from mltk.parity.models import Item, LayerSpec, Manifest, Run, Trial
from mltk.parity.output import run_to_dict


def fixture_run():
    return Run("r", "synthetic", [Item("i", "en", [Trial(str(t), values={"ids": {"b", "a"}})
                                                     for t in range(1, 4)])])


def test_roundtrip_set_and_missing_population(tmp_path):
    run = fixture_run()
    path = tmp_path / "run.json"
    path.write_text(json.dumps(run_to_dict(run)), encoding="utf-8")
    manifest = Manifest({"i": "en", "missing": "en"})
    loaded = load_run(path, manifest=manifest, layers=[LayerSpec("ids", kind="id_set")])
    assert loaded == run
    assert len(manifest.items) == 2


@pytest.mark.parametrize("text", ['{"items":[],"items":[]}', '{"x":NaN}', '{"x":Infinity}'])
def test_reject_duplicate_keys_and_nonfinite_json(tmp_path, text):
    path = tmp_path / "run.json"
    path.write_text(text)
    with pytest.raises(ValueError,
                       match="Duplicate|Nonfinite|Unsupported|wrong type|Failed|disagrees"):
        load_run(path)


@pytest.mark.parametrize("change", [
    "duplicate", "trial_duplicate", "schema", "type", "failed", "unknown", "stratum",
])
def test_reject_corrupt_record(change):
    raw = run_to_dict(fixture_run())
    item = raw["items"][0]
    if change == "duplicate":
        raw["items"].append(item)
    elif change == "trial_duplicate":
        item["trials"].append(item["trials"][0])
    elif change == "schema":
        raw["schema_version"] = "2"
    elif change == "type":
        item["trials"][0]["values"]["ids"] = True
    elif change == "failed":
        item["trials"][0]["status"] = "failed"
    elif change == "unknown":
        item["trials"][0]["trial_id"] = "4"
    else:
        item["stratum"] = "he"
    with pytest.raises(ValueError,
                       match="Duplicate|Nonfinite|Unsupported|wrong type|Failed|disagrees"):
        load_run(raw, manifest=Manifest({"i": "en"}), layers=[LayerSpec("ids", kind="id_set")])


def test_jsonl_truncation_cannot_shrink_manifest(tmp_path):
    raw = run_to_dict(fixture_run())
    item = raw.pop("items")[0]
    raw.update(expected_item_ids=["i", "j"], expected_count=2, expected_trial_ids=["1", "2", "3"])
    path = tmp_path / "r.jsonl"
    path.write_text(json.dumps(raw) + "\n" + json.dumps(item) + "\n")
    with pytest.raises(ValueError, match="expected IDs"):
        load_run(path)
    run = load_run(path, manifest=Manifest({"i": "en", "j": "en"}))
    assert len(run.items) == 1  # independent manifest retains j for reduction
    with pytest.raises(ValueError, match="header disagrees"):
        load_run(path, manifest=Manifest({"i": "en"}))


def test_jsonl_trial_identity_checked_without_external_manifest(tmp_path):
    raw = run_to_dict(fixture_run())
    item = raw.pop("items")[0]
    raw.update(expected_item_ids=["i"], expected_count=1, expected_trial_ids=["1", "2", "3"])
    item["trials"][0]["trial_id"] = "unexpected"
    path = tmp_path / "run.jsonl"
    path.write_text(json.dumps(raw) + "\n" + json.dumps(item))
    with pytest.raises(ValueError, match="trials disagree"):
        load_run(path)


@pytest.mark.parametrize(("field", "value"), [("source_id", ""), ("metadata", {"x": float("nan")})])
def test_invalid_provenance_without_layer_config(field, value):
    raw = run_to_dict(fixture_run())
    raw["items"][0][field] = value
    with pytest.raises(ValueError, match="nonempty|Out of range"):
        load_run(raw)
