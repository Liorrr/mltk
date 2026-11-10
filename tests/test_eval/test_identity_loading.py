"""File identity migration preserves explicit legacy metadata and fingerprints."""

from __future__ import annotations

import csv
import inspect
import json

import pytest

from mltk.eval import EvalTask, ExactMatchScorer, GenerateSolver, load_dataset
from mltk.eval.dataset import DatasetRegistry, EvalDataset
from mltk.eval.task import _load_csv, _load_json


def write_records(tmp_path, extension, records):
    path = tmp_path / f"samples.{extension}"
    if extension == "json":
        path.write_text(json.dumps(records), encoding="utf-8")
    else:
        with path.open("w", newline="", encoding="utf-8") as stream:
            columns = list(dict.fromkeys(key for row in records for key in row))
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(records)
    return path


def load(path, entrypoint, **kwargs):
    if entrypoint == "load_dataset":
        # Preserve all three existing positional arguments.
        return EvalDataset("identity", "1.0.0", load_dataset(path, "input", "target", **kwargs))
    loader = EvalDataset.from_csv if path.suffix == ".csv" else EvalDataset.from_json
    return loader(path, "identity", "1.0.0", **kwargs)


@pytest.mark.parametrize("extension", ["csv", "json"])
@pytest.mark.parametrize("entrypoint", ["EvalDataset", "load_dataset"])
@pytest.mark.parametrize("mode", ["default", "custom", "disabled"])
def test_identity_metadata_fingerprint_and_roundtrips(tmp_path, extension, entrypoint, mode):
    records = [
        {"input": "something", "target": "else", "sample_id": "second", "external": "two"},
        {"input": "q", "target": "a", "sample_id": "first", "external": "one"},
    ]
    path = write_records(tmp_path, extension, records)
    options = {"default": {}, "custom": {"id_column": "external"}, "disabled": {"id_column": None}}
    ds = load(path, entrypoint, **options[mode])
    expected_ids = {
        "default": ["second", "first"],
        "custom": ["two", "one"],
        "disabled": [None, None],
    }
    assert [sample.sample_id for sample in ds.samples] == expected_ids[mode]
    for sample, record in zip(ds.samples, records, strict=True):
        expected = {"sample_id": record["sample_id"], "external": record["external"]}
        if mode != "disabled":
            expected.pop("sample_id" if mode == "default" else "external")
        assert sample.metadata == expected

    # Pre-identity algorithm: sort by input, append input and nonempty target
    # UTF-8 bytes (no metadata); here SHA-256(b"qasomethingelse").
    legacy_hash = "ab853bc114a99cd04e8d82ea0d75a1774eabfa285452be3d1e95b10effee225c"
    assert load(path, entrypoint, id_column=None).fingerprint == legacy_hash
    if mode == "disabled":
        assert ds.fingerprint == legacy_hash
    else:
        assert ds.fingerprint != legacy_hash
        records[0]["sample_id" if mode == "default" else "external"] = "changed-id"
        assert (
            load(
                write_records(tmp_path, extension, records), entrypoint, **options[mode]
            ).fingerprint
            != ds.fingerprint
        )

    restored = EvalDataset.from_dict(ds.to_dict())
    assert restored.samples == ds.samples
    assert restored._compute_fingerprint() == ds.fingerprint
    registry = DatasetRegistry(registry_dir=tmp_path / "registry")
    registry.save(ds)
    registered = registry.load(ds.name, ds.version)
    assert registered.samples == ds.samples
    assert registered.fingerprint == ds.fingerprint
    assert registered._compute_fingerprint() == ds.fingerprint


@pytest.mark.parametrize("entrypoint", ["EvalDataset", "load_dataset"])
@pytest.mark.parametrize("id_column", ["sample_id", "external"])
@pytest.mark.parametrize("invalid", [0, 17, 1.5, False, True, [], {}, ["id"], {"id": "one"}])
def test_json_rejects_non_string_chosen_identity(tmp_path, entrypoint, id_column, invalid):
    path = write_records(tmp_path, "json", [{"input": "q", "target": "a", id_column: invalid}])
    with pytest.raises(ValueError, match=f"JSON item at index 0.*{id_column}.*string or null"):
        load(path, entrypoint, id_column=id_column)
    # Legacy metadata accepts arbitrary JSON values, including falsey values.
    legacy = load(path, entrypoint, id_column=None)
    assert legacy.samples[0].sample_id is None
    assert legacy.samples[0].metadata[id_column] == invalid


@pytest.mark.parametrize("entrypoint", ["EvalDataset", "load_dataset"])
@pytest.mark.parametrize("extension", ["csv", "json"])
@pytest.mark.parametrize("id_column", ["sample_id", "external"])
def test_absent_null_empty_ids_and_scoring_validation(tmp_path, entrypoint, extension, id_column):
    records = [
        {"input": "q", "target": "a", **extra} for extra in ({}, {id_column: None}, {id_column: ""})
    ]
    ds = load(write_records(tmp_path, extension, records), entrypoint, id_column=id_column)
    assert [sample.sample_id for sample in ds.samples] == [None, None, None]
    task = EvalTask("identity", GenerateSolver(), ExactMatchScorer(), ds)
    with pytest.raises(ValueError, match="non-empty string sample_id"):
        task.score({})
    assert task.run(lambda _: "a").metrics["ExactMatchScorer/mean"] == 1.0


@pytest.mark.parametrize("entrypoint", ["EvalDataset", "load_dataset"])
@pytest.mark.parametrize("extension", ["csv", "json"])
@pytest.mark.parametrize("ids", [("one", "two"), ("same", "same"), (" ", "two")])
def test_loaded_ids_preserve_stored_scoring_contract(tmp_path, entrypoint, extension, ids):
    path = write_records(
        tmp_path, extension, [{"input": "q", "target": "a", "external": item_id} for item_id in ids]
    )
    ds = load(path, entrypoint, id_column="external")
    assert [sample.sample_id for sample in ds.samples] == list(ids)
    task = EvalTask("identity", GenerateSolver(), ExactMatchScorer(), ds)
    if ids == ("one", "two"):
        assert task.score({"one": "a", "two": "a"}).metrics["ExactMatchScorer/mean"] == 1.0
        missing = task.score({"one": "a"})
        assert missing.errors["ExactMatchScorer"] == {"errored": 1, "n_valid": 1, "error_rate": 0.5}
    else:
        with pytest.raises(ValueError, match="sample_id"):
            task.score(dict.fromkeys(ids, "a"))
    assert task.run(lambda _: "a").metrics["ExactMatchScorer/mean"] == 1.0


@pytest.mark.parametrize(
    "loader", [EvalDataset.from_csv, EvalDataset.from_json, load_dataset, _load_csv, _load_json]
)
@pytest.mark.parametrize("invalid", [False, 0, [], {}])
def test_id_column_is_keyword_only_and_validated_before_read(tmp_path, loader, invalid):
    parameter = inspect.signature(loader).parameters["id_column"]
    assert parameter.kind == inspect.Parameter.KEYWORD_ONLY
    assert parameter.default == "sample_id"
    path = tmp_path / "absent.json"
    args = (path,) if loader is load_dataset else (path, "input", "target")
    with pytest.raises(ValueError, match="id_column must be a string or None"):
        loader(*args, id_column=invalid)


@pytest.mark.parametrize(("loader", "extension"), [(_load_csv, "csv"), (_load_json, "json")])
def test_private_loaders_forward_custom_and_disabled_identity(tmp_path, loader, extension):
    path = write_records(tmp_path, extension, [{"input": "q", "target": "a", "external": "id"}])
    assert loader(path, "input", "target", id_column="external")[0].sample_id == "id"
    legacy = loader(path, "input", "target", id_column=None)[0]
    assert legacy.sample_id is None
    assert legacy.metadata == {"external": "id"}


@pytest.mark.parametrize("entrypoint", ["EvalDataset", "load_dataset"])
@pytest.mark.parametrize("extension", ["csv", "json"])
def test_only_none_disables_identity_column(tmp_path, entrypoint, extension):
    path = write_records(tmp_path, extension, [{"input": "q", "target": "a", "": "id"}])
    ds = load(path, entrypoint, id_column="")
    assert ds.samples[0].sample_id == "id"
    assert "" not in ds.samples[0].metadata


def test_json_envelope_and_custom_input_keys(tmp_path):
    path = tmp_path / "samples.json"
    path.write_text(
        json.dumps({"samples": [{"prompt": "q", "answer": "a", "key": "id"}]}), encoding="utf-8"
    )
    ds = EvalDataset.from_json(
        path, "identity", "1.0.0", input_key="prompt", target_key="answer", id_column="key"
    )
    assert (ds.samples[0].input, ds.samples[0].target, ds.samples[0].sample_id) == ("q", "a", "id")
    assert ds.samples[0].metadata == {}
