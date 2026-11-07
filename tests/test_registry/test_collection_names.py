"""Collection name validation tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from mltk.registry import load_collection, save_collection


def _source(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    source.mkdir()
    (source / "data.txt").write_text("data", encoding="utf-8")
    return source


@pytest.mark.parametrize("name", ["qa", "my-set_v2", "toxicity.v1", "1.0.0", "2024.01", "latest"])
def test_valid_names_round_trip(tmp_path, name):
    registry = tmp_path / "registry"
    target = tmp_path / "target"
    target.mkdir()
    save_collection(name, _source(tmp_path), registry_dir=registry)
    loaded = load_collection(name, target, registry_dir=registry)
    assert loaded == target / name
    assert (loaded / "data.txt").read_text(encoding="utf-8") == "data"


@pytest.mark.parametrize(
    "name", ["..", "../x", "a/b", r"a\b", "/absolute", "C:x", "", "."]
)
def test_invalid_names_rejected(tmp_path, name):
    source = _source(tmp_path)
    registry = tmp_path / "registry"
    target = tmp_path / "target"
    target.mkdir()

    with pytest.raises(ValueError, match=r"Collection name"):
        save_collection(name, source, registry_dir=registry)
    with pytest.raises(ValueError, match=r"Collection name"):
        load_collection(name, target, registry_dir=registry)


def test_load_rejects_name_escaping_target_and_keeps_files(tmp_path):
    registry = tmp_path / "registry"
    target = tmp_path / "target"
    sentinel = tmp_path / "sentinel"
    target.mkdir()
    sentinel.mkdir()
    (sentinel / "keep.txt").write_text("keep", encoding="utf-8")

    with pytest.raises(ValueError, match=r"Collection name"):
        load_collection("../sentinel", target, registry_dir=registry)

    assert (sentinel / "keep.txt").read_text(encoding="utf-8") == "keep"
