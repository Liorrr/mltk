"""Dataset registry path validation tests."""

from __future__ import annotations

import pytest

from mltk.eval._types import EvalSample
from mltk.eval.dataset import DatasetRegistry, EvalDataset


def _dataset(name="qa", version="1.0.0"):
    return EvalDataset(
        name=name,
        version=version,
        samples=[EvalSample(input="question", target="answer")],
    )


def test_valid_name_and_version_work(tmp_path):
    registry = DatasetRegistry(tmp_path / "registry")
    registry.save(_dataset("my-set_v2", "2024.01"))
    assert registry.exists("my-set_v2", "2024.01")
    assert registry.load("my-set_v2", "2024.01").version == "2024.01"
    assert registry.delete("my-set_v2", "2024.01") is True
    assert not registry.exists("my-set_v2", "2024.01")


def test_delete_rejects_traversal_and_keeps_files(tmp_path):
    root = tmp_path / "registry"
    sentinel = tmp_path / "sentinel"
    root.mkdir()
    sentinel.mkdir()
    (sentinel / "keep.txt").write_text("keep", encoding="utf-8")
    registry = DatasetRegistry(root)

    with pytest.raises(ValueError, match=r"Dataset (name|version|path)"):
        registry.delete("..", "..")
    with pytest.raises(ValueError, match=r"Dataset (name|version|path)"):
        registry.delete("x", "../../y")

    assert (sentinel / "keep.txt").read_text(encoding="utf-8") == "keep"


@pytest.mark.parametrize(
    "name", ["..", "../x", "a/b", r"a\b", "/absolute", "C:x", "", "."]
)
def test_save_rejects_invalid_name(tmp_path, name):
    registry = DatasetRegistry(tmp_path / "registry")
    with pytest.raises(ValueError, match=r"Dataset (name|version|path)"):
        registry.save(_dataset(name))


@pytest.mark.parametrize("method", ["load", "exists"])
def test_load_and_exists_reject_invalid_name(tmp_path, method):
    registry = DatasetRegistry(tmp_path / "registry")
    with pytest.raises(ValueError, match=r"Dataset (name|version|path)"):
        getattr(registry, method)("../x")
