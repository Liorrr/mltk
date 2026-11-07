"""Collection name validation tests."""

from __future__ import annotations

import os
import subprocess
import sys
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
    "name", ["..", "../x", "a/b", r"a\b", "/absolute", "C:x", "", ".", "qa.", "qa "]
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


def _run_cli(*args: str, registry: Path) -> subprocess.CompletedProcess[str]:
    """Run the mltk CLI in a child interpreter (same pattern as test_cli)."""
    code = (
        "import sys; "
        f"sys.argv = ['mltk'] + {list(args)!r}; "
        "from mltk.cli.app import main; main()"
    )
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=120,
        env={**os.environ, "MLTK_REGISTRY_DIR": str(registry)},
    )


def test_cli_push_reports_invalid_name_without_traceback(tmp_path):
    source = _source(tmp_path)
    registry = tmp_path / "registry"

    bad = _run_cli("registry", "push", "../x", "--source", str(source), registry=registry)
    assert bad.returncode == 1
    assert "Error: Collection name" in bad.stdout
    assert "Traceback" not in bad.stderr

    good = _run_cli("registry", "push", "qa", "--source", str(source), registry=registry)
    assert good.returncode == 0, good.stderr
    assert "Saved collection 'qa'" in good.stdout
