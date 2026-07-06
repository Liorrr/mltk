"""Tests for quality-gated dataset registration."""

from __future__ import annotations

from pathlib import Path

from mltk.eval._types import EvalSample
from mltk.eval.dataset import DatasetCard, DatasetRegistry, EvalDataset
from mltk.importer.registry import register_dataset


def _dataset(inputs, *, name="ds", version="0.1.0", source="qa.csv"):
    samples = [EvalSample(input=i, target=f"t-{i}") for i in inputs]
    card = DatasetCard(source=source)
    return EvalDataset(name=name, version=version, samples=samples, card=card)


def test_register_saves_when_gate_passes(tmp_path: Path) -> None:
    ds = _dataset(["q0", "q1", "q2"])
    result = register_dataset(ds, registry_dir=tmp_path)

    assert result.saved is True
    assert result.quality_passed is True
    assert result.path is not None
    assert Path(result.path).exists()
    # The three registry files were written.
    version_dir = tmp_path / "ds" / "0.1.0"
    assert (version_dir / "dataset.json").exists()
    assert (version_dir / "card.json").exists()
    assert (version_dir / "fingerprint.txt").exists()


def test_register_preserves_provenance(tmp_path: Path) -> None:
    ds = _dataset(["q0", "q1"], source="hf://squad")
    register_dataset(ds, registry_dir=tmp_path)

    loaded = DatasetRegistry(tmp_path).load("ds")
    assert loaded.card.source == "hf://squad"


def test_register_blocked_by_duplicate_rate(tmp_path: Path) -> None:
    # Four identical inputs -> duplicate_rate 0.75 > default 0.5 guard.
    ds = _dataset(["dup", "dup", "dup", "dup"])
    result = register_dataset(ds, registry_dir=tmp_path)

    assert result.saved is False
    assert result.quality_passed is False
    assert "duplicate_rate" in result.reason
    assert not (tmp_path / "ds").exists()


def test_register_lenient_defaults_allow_small_unlabeled(tmp_path: Path) -> None:
    # Two samples, no targets -> passes lenient import defaults.
    samples = [EvalSample(input="a"), EvalSample(input="b")]
    ds = EvalDataset(name="small", version="0.1.0", samples=samples)
    result = register_dataset(ds, registry_dir=tmp_path)
    assert result.saved is True


def test_register_tightened_threshold_blocks(tmp_path: Path) -> None:
    ds = _dataset(["q0", "q1"])
    result = register_dataset(ds, registry_dir=tmp_path, min_samples=50)
    assert result.saved is False
    assert result.quality_passed is False
    assert "sample_count" in result.reason


def test_register_existing_version_blocked_without_overwrite(
    tmp_path: Path,
) -> None:
    ds = _dataset(["q0", "q1", "q2"])
    first = register_dataset(ds, registry_dir=tmp_path)
    assert first.saved is True

    second = register_dataset(ds, registry_dir=tmp_path)
    assert second.saved is False
    assert second.quality_passed is True
    assert "already exists" in second.reason


def test_register_overwrite_replaces(tmp_path: Path) -> None:
    ds = _dataset(["q0", "q1", "q2"])
    register_dataset(ds, registry_dir=tmp_path)

    ds2 = _dataset(["q0", "q1", "q2", "q3"])
    result = register_dataset(ds2, registry_dir=tmp_path, overwrite=True)
    assert result.saved is True

    loaded = DatasetRegistry(tmp_path).load("ds")
    assert loaded.sample_count == 4
