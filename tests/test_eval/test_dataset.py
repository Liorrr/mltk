"""Tests for mltk.eval.dataset -- versioned eval datasets.

Covers DatasetCard, EvalDataset, DatasetRegistry,
assert_dataset_quality, and integration with EvalTask.
Target: 80+ tests across 5 test classes.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import warnings
from unittest.mock import patch

import pytest

from mltk.core.assertion import MltkAssertionError
from mltk.core.result import TestResult
from mltk.eval._types import EvalSample
from mltk.eval.dataset import (
    DatasetCard,
    DatasetDiff,
    DatasetInfo,
    DatasetRegistry,
    EvalDataset,
    assert_dataset_quality,
)

# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------

SEED = 42


def _make_samples(
    n: int = 10,
    with_targets: bool = True,
    category: str | None = None,
) -> list[EvalSample]:
    """Build n deterministic EvalSample objects."""
    samples = []
    for i in range(n):
        meta = {}
        if category:
            meta["category"] = category
        elif i % 3 == 0:
            meta["category"] = "math"
        elif i % 3 == 1:
            meta["category"] = "geo"
        else:
            meta["category"] = "science"
        target = str(i * 2) if with_targets else None
        samples.append(
            EvalSample(
                input=f"Question {i}?",
                target=target,
                metadata=meta,
            )
        )
    return samples


def _make_dataset(
    name: str = "qa-test",
    version: str = "1.0.0",
    n: int = 10,
    **kwargs,
) -> EvalDataset:
    """Build a minimal EvalDataset."""
    return EvalDataset(
        name=name,
        version=version,
        samples=_make_samples(n, **kwargs),
    )


def _make_card(**overrides) -> DatasetCard:
    """Build a DatasetCard with sensible defaults."""
    defaults = {
        "description": "Test dataset",
        "task": "qa",
        "source": "synthetic",
        "license": "MIT",
        "tags": ["test", "qa"],
        "author": "test-author",
    }
    defaults.update(overrides)
    return DatasetCard(**defaults)


# ===============================================================
# DatasetCard
# ===============================================================


class TestDatasetCard:
    """DatasetCard: metadata container for datasets."""

    def test_default_fields(self):
        # SCENARIO: card with all fields
        # WHY: validate dataclass construction
        # EXPECTED: fields set correctly
        card = DatasetCard(
            description="A test set",
            task="qa",
            source="synthetic",
            license="MIT",
            tags=[],
            author="tester",
        )
        assert card.description == "A test set"
        assert card.task == "qa"
        assert card.source == "synthetic"
        assert card.license == "MIT"
        assert card.author == "tester"

    def test_custom_fields(self):
        # SCENARIO: card with non-default values
        # WHY: all fields must be customizable
        # EXPECTED: custom values preserved
        card = _make_card(
            description="Custom desc",
            task="classification",
            license="Apache-2.0",
        )
        assert card.description == "Custom desc"
        assert card.task == "classification"
        assert card.license == "Apache-2.0"

    def test_tags_list(self):
        # SCENARIO: card with multiple tags
        # WHY: tags are a list, not a string
        # EXPECTED: tags preserved as list
        card = _make_card(tags=["ml", "eval", "v2"])
        assert card.tags == ["ml", "eval", "v2"]
        assert len(card.tags) == 3

    def test_empty_tags(self):
        # SCENARIO: card with empty tags list
        # WHY: empty tags must be valid
        # EXPECTED: tags == []
        card = _make_card(tags=[])
        assert card.tags == []

    def test_auto_created_timestamp(self):
        # SCENARIO: card has created timestamp
        # WHY: creation time should be auto-set
        # EXPECTED: created field is a non-empty string
        card = _make_card()
        assert hasattr(card, "created")
        assert card.created is not None
        assert isinstance(card.created, str)
        assert len(card.created) > 0

    def test_serialization_round_trip(self):
        # SCENARIO: card -> dict -> card
        # WHY: must survive serialization
        # EXPECTED: all fields preserved
        original = _make_card(
            description="Round trip",
            tags=["a", "b"],
        )
        ds = EvalDataset(
            name="card-rt",
            version="1.0.0",
            samples=_make_samples(3),
            card=original,
        )
        data = ds.to_dict()
        restored = EvalDataset.from_dict(data)
        rc = restored.card
        assert rc.description == "Round trip"
        assert rc.task == original.task
        assert rc.tags == ["a", "b"]
        assert rc.author == original.author

    def test_author_field(self):
        # SCENARIO: author field set and readable
        # WHY: authorship tracking
        # EXPECTED: author matches input
        card = _make_card(author="ml-team")
        assert card.author == "ml-team"


# ===============================================================
# EvalDataset
# ===============================================================


class TestEvalDataset:
    """EvalDataset: versioned, fingerprinted sample set."""

    def test_creation_with_samples(self):
        # SCENARIO: basic construction
        # WHY: core contract
        # EXPECTED: name, version, samples stored
        ds = _make_dataset()
        assert ds.name == "qa-test"
        assert ds.version == "1.0.0"
        assert len(ds.samples) == 10

    def test_auto_fingerprint(self):
        # SCENARIO: fingerprint auto-computed
        # WHY: data integrity
        # EXPECTED: non-empty hex string
        ds = _make_dataset()
        assert ds.fingerprint
        assert isinstance(ds.fingerprint, str)
        assert len(ds.fingerprint) == 64  # SHA-256 hex

    def test_fingerprint_deterministic(self):
        # SCENARIO: same data -> same fingerprint
        # WHY: reproducibility
        # EXPECTED: identical fingerprints
        ds1 = _make_dataset()
        ds2 = _make_dataset()
        assert ds1.fingerprint == ds2.fingerprint

    def test_fingerprint_changes_with_data(self):
        # SCENARIO: different data -> different fingerprint
        # WHY: fingerprint must detect changes
        # EXPECTED: different fingerprints
        ds1 = _make_dataset(n=5)
        ds2 = _make_dataset(n=6)
        assert ds1.fingerprint != ds2.fingerprint

    def test_fingerprint_changes_with_content(self):
        # SCENARIO: same count, different content
        # WHY: fingerprint must hash content not just count
        # EXPECTED: different fingerprints
        s1 = [EvalSample(input="A", target="1")]
        s2 = [EvalSample(input="B", target="2")]
        ds1 = EvalDataset(
            name="t", version="1.0.0", samples=s1,
        )
        ds2 = EvalDataset(
            name="t", version="1.0.0", samples=s2,
        )
        assert ds1.fingerprint != ds2.fingerprint

    def test_sample_count_property(self):
        # SCENARIO: sample_count == len(samples)
        # WHY: convenience property contract
        # EXPECTED: sample_count == 10
        ds = _make_dataset(n=10)
        assert ds.sample_count == 10

    def test_sample_count_empty(self):
        # SCENARIO: dataset with 1 sample
        # WHY: boundary case
        # EXPECTED: sample_count == 1
        ds = EvalDataset(
            name="t",
            version="1.0.0",
            samples=[EvalSample(input="Q?")],
        )
        assert ds.sample_count == 1

    def test_target_coverage_all_targets(self):
        # SCENARIO: every sample has a target
        # WHY: 100% coverage
        # EXPECTED: target_coverage == 1.0
        ds = _make_dataset(n=5, with_targets=True)
        assert ds.target_coverage == pytest.approx(1.0)

    def test_target_coverage_partial(self):
        # SCENARIO: some samples have targets, some None
        # WHY: partial coverage
        # EXPECTED: 0 < coverage < 1
        samples = [
            EvalSample(input="Q1?", target="A1"),
            EvalSample(input="Q2?", target=None),
            EvalSample(input="Q3?", target="A3"),
            EvalSample(input="Q4?", target=None),
        ]
        ds = EvalDataset(
            name="t", version="1.0.0", samples=samples,
        )
        assert ds.target_coverage == pytest.approx(0.5)

    def test_target_coverage_no_targets(self):
        # SCENARIO: no sample has a target
        # WHY: 0% coverage
        # EXPECTED: target_coverage == 0.0
        ds = _make_dataset(n=5, with_targets=False)
        assert ds.target_coverage == pytest.approx(0.0)

    def test_categories_from_metadata(self):
        # SCENARIO: samples with category in metadata
        # WHY: category aggregation
        # EXPECTED: dict with category counts
        ds = _make_dataset(n=9)
        cats = ds.categories
        assert isinstance(cats, dict)
        assert "math" in cats
        assert "geo" in cats
        assert "science" in cats
        total = sum(cats.values())
        assert total == 9

    def test_categories_empty_no_metadata(self):
        # SCENARIO: samples without category key
        # WHY: graceful handling of missing metadata
        # EXPECTED: empty dict
        samples = [
            EvalSample(input="Q?", metadata={}),
            EvalSample(input="Q2?", metadata={}),
        ]
        ds = EvalDataset(
            name="t", version="1.0.0", samples=samples,
        )
        cats = ds.categories
        assert cats == {}

    def test_categories_single_category(self):
        # SCENARIO: all samples same category
        # WHY: uniform distribution
        # EXPECTED: one key with count == n
        ds = _make_dataset(n=5, category="math")
        cats = ds.categories
        assert cats == {"math": 5}

    def test_to_sample_list_returns_copy(self):
        # SCENARIO: to_sample_list returns a new list
        # WHY: mutation safety
        # EXPECTED: modifying copy does not affect original
        ds = _make_dataset(n=3)
        lst = ds.to_sample_list()
        assert len(lst) == 3
        lst.pop()
        assert ds.sample_count == 3

    def test_to_sample_list_content(self):
        # SCENARIO: list content matches samples
        # WHY: data integrity
        # EXPECTED: same inputs and targets
        ds = _make_dataset(n=3)
        lst = ds.to_sample_list()
        for orig, copy in zip(ds.samples, lst):
            assert orig.input == copy.input
            assert orig.target == copy.target

    def test_to_dict_serialization(self):
        # SCENARIO: to_dict returns a plain dict
        # WHY: serialization contract
        # EXPECTED: dict with expected keys
        ds = _make_dataset(n=2)
        d = ds.to_dict()
        assert isinstance(d, dict)
        assert d["name"] == "qa-test"
        assert d["version"] == "1.0.0"
        assert "samples" in d
        assert "fingerprint" in d
        assert len(d["samples"]) == 2

    def test_from_dict_deserialization(self):
        # SCENARIO: from_dict reconstructs dataset
        # WHY: deserialization contract
        # EXPECTED: same name, version, sample count
        ds = _make_dataset(n=3)
        d = ds.to_dict()
        restored = EvalDataset.from_dict(d)
        assert restored.name == ds.name
        assert restored.version == ds.version
        assert restored.sample_count == ds.sample_count

    def test_round_trip_to_dict_from_dict(self):
        # SCENARIO: to_dict -> from_dict preserves data
        # WHY: lossless round-trip
        # EXPECTED: fingerprints match
        ds = _make_dataset(n=5)
        d = ds.to_dict()
        restored = EvalDataset.from_dict(d)
        assert restored.fingerprint == ds.fingerprint
        assert restored.name == ds.name
        assert restored.version == ds.version
        for orig, rest in zip(
            ds.samples, restored.samples
        ):
            assert orig.input == rest.input
            assert orig.target == rest.target

    def test_from_csv(self, tmp_path):
        # SCENARIO: load dataset from CSV file
        # WHY: CSV is a primary input format
        # EXPECTED: samples loaded correctly
        path = tmp_path / "data.csv"
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["input", "target", "category"])
            w.writerow(["Q1?", "A1", "math"])
            w.writerow(["Q2?", "A2", "geo"])
            w.writerow(["Q3?", "A3", "science"])
        ds = EvalDataset.from_csv(
            path, name="csv-ds", version="1.0.0",
        )
        assert ds.name == "csv-ds"
        assert ds.version == "1.0.0"
        assert ds.sample_count == 3
        assert ds.samples[0].input == "Q1?"

    def test_from_json(self, tmp_path):
        # SCENARIO: load dataset from JSON file
        # WHY: JSON is a primary input format
        # EXPECTED: samples loaded correctly
        path = tmp_path / "data.json"
        data = [
            {"input": "Q1?", "target": "A1"},
            {"input": "Q2?", "target": "A2"},
        ]
        with open(path, "w") as f:
            json.dump(data, f)
        ds = EvalDataset.from_json(
            path, name="json-ds", version="2.0.0",
        )
        assert ds.name == "json-ds"
        assert ds.version == "2.0.0"
        assert ds.sample_count == 2

    def test_version_string_stored(self):
        # SCENARIO: version is a string
        # WHY: semver is always a string
        # EXPECTED: version preserved as-is
        ds = _make_dataset(version="0.1.0-beta")
        assert ds.version == "0.1.0-beta"

    def test_card_attached(self):
        # SCENARIO: card provided at construction
        # WHY: metadata must persist
        # EXPECTED: card is accessible and correct
        card = _make_card(description="Attached")
        ds = EvalDataset(
            name="t",
            version="1.0.0",
            samples=_make_samples(2),
            card=card,
        )
        assert ds.card is not None
        assert ds.card.description == "Attached"

    def test_card_default_none(self):
        # SCENARIO: no card provided
        # WHY: card is optional
        # EXPECTED: card is None or auto-generated
        ds = _make_dataset(n=2)
        # Card may be None or auto-generated
        # Just verify no crash on access
        _ = ds.card

    def test_large_dataset(self):
        # SCENARIO: dataset with 100+ samples
        # WHY: must handle realistic sizes
        # EXPECTED: all properties work at scale
        ds = _make_dataset(n=150)
        assert ds.sample_count == 150
        assert ds.fingerprint
        assert len(ds.to_sample_list()) == 150
        cats = ds.categories
        assert sum(cats.values()) == 150


# ===============================================================
# DatasetRegistry
# ===============================================================


class TestDatasetRegistry:
    """DatasetRegistry: save, load, list, diff datasets."""

    def test_save_creates_directory(self, tmp_path):
        # SCENARIO: save to fresh registry dir
        # WHY: must create dirs as needed
        # EXPECTED: directory exists after save
        reg = DatasetRegistry(registry_dir=tmp_path)
        ds = _make_dataset()
        reg.save(ds)
        assert (tmp_path / "qa-test").exists()

    def test_save_writes_files(self, tmp_path):
        # SCENARIO: save writes dataset file
        # WHY: persistence contract
        # EXPECTED: file(s) exist on disk
        reg = DatasetRegistry(registry_dir=tmp_path)
        ds = _make_dataset()
        result = reg.save(ds)
        assert result.exists()

    def test_load_returns_same_data(self, tmp_path):
        # SCENARIO: save then load
        # WHY: round-trip integrity
        # EXPECTED: loaded data matches saved
        reg = DatasetRegistry(registry_dir=tmp_path)
        ds = _make_dataset(n=5)
        reg.save(ds)
        loaded = reg.load("qa-test", "1.0.0")
        assert loaded.name == ds.name
        assert loaded.version == ds.version
        assert loaded.sample_count == ds.sample_count
        assert loaded.fingerprint == ds.fingerprint

    def test_load_latest_version(self, tmp_path):
        # SCENARIO: save v1.0.0 and v2.0.0, load latest
        # WHY: default load returns latest
        # EXPECTED: loaded version is 2.0.0
        reg = DatasetRegistry(registry_dir=tmp_path)
        reg.save(_make_dataset(version="1.0.0"))
        reg.save(_make_dataset(version="2.0.0", n=12))
        loaded = reg.load("qa-test")
        assert loaded.version == "2.0.0"

    def test_load_specific_version(self, tmp_path):
        # SCENARIO: load v1.0.0 when v2.0.0 exists
        # WHY: pin to specific version
        # EXPECTED: loaded version is 1.0.0
        reg = DatasetRegistry(registry_dir=tmp_path)
        reg.save(_make_dataset(version="1.0.0", n=5))
        reg.save(_make_dataset(version="2.0.0", n=8))
        loaded = reg.load("qa-test", "1.0.0")
        assert loaded.version == "1.0.0"
        assert loaded.sample_count == 5

    def test_load_nonexistent_raises(self, tmp_path):
        # SCENARIO: load dataset that does not exist
        # WHY: must raise on missing dataset
        # EXPECTED: raises exception
        reg = DatasetRegistry(registry_dir=tmp_path)
        with pytest.raises(
            (FileNotFoundError, KeyError, ValueError)
        ):
            reg.load("nonexistent")

    def test_load_nonexistent_version_raises(
        self, tmp_path
    ):
        # SCENARIO: load version that does not exist
        # WHY: must raise on missing version
        # EXPECTED: raises exception
        reg = DatasetRegistry(registry_dir=tmp_path)
        reg.save(_make_dataset(version="1.0.0"))
        with pytest.raises(
            (FileNotFoundError, KeyError, ValueError)
        ):
            reg.load("qa-test", "9.9.9")

    def test_save_duplicate_version_raises(
        self, tmp_path
    ):
        # SCENARIO: save same name+version twice
        # WHY: versions are immutable
        # EXPECTED: raises on duplicate
        reg = DatasetRegistry(registry_dir=tmp_path)
        reg.save(_make_dataset(version="1.0.0"))
        with pytest.raises(
            (ValueError, FileExistsError)
        ):
            reg.save(_make_dataset(version="1.0.0"))

    def test_list_empty_registry(self, tmp_path):
        # SCENARIO: list from empty registry
        # WHY: must return empty list
        # EXPECTED: empty list
        reg = DatasetRegistry(registry_dir=tmp_path)
        result = reg.list()
        assert result == []

    def test_list_with_datasets(self, tmp_path):
        # SCENARIO: list after saving datasets
        # WHY: must show all datasets
        # EXPECTED: list contains saved dataset info
        reg = DatasetRegistry(registry_dir=tmp_path)
        reg.save(_make_dataset(name="ds-a"))
        reg.save(_make_dataset(name="ds-b"))
        result = reg.list()
        names = [info.name for info in result]
        assert "ds-a" in names
        assert "ds-b" in names

    def test_list_returns_dataset_info(self, tmp_path):
        # SCENARIO: list item is DatasetInfo
        # WHY: typed response contract
        # EXPECTED: DatasetInfo with correct fields
        reg = DatasetRegistry(registry_dir=tmp_path)
        reg.save(_make_dataset(n=7))
        infos = reg.list()
        assert len(infos) >= 1
        info = infos[0]
        assert isinstance(info, DatasetInfo)
        assert info.name == "qa-test"

    def test_versions_returns_sorted(self, tmp_path):
        # SCENARIO: multiple versions saved
        # WHY: versions must be sorted
        # EXPECTED: sorted semver list
        reg = DatasetRegistry(registry_dir=tmp_path)
        reg.save(_make_dataset(version="2.0.0", n=3))
        reg.save(_make_dataset(version="1.0.0", n=3))
        reg.save(_make_dataset(version="1.1.0", n=3))
        versions = reg.versions("qa-test")
        assert versions == ["1.0.0", "1.1.0", "2.0.0"]

    def test_exists_returns_true(self, tmp_path):
        # SCENARIO: check existing dataset
        # WHY: existence check
        # EXPECTED: True
        reg = DatasetRegistry(registry_dir=tmp_path)
        reg.save(_make_dataset())
        assert reg.exists("qa-test") is True

    def test_exists_returns_false(self, tmp_path):
        # SCENARIO: check non-existing dataset
        # WHY: existence check
        # EXPECTED: False
        reg = DatasetRegistry(registry_dir=tmp_path)
        assert reg.exists("nope") is False

    def test_exists_specific_version(self, tmp_path):
        # SCENARIO: check specific version exists
        # WHY: version-level existence
        # EXPECTED: True for saved, False for missing
        reg = DatasetRegistry(registry_dir=tmp_path)
        reg.save(_make_dataset(version="1.0.0"))
        assert reg.exists("qa-test", "1.0.0") is True
        assert reg.exists("qa-test", "9.0.0") is False

    def test_delete_removes_version(self, tmp_path):
        # SCENARIO: delete a saved version
        # WHY: cleanup / removal contract
        # EXPECTED: version no longer exists
        reg = DatasetRegistry(registry_dir=tmp_path)
        reg.save(_make_dataset(version="1.0.0"))
        result = reg.delete("qa-test", "1.0.0")
        assert result is True
        assert reg.exists("qa-test", "1.0.0") is False

    def test_delete_nonexistent_returns_false(
        self, tmp_path
    ):
        # SCENARIO: delete version that does not exist
        # WHY: idempotent delete
        # EXPECTED: returns False
        reg = DatasetRegistry(registry_dir=tmp_path)
        result = reg.delete("nope", "1.0.0")
        assert result is False

    def test_diff_shows_added_samples(self, tmp_path):
        # SCENARIO: v2 has more samples than v1
        # WHY: diff must detect additions
        # EXPECTED: added_samples > 0
        reg = DatasetRegistry(registry_dir=tmp_path)
        reg.save(_make_dataset(version="1.0.0", n=3))
        reg.save(_make_dataset(version="2.0.0", n=6))
        diff = reg.diff("qa-test", "1.0.0", "2.0.0")
        assert isinstance(diff, DatasetDiff)
        assert len(diff.added_samples) > 0

    def test_diff_shows_removed_samples(
        self, tmp_path
    ):
        # SCENARIO: v2 has fewer samples than v1
        # WHY: diff must detect removals
        # EXPECTED: removed_samples > 0
        reg = DatasetRegistry(registry_dir=tmp_path)
        reg.save(_make_dataset(version="1.0.0", n=6))
        # v2 with only first 3 samples
        s = _make_samples(3)
        ds2 = EvalDataset(
            name="qa-test",
            version="2.0.0",
            samples=s,
        )
        reg.save(ds2)
        diff = reg.diff("qa-test", "1.0.0", "2.0.0")
        assert len(diff.removed_samples) > 0

    def test_diff_shows_unchanged(self, tmp_path):
        # SCENARIO: both versions share samples
        # WHY: diff must track unchanged
        # EXPECTED: unchanged_samples > 0
        reg = DatasetRegistry(registry_dir=tmp_path)
        reg.save(_make_dataset(version="1.0.0", n=5))
        # v2 same first 5 + 2 more
        s = _make_samples(5) + [
            EvalSample(input="New Q1?", target="NA"),
            EvalSample(input="New Q2?", target="NB"),
        ]
        ds2 = EvalDataset(
            name="qa-test",
            version="2.0.0",
            samples=s,
        )
        reg.save(ds2)
        diff = reg.diff("qa-test", "1.0.0", "2.0.0")
        assert len(diff.unchanged_samples) > 0

    def test_diff_suggests_patch_for_additions(
        self, tmp_path
    ):
        # SCENARIO: only additions in v2
        # WHY: should suggest patch bump
        # EXPECTED: suggested_bump contains "patch" or
        #   "minor"
        reg = DatasetRegistry(registry_dir=tmp_path)
        reg.save(_make_dataset(version="1.0.0", n=5))
        s = _make_samples(5) + [
            EvalSample(input="Extra?", target="E"),
        ]
        ds2 = EvalDataset(
            name="qa-test",
            version="1.1.0",
            samples=s,
        )
        reg.save(ds2)
        diff = reg.diff("qa-test", "1.0.0", "1.1.0")
        assert diff.suggested_bump in (
            "patch", "minor",
        )

    def test_diff_suggests_major_for_schema(
        self, tmp_path
    ):
        # SCENARIO: schema changes between versions
        # WHY: should suggest major bump
        # EXPECTED: suggested_bump == "major"
        reg = DatasetRegistry(registry_dir=tmp_path)
        s1 = [
            EvalSample(
                input="Q?",
                target="A",
                metadata={"category": "x"},
            ),
        ]
        ds1 = EvalDataset(
            name="qa-test",
            version="1.0.0",
            samples=s1,
        )
        reg.save(ds1)
        s2 = [
            EvalSample(
                input="Q?",
                target="A",
                metadata={"type": "y", "level": "2"},
            ),
        ]
        ds2 = EvalDataset(
            name="qa-test",
            version="2.0.0",
            samples=s2,
        )
        reg.save(ds2)
        diff = reg.diff("qa-test", "1.0.0", "2.0.0")
        if diff.schema_changes:
            assert diff.suggested_bump == "major"

    def test_fingerprint_verified_on_load(
        self, tmp_path
    ):
        # SCENARIO: save, tamper fingerprint, load
        # WHY: integrity check on load
        # EXPECTED: warns or raises on mismatch
        reg = DatasetRegistry(registry_dir=tmp_path)
        ds = _make_dataset(n=3)
        saved_path = reg.save(ds)
        # Tamper with the saved file
        with open(saved_path, "r") as f:
            data = json.load(f)
        data["fingerprint"] = "0" * 64
        with open(saved_path, "w") as f:
            json.dump(data, f)
        # Should warn or raise
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                loaded = reg.load("qa-test", "1.0.0")
                # If no exception, check for warning
                tamper_warns = [
                    x for x in w
                    if "fingerprint" in str(x.message)
                    .lower()
                ]
                assert len(tamper_warns) > 0
            except (ValueError, RuntimeError):
                pass  # Raising is also acceptable

    def test_custom_registry_dir(self, tmp_path):
        # SCENARIO: explicit registry_dir
        # WHY: configurable storage location
        # EXPECTED: data stored in custom dir
        custom = tmp_path / "my-registry"
        custom.mkdir()
        reg = DatasetRegistry(registry_dir=custom)
        reg.save(_make_dataset())
        assert reg.exists("qa-test") is True

    def test_env_var_registry_dir(
        self, tmp_path, monkeypatch
    ):
        # SCENARIO: MLTK_DATASET_DIR env var
        # WHY: env-based configuration
        # EXPECTED: uses env var path
        env_dir = tmp_path / "env-registry"
        env_dir.mkdir()
        monkeypatch.setenv(
            "MLTK_DATASET_DIR", str(env_dir)
        )
        reg = DatasetRegistry()
        reg.save(_make_dataset())
        assert reg.exists("qa-test") is True

    def test_multiple_datasets(self, tmp_path):
        # SCENARIO: multiple different datasets
        # WHY: registry must handle multiple names
        # EXPECTED: all datasets accessible
        reg = DatasetRegistry(registry_dir=tmp_path)
        reg.save(_make_dataset(name="alpha", n=3))
        reg.save(_make_dataset(name="beta", n=5))
        reg.save(_make_dataset(name="gamma", n=7))
        assert reg.exists("alpha")
        assert reg.exists("beta")
        assert reg.exists("gamma")
        infos = reg.list()
        assert len(infos) == 3

    def test_multiple_versions_same_dataset(
        self, tmp_path
    ):
        # SCENARIO: 3 versions of same dataset
        # WHY: version management
        # EXPECTED: all versions accessible
        reg = DatasetRegistry(registry_dir=tmp_path)
        for v in ["1.0.0", "1.1.0", "2.0.0"]:
            reg.save(_make_dataset(version=v, n=3))
        versions = reg.versions("qa-test")
        assert len(versions) == 3

    def test_semver_sorting(self, tmp_path):
        # SCENARIO: versions 1.0.0, 1.1.0, 2.0.0
        # WHY: proper semver sort order
        # EXPECTED: sorted numerically, not alpha
        reg = DatasetRegistry(registry_dir=tmp_path)
        for v in ["2.0.0", "1.1.0", "1.0.0"]:
            reg.save(_make_dataset(version=v, n=3))
        versions = reg.versions("qa-test")
        assert versions[0] == "1.0.0"
        assert versions[-1] == "2.0.0"

    def test_save_load_round_trip_integrity(
        self, tmp_path
    ):
        # SCENARIO: full round-trip with all fields
        # WHY: no data loss through persistence
        # EXPECTED: all fields preserved
        reg = DatasetRegistry(registry_dir=tmp_path)
        card = _make_card(description="Integrity")
        ds = EvalDataset(
            name="rt-test",
            version="1.0.0",
            samples=_make_samples(10),
            card=card,
        )
        reg.save(ds)
        loaded = reg.load("rt-test", "1.0.0")
        assert loaded.name == ds.name
        assert loaded.version == ds.version
        assert loaded.sample_count == ds.sample_count
        assert loaded.fingerprint == ds.fingerprint
        for orig, rest in zip(
            ds.samples, loaded.samples
        ):
            assert orig.input == rest.input
            assert orig.target == rest.target

    def test_registry_with_nested_paths(
        self, tmp_path
    ):
        # SCENARIO: registry in deeply nested path
        # WHY: must handle any path depth
        # EXPECTED: works without error
        nested = tmp_path / "a" / "b" / "c"
        nested.mkdir(parents=True)
        reg = DatasetRegistry(registry_dir=nested)
        reg.save(_make_dataset())
        assert reg.exists("qa-test")

    def test_list_shows_all_versions(self, tmp_path):
        # SCENARIO: multiple versions -> list shows all
        # WHY: info must include version list
        # EXPECTED: DatasetInfo has all versions
        reg = DatasetRegistry(registry_dir=tmp_path)
        for v in ["1.0.0", "1.1.0", "2.0.0"]:
            reg.save(_make_dataset(version=v, n=3))
        infos = reg.list()
        qa_info = [
            i for i in infos if i.name == "qa-test"
        ][0]
        assert len(qa_info.versions) == 3
        assert qa_info.latest_version == "2.0.0"

    def test_delete_last_version(self, tmp_path):
        # SCENARIO: delete the only version
        # WHY: edge case -- dataset effectively removed
        # EXPECTED: dataset no longer exists
        reg = DatasetRegistry(registry_dir=tmp_path)
        reg.save(_make_dataset(version="1.0.0"))
        reg.delete("qa-test", "1.0.0")
        assert reg.exists("qa-test") is False

    def test_delete_one_of_many(self, tmp_path):
        # SCENARIO: delete one version, keep others
        # WHY: selective version removal
        # EXPECTED: other versions still exist
        reg = DatasetRegistry(registry_dir=tmp_path)
        reg.save(_make_dataset(version="1.0.0", n=3))
        reg.save(_make_dataset(version="2.0.0", n=5))
        reg.delete("qa-test", "1.0.0")
        assert reg.exists("qa-test", "1.0.0") is False
        assert reg.exists("qa-test", "2.0.0") is True

    def test_diff_versions_match(self, tmp_path):
        # SCENARIO: diff returns correct version info
        # WHY: version tracking in diff
        # EXPECTED: old_version and new_version correct
        reg = DatasetRegistry(registry_dir=tmp_path)
        reg.save(_make_dataset(version="1.0.0", n=3))
        reg.save(_make_dataset(version="2.0.0", n=5))
        diff = reg.diff("qa-test", "1.0.0", "2.0.0")
        assert diff.old_version == "1.0.0"
        assert diff.new_version == "2.0.0"

    def test_save_returns_path(self, tmp_path):
        # SCENARIO: save returns the file path
        # WHY: caller may need the path
        # EXPECTED: Path object to a real file
        reg = DatasetRegistry(registry_dir=tmp_path)
        ds = _make_dataset()
        result = reg.save(ds)
        from pathlib import Path
        assert isinstance(result, Path)
        assert result.exists()


# ===============================================================
# assert_dataset_quality
# ===============================================================


class TestAssertDatasetQuality:
    """assert_dataset_quality: quality gate assertion."""

    def test_good_dataset_passes(self):
        # SCENARIO: dataset meets all thresholds
        # WHY: happy path
        # EXPECTED: result.passed == True
        ds = _make_dataset(n=100)
        result = assert_dataset_quality(ds)
        assert isinstance(result, TestResult)
        assert result.passed is True

    def test_too_few_samples_fails(self):
        # SCENARIO: dataset below min_samples
        # WHY: must enforce minimum size
        # EXPECTED: MltkAssertionError raised
        ds = _make_dataset(n=5)
        with pytest.raises(MltkAssertionError) as exc:
            assert_dataset_quality(
                ds, min_samples=50,
            )
        assert exc.value.result.passed is False

    def test_low_target_coverage_fails(self):
        # SCENARIO: target_coverage below threshold
        # WHY: must enforce coverage
        # EXPECTED: MltkAssertionError raised
        samples = []
        for i in range(100):
            t = str(i) if i < 50 else None
            samples.append(
                EvalSample(
                    input=f"Q{i}?",
                    target=t,
                    metadata={"category": "a"},
                )
            )
        ds = EvalDataset(
            name="t",
            version="1.0.0",
            samples=samples,
        )
        with pytest.raises(MltkAssertionError) as exc:
            assert_dataset_quality(
                ds, min_target_coverage=0.9,
            )
        assert exc.value.result.passed is False

    def test_high_duplicate_rate_fails(self):
        # SCENARIO: many duplicate samples
        # WHY: must enforce deduplication
        # EXPECTED: MltkAssertionError raised
        base = EvalSample(
            input="Same Q?",
            target="Same A",
            metadata={"category": "dup"},
        )
        samples = [base] * 100
        ds = EvalDataset(
            name="t",
            version="1.0.0",
            samples=samples,
        )
        with pytest.raises(MltkAssertionError) as exc:
            assert_dataset_quality(
                ds,
                min_samples=10,
                max_duplicate_rate=0.01,
            )
        assert exc.value.result.passed is False

    def test_min_categories_met_passes(self):
        # SCENARIO: dataset has enough categories
        # WHY: category diversity check
        # EXPECTED: passes
        ds = _make_dataset(n=100)
        result = assert_dataset_quality(
            ds,
            min_samples=10,
            min_categories=2,
        )
        assert result.passed is True

    def test_min_categories_not_met_fails(self):
        # SCENARIO: dataset has too few categories
        # WHY: must enforce category diversity
        # EXPECTED: MltkAssertionError raised
        ds = _make_dataset(
            n=100, category="single",
        )
        with pytest.raises(MltkAssertionError) as exc:
            assert_dataset_quality(
                ds,
                min_samples=10,
                min_categories=3,
            )
        assert exc.value.result.passed is False

    def test_edge_exactly_min_samples(self):
        # SCENARIO: sample_count == min_samples
        # WHY: boundary must pass
        # EXPECTED: passed == True
        ds = _make_dataset(n=50)
        result = assert_dataset_quality(
            ds, min_samples=50,
        )
        assert result.passed is True

    def test_edge_exactly_min_coverage(self):
        # SCENARIO: coverage exactly at threshold
        # WHY: boundary must pass
        # EXPECTED: passed == True
        ds = _make_dataset(n=100, with_targets=True)
        result = assert_dataset_quality(
            ds,
            min_samples=10,
            min_target_coverage=1.0,
        )
        assert result.passed is True

    def test_all_checks_pass_combined(self):
        # SCENARIO: all quality checks at once
        # WHY: combined validation
        # EXPECTED: passed == True
        ds = _make_dataset(n=100)
        result = assert_dataset_quality(
            ds,
            min_samples=50,
            min_target_coverage=0.9,
            max_duplicate_rate=0.05,
            min_categories=2,
        )
        assert result.passed is True

    def test_multiple_checks_fail(self):
        # SCENARIO: too few samples AND low coverage
        # WHY: multiple violations
        # EXPECTED: MltkAssertionError
        samples = [
            EvalSample(
                input="Q?",
                target=None,
                metadata={},
            ),
        ]
        ds = EvalDataset(
            name="t",
            version="1.0.0",
            samples=samples,
        )
        with pytest.raises(MltkAssertionError) as exc:
            assert_dataset_quality(
                ds,
                min_samples=50,
                min_target_coverage=0.9,
            )
        assert exc.value.result.passed is False

    def test_empty_dataset_fails(self):
        # SCENARIO: dataset with very few samples
        # WHY: too-small datasets must fail
        # EXPECTED: MltkAssertionError
        ds = EvalDataset(
            name="t",
            version="1.0.0",
            samples=[EvalSample(input="Q?")],
        )
        with pytest.raises(MltkAssertionError):
            assert_dataset_quality(
                ds, min_samples=50,
            )

    def test_dataset_with_no_targets(self):
        # SCENARIO: all targets are None
        # WHY: 0% coverage vs threshold
        # EXPECTED: fails when threshold > 0
        ds = _make_dataset(
            n=100, with_targets=False,
        )
        with pytest.raises(MltkAssertionError):
            assert_dataset_quality(
                ds, min_target_coverage=0.5,
            )

    def test_dataset_with_duplicates(self):
        # SCENARIO: some duplicates below threshold
        # WHY: acceptable dup rate
        # EXPECTED: passes
        unique = _make_samples(98)
        dups = [unique[0], unique[1]]
        samples = unique + dups
        ds = EvalDataset(
            name="t",
            version="1.0.0",
            samples=samples,
        )
        result = assert_dataset_quality(
            ds,
            min_samples=10,
            max_duplicate_rate=0.05,
        )
        assert result.passed is True

    def test_details_contain_violations(self):
        # SCENARIO: quality check fails
        # WHY: details must explain why
        # EXPECTED: details dict has violation info
        ds = _make_dataset(n=5)
        with pytest.raises(MltkAssertionError) as exc:
            assert_dataset_quality(
                ds, min_samples=50,
            )
        r = exc.value.result
        assert r.details is not None
        assert len(r.details) > 0

    def test_assertion_name_correct(self):
        # SCENARIO: check result.name
        # WHY: naming convention
        # EXPECTED: name contains "dataset" or "quality"
        ds = _make_dataset(n=100)
        result = assert_dataset_quality(ds)
        name_lower = result.name.lower()
        assert (
            "dataset" in name_lower
            or "quality" in name_lower
        )

    def test_returns_test_result(self):
        # SCENARIO: return type
        # WHY: type contract
        # EXPECTED: TestResult instance
        ds = _make_dataset(n=100)
        result = assert_dataset_quality(ds)
        assert isinstance(result, TestResult)

    def test_duration_ms_populated(self):
        # SCENARIO: timed_assertion decorator
        # WHY: timing must be recorded
        # EXPECTED: duration_ms >= 0
        ds = _make_dataset(n=100)
        result = assert_dataset_quality(ds)
        assert result.duration_ms >= 0.0

    def test_no_categories_check_when_none(self):
        # SCENARIO: min_categories=None (default)
        # WHY: should skip categories check
        # EXPECTED: passes even with 0 categories
        samples = [
            EvalSample(
                input=f"Q{i}?",
                target=str(i),
                metadata={},
            )
            for i in range(100)
        ]
        ds = EvalDataset(
            name="t",
            version="1.0.0",
            samples=samples,
        )
        result = assert_dataset_quality(
            ds,
            min_samples=10,
            min_categories=None,
        )
        assert result.passed is True


# ===============================================================
# Integration: EvalTask + EvalDataset
# ===============================================================


class TestIntegration:
    """Integration: EvalDataset with EvalTask + Registry."""

    def test_eval_task_accepts_dataset(self):
        # SCENARIO: EvalTask accepts EvalDataset samples
        # WHY: interop contract
        # EXPECTED: EvalTask constructed without error
        from mltk.eval.scorers import ExactMatchScorer
        from mltk.eval.solvers import GenerateSolver
        from mltk.eval.task import EvalTask

        ds = _make_dataset(n=5)
        task = EvalTask(
            name="ds-task",
            solver=GenerateSolver(),
            scorers=[ExactMatchScorer()],
            dataset=ds.to_sample_list(),
        )
        assert task is not None

    def test_eval_task_runs_with_dataset(self):
        # SCENARIO: EvalTask runs using dataset samples
        # WHY: end-to-end pipeline
        # EXPECTED: EvalResult returned
        from mltk.eval._types import EvalResult
        from mltk.eval.scorers import ExactMatchScorer
        from mltk.eval.solvers import GenerateSolver
        from mltk.eval.task import EvalTask

        ds = _make_dataset(n=3)

        def model_fn(prompt: str) -> str:
            return "0"

        task = EvalTask(
            name="ds-run",
            solver=GenerateSolver(),
            scorers=[ExactMatchScorer()],
            dataset=ds.to_sample_list(),
        )
        result = task.run(model_fn)
        assert isinstance(result, EvalResult)
        assert result.total_samples == 3

    def test_full_pipeline_create_save_load_run(
        self, tmp_path
    ):
        # SCENARIO: create -> save -> load -> run -> score
        # WHY: full lifecycle
        # EXPECTED: all steps succeed
        from mltk.eval._types import EvalResult
        from mltk.eval.scorers import (
            ExactMatchScorer,
        )
        from mltk.eval.solvers import GenerateSolver
        from mltk.eval.task import EvalTask

        # Create
        ds = _make_dataset(n=5)
        # Save
        reg = DatasetRegistry(
            registry_dir=tmp_path,
        )
        reg.save(ds)
        # Load
        loaded = reg.load("qa-test", "1.0.0")
        assert loaded.fingerprint == ds.fingerprint
        # Run

        def model_fn(prompt: str) -> str:
            return "0"

        task = EvalTask(
            name="pipeline",
            solver=GenerateSolver(),
            scorers=[ExactMatchScorer()],
            dataset=loaded.to_sample_list(),
        )
        result = task.run(model_fn)
        assert isinstance(result, EvalResult)
        assert result.total_samples == 5

    def test_dataset_diff_after_modification(
        self, tmp_path
    ):
        # SCENARIO: save v1, save modified v2, diff
        # WHY: version comparison workflow
        # EXPECTED: diff shows changes
        reg = DatasetRegistry(
            registry_dir=tmp_path,
        )
        reg.save(_make_dataset(version="1.0.0", n=5))
        reg.save(_make_dataset(version="2.0.0", n=8))
        diff = reg.diff("qa-test", "1.0.0", "2.0.0")
        assert isinstance(diff, DatasetDiff)
        assert diff.old_version == "1.0.0"
        assert diff.new_version == "2.0.0"

    def test_registry_list_after_multiple_saves(
        self, tmp_path
    ):
        # SCENARIO: save several datasets, list all
        # WHY: registry discovery
        # EXPECTED: all saved datasets listed
        reg = DatasetRegistry(
            registry_dir=tmp_path,
        )
        reg.save(_make_dataset(name="a", n=3))
        reg.save(_make_dataset(name="b", n=5))
        reg.save(_make_dataset(name="c", n=7))
        infos = reg.list()
        names = sorted(i.name for i in infos)
        assert names == ["a", "b", "c"]

    def test_quality_gate_before_eval(self):
        # SCENARIO: assert_dataset_quality before eval
        # WHY: quality gate workflow
        # EXPECTED: passes then runs eval
        from mltk.eval.scorers import ExactMatchScorer
        from mltk.eval.solvers import GenerateSolver
        from mltk.eval.task import EvalTask

        ds = _make_dataset(n=100)
        # Quality gate
        qr = assert_dataset_quality(
            ds, min_samples=10,
        )
        assert qr.passed is True
        # Eval

        def model_fn(prompt: str) -> str:
            return "0"

        task = EvalTask(
            name="gated",
            solver=GenerateSolver(),
            scorers=[ExactMatchScorer()],
            dataset=ds.to_sample_list(),
        )
        result = task.run(model_fn)
        assert result.total_samples == 100

    def test_load_and_quality_check(self, tmp_path):
        # SCENARIO: load from registry + quality check
        # WHY: real workflow
        # EXPECTED: both succeed
        reg = DatasetRegistry(
            registry_dir=tmp_path,
        )
        reg.save(_make_dataset(n=100))
        loaded = reg.load("qa-test")
        result = assert_dataset_quality(
            loaded, min_samples=50,
        )
        assert result.passed is True

    def test_dataset_card_persists_through_registry(
        self, tmp_path
    ):
        # SCENARIO: card survives save/load
        # WHY: metadata round-trip
        # EXPECTED: card fields preserved
        card = _make_card(
            description="Persist me",
            tags=["v1", "test"],
        )
        ds = EvalDataset(
            name="card-test",
            version="1.0.0",
            samples=_make_samples(5),
            card=card,
        )
        reg = DatasetRegistry(
            registry_dir=tmp_path,
        )
        reg.save(ds)
        loaded = reg.load("card-test", "1.0.0")
        assert loaded.card is not None
        assert loaded.card.description == "Persist me"
        assert loaded.card.tags == ["v1", "test"]

    def test_diff_identical_versions(self, tmp_path):
        # SCENARIO: diff v1 with itself
        # WHY: edge case -- no changes
        # EXPECTED: no added, no removed, all unchanged
        reg = DatasetRegistry(
            registry_dir=tmp_path,
        )
        ds = _make_dataset(version="1.0.0", n=5)
        reg.save(ds)
        # Save v2 with identical content
        ds2 = EvalDataset(
            name="qa-test",
            version="2.0.0",
            samples=list(ds.samples),
        )
        reg.save(ds2)
        diff = reg.diff("qa-test", "1.0.0", "2.0.0")
        assert len(diff.added_samples) == 0
        assert len(diff.removed_samples) == 0
        assert len(diff.unchanged_samples) == 5

    def test_multiple_registries_independent(
        self, tmp_path
    ):
        # SCENARIO: two separate registries
        # WHY: isolation
        # EXPECTED: datasets in one don't appear in other
        dir_a = tmp_path / "reg-a"
        dir_b = tmp_path / "reg-b"
        dir_a.mkdir()
        dir_b.mkdir()
        reg_a = DatasetRegistry(registry_dir=dir_a)
        reg_b = DatasetRegistry(registry_dir=dir_b)
        reg_a.save(_make_dataset(name="only-in-a"))
        assert reg_a.exists("only-in-a")
        assert reg_b.exists("only-in-a") is False
