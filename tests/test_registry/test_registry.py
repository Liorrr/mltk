"""Tests for mltk.registry — save, load, and list test resource collections.

The registry is a local directory store that lets teams share test fixture files
by name.  These tests cover the full lifecycle: saving, loading, listing, and
roundtrip fidelity, as well as error paths and custom registry locations.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mltk.registry import list_collections, load_collection, save_collection
from mltk.registry.manifest import CollectionManifest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_source(tmp_path: Path, files: dict[str, str] | None = None) -> Path:
    """Create a temporary source directory with the given file->content mapping."""
    src = tmp_path / "source"
    src.mkdir(parents=True, exist_ok=True)
    default = {"data.csv": "id,value\n1,2\n", "schema.yaml": "columns:\n  id: int\n"}
    for name, content in (files or default).items():
        (src / name).write_text(content, encoding="utf-8")
    return src


# ---------------------------------------------------------------------------
# save_collection
# ---------------------------------------------------------------------------

class TestSaveCollection:
    """Tests for save_collection()."""

    def test_save_collection_creates_dir(self, tmp_path: Path) -> None:
        # SCENARIO: Save a collection to a custom registry directory.
        # WHY: The collection subdirectory must be created so subsequent
        #      load/list operations can discover it.
        # EXPECTED: {registry}/{name}/ directory exists after save.
        registry = tmp_path / "reg"
        src = _make_source(tmp_path)
        result = save_collection("mykit", src, registry_dir=registry)
        assert result.is_dir()
        assert result == registry / "mykit"

    def test_save_collection_creates_manifest(self, tmp_path: Path) -> None:
        # SCENARIO: Save a collection and check that manifest.json is written.
        # WHY: manifest.json is the anchor that list_collections() uses to
        #      discover and parse collection metadata.
        # EXPECTED: manifest.json exists inside the collection directory.
        registry = tmp_path / "reg"
        src = _make_source(tmp_path)
        col_dir = save_collection("kit", src, registry_dir=registry)
        assert (col_dir / "manifest.json").is_file()

    def test_save_collection_copies_files(self, tmp_path: Path) -> None:
        # SCENARIO: Source directory contains multiple files.
        # WHY: The whole point of save_collection is to preserve fixture files
        #      so they can be restored on another machine or in CI.
        # EXPECTED: All source files appear in the collection directory.
        registry = tmp_path / "reg"
        src = _make_source(tmp_path, {"a.csv": "1,2", "b.yaml": "key: val"})
        col_dir = save_collection("fixtures", src, registry_dir=registry)
        assert (col_dir / "a.csv").is_file()
        assert (col_dir / "b.yaml").is_file()

    def test_save_collection_manifest_content(self, tmp_path: Path) -> None:
        # SCENARIO: Save with explicit name, version, description, and tags.
        # WHY: Downstream tooling (list_collections, CLI list) surfaces these
        #      fields to users — they must be stored accurately.
        # EXPECTED: Parsed manifest reflects the supplied metadata.
        registry = tmp_path / "reg"
        src = _make_source(tmp_path)
        col_dir = save_collection(
            "annotated",
            src,
            registry_dir=registry,
            description="Smoke test fixtures",
            version="2.0",
            tags=["smoke", "data"],
        )
        import json
        data = json.loads((col_dir / "manifest.json").read_text())
        assert data["name"] == "annotated"
        assert data["version"] == "2.0"
        assert data["description"] == "Smoke test fixtures"
        assert set(data["tags"]) == {"smoke", "data"}

    def test_save_collection_manifest_files_list(self, tmp_path: Path) -> None:
        # SCENARIO: Manifest records the relative paths of all copied files.
        # WHY: The files list lets callers verify what is inside a collection
        #      without unpacking it.
        # EXPECTED: manifest["files"] contains the names of the source files.
        registry = tmp_path / "reg"
        src = _make_source(tmp_path, {"train.csv": "x,y\n1,0\n", "test.csv": "x,y\n2,1\n"})
        col_dir = save_collection("splits", src, registry_dir=registry)
        import json
        data = json.loads((col_dir / "manifest.json").read_text())
        assert "train.csv" in data["files"]
        assert "test.csv" in data["files"]

    def test_save_collection_missing_source_raises(self, tmp_path: Path) -> None:
        # SCENARIO: The source_dir argument points to a non-existent path.
        # WHY: Silent failure (empty collection) would be harder to debug than
        #      an immediate, descriptive error.
        # EXPECTED: FileNotFoundError is raised.
        registry = tmp_path / "reg"
        with pytest.raises(FileNotFoundError):
            save_collection("bad", tmp_path / "no_such_dir", registry_dir=registry)


# ---------------------------------------------------------------------------
# load_collection
# ---------------------------------------------------------------------------

class TestLoadCollection:
    """Tests for load_collection()."""

    def test_load_collection_copies_to_target(self, tmp_path: Path) -> None:
        # SCENARIO: A collection is saved then loaded into a different directory.
        # WHY: The primary use-case — restore fixtures in a fresh workspace.
        # EXPECTED: Files appear at {target}/{name}/ after load.
        registry = tmp_path / "reg"
        src = _make_source(tmp_path, {"fixture.csv": "a,b\n1,2\n"})
        save_collection("restore_me", src, registry_dir=registry)

        target = tmp_path / "workspace"
        target.mkdir()
        loaded = load_collection("restore_me", target, registry_dir=registry)

        assert loaded == target / "restore_me"
        assert (loaded / "fixture.csv").is_file()
        assert (loaded / "fixture.csv").read_text() == "a,b\n1,2\n"

    def test_load_collection_not_found(self, tmp_path: Path) -> None:
        # SCENARIO: Caller tries to load a collection that was never saved.
        # WHY: A clear error message prevents silent use of stale/wrong data.
        # EXPECTED: ValueError is raised with the collection name mentioned.
        registry = tmp_path / "empty_reg"
        registry.mkdir()
        with pytest.raises(ValueError, match="ghost"):
            load_collection("ghost", tmp_path / "out", registry_dir=registry)

    def test_load_collection_includes_manifest(self, tmp_path: Path) -> None:
        # SCENARIO: Load a previously saved collection.
        # WHY: The manifest is part of the collection payload; including it
        #      lets a loaded workspace self-describe.
        # EXPECTED: manifest.json exists in the loaded directory.
        registry = tmp_path / "reg"
        src = _make_source(tmp_path)
        save_collection("with_manifest", src, registry_dir=registry)
        loaded = load_collection("with_manifest", tmp_path / "out", registry_dir=registry)
        assert (loaded / "manifest.json").is_file()


# ---------------------------------------------------------------------------
# list_collections
# ---------------------------------------------------------------------------

class TestListCollections:
    """Tests for list_collections()."""

    def test_list_collections_empty(self, tmp_path: Path) -> None:
        # SCENARIO: Registry directory exists but contains no collections.
        # WHY: list_collections() must return an empty list, not raise, when
        #      the registry is freshly initialised with nothing in it.
        # EXPECTED: Empty list returned.
        registry = tmp_path / "empty"
        registry.mkdir()
        assert list_collections(registry_dir=registry) == []

    def test_list_collections_nonexistent_dir(self, tmp_path: Path) -> None:
        # SCENARIO: Registry directory does not exist at all.
        # WHY: A first-time user who has never saved a collection should not
        #      see a crash — just an empty list.
        # EXPECTED: Empty list returned without raising.
        result = list_collections(registry_dir=tmp_path / "never_created")
        assert result == []

    def test_list_collections_multiple(self, tmp_path: Path) -> None:
        # SCENARIO: Three collections are saved; list_collections is called.
        # WHY: Teams may have many named collections; listing them all is a
        #      core discovery workflow.
        # EXPECTED: Returns a manifest for every saved collection, sorted by name.
        registry = tmp_path / "reg"
        for cname in ("zebra", "alpha", "middle"):
            src = _make_source(tmp_path / cname)
            save_collection(cname, src, registry_dir=registry)

        manifests = list_collections(registry_dir=registry)
        names = [m.name for m in manifests]
        assert names == sorted(names)  # alphabetically sorted
        assert set(names) == {"zebra", "alpha", "middle"}

    def test_list_collections_returns_manifests(self, tmp_path: Path) -> None:
        # SCENARIO: A saved collection has tags and description; list returns them.
        # WHY: Callers may filter by tags or display descriptions in a UI.
        # EXPECTED: Returned manifest objects carry the correct metadata.
        registry = tmp_path / "reg"
        src = _make_source(tmp_path)
        save_collection(
            "tagged",
            src,
            registry_dir=registry,
            description="Tagged fixture set",
            tags=["cv", "regression"],
        )
        manifests = list_collections(registry_dir=registry)
        assert len(manifests) == 1
        m = manifests[0]
        assert isinstance(m, CollectionManifest)
        assert m.name == "tagged"
        assert m.description == "Tagged fixture set"
        assert "cv" in m.tags


# ---------------------------------------------------------------------------
# End-to-end / cross-function tests
# ---------------------------------------------------------------------------

class TestRoundtrip:
    """Roundtrip and integration tests."""

    def test_save_load_roundtrip(self, tmp_path: Path) -> None:
        # SCENARIO: Save a collection, load it, verify file contents are intact.
        # WHY: File contents must be byte-for-byte identical after roundtrip —
        #      any corruption would invalidate fixtures used in CI.
        # EXPECTED: Loaded file contents match the originals exactly.
        registry = tmp_path / "reg"
        src = _make_source(tmp_path, {
            "train.csv": "feature,label\n0.5,0\n0.9,1\n",
            "readme.txt": "Test fixtures v2",
        })
        save_collection("roundtrip", src, registry_dir=registry)

        out = tmp_path / "restored"
        out.mkdir()
        loaded_dir = load_collection("roundtrip", out, registry_dir=registry)

        assert (loaded_dir / "train.csv").read_text() == "feature,label\n0.5,0\n0.9,1\n"
        assert (loaded_dir / "readme.txt").read_text() == "Test fixtures v2"

    def test_custom_registry_dir(self, tmp_path: Path) -> None:
        # SCENARIO: User specifies a non-default registry path for all operations.
        # WHY: CI pipelines often use project-local registries to avoid
        #      polluting the user home directory.
        # EXPECTED: All three functions operate correctly on the custom path.
        custom_reg = tmp_path / "custom_registry"
        src = _make_source(tmp_path)

        col_dir = save_collection("custom", src, registry_dir=custom_reg)
        assert col_dir.parent == custom_reg

        manifests = list_collections(registry_dir=custom_reg)
        assert len(manifests) == 1
        assert manifests[0].name == "custom"

        out = tmp_path / "out"
        out.mkdir()
        loaded = load_collection("custom", out, registry_dir=custom_reg)
        assert loaded.is_dir()

    def test_collection_with_tags(self, tmp_path: Path) -> None:
        # SCENARIO: Save a collection with tags; load it; read manifest tags.
        # WHY: Tags enable filtered queries (e.g. "find all 'nlp' collections").
        #      They must survive save→load→list roundtrip.
        # EXPECTED: Tags present in both the saved manifest and listed manifest.
        registry = tmp_path / "reg"
        src = _make_source(tmp_path)
        save_collection("nlp_fixtures", src, registry_dir=registry, tags=["nlp", "bert"])

        manifests = list_collections(registry_dir=registry)
        assert len(manifests) == 1
        assert set(manifests[0].tags) == {"nlp", "bert"}

        # Tags also survive a load (manifest.json is copied)
        import json
        out = tmp_path / "out"
        out.mkdir()
        loaded = load_collection("nlp_fixtures", out, registry_dir=registry)
        manifest_data = json.loads((loaded / "manifest.json").read_text())
        assert set(manifest_data["tags"]) == {"nlp", "bert"}
