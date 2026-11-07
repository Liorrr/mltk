"""Test collection management — save, load, list test resource collections."""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from mltk.registry.manifest import CollectionManifest

# Default registry directory — overridable via MLTK_REGISTRY_DIR env var
_DEFAULT_REGISTRY = Path.home() / ".mltk" / "registry"


def _validate_component(value: str, kind: str) -> str:
    """Validate a caller-supplied single filesystem name component."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{kind} must be a non-empty string")
    if value in {".", ".."}:
        raise ValueError(f"{kind} cannot be '.' or '..'")
    if any(char in value for char in ("/", "\\", "\x00", ":")):
        raise ValueError(f"{kind} must be a single relative path component")
    if Path(value).is_absolute():
        raise ValueError(f"{kind} must be a relative path component")
    return value


def _path_inside(root: Path, component: str, kind: str) -> Path:
    """Return a validated component path that remains inside *root*."""
    root_resolved = root.resolve()
    candidate = (root_resolved / _validate_component(component, kind)).resolve()
    if candidate != root_resolved and not candidate.is_relative_to(root_resolved):
        raise ValueError(f"{kind} resolves outside its root")
    return candidate


def _resolve_registry(registry_dir: str | Path | None) -> Path:
    """Return the effective registry directory.

    Priority: explicit argument > ``MLTK_REGISTRY_DIR`` env var > default.
    """
    if registry_dir is not None:
        return Path(registry_dir)
    env = os.environ.get("MLTK_REGISTRY_DIR")
    if env:
        return Path(env)
    return _DEFAULT_REGISTRY


def save_collection(
    name: str,
    source_dir: str | Path,
    registry_dir: str | Path | None = None,
    description: str = "",
    version: str = "1.0",
    tags: list[str] | None = None,
) -> Path:
    """Save a directory of test files as a named collection in the registry.

    Copies all files from *source_dir* into ``{registry_dir}/{name}/`` and
    writes a ``manifest.json`` alongside them.

    Args:
        name: Unique name for the collection.
        source_dir: Directory whose contents will be saved.
        registry_dir: Override the registry root directory.  Falls back to the
            ``MLTK_REGISTRY_DIR`` environment variable or ``~/.mltk/registry/``.
        description: Human-readable description stored in the manifest.
        version: Version string stored in the manifest (default ``"1.0"``).
        tags: Optional list of labels for the collection.

    Returns:
        Path to the newly created collection directory inside the registry.

    Raises:
        FileNotFoundError: If *source_dir* does not exist.
        ValueError: If *name* is not a single relative path component.
    """
    source_path = Path(source_dir)
    if not source_path.exists():
        raise FileNotFoundError(f"Source directory not found: {source_path}")

    registry_path = _resolve_registry(registry_dir).resolve()
    collection_dir = _path_inside(registry_path, name, "Collection name")
    collection_dir.mkdir(parents=True, exist_ok=True)

    # Copy all files from source_dir (non-recursive, flat copy of root files
    # plus recursive copy of subdirs)
    copied_files: list[str] = []
    for item in source_path.iterdir():
        dest = collection_dir / item.name
        if item.is_dir():
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(item, dest)
            # Record contained files relative to collection_dir
            for sub in dest.rglob("*"):
                if sub.is_file():
                    copied_files.append(str(sub.relative_to(collection_dir)))
        else:
            shutil.copy2(item, dest)
            copied_files.append(item.name)

    manifest = CollectionManifest(
        name=name,
        version=version,
        description=description,
        tags=list(tags) if tags else [],
        files=sorted(copied_files),
    )

    manifest_path = collection_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest.to_dict(), indent=2), encoding="utf-8")

    return collection_dir


def load_collection(
    name: str,
    target_dir: str | Path,
    registry_dir: str | Path | None = None,
) -> Path:
    """Load a collection from the registry into a target directory.

    Copies ``{registry_dir}/{name}/*`` into ``{target_dir}/{name}/``.
    The ``manifest.json`` is included in the copy.

    Args:
        name: Name of the collection to load.
        target_dir: Destination root directory.  The collection will be placed
            at ``{target_dir}/{name}/``.
        registry_dir: Override the registry root directory.

    Returns:
        Path to the loaded collection directory (``{target_dir}/{name}/``).

    Raises:
        ValueError: If *name* is not a single relative path component, the
            target path escapes *target_dir*, or no collection with *name*
            exists in the registry.
    """
    registry_path = _resolve_registry(registry_dir).resolve()
    collection_dir = _path_inside(registry_path, name, "Collection name")

    if not collection_dir.exists() or not (collection_dir / "manifest.json").exists():
        raise ValueError(
            f"Collection '{name}' not found in registry at {registry_path}"
        )

    target_path = Path(target_dir).resolve()
    dest = _path_inside(target_path, name, "Collection name")
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(collection_dir, dest)

    return dest


def list_collections(
    registry_dir: str | Path | None = None,
) -> list[CollectionManifest]:
    """List all collections in the registry.

    Reads every ``manifest.json`` found directly inside *registry_dir* and
    returns the parsed manifests sorted by collection name.

    Args:
        registry_dir: Override the registry root directory.

    Returns:
        List of :class:`~mltk.registry.manifest.CollectionManifest` instances,
        sorted alphabetically by name.  Returns an empty list if the registry
        directory does not exist or contains no valid collections.
    """
    registry_path = _resolve_registry(registry_dir)
    if not registry_path.exists():
        return []

    manifests: list[CollectionManifest] = []
    for manifest_file in registry_path.glob("*/manifest.json"):
        try:
            data = json.loads(manifest_file.read_text(encoding="utf-8"))
            manifests.append(CollectionManifest.from_dict(data))
        except (json.JSONDecodeError, KeyError):
            # Skip malformed manifests rather than crashing
            continue

    return sorted(manifests, key=lambda m: m.name)
