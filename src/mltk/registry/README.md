# mltk.registry

## Overview
Local registry of named test-resource collections (default `~/.mltk/registry/`, override with `MLTK_REGISTRY_DIR`).

## Interface
- `save_collection(name, source_dir, ...)`, `load_collection(name, target_dir, ...)`, `list_collections(...)`.
- `CollectionManifest` with `to_dict` / `from_dict`.

## Invariants
- Each collection directory carries a JSON manifest.
- An explicit registry directory argument takes precedence over the environment variable and default.

## Side Effects
- Creates directories and copies files; writes the manifest JSON.

## Verification
- Run: `python -m pytest tests/test_registry -q`
