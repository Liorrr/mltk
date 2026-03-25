# Test Registry

Share and reuse test resource collections across projects and CI pipelines.

**Module:** `mltk.registry`

---

## Overview

The Test Registry is a local directory store (``~/.mltk/registry/`` by default) that lets you **name**, **save**, and **restore** directories of test fixture files.  Instead of committing large binary fixtures to source control or reinventing fixture distribution each project, you save them once and pull them anywhere.

```
~/.mltk/registry/
  my_fixtures/
    manifest.json       ← auto-generated metadata
    train.csv
    schema.yaml
  cv_benchmarks/
    manifest.json
    images/
```

Override the default location with the `MLTK_REGISTRY_DIR` environment variable.

---

## Quick Start

```python
from mltk.registry import save_collection, load_collection, list_collections

# 1. Save a directory of test files as a named collection
save_collection(
    "my_fixtures",
    source_dir="tests/fixtures",
    description="Smoke-test CSV + schema files",
    tags=["smoke", "data"],
)

# 2. Restore the collection in another project / CI job
load_collection("my_fixtures", target_dir="tests/")
# → files appear at tests/my_fixtures/

# 3. Browse the registry
for manifest in list_collections():
    print(f"{manifest.name:20s}  v{manifest.version}  {manifest.tags}")
```

---

## CLI

```bash
# Save current directory as a collection
mltk registry push my_fixtures

# Save a specific directory
mltk registry push my_fixtures --source tests/fixtures

# Restore a collection here
mltk registry pull my_fixtures

# Pull into a specific directory
mltk registry pull my_fixtures --target /tmp/workspace

# List all saved collections
mltk registry list
```

---

## API Reference

### `save_collection`

```python
save_collection(
    name: str,
    source_dir: str | Path,
    registry_dir: str | Path | None = None,
    description: str = "",
    version: str = "1.0",
    tags: list[str] | None = None,
) -> Path
```

Save a directory of test files as a named collection.

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Unique collection identifier. |
| `source_dir` | `str \| Path` | Directory to copy into the registry. |
| `registry_dir` | `str \| Path \| None` | Override registry location. |
| `description` | `str` | Human-readable description stored in the manifest. |
| `version` | `str` | Version string (default `"1.0"`). |
| `tags` | `list[str] \| None` | Labels for filtering collections. |

**Returns:** `Path` to the collection directory inside the registry.

**Raises:** `FileNotFoundError` if `source_dir` does not exist.

---

### `load_collection`

```python
load_collection(
    name: str,
    target_dir: str | Path,
    registry_dir: str | Path | None = None,
) -> Path
```

Restore a collection from the registry into a target directory.

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Name of the collection to restore. |
| `target_dir` | `str \| Path` | Destination root; collection lands at `{target_dir}/{name}/`. |
| `registry_dir` | `str \| Path \| None` | Override registry location. |

**Returns:** `Path` to the loaded collection directory (`{target_dir}/{name}/`).

**Raises:** `ValueError` if the collection does not exist.

---

### `list_collections`

```python
list_collections(
    registry_dir: str | Path | None = None,
) -> list[CollectionManifest]
```

List all collections in the registry, sorted alphabetically by name.

**Returns:** List of [`CollectionManifest`](#collectionmanifest) instances.  Empty list if the registry does not exist or contains no valid collections.

---

### `CollectionManifest`

Dataclass describing a saved collection.

```python
@dataclass
class CollectionManifest:
    name: str
    version: str = "1.0"
    description: str = ""
    author: str = ""
    created: str          # ISO-8601 timestamp
    files: list[str]      # relative paths of included files
    tags: list[str]
```

**Methods:**

- `to_dict() -> dict` — serialize to a JSON-compatible dictionary.
- `CollectionManifest.from_dict(data: dict) -> CollectionManifest` — deserialize from a dictionary.

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MLTK_REGISTRY_DIR` | `~/.mltk/registry` | Override the registry root directory. |

---

## Examples

### CI Pipeline Workflow

```yaml
# .github/workflows/test.yml
- name: Restore test fixtures
  run: mltk registry pull integration_fixtures --target tests/

- name: Run tests
  run: pytest tests/
```

### Tagging and Filtering

```python
# Save domain-specific collections
save_collection("bert_fixtures", "tests/nlp/fixtures", tags=["nlp", "bert"])
save_collection("resnet_fixtures", "tests/cv/fixtures", tags=["cv", "resnet"])

# Filter by tag in Python
nlp_kits = [m for m in list_collections() if "nlp" in m.tags]
```

### Custom Registry Location

```python
import os
os.environ["MLTK_REGISTRY_DIR"] = "/shared/mltk-registry"

save_collection("shared_fixtures", "tests/fixtures")
# → /shared/mltk-registry/shared_fixtures/
```

---
