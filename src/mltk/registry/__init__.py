"""Test Resource Registry — save, load, and list test collections.

The registry stores named collections of test resource files in a local
directory (``~/.mltk/registry/`` by default).  Override the location with the
``MLTK_REGISTRY_DIR`` environment variable.

Quick start::

    from mltk.registry import save_collection, load_collection, list_collections

    # Save a directory of fixture files as "my_fixtures"
    save_collection("my_fixtures", source_dir="tests/fixtures")

    # Load it elsewhere
    load_collection("my_fixtures", target_dir="/tmp/project/tests")

    # List everything in the registry
    for manifest in list_collections():
        print(manifest.name, manifest.version)
"""

from mltk.registry.collection import list_collections, load_collection, save_collection
from mltk.registry.manifest import CollectionManifest

__all__ = ["save_collection", "load_collection", "list_collections", "CollectionManifest"]
