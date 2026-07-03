"""Guards the architectural invariant that mltk.importer stays optional.

``src/mltk/__init__.py`` must never import ``mltk.importer``, so that a
plain ``import mltk`` never pulls in the optional ``datasets``/``pyarrow``
heavy dependencies (mirrors the ``mltk.cost`` package).
"""

from __future__ import annotations

from pathlib import Path


def test_top_level_init_does_not_import_importer_package():
    # SCENARIO: inspect src/mltk/__init__.py's source directly
    # WHY: a plain `import mltk` must never require `datasets`/`pyarrow`
    # EXPECTED: no reference to the importer package
    init_source = Path(__file__).parents[2] / "src" / "mltk" / "__init__.py"
    content = init_source.read_text(encoding="utf-8")
    assert "mltk.importer" not in content
    assert "from mltk import importer" not in content


def test_importer_package_is_independently_importable():
    # SCENARIO: import mltk.importer directly
    # WHY: the package must stand alone, reachable via its own import
    # EXPECTED: succeeds and exposes the documented public API
    import mltk.importer as importer_pkg

    assert hasattr(importer_pkg, "DatasetImporter")
    assert hasattr(importer_pkg, "ColumnMapping")
    assert hasattr(importer_pkg, "ColumnRole")
    assert hasattr(importer_pkg, "ImportResult")
    assert hasattr(importer_pkg, "auto_map_columns")
