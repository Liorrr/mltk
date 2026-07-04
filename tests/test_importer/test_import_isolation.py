"""Guards the architectural invariant that mltk.importer stays optional.

``src/mltk/__init__.py`` must never import ``mltk.importer``, so that a
plain ``import mltk`` never pulls in the optional ``datasets``/``pyarrow``
heavy dependencies (mirrors the ``mltk.cost`` package).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_top_level_init_does_not_import_importer_package():
    # SCENARIO: inspect src/mltk/__init__.py's source directly
    # WHY: fast lexical guard, kept as a secondary check -- the real
    #   invariant (a plain `import mltk` never loading `mltk.importer`
    #   or `datasets` at runtime) is proven by the subprocess test below,
    #   since this in-process test suite may have already imported
    #   `mltk.importer` via other test modules by the time this runs
    # EXPECTED: no reference to the importer package
    init_source = Path(__file__).parents[2] / "src" / "mltk" / "__init__.py"
    content = init_source.read_text(encoding="utf-8")
    assert "mltk.importer" not in content
    assert "from mltk import importer" not in content


def test_plain_import_mltk_does_not_load_importer_or_datasets():
    # SCENARIO: `import mltk` in a FRESH interpreter subprocess
    # WHY: this MUST run in a subprocess -- by the time any test in this
    #   process's pytest session runs, `mltk.importer` (and therefore
    #   `datasets`, if installed) may already sit in `sys.modules` from
    #   an earlier test module, so an in-process check of `sys.modules`
    #   here would prove nothing either way. A subprocess is the only
    #   way to observe what a truly plain `import mltk` pulls in.
    # EXPECTED: subprocess exits 0 -- neither `mltk.importer` nor
    #   `datasets` is loaded merely by importing the top-level package
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import mltk; "
            "assert 'mltk.importer' not in sys.modules; "
            "assert 'datasets' not in sys.modules",
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr


def test_importer_package_is_independently_importable():
    # SCENARIO: import mltk.importer directly and check its public API
    #   surface
    # WHY: this is a public-API-surface check, NOT proof of independent
    #   importability -- `mltk.importer` may already be cached in
    #   `sys.modules` by the time this in-process test runs (e.g. other
    #   test modules in this same suite import it first), so this test
    #   cannot by itself demonstrate the isolation invariant. That
    #   invariant is covered by the subprocess test above; this test
    #   only confirms the documented public API is present once imported.
    # EXPECTED: succeeds and exposes the documented public API
    import mltk.importer as importer_pkg

    assert hasattr(importer_pkg, "DatasetImporter")
    assert hasattr(importer_pkg, "ColumnMapping")
    assert hasattr(importer_pkg, "ColumnRole")
    assert hasattr(importer_pkg, "ImportResult")
    assert hasattr(importer_pkg, "auto_map_columns")
