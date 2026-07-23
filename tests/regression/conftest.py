"""Fixtures for the slim claim-integrity canary suite.

MCP scan honesty reuses the mock FastMCP helpers from ``tests/test_mcp``
so we do not depend on the real ``mcp`` package or re-import other test
modules' cases.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

# Sibling package under tests/ is not a top-level install; make it importable.
_TESTS_ROOT = Path(__file__).resolve().parent.parent
if str(_TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_TESTS_ROOT))

from test_mcp._helpers import (  # noqa: E402
    import_server,
    make_mcp_modules,
    registered_tools,
)


@pytest.fixture
def mcp_scan_registered() -> Iterator[None]:
    """Register MCP server tools (including mltk_scan) for one test.

    Mirrors ``tests/test_mcp/conftest.py`` but is opt-in (not autouse) so
    non-MCP canaries stay free of sys.modules surgery.
    """
    modules = make_mcp_modules()
    saved = {name: sys.modules.get(name) for name in modules}
    sys.modules.update(modules)
    registered_tools.clear()
    try:
        server = import_server()
        server.create_server()
        yield
    finally:
        for name, orig in saved.items():
            if orig is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = orig
        registered_tools.clear()
