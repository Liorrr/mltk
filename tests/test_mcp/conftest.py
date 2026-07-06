"""Pytest fixtures for MCP server tests.

Auto-creates mock MCP server with all 13 tools registered
before every test. No real ``mcp`` package needed.
"""
from __future__ import annotations

import sys

import pytest

from ._helpers import (
    import_server,
    make_mcp_modules,
    registered_tools,
)


@pytest.fixture(autouse=True)
def mcp_server():
    """Inject mock mcp package, import server, create it.

    Populates ``registered_tools`` with all 11 tool functions
    so that ``call_tool()`` works in every test.

    Injects/restores ONLY the three mock ``mcp*`` keys — never
    ``patch.dict("sys.modules", ...)``, whose teardown evicts every
    module first imported during the test while parent packages keep
    attribute references to the evicted objects. Python 3.10's
    ``unittest.mock`` resolves dotted patch targets via getattr chains
    through those parent attributes (3.11+ uses importlib), so later
    tests would patch a stale evicted module while the code under test
    imports a fresh one — making patches silently no-op.
    """
    modules = make_mcp_modules()
    saved = {name: sys.modules.get(name) for name in modules}
    sys.modules.update(modules)
    registered_tools.clear()
    try:
        server = import_server()
        yield server.create_server()
    finally:
        for name, orig in saved.items():
            if orig is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = orig
        registered_tools.clear()
