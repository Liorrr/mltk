"""Pytest fixtures for MCP server tests.

Auto-creates mock MCP server with all 8 tools registered
before every test. No real ``mcp`` package needed.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from ._helpers import (
    import_server,
    make_mcp_modules,
    registered_tools,
)


@pytest.fixture(autouse=True)
def mcp_server():
    """Inject mock mcp package, import server, create it.

    Populates ``registered_tools`` with all 8 tool functions
    so that ``call_tool()`` works in every test.
    """
    modules = make_mcp_modules()
    registered_tools.clear()
    with patch.dict("sys.modules", modules):
        server = import_server()
        srv = server.create_server()
        yield srv
    registered_tools.clear()
