"""Tests for Asana adapter — all API calls mocked."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from mltk.integrations.asana_adapter import AsanaAdapter


def _mock_response(data: dict) -> MagicMock:
    mock = MagicMock()
    mock.read.return_value = json.dumps(data).encode()
    mock.__enter__ = lambda s: s
    mock.__exit__ = MagicMock(return_value=False)
    return mock


def test_create_task():
    adapter = AsanaAdapter(token="test-token", workspace_gid="WS-1")
    resp = {"data": {"gid": "12345", "name": "Bug"}}
    with patch("urllib.request.urlopen", return_value=_mock_response(resp)):
        result = adapter.create_issue("proj-gid", "Bug title", "Description")
    assert result == "12345"


def test_search_tasks():
    adapter = AsanaAdapter(token="test-token", workspace_gid="WS-1")
    resp = {"data": [{"gid": "1", "name": "Task A"}]}
    with patch("urllib.request.urlopen", return_value=_mock_response(resp)):
        results = adapter.search_issues("drift")
    assert len(results) == 1


def test_update_task():
    adapter = AsanaAdapter(token="test-token")
    resp = {"data": {"gid": "12345"}}
    with patch("urllib.request.urlopen", return_value=_mock_response(resp)):
        ok = adapter.update_issue("12345", {"title": "Updated"})
    assert ok is True


def test_api_error_returns_empty():
    adapter = AsanaAdapter(token="test-token")
    with patch("urllib.request.urlopen", side_effect=Exception("timeout")):
        result = adapter.create_issue("proj", "Title", "Desc")
    assert result == ""
