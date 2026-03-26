"""Tests for Linear.app adapter — all API calls mocked."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from mltk.integrations.linear_adapter import LinearAdapter


def _mock_response(data: dict) -> MagicMock:
    mock = MagicMock()
    mock.read.return_value = json.dumps(data).encode()
    mock.__enter__ = lambda s: s
    mock.__exit__ = MagicMock(return_value=False)
    return mock


def test_create_issue():
    adapter = LinearAdapter(api_key="test-key", team_id="TEAM-1")
    resp = {"data": {"issueCreate": {"issue": {"identifier": "LIN-42", "url": "https://linear.app/..."}}}}
    with patch("urllib.request.urlopen", return_value=_mock_response(resp)):
        result = adapter.create_issue("proj", "Bug title", "Description")
    assert result == "LIN-42"


def test_search_issues():
    adapter = LinearAdapter(api_key="test-key")
    resp = {"data": {"issueSearch": {"nodes": [{"identifier": "LIN-1", "title": "Test"}]}}}
    with patch("urllib.request.urlopen", return_value=_mock_response(resp)):
        results = adapter.search_issues("drift")
    assert len(results) == 1
    assert results[0]["identifier"] == "LIN-1"


def test_update_issue():
    adapter = LinearAdapter(api_key="test-key")
    resp = {"data": {"issueUpdate": {"success": True}}}
    with patch("urllib.request.urlopen", return_value=_mock_response(resp)):
        ok = adapter.update_issue("abc-123", {"title": "New title"})
    assert ok is True


def test_api_error_returns_empty():
    adapter = LinearAdapter(api_key="test-key")
    with patch("urllib.request.urlopen", side_effect=Exception("timeout")):
        result = adapter.create_issue("proj", "Title", "Desc")
    assert result == ""
