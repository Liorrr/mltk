"""Tests for GitHubIssuesAdapter."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from mltk.integrations.github_adapter import GitHubIssuesAdapter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_response(status: int, body: dict) -> MagicMock:
    """Build a mock urllib response context manager."""
    resp = MagicMock()
    resp.status = status
    resp.read.return_value = json.dumps(body).encode()
    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=resp)
    cm.__exit__ = MagicMock(return_value=False)
    return cm


def _make_http_error(code: int, body: dict):
    import io
    import urllib.error

    err = urllib.error.HTTPError(
        url="https://api.github.com/repos/owner/repo/issues",
        code=code,
        msg="Error",
        hdrs=None,  # type: ignore[arg-type]
        fp=io.BytesIO(json.dumps(body).encode()),
    )
    return err


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestGitHubIssuesAdapterCreateIssue:
    # SCENARIO: successful issue creation
    # WHY: verify happy path — URL is extracted from the API response
    # EXPECTED: returns the html_url from the response body
    def test_create_issue(self):
        adapter = GitHubIssuesAdapter("owner/repo", token="ghp_test")
        response_body = {
            "number": 42,
            "html_url": "https://github.com/owner/repo/issues/42",
        }
        mock_cm = _make_response(201, response_body)

        with patch("urllib.request.urlopen", return_value=mock_cm):
            url = adapter.create_issue(
                project="repo",
                title="Drift detected",
                description="PSI > 0.2 on feature_age",
            )

        assert url == "https://github.com/owner/repo/issues/42"

    # SCENARIO: create issue with labels and assignees in fields
    # WHY: ensure extra fields are forwarded in the request payload
    # EXPECTED: request body contains labels and assignees
    def test_create_issue_with_fields(self):
        adapter = GitHubIssuesAdapter("owner/repo", token="ghp_test")
        response_body = {
            "number": 7,
            "html_url": "https://github.com/owner/repo/issues/7",
        }
        mock_cm = _make_response(201, response_body)

        captured_data: list[bytes] = []

        def capturing_urlopen(req):
            captured_data.append(req.data)
            return mock_cm

        with patch("urllib.request.urlopen", side_effect=capturing_urlopen):
            adapter.create_issue(
                project="repo",
                title="Bias alert",
                description="Demographic parity gap > 0.1",
                fields={"labels": ["bias", "ml"], "assignees": ["ml-bot"]},
            )

        assert len(captured_data) == 1
        payload = json.loads(captured_data[0])
        assert payload["labels"] == ["bias", "ml"]
        assert payload["assignees"] == ["ml-bot"]


class TestGitHubIssuesAdapterSearchIssues:
    # SCENARIO: search returns multiple issues
    # WHY: verify result normalisation to key/summary/url/state dict
    # EXPECTED: list of normalised dicts with correct fields
    def test_search_issues(self):
        adapter = GitHubIssuesAdapter("owner/repo", token="ghp_test")
        response_body = {
            "total_count": 2,
            "items": [
                {
                    "number": 10,
                    "title": "Drift: feature_age",
                    "html_url": "https://github.com/owner/repo/issues/10",
                    "state": "open",
                },
                {
                    "number": 11,
                    "title": "Drift: feature_salary",
                    "html_url": "https://github.com/owner/repo/issues/11",
                    "state": "closed",
                },
            ],
        }
        mock_cm = _make_response(200, response_body)

        with patch("urllib.request.urlopen", return_value=mock_cm):
            results = adapter.search_issues("drift")

        assert len(results) == 2
        assert results[0]["key"] == "10"
        assert results[0]["summary"] == "Drift: feature_age"
        assert results[0]["state"] == "open"
        assert results[1]["key"] == "11"

    # SCENARIO: search returns empty result set
    # WHY: edge case — no matching issues should yield empty list, not error
    # EXPECTED: empty list returned
    def test_search_issues_empty(self):
        adapter = GitHubIssuesAdapter("owner/repo", token="ghp_test")
        mock_cm = _make_response(200, {"total_count": 0, "items": []})

        with patch("urllib.request.urlopen", return_value=mock_cm):
            results = adapter.search_issues("nonexistent_query_xyz")

        assert results == []


class TestGitHubIssuesAdapterUpdateIssue:
    # SCENARIO: update state and add a comment
    # WHY: verify that both PATCH and POST calls succeed for combined updates
    # EXPECTED: returns True, two HTTP calls are made
    def test_update_issue(self):
        adapter = GitHubIssuesAdapter("owner/repo", token="ghp_test")
        mock_cm = _make_response(200, {"number": 42, "state": "closed"})

        call_count = 0

        def side_effect(_req):
            nonlocal call_count
            call_count += 1
            return mock_cm

        with patch("urllib.request.urlopen", side_effect=side_effect):
            result = adapter.update_issue(
                "42",
                {"state": "closed", "comment": "Fixed in v2.3"},
            )

        assert result is True
        assert call_count == 2  # one for comment POST, one for PATCH

    # SCENARIO: update with only a comment (no patch fields)
    # WHY: comment-only update should make exactly one POST call
    # EXPECTED: returns True, one HTTP call made
    def test_update_issue_comment_only(self):
        adapter = GitHubIssuesAdapter("owner/repo", token="ghp_test")
        mock_cm = _make_response(201, {"id": 999})

        call_count = 0

        def side_effect(_req):
            nonlocal call_count
            call_count += 1
            return mock_cm

        with patch("urllib.request.urlopen", side_effect=side_effect):
            result = adapter.update_issue("5", {"comment": "Acknowledged"})

        assert result is True
        assert call_count == 1


class TestGitHubIssuesAdapterTokenHandling:
    # SCENARIO: no token passed, but GITHUB_TOKEN env var is set
    # WHY: ensure environment variable fallback works correctly
    # EXPECTED: token is read from env var and used in Authorization header
    def test_missing_token_uses_env_var(self, monkeypatch):
        monkeypatch.setenv("GITHUB_TOKEN", "ghp_from_env")
        adapter = GitHubIssuesAdapter("owner/repo")
        assert adapter.token == "ghp_from_env"

    # SCENARIO: neither token arg nor env var is set
    # WHY: adapter should still construct without error (token is None, unauthed)
    # EXPECTED: token is None, header omits Authorization
    def test_no_token_no_env_var(self, monkeypatch):
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        adapter = GitHubIssuesAdapter("owner/repo")
        assert adapter.token is None
        headers = adapter._headers()
        assert "Authorization" not in headers


class TestGitHubIssuesAdapterApiError:
    # SCENARIO: GitHub API returns 422 (validation error)
    # WHY: HTTP errors from the API should be handled gracefully, not crash
    # EXPECTED: RuntimeError raised with status code in message
    def test_api_error_raises_runtime_error(self):
        import io
        import urllib.error

        adapter = GitHubIssuesAdapter("owner/repo", token="ghp_test")
        err = urllib.error.HTTPError(
            url="https://api.github.com/repos/owner/repo/issues",
            code=422,
            msg="Unprocessable Entity",
            hdrs=None,  # type: ignore[arg-type]
            fp=io.BytesIO(json.dumps({"message": "Validation Failed"}).encode()),
        )

        with patch("urllib.request.urlopen", side_effect=err):
            with pytest.raises(RuntimeError, match="422"):
                adapter.create_issue("repo", "Bad issue", "bad body")

    # SCENARIO: search API returns non-200 (e.g., rate limited)
    # WHY: search errors should return empty list, not crash
    # EXPECTED: empty list returned on non-200 search response
    def test_search_api_error_returns_empty(self):
        import io
        import urllib.error

        adapter = GitHubIssuesAdapter("owner/repo", token="ghp_test")
        err = urllib.error.HTTPError(
            url="https://api.github.com/search/issues",
            code=403,
            msg="Forbidden",
            hdrs=None,  # type: ignore[arg-type]
            fp=io.BytesIO(json.dumps({"message": "rate limit exceeded"}).encode()),
        )

        with patch("urllib.request.urlopen", side_effect=err):
            results = adapter.search_issues("drift")

        assert results == []
