"""Tests for mltk GitHub CI/CD integration helpers."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from mltk.server.github_ci import (
    _github_api,
    create_check_run,
    format_pr_comment,
    post_pr_comment,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

RESULTS_ALL_PASS = {
    "total": 3,
    "passed": 3,
    "failed": 0,
    "score": 100.0,
    "results": [
        {"name": "test_schema", "passed": True, "severity": "info",
         "message": "schema ok", "duration_ms": 8.0},
        {"name": "test_drift", "passed": True, "severity": "info",
         "message": "no drift", "duration_ms": 12.0},
        {"name": "test_bias", "passed": True, "severity": "info",
         "message": "bias within threshold", "duration_ms": 5.0},
    ],
}

RESULTS_WITH_FAILURES = {
    "total": 4,
    "passed": 2,
    "failed": 2,
    "score": 50.0,
    "results": [
        {"name": "test_schema", "passed": True, "severity": "info",
         "message": "schema ok", "duration_ms": 8.0},
        {"name": "test_drift", "passed": False, "severity": "error",
         "message": "PSI=0.18 exceeds threshold", "duration_ms": 42.0},
        {"name": "test_bias", "passed": True, "severity": "info",
         "message": "bias ok", "duration_ms": 5.0},
        {"name": "test_distribution", "passed": False, "severity": "warning",
         "message": "distribution shift detected", "duration_ms": 21.0},
    ],
}


# ---------------------------------------------------------------------------
# format_pr_comment — no network calls needed
# ---------------------------------------------------------------------------


def test_format_pr_comment_produces_markdown():
    # SCENARIO: call format_pr_comment with a mixed-result dict
    # WHY: downstream callers (post_pr_comment, users) rely on GitHub-flavored
    #      Markdown being well-formed and non-empty
    # EXPECTED: output is a non-empty string containing the mltk header
    comment = format_pr_comment(RESULTS_WITH_FAILURES)
    assert isinstance(comment, str)
    assert len(comment) > 0
    assert "## " in comment
    assert "mltk" in comment


def test_format_pr_comment_all_pass():
    # SCENARIO: format a result dict where every test passed
    # WHY: the "all pass" path must NOT include a failures table or failure
    #      language, so reviewers are not confused by phantom issues
    # EXPECTED: comment contains "All tests passed", no "Failed Tests" section,
    #           score shows 100.0%
    comment = format_pr_comment(RESULTS_ALL_PASS)
    assert "All tests passed" in comment
    assert "Failed Tests" not in comment
    assert "100.0%" in comment
    assert "3/3" in comment


def test_format_pr_comment_with_failures():
    # SCENARIO: format a result dict with two failing tests
    # WHY: reviewers must be able to see exactly which tests failed, their
    #      severity, and the failure message without leaving the PR
    # EXPECTED: "Failed Tests" section is present, both failing test names
    #           appear in the table, passing tests are excluded, score is 50%
    comment = format_pr_comment(RESULTS_WITH_FAILURES)
    assert "Failed Tests" in comment
    assert "test_drift" in comment
    assert "test_distribution" in comment
    # Passing tests must NOT appear in the failure table
    assert "test_schema" not in comment.split("Failed Tests")[1]
    assert "50.0%" in comment
    assert "2/4" in comment


def test_format_pr_comment_footer():
    # SCENARIO: inspect the footer of a formatted comment
    # WHY: the "Posted by mltk" attribution helps readers trace the comment
    #      back to the tool and surfaces the current version
    # EXPECTED: footer line references mltk and the version string
    comment = format_pr_comment(RESULTS_ALL_PASS)
    assert "mltk" in comment.split("---")[-1]
    assert "v0.6.0" in comment


def test_format_pr_comment_empty_results_list():
    # SCENARIO: results dict has counts but an empty per-test list
    # WHY: callers may omit the 'results' key (e.g. aggregated summaries);
    #      the formatter must not crash on a missing or empty list
    # EXPECTED: comment renders without error; no "Failed Tests" section
    minimal = {"total": 5, "passed": 5, "failed": 0, "score": 100.0}
    comment = format_pr_comment(minimal)
    assert "5/5" in comment
    assert "Failed Tests" not in comment


def test_format_pr_comment_pipe_escape():
    # SCENARIO: a test message contains a pipe character
    # WHY: pipes inside a GitHub Markdown table cell break the table layout;
    #      the formatter must escape them
    # EXPECTED: the rendered table cell uses \| instead of a bare |
    results = {
        "total": 1,
        "passed": 0,
        "failed": 1,
        "score": 0.0,
        "results": [
            {"name": "test_x", "passed": False, "severity": "error",
             "message": "got 0.9 | expected 1.0"},
        ],
    }
    comment = format_pr_comment(results)
    assert "\\|" in comment


# ---------------------------------------------------------------------------
# post_pr_comment — urllib mocked
# ---------------------------------------------------------------------------


def _make_mock_response(payload: dict) -> MagicMock:  # type: ignore[type-arg]
    """Return a context-manager-compatible mock for urllib.request.urlopen."""
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps(payload).encode()
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


def test_post_pr_comment_calls_api():
    # SCENARIO: post_pr_comment is called with valid repo, PR, results, token
    # WHY: the function must hit the correct GitHub Issues Comments endpoint
    #      with the right Authorization header and a non-empty body
    # EXPECTED: urlopen is called once; the request targets the correct URL;
    #           Authorization header contains the token; return value is True
    mock_resp = _make_mock_response({"id": 42, "html_url": "https://github.com/..."})

    with patch("urllib.request.urlopen", return_value=mock_resp) as mock_open:
        result = post_pr_comment(
            repo="owner/repo",
            pr_number=7,
            results=RESULTS_ALL_PASS,
            token="ghp_test_token",
        )

    assert result is True
    mock_open.assert_called_once()
    req = mock_open.call_args[0][0]  # first positional arg = Request object
    assert "owner/repo" in req.full_url
    assert "/issues/7/comments" in req.full_url
    assert req.get_header("Authorization") == "Bearer ghp_test_token"
    sent_body = json.loads(req.data)
    assert "body" in sent_body
    assert len(sent_body["body"]) > 0


def test_post_pr_comment_returns_false_on_http_error():
    # SCENARIO: the GitHub API responds with a 403 Forbidden
    # WHY: network/auth errors must not propagate as exceptions into the
    #      caller's CI pipeline; the function should degrade gracefully
    # EXPECTED: post_pr_comment returns False without raising
    import urllib.error

    with patch(
        "urllib.request.urlopen",
        side_effect=urllib.error.HTTPError(
            url="https://api.github.com/...",
            code=403,
            msg="Forbidden",
            hdrs=None,  # type: ignore[arg-type]
            fp=None,
        ),
    ):
        result = post_pr_comment("owner/repo", 1, RESULTS_ALL_PASS, "bad_token")

    assert result is False


# ---------------------------------------------------------------------------
# create_check_run — urllib mocked
# ---------------------------------------------------------------------------


def test_create_check_run_success_conclusion():
    # SCENARIO: create_check_run is called with an all-passing results dict
    # WHY: GitHub status checks must report "success" when every test passes
    #      so that branch protection rules can enforce green CI
    # EXPECTED: urlopen called once; payload conclusion is "success";
    #           function returns True
    mock_resp = _make_mock_response({"id": 99})

    with patch("urllib.request.urlopen", return_value=mock_resp) as mock_open:
        result = create_check_run(
            repo="owner/repo",
            sha="abc123def456",
            results=RESULTS_ALL_PASS,
            token="ghp_test_token",
        )

    assert result is True
    mock_open.assert_called_once()
    req = mock_open.call_args[0][0]
    sent_body = json.loads(req.data)
    assert sent_body["conclusion"] == "success"
    assert sent_body["head_sha"] == "abc123def456"
    assert sent_body["status"] == "completed"


def test_create_check_run_failure_conclusion():
    # SCENARIO: create_check_run is called with results that include failures
    # WHY: GitHub status checks must report "failure" to block PR merges when
    #      any mltk test fails — a wrong conclusion could let bad code through
    # EXPECTED: payload conclusion is "failure"; annotations list is non-empty;
    #           function returns True
    mock_resp = _make_mock_response({"id": 100})

    with patch("urllib.request.urlopen", return_value=mock_resp) as mock_open:
        result = create_check_run(
            repo="owner/repo",
            sha="deadbeef",
            results=RESULTS_WITH_FAILURES,
            token="ghp_test_token",
        )

    assert result is True
    req = mock_open.call_args[0][0]
    sent_body = json.loads(req.data)
    assert sent_body["conclusion"] == "failure"
    annotations = sent_body["output"]["annotations"]
    assert len(annotations) == 2
    annotation_titles = {a["title"] for a in annotations}
    assert "test_drift" in annotation_titles
    assert "test_distribution" in annotation_titles


def test_create_check_run_custom_name():
    # SCENARIO: caller provides a custom name for the check run
    # WHY: projects may run multiple mltk suites (e.g. "mltk-unit",
    #      "mltk-integration") and need distinct names in the GitHub UI
    # EXPECTED: the 'name' field in the request body matches the custom name
    mock_resp = _make_mock_response({"id": 101})

    with patch("urllib.request.urlopen", return_value=mock_resp) as mock_open:
        create_check_run(
            repo="owner/repo",
            sha="cafebabe",
            results=RESULTS_ALL_PASS,
            token="ghp_test_token",
            name="mltk-integration",
        )

    req = mock_open.call_args[0][0]
    sent_body = json.loads(req.data)
    assert sent_body["name"] == "mltk-integration"


def test_create_check_run_returns_false_on_url_error():
    # SCENARIO: the GitHub API is unreachable (network timeout / DNS failure)
    # WHY: CI pipelines must not crash due to transient connectivity issues;
    #      the function should degrade gracefully
    # EXPECTED: create_check_run returns False without raising
    import urllib.error

    with patch(
        "urllib.request.urlopen",
        side_effect=urllib.error.URLError("network unreachable"),
    ):
        result = create_check_run("owner/repo", "sha", RESULTS_ALL_PASS, "token")

    assert result is False


# ---------------------------------------------------------------------------
# _github_api — low-level helper
# ---------------------------------------------------------------------------


def test_github_api_sets_headers():
    # SCENARIO: call _github_api directly and inspect the outgoing request
    # WHY: all GitHub API calls must carry the Authorization, Accept, and
    #      User-Agent headers; missing headers cause silent 4xx failures
    # EXPECTED: request object has correct Authorization, Accept, User-Agent
    mock_resp = _make_mock_response({"ok": True})

    with patch("urllib.request.urlopen", return_value=mock_resp) as mock_open:
        _github_api("GET", "https://api.github.com/repos/x/y", "tok_abc")

    req = mock_open.call_args[0][0]
    assert req.get_header("Authorization") == "Bearer tok_abc"
    assert "github" in req.get_header("Accept").lower()
    assert req.get_header("User-agent").startswith("mltk")


def test_github_api_encodes_body():
    # SCENARIO: _github_api is called with a dict body
    # WHY: the body must be JSON-encoded as UTF-8 bytes; wrong encoding causes
    #      GitHub to reject the request with a 422
    # EXPECTED: request data is the JSON-encoded version of the input dict
    mock_resp = _make_mock_response({})
    payload = {"key": "value", "number": 42}

    with patch("urllib.request.urlopen", return_value=mock_resp) as mock_open:
        _github_api("POST", "https://api.github.com/x", "tok", body=payload)

    req = mock_open.call_args[0][0]
    assert json.loads(req.data) == payload


def test_github_api_empty_response():
    # SCENARIO: GitHub returns an empty body (e.g. 204 No Content)
    # WHY: the helper must not crash on an empty response; some GitHub
    #      endpoints return no body on success
    # EXPECTED: _github_api returns an empty dict without raising
    mock_resp = MagicMock()
    mock_resp.read.return_value = b""
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)

    with patch("urllib.request.urlopen", return_value=mock_resp):
        result = _github_api("DELETE", "https://api.github.com/x", "tok")

    assert result == {}
