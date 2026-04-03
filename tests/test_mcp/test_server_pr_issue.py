"""Tests for mltk_create_pr and mltk_create_issue MCP tools (S90).

Covers happy paths, validation errors, external-dependency failures,
and response-shape contracts. All network / git / adapter calls are
mocked so tests run offline with no side-effects.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from ._helpers import assert_error, assert_ok, call_tool

# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

# Minimal valid finding JSON (scanner_name + nested result object)
_FINDING_JSON = json.dumps({
    "scanner_name": "drift",
    "result": {
        "name": "psi_check",
        "passed": False,
        "severity": "warning",
        "message": "PSI exceeds threshold",
    },
})

# Minimal valid fix JSON
_FIX_JSON = json.dumps({
    "category": "code",
    "title": "Retrain on recent data",
    "description": "Drift detected — retrain with a newer window.",
    "confidence": "high",
    "code_snippet": "model.fit(X_recent, y_recent)",
})

_REPO = "owner/repo"
_PR_URL = "https://github.com/owner/repo/pull/7"
_ISSUE_URL = "https://github.com/owner/repo/issues/42"
_ISSUE_KEY = "42"

_GH_CONFIG_JSON = json.dumps({"repo": _REPO, "token": "ghp_test"})
_JIRA_CONFIG_JSON = json.dumps({
    "url": "https://myorg.atlassian.net",
    "email": "bot@example.com",
    "token": "ATATT3_test",
})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pr_result(
    url: str = _PR_URL,
    branch: str = "mltk/fix-drift-abc12345",
    number: int = 7,
    draft: bool = True,
) -> MagicMock:
    """Build a mock PullRequestResult."""
    r = MagicMock()
    r.url = url
    r.branch = branch
    r.number = number
    r.draft = draft
    return r


# ---------------------------------------------------------------------------
# mltk_create_pr
# ---------------------------------------------------------------------------

class TestMltkCreatePr:
    """Tests for the mltk_create_pr MCP tool."""

    # Patch order (bottom-up decorator order → first arg is innermost):
    # git_available, find_git_root, GitHubIssuesAdapter, PullRequestGenerator

    @patch("mltk.integrations.pr_generator.PullRequestGenerator")
    @patch("mltk.integrations.github_adapter.GitHubIssuesAdapter")
    @patch("mltk.experiment.worktree.find_git_root", return_value=Path("/repo"))
    @patch("mltk.experiment.worktree.git_available", return_value=True)
    def test_create_pr_happy_path(
        self, mock_git_avail, mock_git_root, mock_adapter, mock_gen
    ):
        # SCENARIO: All deps available, valid finding + fix
        # WHY: Core happy path — should return PR metadata
        # EXPECTED: status=ok, url/branch/number present
        mock_gen_inst = MagicMock()
        mock_gen.return_value = mock_gen_inst
        mock_gen_inst.create_pr.return_value = _make_pr_result()

        result = call_tool(
            "mltk_create_pr",
            finding_json=_FINDING_JSON,
            fix_json=_FIX_JSON,
            repo=_REPO,
        )

        assert_ok(result)
        assert result["url"] == _PR_URL
        assert "branch" in result
        assert result["number"] == 7

    @patch("mltk.integrations.pr_generator.PullRequestGenerator")
    @patch("mltk.integrations.github_adapter.GitHubIssuesAdapter")
    @patch("mltk.experiment.worktree.find_git_root", return_value=Path("/repo"))
    @patch("mltk.experiment.worktree.git_available", return_value=True)
    def test_create_pr_empty_finding_json(
        self, mock_git_avail, mock_git_root, mock_adapter, mock_gen
    ):
        # SCENARIO: finding_json is an empty string
        # WHY: Must reject before attempting any git/GitHub ops
        # EXPECTED: status=error
        result = call_tool(
            "mltk_create_pr",
            finding_json="",
            fix_json=_FIX_JSON,
            repo=_REPO,
        )

        assert_error(result)

    @patch("mltk.integrations.pr_generator.PullRequestGenerator")
    @patch("mltk.integrations.github_adapter.GitHubIssuesAdapter")
    @patch("mltk.experiment.worktree.find_git_root", return_value=Path("/repo"))
    @patch("mltk.experiment.worktree.git_available", return_value=True)
    def test_create_pr_invalid_json(
        self, mock_git_avail, mock_git_root, mock_adapter, mock_gen
    ):
        # SCENARIO: finding_json is malformed JSON
        # WHY: Must surface parse error cleanly
        # EXPECTED: status=error
        result = call_tool(
            "mltk_create_pr",
            finding_json="{not valid json",
            fix_json=_FIX_JSON,
            repo=_REPO,
        )

        assert_error(result)

    @patch("mltk.integrations.pr_generator.PullRequestGenerator")
    @patch("mltk.integrations.github_adapter.GitHubIssuesAdapter")
    @patch("mltk.experiment.worktree.find_git_root", return_value=Path("/repo"))
    @patch("mltk.experiment.worktree.git_available", return_value=False)
    def test_create_pr_no_git(
        self, mock_git_avail, mock_git_root, mock_adapter, mock_gen
    ):
        # SCENARIO: git_available() returns False
        # WHY: Cannot create a PR without git — must abort early
        # EXPECTED: status=error, message mentions "git"
        result = call_tool(
            "mltk_create_pr",
            finding_json=_FINDING_JSON,
            fix_json=_FIX_JSON,
            repo=_REPO,
        )

        assert_error(result)
        assert "git" in result["error"].lower()

    @patch("mltk.integrations.pr_generator.PullRequestGenerator")
    @patch("mltk.integrations.github_adapter.GitHubIssuesAdapter")
    @patch(
        "mltk.experiment.worktree.find_git_root",
        side_effect=FileNotFoundError("Not inside a git repository"),
    )
    @patch("mltk.experiment.worktree.git_available", return_value=True)
    def test_create_pr_not_in_repo(
        self, mock_git_avail, mock_git_root, mock_adapter, mock_gen
    ):
        # SCENARIO: find_git_root raises FileNotFoundError
        # WHY: Must detect non-repo CWD and surface actionable error
        # EXPECTED: status=error, message mentions "repository"
        result = call_tool(
            "mltk_create_pr",
            finding_json=_FINDING_JSON,
            fix_json=_FIX_JSON,
            repo=_REPO,
        )

        assert_error(result)
        assert "repository" in result["error"].lower()

    @patch("mltk.integrations.pr_generator.PullRequestGenerator")
    @patch("mltk.integrations.github_adapter.GitHubIssuesAdapter")
    @patch("mltk.experiment.worktree.find_git_root", return_value=Path("/repo"))
    @patch("mltk.experiment.worktree.git_available", return_value=True)
    def test_create_pr_draft_false(
        self, mock_git_avail, mock_git_root, mock_adapter, mock_gen
    ):
        # SCENARIO: Caller sets draft=False
        # WHY: draft flag must be forwarded to generator.create_pr
        # EXPECTED: create_pr called with draft=False; response reflects it
        mock_gen_inst = MagicMock()
        mock_gen.return_value = mock_gen_inst
        mock_gen_inst.create_pr.return_value = _make_pr_result(draft=False)

        result = call_tool(
            "mltk_create_pr",
            finding_json=_FINDING_JSON,
            fix_json=_FIX_JSON,
            repo=_REPO,
            draft=False,
        )

        assert_ok(result)
        assert result["draft"] is False
        _, call_kwargs = mock_gen_inst.create_pr.call_args
        assert call_kwargs.get("draft") is False or (
            mock_gen_inst.create_pr.call_args[0]
            and mock_gen_inst.create_pr.call_args[0][3] is False
        )

    @patch("mltk.integrations.pr_generator.PullRequestGenerator")
    @patch("mltk.integrations.github_adapter.GitHubIssuesAdapter")
    @patch("mltk.experiment.worktree.find_git_root", return_value=Path("/repo"))
    @patch("mltk.experiment.worktree.git_available", return_value=True)
    def test_create_pr_push_failure(
        self, mock_git_avail, mock_git_root, mock_adapter, mock_gen
    ):
        # SCENARIO: generator.create_pr raises RuntimeError (push failed)
        # WHY: Push errors must be surfaced as error responses, not exceptions
        # EXPECTED: status=error
        mock_gen_inst = MagicMock()
        mock_gen.return_value = mock_gen_inst
        mock_gen_inst.create_pr.side_effect = RuntimeError(
            "Failed to push branch 'mltk/fix-drift-abc' to remote"
        )

        result = call_tool(
            "mltk_create_pr",
            finding_json=_FINDING_JSON,
            fix_json=_FIX_JSON,
            repo=_REPO,
        )

        assert_error(result)

    @patch("mltk.integrations.pr_generator.PullRequestGenerator")
    @patch("mltk.integrations.github_adapter.GitHubIssuesAdapter")
    @patch("mltk.experiment.worktree.find_git_root", return_value=Path("/repo"))
    @patch("mltk.experiment.worktree.git_available", return_value=True)
    def test_create_pr_response_shape(
        self, mock_git_avail, mock_git_root, mock_adapter, mock_gen
    ):
        # SCENARIO: Successful PR creation
        # WHY: Response contract must include all 5 expected keys
        # EXPECTED: url, branch, number, draft, suggested_next_step present
        mock_gen_inst = MagicMock()
        mock_gen.return_value = mock_gen_inst
        mock_gen_inst.create_pr.return_value = _make_pr_result()

        result = call_tool(
            "mltk_create_pr",
            finding_json=_FINDING_JSON,
            fix_json=_FIX_JSON,
            repo=_REPO,
        )

        assert_ok(result)
        for key in ("url", "branch", "number", "draft", "suggested_next_step"):
            assert key in result, f"Missing key: {key!r}"
        assert isinstance(result["suggested_next_step"], str)
        assert len(result["suggested_next_step"]) > 0


# ---------------------------------------------------------------------------
# mltk_create_issue
# ---------------------------------------------------------------------------

class TestMltkCreateIssue:
    """Tests for the mltk_create_issue MCP tool."""

    @patch("mltk.integrations.issue_linker.IssueLinker")
    @patch("mltk.integrations.github_adapter.GitHubIssuesAdapter")
    def test_create_issue_github_happy_path(self, mock_adapter, mock_linker):
        # SCENARIO: tracker="github", config supplies repo + token
        # WHY: Core happy path for GitHub Issues backend
        # EXPECTED: status=ok, issue_url contains issues path
        mock_linker_inst = MagicMock()
        mock_linker.return_value = mock_linker_inst
        mock_linker_inst.create_from_finding.return_value = _ISSUE_URL

        result = call_tool(
            "mltk_create_issue",
            finding_json=_FINDING_JSON,
            tracker="github",
            config_json=_GH_CONFIG_JSON,
        )

        assert_ok(result)
        assert "issues/42" in result["issue_url"]

    @patch("mltk.integrations.issue_linker.IssueLinker")
    @patch("mltk.integrations.jira_adapter.JiraAdapter")
    def test_create_issue_jira_happy_path(self, mock_jira, mock_linker):
        # SCENARIO: tracker="jira", config supplies url/email/token
        # WHY: Jira backend must be wired up and return a key
        # EXPECTED: status=ok, issue_key present (e.g., "ML-42")
        mock_linker_inst = MagicMock()
        mock_linker.return_value = mock_linker_inst
        mock_linker_inst.create_from_finding.return_value = "ML-42"

        result = call_tool(
            "mltk_create_issue",
            finding_json=_FINDING_JSON,
            tracker="jira",
            project="ML",
            config_json=_JIRA_CONFIG_JSON,
        )

        assert_ok(result)
        assert result["issue_key"] == "ML-42"

    @patch("mltk.integrations.issue_linker.IssueLinker")
    @patch("mltk.integrations.github_adapter.GitHubIssuesAdapter")
    def test_create_issue_invalid_tracker(self, mock_adapter, mock_linker):
        # SCENARIO: tracker="gitlab" is not a supported value
        # WHY: Unknown trackers must be rejected with actionable error
        # EXPECTED: status=error, message mentions "tracker"
        result = call_tool(
            "mltk_create_issue",
            finding_json=_FINDING_JSON,
            tracker="gitlab",
            config_json=_GH_CONFIG_JSON,
        )

        assert_error(result)
        assert "tracker" in result["error"].lower()

    @patch("mltk.integrations.issue_linker.IssueLinker")
    @patch("mltk.integrations.github_adapter.GitHubIssuesAdapter")
    def test_create_issue_empty_finding_json(self, mock_adapter, mock_linker):
        # SCENARIO: finding_json is an empty string
        # WHY: Must reject before constructing any adapter
        # EXPECTED: status=error
        result = call_tool(
            "mltk_create_issue",
            finding_json="",
            tracker="github",
            config_json=_GH_CONFIG_JSON,
        )

        assert_error(result)

    @patch("mltk.integrations.issue_linker.IssueLinker")
    @patch("mltk.integrations.github_adapter.GitHubIssuesAdapter")
    def test_create_issue_dedup_skip(self, mock_adapter, mock_linker):
        # SCENARIO: create_from_finding returns None (dedup decided to skip)
        # WHY: Dedup skip must be surfaced as ok with null issue_url
        # EXPECTED: status=ok, issue_url is None
        mock_linker_inst = MagicMock()
        mock_linker.return_value = mock_linker_inst
        mock_linker_inst.create_from_finding.return_value = None

        result = call_tool(
            "mltk_create_issue",
            finding_json=_FINDING_JSON,
            tracker="github",
            config_json=_GH_CONFIG_JSON,
        )

        assert_ok(result)
        assert result["issue_url"] is None

    @patch("mltk.integrations.issue_linker.IssueLinker")
    @patch("mltk.integrations.github_adapter.GitHubIssuesAdapter")
    def test_create_issue_with_pr_link(self, mock_adapter, mock_linker):
        # SCENARIO: pr_url is provided alongside the finding
        # WHY: Tool must call linker.link_pr after creating the issue
        # EXPECTED: status=ok, linked_pr equals the provided pr_url
        mock_linker_inst = MagicMock()
        mock_linker.return_value = mock_linker_inst
        mock_linker_inst.create_from_finding.return_value = _ISSUE_URL
        mock_linker_inst.link_pr.return_value = True

        result = call_tool(
            "mltk_create_issue",
            finding_json=_FINDING_JSON,
            tracker="github",
            config_json=_GH_CONFIG_JSON,
            pr_url=_PR_URL,
        )

        assert_ok(result)
        assert result["linked_pr"] == _PR_URL
        mock_linker_inst.link_pr.assert_called_once()

    @patch("mltk.integrations.issue_linker.IssueLinker")
    @patch("mltk.integrations.github_adapter.GitHubIssuesAdapter")
    def test_create_issue_response_shape(self, mock_adapter, mock_linker):
        # SCENARIO: Successful issue creation
        # WHY: Response contract must expose all 4 expected keys
        # EXPECTED: issue_key, issue_url, linked_pr, suggested_next_step
        mock_linker_inst = MagicMock()
        mock_linker.return_value = mock_linker_inst
        mock_linker_inst.create_from_finding.return_value = _ISSUE_URL

        result = call_tool(
            "mltk_create_issue",
            finding_json=_FINDING_JSON,
            tracker="github",
            config_json=_GH_CONFIG_JSON,
        )

        assert_ok(result)
        for key in ("issue_key", "issue_url", "linked_pr", "suggested_next_step"):
            assert key in result, f"Missing key: {key!r}"
        assert isinstance(result["suggested_next_step"], str)
        assert len(result["suggested_next_step"]) > 0

    @patch("mltk.integrations.issue_linker.IssueLinker")
    @patch("mltk.integrations.github_adapter.GitHubIssuesAdapter")
    def test_create_issue_invalid_finding_json(self, mock_adapter, mock_linker):
        # SCENARIO: finding_json is malformed JSON
        # WHY: Parse error must be caught and returned as error response
        # EXPECTED: status=error
        result = call_tool(
            "mltk_create_issue",
            finding_json="{broken",
            tracker="github",
            config_json=_GH_CONFIG_JSON,
        )

        assert_error(result)

    @patch("mltk.integrations.issue_linker.IssueLinker")
    @patch("mltk.integrations.github_adapter.GitHubIssuesAdapter")
    def test_create_issue_no_pr_link_skips_link_pr(self, mock_adapter, mock_linker):
        # SCENARIO: pr_url not provided (empty string default)
        # WHY: link_pr must NOT be called when no PR URL is given
        # EXPECTED: status=ok, linker.link_pr not called, linked_pr is falsy
        mock_linker_inst = MagicMock()
        mock_linker.return_value = mock_linker_inst
        mock_linker_inst.create_from_finding.return_value = _ISSUE_URL

        result = call_tool(
            "mltk_create_issue",
            finding_json=_FINDING_JSON,
            tracker="github",
            config_json=_GH_CONFIG_JSON,
        )

        assert_ok(result)
        mock_linker_inst.link_pr.assert_not_called()
        assert not result["linked_pr"]

    @patch("mltk.integrations.issue_linker.IssueLinker")
    @patch("mltk.integrations.github_adapter.GitHubIssuesAdapter")
    def test_create_issue_adapter_error(self, mock_adapter, mock_linker):
        # SCENARIO: linker.create_from_finding raises RuntimeError
        # WHY: Adapter-level errors must be caught and returned as error
        # EXPECTED: status=error
        mock_linker_inst = MagicMock()
        mock_linker.return_value = mock_linker_inst
        mock_linker_inst.create_from_finding.side_effect = RuntimeError(
            "GitHub API error 422"
        )

        result = call_tool(
            "mltk_create_issue",
            finding_json=_FINDING_JSON,
            tracker="github",
            config_json=_GH_CONFIG_JSON,
        )

        assert_error(result)

    @patch("mltk.integrations.issue_linker.IssueLinker")
    @patch("mltk.integrations.github_adapter.GitHubIssuesAdapter")
    def test_create_issue_invalid_config_json(self, mock_adapter, mock_linker):
        # SCENARIO: config_json is not valid JSON
        # WHY: Config parse error must be caught before adapter construction
        # EXPECTED: status=error
        result = call_tool(
            "mltk_create_issue",
            finding_json=_FINDING_JSON,
            tracker="github",
            config_json="{not json",
        )

        assert_error(result)
