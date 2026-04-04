"""GitHub Issues adapter — create/search/update issues via GitHub API.

Uses urllib (stdlib) — no external dependencies required.
Auth via personal access token or GITHUB_TOKEN environment variable.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from mltk.integrations.adapter import IssueTrackerAdapter

_GITHUB_API = "https://api.github.com"


class GitHubIssuesAdapter(IssueTrackerAdapter):
    """GitHub Issues integration using REST API v3.

    No external dependencies — uses urllib (stdlib) for all HTTP calls.
    Authenticates via personal access token (PAT) or GITHUB_TOKEN env var.

    Args:
        repo: Repository in ``owner/name`` format (e.g., ``"acme/ml-service"``).
        token: Personal access token. Falls back to ``GITHUB_TOKEN`` env var.

    Example:
        >>> adapter = GitHubIssuesAdapter("myorg/myrepo", token="ghp_...")
        >>> url = adapter.create_issue("myrepo", "Drift detected", "PSI > 0.2 on feature X")
    """

    def __init__(self, repo: str, token: str | None = None) -> None:
        self.repo = repo
        self.token = token or os.environ.get("GITHUB_TOKEN")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        """Build request headers, including auth if token is set."""
        headers: dict[str, str] = {
            "Accept": "application/vnd.github+json",
            "Content-Type": "application/json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if self.token:
            headers["Authorization"] = f"token {self.token}"
        return headers

    def _request(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
    ) -> tuple[int, Any]:
        """Execute an HTTP request against the GitHub API.

        Returns:
            Tuple of (status_code, response_data).
        """
        url = f"{_GITHUB_API}{path}"
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(url, data=data, headers=self._headers(), method=method)
        try:
            with urllib.request.urlopen(req) as resp:
                return resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            try:
                err_body = json.loads(exc.read())
            except (json.JSONDecodeError, ValueError):
                err_body = {}
            return exc.code, err_body

    # ------------------------------------------------------------------
    # IssueTrackerAdapter implementation
    # ------------------------------------------------------------------

    def create_issue(
        self,
        project: str,
        title: str,
        description: str,
        fields: dict[str, Any] | None = None,
    ) -> str:
        """Create a GitHub issue.

        Args:
            project: Ignored for GitHub (repo is set at construction time).
                     Accepted for interface compatibility.
            title: Issue title.
            description: Issue body (Markdown supported).
            fields: Optional extras — ``labels`` (list[str]),
                    ``assignees`` (list[str]), ``milestone`` (int).

        Returns:
            URL of the created issue (e.g., ``"https://github.com/owner/repo/issues/42"``).

        Raises:
            RuntimeError: If the GitHub API returns a non-2xx status.
        """
        payload: dict[str, Any] = {"title": title, "body": description}
        if fields:
            for key in ("labels", "assignees", "milestone"):
                if key in fields:
                    payload[key] = fields[key]

        status, data = self._request("POST", f"/repos/{self.repo}/issues", payload)
        if status not in (200, 201):
            message = data.get("message", "unknown error") if isinstance(data, dict) else str(data)
            raise RuntimeError(f"GitHub API error {status}: {message}")

        return str(data.get("html_url", ""))

    def search_issues(self, query: str) -> list[dict[str, Any]]:
        """Search GitHub issues using the search API.

        Args:
            query: Search query string. GitHub search qualifiers are supported
                   (e.g., ``"drift repo:myorg/myrepo is:open"``).
                   The adapter automatically scopes the search to ``repo:{self.repo}``
                   if no ``repo:`` qualifier is already present.

        Returns:
            List of issue dicts, each containing at least ``key`` (issue number
            as string) and ``summary`` (issue title), plus ``url`` and ``state``.
        """
        scoped_query = query
        if "repo:" not in query:
            scoped_query = f"{query} repo:{self.repo}"

        encoded = urllib.parse.quote(scoped_query)
        status, data = self._request("GET", f"/search/issues?q={encoded}&per_page=50")

        if status != 200:
            return []

        items = data.get("items", []) if isinstance(data, dict) else []
        return [
            {
                "key": str(item["number"]),
                "summary": item.get("title", ""),
                "url": item.get("html_url", ""),
                "state": item.get("state", ""),
            }
            for item in items
        ]

    def create_pull_request(
        self,
        head: str,
        base: str,
        title: str,
        body: str = "",
        draft: bool = True,
        labels: list[str] | None = None,
    ) -> dict[str, Any]:
        """Create a GitHub pull request.

        Args:
            head: Branch containing the changes.
            base: Branch to merge into (e.g., ``"main"``).
            title: PR title.
            body: PR description (Markdown supported).
            draft: Create as draft PR when ``True`` (default).
            labels: Optional label names to attach after creation.

        Returns:
            Full GitHub API response dict (includes ``html_url``,
            ``number``, ``draft``, etc.).

        Raises:
            RuntimeError: If the GitHub API returns a non-2xx status.
        """
        payload: dict[str, Any] = {
            "title": title,
            "head": head,
            "base": base,
            "body": body,
            "draft": draft,
        }

        status, data = self._request("POST", f"/repos/{self.repo}/pulls", payload)
        if status not in (200, 201):
            message = data.get("message", "unknown error") if isinstance(data, dict) else str(data)
            raise RuntimeError(f"GitHub API error {status}: {message}")

        # Attach labels via the Issues API (PRs share the issues endpoint)
        if labels and isinstance(data, dict):
            pr_number = data.get("number")
            if pr_number:
                self._request(
                    "POST",
                    f"/repos/{self.repo}/issues/{pr_number}/labels",
                    {"labels": labels},
                )

        return data if isinstance(data, dict) else {}

    def update_issue(self, issue_id: str, updates: dict[str, Any]) -> bool:
        """Update a GitHub issue or add a comment.

        Supported ``updates`` keys:

        - ``comment`` (str): Add a comment to the issue.
        - ``state`` (str): ``"open"`` or ``"closed"``.
        - ``labels`` (list[str]): Replace the issue's labels.
        - ``title`` (str): Update the issue title.
        - ``body`` (str): Update the issue body.

        Args:
            issue_id: Issue number as a string (e.g., ``"42"``).
            updates: Dict of fields to update.

        Returns:
            True if all requested updates succeeded, False on any error.
        """
        success = True

        # Post a comment if requested
        if "comment" in updates:
            status, _ = self._request(
                "POST",
                f"/repos/{self.repo}/issues/{issue_id}/comments",
                {"body": updates["comment"]},
            )
            if status not in (200, 201):
                success = False

        # Patch the issue itself for everything else
        patch_fields = {k: v for k, v in updates.items() if k != "comment"}
        if patch_fields:
            status, _ = self._request(
                "PATCH",
                f"/repos/{self.repo}/issues/{issue_id}",
                patch_fields,
            )
            if status not in (200, 201):
                success = False

        return success
