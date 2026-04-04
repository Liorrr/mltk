"""Jira Cloud/Server adapter — creates and manages issues via Jira REST API.

Requires: pip install jira
"""

from __future__ import annotations

import logging
from typing import Any

from mltk.integrations.adapter import IssueTrackerAdapter

logger = logging.getLogger(__name__)


class JiraAdapter(IssueTrackerAdapter):
    """Jira Cloud and Data Center adapter.

    Args:
        instance_url: Jira instance URL (e.g., "https://myorg.atlassian.net").
        email: User email for authentication.
        api_token: API token (generate at id.atlassian.com).
        custom_fields: Mapping of ML field names to Jira custom field IDs.

    Example:
        >>> adapter = JiraAdapter(
        ...     instance_url="https://mycompany.atlassian.net",
        ...     email="ml-bot@mycompany.com",
        ...     api_token="ATATT3...",
        ... )
        >>> key = adapter.create_issue("ML", "Drift detected", "PSI > 0.2 on feature X")
    """

    def __init__(
        self,
        instance_url: str,
        email: str,
        api_token: str,
        custom_fields: dict[str, str] | None = None,
    ) -> None:
        self.instance_url = instance_url
        self.email = email
        self.api_token = api_token
        self.custom_fields = custom_fields or {}
        self._jira: Any = None

    def _get_client(self) -> Any:
        """Lazy-initialize Jira client."""
        if self._jira is None:
            try:
                from jira import JIRA
            except ImportError as err:
                raise ImportError(
                    "jira is required for JiraAdapter. "
                    "Install it with: pip install jira"
                ) from err
            self._jira = JIRA(
                self.instance_url,
                basic_auth=(self.email, self.api_token),
            )
        return self._jira

    def create_issue(
        self,
        project: str,
        title: str,
        description: str,
        fields: dict[str, Any] | None = None,
    ) -> str:
        """Create a Jira issue.

        Args:
            project: Jira project key.
            title: Issue summary.
            description: Issue description.
            fields: Additional fields including ML-specific custom fields.

        Returns:
            Jira issue key (e.g., "ML-42").
        """
        client = self._get_client()
        issue_dict: dict[str, Any] = {
            "project": {"key": project},
            "summary": title,
            "description": description,
            "issuetype": {"name": "Bug"},
        }

        # Map ML custom fields
        if fields:
            for field_name, value in fields.items():
                if field_name in self.custom_fields:
                    issue_dict[self.custom_fields[field_name]] = value

        issue = client.create_issue(fields=issue_dict)
        return str(issue.key)

    def search_issues(self, query: str) -> list[dict[str, Any]]:
        """Search Jira issues with JQL.

        Args:
            query: JQL query string.

        Returns:
            List of issue dicts.
        """
        client = self._get_client()
        issues = client.search_issues(query, maxResults=50)
        return [
            {"key": str(i.key), "summary": str(i.fields.summary)}
            for i in issues
        ]

    def add_remote_link(
        self,
        issue_id: str,
        url: str,
        title: str,
    ) -> bool:
        """Add a remote link (e.g. a PR URL) to a Jira issue.

        Uses the Jira remote-links API to create a web link visible in
        the issue's *Development* panel.

        Args:
            issue_id: Jira issue key (e.g., ``"ML-42"``).
            url: Full URL to link (e.g., a GitHub PR URL).
            title: Display title for the link.

        Returns:
            ``True`` if the link was created successfully.
        """
        try:
            client = self._get_client()
            client.add_remote_link(
                issue_id,
                {"url": url, "title": title},
            )
            return True
        except Exception:
            logger.warning("Failed to add remote link to %s", issue_id, exc_info=True)
            return False

    def update_issue(self, issue_id: str, updates: dict[str, Any]) -> bool:
        """Update a Jira issue.

        Args:
            issue_id: Jira issue key.
            updates: Fields to update.

        Returns:
            True if successful.
        """
        try:
            client = self._get_client()

            # Handle "comment" key separately — Jira SDK uses
            # add_comment(), not a field update keyword.
            if "comment" in updates:
                client.add_comment(issue_id, updates["comment"])

            patch_fields = {k: v for k, v in updates.items() if k != "comment"}
            if patch_fields:
                issue = client.issue(issue_id)
                issue.update(**patch_fields)

            return True
        except Exception:
            logger.warning("Failed to update issue %s", issue_id, exc_info=True)
            return False
