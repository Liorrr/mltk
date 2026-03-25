"""Issue tracker adapter — vendor-agnostic base class for PM integrations.

Implement this for Jira, Linear, GitHub Issues, Asana, etc.
mltk code only depends on this interface, never on a specific vendor.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class IssueTrackerAdapter(ABC):
    """Abstract base class for issue tracking integrations.

    Example:
        >>> class MyAdapter(IssueTrackerAdapter):
        ...     def create_issue(self, project, title, description, fields):
        ...         return "TICKET-123"
    """

    @abstractmethod
    def create_issue(
        self,
        project: str,
        title: str,
        description: str,
        fields: dict[str, Any] | None = None,
    ) -> str:
        """Create a new issue. Returns issue key/ID.

        Args:
            project: Project key (e.g., "ML").
            title: Issue title/summary.
            description: Issue description/body.
            fields: Additional fields (severity, labels, custom ML fields).

        Returns:
            Issue key or ID string.
        """

    @abstractmethod
    def search_issues(self, query: str) -> list[dict[str, Any]]:
        """Search for matching issues.

        Args:
            query: Search query (JQL for Jira, text for others).

        Returns:
            List of issue dicts with at least 'key' and 'summary'.
        """

    @abstractmethod
    def update_issue(self, issue_id: str, updates: dict[str, Any]) -> bool:
        """Update an existing issue.

        Args:
            issue_id: Issue key/ID.
            updates: Fields to update.

        Returns:
            True if updated successfully.
        """
