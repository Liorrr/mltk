"""Linear.app issue tracker integration via GraphQL API."""

from __future__ import annotations

import json
import os
import urllib.request
from typing import Any

from mltk.integrations.adapter import IssueTrackerAdapter

_GRAPHQL_URL = "https://api.linear.app/graphql"


class LinearAdapter(IssueTrackerAdapter):
    """Linear issue tracker using GraphQL API.

    Auth via constructor arg or ``LINEAR_API_KEY`` env var.
    """

    def __init__(
        self,
        api_key: str | None = None,
        team_id: str | None = None,
    ) -> None:
        self.api_key = api_key or os.environ.get("LINEAR_API_KEY", "")
        self.team_id = team_id or ""

    def _graphql(self, query: str, variables: dict | None = None) -> dict:
        body = {"query": query}
        if variables:
            body["variables"] = variables
        data = json.dumps(body).encode()
        req = urllib.request.Request(
            _GRAPHQL_URL,
            data=data,
            headers={
                "Authorization": self.api_key,
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read())
        except Exception:
            return {}

    def create_issue(
        self,
        project: str,
        title: str,
        description: str,
        fields: dict[str, Any] | None = None,
    ) -> str:
        """Create a Linear issue. Returns issue identifier."""
        team = self.team_id or project
        mutation = """
        mutation($title: String!, $desc: String!, $teamId: String!) {
            issueCreate(input: {title: $title, description: $desc, teamId: $teamId}) {
                issue { identifier url }
            }
        }
        """
        result = self._graphql(
            mutation,
            {"title": title, "desc": description, "teamId": team},
        )
        issue = (
            result.get("data", {})
            .get("issueCreate", {})
            .get("issue", {})
        )
        return issue.get("identifier", "")

    def search_issues(self, query: str) -> list[dict]:
        """Search Linear issues by text query."""
        gql = """
        query($q: String!) {
            issueSearch(query: $q, first: 20) {
                nodes { identifier title state { name } url }
            }
        }
        """
        result = self._graphql(gql, {"q": query})
        nodes = (
            result.get("data", {})
            .get("issueSearch", {})
            .get("nodes", [])
        )
        return nodes

    def update_issue(
        self, issue_id: str, updates: dict[str, Any]
    ) -> bool:
        """Update a Linear issue."""
        mutation = """
        mutation($id: String!, $title: String, $desc: String) {
            issueUpdate(id: $id, input: {title: $title, description: $desc}) {
                success
            }
        }
        """
        result = self._graphql(
            mutation,
            {
                "id": issue_id,
                "title": updates.get("title"),
                "desc": updates.get("description"),
            },
        )
        return (
            result.get("data", {})
            .get("issueUpdate", {})
            .get("success", False)
        )
