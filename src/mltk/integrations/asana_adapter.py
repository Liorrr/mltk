"""Asana project management integration via REST API."""

from __future__ import annotations

import json
import os
import urllib.request
from typing import Any

from mltk.integrations.adapter import IssueTrackerAdapter

_BASE_URL = "https://app.asana.com/api/1.0"


class AsanaAdapter(IssueTrackerAdapter):
    """Asana task management using REST API.

    Auth via constructor arg or ``ASANA_TOKEN`` env var.
    """

    def __init__(
        self,
        token: str | None = None,
        workspace_gid: str | None = None,
    ) -> None:
        self.token = token or os.environ.get("ASANA_TOKEN", "")
        self.workspace_gid = workspace_gid or ""

    def _request(
        self, method: str, path: str, body: dict | None = None
    ) -> dict:
        url = f"{_BASE_URL}{path}"
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }
        req = urllib.request.Request(url, method=method, headers=headers)
        if body:
            req.data = json.dumps({"data": body}).encode()
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
        """Create an Asana task. Returns task GID."""
        body: dict[str, Any] = {
            "name": title,
            "notes": description,
        }
        if project:
            body["projects"] = [project]
        if self.workspace_gid:
            body["workspace"] = self.workspace_gid
        if fields:
            body.update(fields)

        result = self._request("POST", "/tasks", body)
        return result.get("data", {}).get("gid", "")

    def search_issues(self, query: str) -> list[dict]:
        """Search Asana tasks in workspace."""
        if not self.workspace_gid:
            return []
        path = (
            f"/workspaces/{self.workspace_gid}/tasks/search"
            f"?text={urllib.request.quote(query)}"
        )
        result = self._request("GET", path)
        return result.get("data", [])

    def update_issue(
        self, issue_id: str, updates: dict[str, Any]
    ) -> bool:
        """Update an Asana task."""
        body: dict[str, Any] = {}
        if "title" in updates:
            body["name"] = updates["title"]
        if "description" in updates:
            body["notes"] = updates["description"]
        if "completed" in updates:
            body["completed"] = updates["completed"]
        body.update(
            {k: v for k, v in updates.items() if k not in ("title", "description", "completed")}
        )
        result = self._request("PUT", f"/tasks/{issue_id}", body)
        return bool(result.get("data", {}).get("gid"))
