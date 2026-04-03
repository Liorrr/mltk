"""Issue linker — create tracker tickets from scan findings and link PRs.

Bridges :class:`ScanFinding` objects to any :class:`IssueTrackerAdapter`
(Jira, GitHub Issues, Linear, Asana) with deduplication and template
rendering.
"""

from __future__ import annotations

from typing import Any

from mltk.integrations.adapter import IssueTrackerAdapter
from mltk.integrations.dedup import TicketDecisionEngine
from mltk.integrations.templates import render_ticket
from mltk.scan.finding import ScanFinding

_SEV_TO_DEDUP: dict[str, str] = {
    "CRITICAL": "CRITICAL",
    "WARNING": "HIGH",
    "INFO": "LOW",
}


def _finding_to_failure(finding: ScanFinding) -> dict[str, Any]:
    """Convert a :class:`ScanFinding` to the failure dict expected by
    :class:`TicketDecisionEngine` and :func:`render_ticket`.

    The :class:`~mltk.core.result.Severity` enum uses ``CRITICAL``,
    ``WARNING``, ``INFO`` but :class:`TicketDecisionEngine` expects
    ``CRITICAL``, ``HIGH``, ``MEDIUM``, ``LOW``.  This function
    translates between the two vocabularies.

    Args:
        finding: The scan finding to convert.

    Returns:
        A dict with keys ``test_name``, ``severity``, ``assertion_type``,
        ``metric_name``, ``message``, and ``timestamp``.
    """
    timestamp = ""
    if hasattr(finding.result, "timestamp") and finding.result.timestamp is not None:
        timestamp = finding.result.timestamp.isoformat()

    raw_sev = finding.result.severity.value.upper()

    return {
        "test_name": finding.result.name,
        "severity": _SEV_TO_DEDUP.get(raw_sev, raw_sev),
        "assertion_type": finding.scanner_name,
        "metric_name": finding.scanner_name,
        "message": finding.result.message,
        "timestamp": timestamp,
    }


class IssueLinker:
    """Thin coordination layer that turns scan findings into tracker tickets.

    Creates issues via an :class:`IssueTrackerAdapter`, applies deduplication
    via :class:`TicketDecisionEngine`, and renders ticket content via
    :func:`render_ticket`.  Can also link a pull-request URL to an existing
    issue by adding a comment.

    Args:
        adapter: The issue tracker adapter to use for creating/updating issues.
        dedup: Deduplication engine.  A default
            :class:`TicketDecisionEngine` is created when *None* is passed.

    Example:
        >>> linker = IssueLinker(adapter=my_adapter)
        >>> issue_key = linker.create_from_finding(finding, project="ML")
        >>> linker.link_pr(issue_key, "https://github.com/owner/repo/pull/7")
    """

    def __init__(
        self,
        adapter: IssueTrackerAdapter,
        dedup: TicketDecisionEngine | None = None,
    ) -> None:
        self._adapter = adapter
        self._dedup = dedup if dedup is not None else TicketDecisionEngine()

    def create_from_finding(
        self,
        finding: ScanFinding,
        project: str,
        template: str = "default",
    ) -> str | None:
        """Create a tracker issue from a scan finding.

        The finding is first converted to a failure dict, checked for
        deduplication, rendered with the requested template, and then
        submitted to the adapter.

        Args:
            finding: The scan finding that triggered this ticket.
            project: Project key or identifier forwarded to the adapter
                (e.g. ``"ML"`` for Jira, ``"owner/repo"`` for GitHub).
            template: Template name passed to :func:`render_ticket`.
                Supported values: ``"data_quality"``, ``"model_regression"``,
                ``"drift_detection"``, ``"bias_violation"``, ``"default"``.

        Returns:
            The issue key / URL returned by the adapter, or ``None`` if
            deduplication decided the ticket should be skipped.
        """
        failure = _finding_to_failure(finding)

        if not self._dedup.should_create(failure):
            return None

        ticket = render_ticket(template, **failure)
        return self._adapter.create_issue(
            project,
            ticket["title"],
            ticket["description"],
        )

    def link_pr(self, issue_id: str, pr_url: str) -> bool:
        """Add a pull-request link as a comment on an existing issue.

        Args:
            issue_id: The issue key / ID to update.
            pr_url: Full URL of the pull request.

        Returns:
            ``True`` if the adapter reported a successful update,
            ``False`` otherwise.
        """
        comment = f"**Fix PR:** {pr_url}\n\n*Linked by mltk experiment runner.*"
        return self._adapter.update_issue(issue_id, {"comment": comment})
