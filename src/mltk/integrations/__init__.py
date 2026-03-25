"""PM tool integrations — Jira, Linear, GitHub Issues (adapter pattern)."""

from mltk.integrations.adapter import IssueTrackerAdapter
from mltk.integrations.dedup import TicketDecisionEngine
from mltk.integrations.jira_adapter import JiraAdapter
from mltk.integrations.templates import render_ticket

__all__ = [
    "IssueTrackerAdapter",
    "JiraAdapter",
    "TicketDecisionEngine",
    "render_ticket",
]
