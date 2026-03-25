"""PM tool integrations — Jira, Linear, GitHub Issues (adapter pattern), MLflow, and Slack."""

from mltk.integrations.adapter import IssueTrackerAdapter
from mltk.integrations.dedup import TicketDecisionEngine
from mltk.integrations.github_adapter import GitHubIssuesAdapter
from mltk.integrations.jira_adapter import JiraAdapter
from mltk.integrations.mlflow_logger import MlflowLogger
from mltk.integrations.slack import format_slack_message, notify_slack
from mltk.integrations.templates import render_ticket

__all__ = [
    "IssueTrackerAdapter",
    "GitHubIssuesAdapter",
    "JiraAdapter",
    "MlflowLogger",
    "TicketDecisionEngine",
    "format_slack_message",
    "notify_slack",
    "render_ticket",
]
