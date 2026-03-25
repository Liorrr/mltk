"""Slack notifications — post test failure summaries via incoming webhook.

No external dependencies — uses urllib (stdlib) for HTTP POST.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from mltk.core.result import TestSuite


def notify_slack(
    webhook_url: str,
    suite: TestSuite | None = None,
    message: str | None = None,
    channel: str | None = None,
) -> bool:
    """Post a message to Slack via an incoming webhook.

    If ``suite`` is provided, auto-generates a rich Block Kit summary:
    ``"mltk: 45/50 passed (90%) | 5 failures: [list]"``

    If both ``suite`` and ``message`` are provided, the custom ``message``
    is shown as a context block beneath the suite summary.

    Args:
        webhook_url: Slack incoming webhook URL
                     (e.g., ``"https://hooks.slack.com/services/T.../B.../..."``).
        suite: ``TestSuite`` to summarise. Takes priority for payload generation.
        message: Custom plain-text message (used when no suite is provided,
                 or appended as context when a suite is also provided).
        channel: Override the webhook's default channel (optional).

    Returns:
        ``True`` if Slack returned HTTP 200, ``False`` otherwise.
    """
    if suite is not None:
        payload = format_slack_message(suite)
        if message:
            payload.setdefault("blocks", []).append(
                {
                    "type": "context",
                    "elements": [{"type": "mrkdwn", "text": message}],
                }
            )
    elif message is not None:
        payload = {"text": message}
    else:
        raise ValueError("Either suite or message must be provided")

    if channel:
        payload["channel"] = channel

    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        webhook_url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status == 200
    except urllib.error.HTTPError:
        return False
    except urllib.error.URLError:
        return False


def format_slack_message(suite: TestSuite) -> dict[str, Any]:
    """Format a ``TestSuite`` as a Slack Block Kit webhook payload.

    Structure:
    - Header block: "mltk Test Results"
    - Section block: pass/fail counts + score percentage
    - Section block (if failures): bullet list of failed test names
    - Colour attachment border: green (all pass) or red (any failure)

    Args:
        suite: The test suite to format.

    Returns:
        Dict suitable for POSTing to a Slack incoming webhook.
    """
    all_pass = suite.failed_count == 0
    color = "#22c55e" if all_pass else "#ef4444"
    status_emoji = ":white_check_mark:" if all_pass else ":x:"
    score_text = f"{suite.passed_count}/{suite.total} passed ({suite.score:.0f}%)"

    header_block: dict[str, Any] = {
        "type": "header",
        "text": {
            "type": "plain_text",
            "text": "mltk Test Results",
            "emoji": True,
        },
    }

    summary_block: dict[str, Any] = {
        "type": "section",
        "text": {
            "type": "mrkdwn",
            "text": f"{status_emoji} *{score_text}*",
        },
        "fields": [
            {"type": "mrkdwn", "text": f"*Passed:* {suite.passed_count}"},
            {"type": "mrkdwn", "text": f"*Failed:* {suite.failed_count}"},
            {"type": "mrkdwn", "text": f"*Total:* {suite.total}"},
            {"type": "mrkdwn", "text": f"*Score:* {suite.score:.1f}%"},
        ],
    }

    blocks: list[dict[str, Any]] = [header_block, summary_block]

    if suite.failed_count > 0:
        failed_names = [r.name for r in suite.results if not r.passed]
        bullets = "\n".join(f"• {name}" for name in failed_names[:20])
        if len(failed_names) > 20:
            bullets += f"\n_... and {len(failed_names) - 20} more_"

        failures_block: dict[str, Any] = {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*Failed tests:*\n{bullets}",
            },
        }
        blocks.append(failures_block)

    # Slack attachment-style coloured border (legacy attachments support colour)
    return {
        "attachments": [
            {
                "color": color,
                "blocks": blocks,
            }
        ]
    }
