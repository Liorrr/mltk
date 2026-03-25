"""Tests for Slack notification integration."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from mltk.core.result import Severity, TestResult, TestSuite
from mltk.integrations.slack import format_slack_message, notify_slack

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_suite(passed: int = 4, failed: int = 1) -> TestSuite:
    """Build a synthetic TestSuite with the given pass/fail counts."""
    suite = TestSuite()
    for i in range(passed):
        suite.add(
            TestResult(
                name=f"assert_check_{i}",
                passed=True,
                severity=Severity.CRITICAL,
                message="OK",
            )
        )
    for i in range(failed):
        suite.add(
            TestResult(
                name=f"assert_failing_{i}",
                passed=False,
                severity=Severity.CRITICAL,
                message="threshold exceeded",
            )
        )
    return suite


def _mock_urlopen(status: int = 200) -> MagicMock:
    """Return a mock urlopen that responds with the given status."""
    resp = MagicMock()
    resp.status = status
    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=resp)
    cm.__exit__ = MagicMock(return_value=False)
    return cm


# ---------------------------------------------------------------------------
# notify_slack tests
# ---------------------------------------------------------------------------

class TestNotifySlackSuite:
    # SCENARIO: notify_slack called with a TestSuite that has failures
    # WHY: primary use-case — auto-summary should be sent when tests fail
    # EXPECTED: returns True, urllib.request.urlopen is called once
    def test_notify_slack_suite(self):
        suite = _make_suite(passed=4, failed=1)
        mock_cm = _mock_urlopen(200)

        with patch("urllib.request.urlopen", return_value=mock_cm) as mock_open:
            result = notify_slack(webhook_url="https://hooks.slack.com/test", suite=suite)

        assert result is True
        mock_open.assert_called_once()

    # SCENARIO: notify_slack called with a plain custom message (no suite)
    # WHY: standalone message mode must work without a TestSuite
    # EXPECTED: returns True, payload has "text" key not "attachments"
    def test_notify_slack_message(self):
        mock_cm = _mock_urlopen(200)
        captured: list[bytes] = []

        def capture_urlopen(req):
            captured.append(req.data)
            return mock_cm

        with patch("urllib.request.urlopen", side_effect=capture_urlopen):
            result = notify_slack(
                webhook_url="https://hooks.slack.com/test",
                message="Nightly tests complete",
            )

        assert result is True
        import json
        payload = json.loads(captured[0])
        assert "text" in payload
        assert payload["text"] == "Nightly tests complete"

    # SCENARIO: notify_slack raises ValueError when neither suite nor message is given
    # WHY: calling with no content is a programming error that should fail loudly
    # EXPECTED: ValueError raised
    def test_notify_slack_no_args_raises(self):
        with pytest.raises(ValueError, match="suite.*message"):
            notify_slack(webhook_url="https://hooks.slack.com/test")


# ---------------------------------------------------------------------------
# format_slack_message tests
# ---------------------------------------------------------------------------

class TestFormatSlackMessagePass:
    # SCENARIO: all tests pass
    # WHY: all-pass suites should use green colour and no failures block
    # EXPECTED: color is green, no failures section in blocks
    def test_format_message_pass(self):
        suite = _make_suite(passed=5, failed=0)
        payload = format_slack_message(suite)

        assert "attachments" in payload
        attachment = payload["attachments"][0]
        assert attachment["color"] == "#22c55e"

        # No failures block — only header + summary
        block_texts = [
            b.get("text", {}).get("text", "") if isinstance(b.get("text"), dict) else ""
            for b in attachment["blocks"]
        ]
        assert not any("Failed tests" in t for t in block_texts)

    # SCENARIO: all tests pass, summary text contains correct counts
    # WHY: the summary block must accurately reflect the suite state
    # EXPECTED: summary mentions "5/5" and "100%"
    def test_format_message_pass_summary_text(self):
        suite = _make_suite(passed=5, failed=0)
        payload = format_slack_message(suite)
        blocks = payload["attachments"][0]["blocks"]
        # Find the section block with the summary
        summary_block = next(b for b in blocks if b["type"] == "section")
        text = summary_block["text"]["text"]
        assert "5/5" in text
        assert "100%" in text


class TestFormatSlackMessageFail:
    # SCENARIO: suite has failures
    # WHY: failures should trigger red colour and a list of failed test names
    # EXPECTED: color is red, failed test names appear in a block
    def test_format_message_fail(self):
        suite = _make_suite(passed=3, failed=2)
        payload = format_slack_message(suite)

        attachment = payload["attachments"][0]
        assert attachment["color"] == "#ef4444"

        blocks = attachment["blocks"]
        all_text = " ".join(
            b.get("text", {}).get("text", "") if isinstance(b.get("text"), dict) else ""
            for b in blocks
        )
        assert "Failed tests" in all_text
        assert "assert_failing_0" in all_text
        assert "assert_failing_1" in all_text

    # SCENARIO: format_message_fail with more than 20 failures
    # WHY: long failure lists should be truncated with "... and N more"
    # EXPECTED: only 20 bullets shown, remainder counted in trailing text
    def test_format_message_many_failures_truncated(self):
        suite = TestSuite()
        for i in range(25):
            suite.add(
                TestResult(
                    name=f"assert_fail_{i}",
                    passed=False,
                    severity=Severity.CRITICAL,
                    message="failed",
                )
            )
        payload = format_slack_message(suite)
        all_text = " ".join(
            b.get("text", {}).get("text", "") if isinstance(b.get("text"), dict) else ""
            for b in payload["attachments"][0]["blocks"]
        )
        assert "and 5 more" in all_text


# ---------------------------------------------------------------------------
# Webhook error handling
# ---------------------------------------------------------------------------

class TestNotifySlackWebhookError:
    # SCENARIO: Slack webhook returns HTTP 400 (bad request)
    # WHY: non-200 HTTP status from Slack should not raise, just return False
    # EXPECTED: returns False
    def test_webhook_http_error_returns_false(self):
        import io
        import urllib.error

        err = urllib.error.HTTPError(
            url="https://hooks.slack.com/test",
            code=400,
            msg="Bad Request",
            hdrs=None,  # type: ignore[arg-type]
            fp=io.BytesIO(b"invalid_payload"),
        )

        with patch("urllib.request.urlopen", side_effect=err):
            result = notify_slack(
                webhook_url="https://hooks.slack.com/test",
                message="test",
            )

        assert result is False

    # SCENARIO: network is unreachable (URLError)
    # WHY: connection failures should also return False, not raise
    # EXPECTED: returns False
    def test_webhook_url_error_returns_false(self):
        import urllib.error

        err = urllib.error.URLError("Network unreachable")

        with patch("urllib.request.urlopen", side_effect=err):
            result = notify_slack(
                webhook_url="https://hooks.slack.com/test",
                message="test",
            )

        assert result is False
