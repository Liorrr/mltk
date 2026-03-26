"""Webhook dispatch — notify external services on test run events."""
from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass


@dataclass
class WebhookConfig:
    """Configuration for a registered webhook."""

    id: int
    url: str
    events: list[str]  # ["on_failure", "on_success", "on_drift"]
    project: str | None = None


def send_webhook(url: str, payload: dict, headers: dict | None = None) -> bool:  # type: ignore[type-arg]
    """POST JSON payload to webhook URL. Returns True on success."""
    try:
        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json", **(headers or {})},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:  # noqa: S310
            return resp.status == 200
    except Exception:  # noqa: BLE001
        return False


def should_fire(config: WebhookConfig, run_data: dict) -> bool:  # type: ignore[type-arg]
    """Check if a webhook should fire for this run based on its event subscriptions."""
    has_failures = run_data.get("failed", 0) > 0
    if "on_failure" in config.events and has_failures:
        return True
    if "on_success" in config.events and not has_failures:
        return True
    return False
