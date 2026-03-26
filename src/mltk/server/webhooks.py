"""Webhook dispatch — notify external services on test run events."""
from __future__ import annotations

import ipaddress
import json
import urllib.parse
import urllib.request
from dataclasses import dataclass


@dataclass
class WebhookConfig:
    """Configuration for a registered webhook."""

    id: int
    url: str
    events: list[str]  # ["on_failure", "on_success", "on_drift"]
    project: str | None = None


_PRIVATE_NETWORKS = [
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),
]


def validate_webhook_url(url: str) -> bool:
    """Return True if the URL is safe to dispatch to.

    Rejects:
    - Non-http(s) schemes (file://, ftp://, etc.)
    - Missing or empty netloc
    - localhost hostname variants
    - Private/loopback IP addresses (SSRF mitigation)

    Args:
        url: The candidate webhook URL.

    Returns:
        True if the URL passes all safety checks, False otherwise.
    """
    try:
        parsed = urllib.parse.urlparse(url)
    except Exception:  # noqa: BLE001
        return False

    if parsed.scheme not in ("http", "https"):
        return False

    host = parsed.hostname
    if not host:
        return False

    # Reject bare localhost names
    if host.lower() in ("localhost", "localhost.localdomain"):
        return False

    # Reject private/loopback IP addresses
    try:
        addr = ipaddress.ip_address(host)
        for net in _PRIVATE_NETWORKS:
            if addr in net:
                return False
    except ValueError:
        # host is a domain name, not an IP — allow it
        pass

    return True


def send_webhook(url: str, payload: dict, headers: dict | None = None) -> bool:  # type: ignore[type-arg]
    """POST JSON payload to webhook URL. Returns True on success.

    Validates the URL before dispatching to prevent SSRF attacks.
    """
    if not validate_webhook_url(url):
        return False
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
