"""Webhook dispatch — notify external services on test run events."""
from __future__ import annotations

import ipaddress
import json
import socket
import urllib.error
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


def _is_private_ip(ip_str: str) -> bool:
    """Return True if *ip_str* belongs to a private/loopback network.

    Args:
        ip_str: An IPv4 or IPv6 address string.

    Returns:
        True if the address falls within any blocked network, False otherwise.
    """
    try:
        addr = ipaddress.ip_address(ip_str)
    except ValueError:
        return False
    return any(addr in net for net in _PRIVATE_NETWORKS)


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
    if _is_private_ip(host):
        return False

    return True


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Handler that raises on any redirect, preventing SSRF bypass via
    3xx responses that redirect to internal addresses."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[override]  # noqa: ARG002
        raise urllib.error.HTTPError(
            req.full_url, code, "Redirects are not allowed for webhooks", headers, fp,
        )


def send_webhook(url: str, payload: dict, headers: dict | None = None) -> bool:  # type: ignore[type-arg]
    """POST JSON payload to webhook URL. Returns True on success.

    Validates the URL before dispatching to prevent SSRF attacks.
    To mitigate DNS rebinding attacks the hostname is resolved
    *immediately before* the request and the resolved IP is checked
    against the private-network blocklist a second time.  This closes
    the TOCTOU gap where an attacker's DNS returns a safe IP during
    ``validate_webhook_url()`` then switches to ``127.0.0.1`` at
    dispatch time.  Redirect following is also disabled so a 3xx
    cannot bypass the checks.
    """
    if not validate_webhook_url(url):
        return False
    try:
        parsed = urllib.parse.urlparse(url)
        host = parsed.hostname
        port = parsed.port or (443 if parsed.scheme == "https" else 80)

        # --- DNS-rebinding defence -----------------------------------
        # Resolve the hostname NOW and validate the resolved IP before
        # handing the request to urllib which would resolve again.
        if host:
            addrinfo = socket.getaddrinfo(host, port)
            if not addrinfo:
                return False
            resolved_ip = addrinfo[0][4][0]
            if _is_private_ip(resolved_ip):
                return False

        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json", **(headers or {})},
        )
        # Use _NoRedirectHandler so 3xx responses raise instead of
        # following the redirect to a potentially internal address.
        opener = urllib.request.build_opener(_NoRedirectHandler)
        with opener.open(req, timeout=10) as resp:  # noqa: S310
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
