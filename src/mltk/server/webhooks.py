"""Webhook dispatch with URL and connect-time target checks."""
from __future__ import annotations

import http.client
import ipaddress
import json
import os
import socket
import ssl
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass


@dataclass
class WebhookConfig:
    """Configuration for a registered webhook."""

    id: int
    url: str
    events: list[str]
    project: str | None = None


_ALWAYS_BLOCKED_NETWORKS = (
    ipaddress.ip_network("0.0.0.0/8"), ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("169.254.0.0/16"), ipaddress.ip_network("224.0.0.0/4"),
    ipaddress.ip_network("::/96"), ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fe80::/10"), ipaddress.ip_network("ff00::/8"),
    ipaddress.ip_network("64:ff9b::/96"), ipaddress.ip_network("2001::/32"),
)
_PRIVATE_NETWORKS = (
    ipaddress.ip_network("10.0.0.0/8"), ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"), ipaddress.ip_network("100.64.0.0/10"),
    ipaddress.ip_network("fc00::/7"),
    # Deprecated IPv6 site-local; ipaddress reports it as global.
    ipaddress.ip_network("fec0::/10"),
)


def allow_private_targets() -> bool:
    """Return whether configured private webhook targets are allowed."""
    return os.environ.get("MLTK_SERVER_WEBHOOK_ALLOW_PRIVATE", "").strip().lower() in {
        "1", "true", "yes", "on",
    }


def _unwrapped_addresses(addr: ipaddress._BaseAddress) -> list[ipaddress._BaseAddress]:
    """Return an address plus any IPv4 address embedded in it.

    IPv4-mapped (``::ffff:a.b.c.d``) and 6to4 (``2002::/16``) IPv6 addresses
    carry an IPv4 address, which is checked as well. IPv4 addresses have no
    embedded form.
    """
    result = [addr]
    current = addr
    while isinstance(current, ipaddress.IPv6Address):
        inner = current.ipv4_mapped or current.sixtofour
        if inner is None or inner in result:
            break
        result.append(inner)
        current = inner
    return result


def is_allowed_address(ip: str, *, allow_private: bool) -> bool:
    """Return whether an IP address is an allowed webhook destination."""
    if "%" in ip:
        return False
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return False
    for candidate in _unwrapped_addresses(addr):
        if any(candidate in network for network in _ALWAYS_BLOCKED_NETWORKS):
            return False
        if candidate.is_loopback or candidate.is_link_local or candidate.is_unspecified:
            return False
        if candidate.is_multicast or candidate.is_reserved:
            return False
        if any(candidate in network for network in _PRIVATE_NETWORKS):
            if not allow_private:
                return False
            continue
        if not candidate.is_global:
            return False
    return True


def _is_private_ip(ip_str: str) -> bool:
    """Return True if *ip_str* is an IP address refused by default.

    Kept for compatibility: a value that is not an IP address (such as a
    hostname) returns False, as before. Hostnames are checked by resolving
    them in :func:`validate_webhook_url`.
    """
    try:
        ipaddress.ip_address(ip_str)
    except ValueError:
        return False
    return not is_allowed_address(ip_str, allow_private=False)


def validate_webhook_url(url: str) -> bool:
    """Validate scheme, syntax, and every resolved webhook destination address."""
    try:
        parsed = urllib.parse.urlsplit(url)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            return False
        if any(char in parsed.netloc for char in ("%", "\\")) or any(
            ord(char) <= 0x20 or ord(char) == 0x7F for char in parsed.netloc
        ):
            return False
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        host = parsed.hostname
        if host.lower() in ("localhost", "localhost.localdomain"):
            return False
        try:
            ipaddress.ip_address(host)
        except ValueError:
            results = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
            return bool(results) and all(
                is_allowed_address(info[4][0], allow_private=allow_private_targets())
                for info in results
            )
        return is_allowed_address(host, allow_private=allow_private_targets())
    except (OSError, ValueError, UnicodeError):
        return False


def _checked_create_connection(
    address: tuple[str, int],
    timeout: object = socket._GLOBAL_DEFAULT_TIMEOUT,  # noqa: SLF001
    source_address: tuple[str, int] | None = None,
) -> socket.socket:
    """Resolve *address*, check every result, then connect to a checked one.

    This runs at connect time, so the addresses checked are the ones used:
    a DNS answer that changes between registration and delivery cannot move
    the connection to a refused address.
    """
    host, port = address
    results = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    allow_private = allow_private_targets()
    ips = [info[4][0] for info in results]
    if not ips or not all(is_allowed_address(ip, allow_private=allow_private) for ip in ips):
        raise OSError("Webhook target address is not allowed")
    last_error: OSError | None = None
    for ip in dict.fromkeys(ips):
        try:
            return socket.create_connection((ip, port), timeout, source_address)
        except OSError as exc:
            last_error = exc
    raise last_error or OSError("Webhook target could not be reached")


class _CheckedHTTPConnection(http.client.HTTPConnection):
    """HTTP connection that checks the target addresses when it connects."""

    def __init__(self, host: str, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        super().__init__(host, *args, **kwargs)
        self._create_connection = _checked_create_connection


class _CheckedHTTPSConnection(http.client.HTTPSConnection):
    """HTTPS connection with the same connect-time check.

    TLS still uses the original hostname for SNI and certificate checks.
    """

    def __init__(self, host: str, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        super().__init__(host, *args, **kwargs)
        self._create_connection = _checked_create_connection


class _CheckedHTTPHandler(urllib.request.HTTPHandler):
    """urllib handler using checked HTTP connections."""

    def do_open(self, http_class, req, **http_conn_args):  # type: ignore[no-untyped-def]
        return super().do_open(_CheckedHTTPConnection, req, **http_conn_args)


class _CheckedHTTPSHandler(urllib.request.HTTPSHandler):
    """urllib handler using checked HTTPS connections."""

    def do_open(self, http_class, req, **http_conn_args):  # type: ignore[no-untyped-def]
        return super().do_open(_CheckedHTTPSConnection, req, **http_conn_args)


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Handler that refuses all redirects."""

    def redirect_request(
        self, req, fp, code, msg, headers, newurl,
    ):  # type: ignore[override]  # noqa: ARG002
        raise urllib.error.HTTPError(
            req.full_url, code, "Redirects are not allowed for webhooks", headers, fp,
        )


def deliver_json(
    url: str,
    payload: dict,
    *,
    headers: dict | None = None,
    timeout: float = 10.0,
) -> bool:
    """POST JSON through checked, proxy-free, non-redirecting connections."""
    try:
        req = urllib.request.Request(
            url, data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", **(headers or {})}, method="POST",
        )
        opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}), _NoRedirectHandler,
            _CheckedHTTPHandler,
            _CheckedHTTPSHandler(context=ssl.create_default_context()),
        )
        with opener.open(req, timeout=timeout) as response:  # noqa: S310
            return 200 <= response.status < 300
    except Exception:  # noqa: BLE001
        return False


def send_webhook(url: str, payload: dict, headers: dict | None = None) -> bool:
    """Validate a webhook URL, then deliver its JSON payload without redirects or proxies."""
    if not validate_webhook_url(url):
        return False
    return deliver_json(url, payload, headers=headers)


def should_fire(config: WebhookConfig, run_data: dict) -> bool:
    """Check if a webhook should fire for this run based on event subscriptions."""
    has_failures = run_data.get("failed", 0) > 0
    if "on_failure" in config.events and has_failures:
        return True
    if "on_success" in config.events and not has_failures:
        return True
    return False
