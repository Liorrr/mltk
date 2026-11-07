"""Helpers for displaying sensitive endpoint configuration safely."""
from __future__ import annotations

from urllib.parse import urlsplit


def mask_webhook_url(url: str) -> str:
    """Return a webhook URL with its path and query replaced by ``...``."""
    try:
        parsed = urlsplit(url)
        if not parsed.scheme or not parsed.hostname:
            return "..."
        host = parsed.hostname
        if ":" in host:
            host = f"[{host}]"
        if parsed.port is not None:
            host += f":{parsed.port}"
        return f"{parsed.scheme}://{host}/..."
    except (ValueError, TypeError):
        return "..."
