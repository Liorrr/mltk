"""Tests for webhook URL display masking."""
from __future__ import annotations

import pytest

from mltk.server.redact import mask_webhook_url


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://example.com:8443/hooks/token", "https://example.com:8443/..."),
        ("https://[2001:db8::1]/hooks/token", "https://[2001:db8::1]/..."),
        ("http://example.com/hooks/token", "http://example.com/..."),
        ("https://example.com", "https://example.com/..."),
        ("not a url", "..."),
    ],
)
def test_mask_webhook_url_forms(url: str, expected: str) -> None:
    assert mask_webhook_url(url) == expected
