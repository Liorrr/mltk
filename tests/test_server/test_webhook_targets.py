"""Target policy and transport tests for webhook delivery."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from mltk.server import webhooks


def _info(address: str, port: int = 80) -> tuple:
    return (2, 1, 6, "", (address, port))


@pytest.mark.parametrize("address", ["8.8.8.8", "2606:4700:4700::1111"])
def test_allows_public_targets(address: str) -> None:
    assert webhooks.is_allowed_address(address, allow_private=False)


def test_allows_public_hostname(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(webhooks.socket, "getaddrinfo", lambda *args, **kwargs: [_info("8.8.8.8")])
    assert webhooks.validate_webhook_url("http://public.example.test/hook")


@pytest.mark.parametrize("address", [
    "169.254.169.254", "0.0.0.0", "0.1.2.3", "100.64.0.1", "127.0.0.1",
    "10.0.0.5", "192.168.1.1", "fe80::1", "::1", "::ffff:127.0.0.1",
    "::ffff:169.254.169.254", "2002:7f00:1::1", "64:ff9b::7f00:1", "::127.0.0.1",
])
def test_refuses_internal_targets(address: str) -> None:
    assert not webhooks.is_allowed_address(address, allow_private=False)


def test_hostname_resolution_requires_every_address(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(webhooks.socket, "getaddrinfo", lambda *args, **kwargs: [
        _info("8.8.8.8"), _info("127.0.0.1"),
    ])
    assert not webhooks.validate_webhook_url("http://example.test/hook")


def test_unresolvable_host_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args, **kwargs):
        raise OSError("no DNS")
    monkeypatch.setattr(webhooks.socket, "getaddrinfo", fail)
    assert not webhooks.validate_webhook_url("http://example.test/hook")


@pytest.mark.parametrize("url", [
    "ftp://example.test/", "file:///tmp/x", "http:///missing", "http://example%25.com/",
    r"http://example.com\\hook", "http://example.com:bad/", "http://localhost/",
])
def test_bad_url_forms_are_refused(url: str) -> None:
    assert not webhooks.validate_webhook_url(url)


def test_private_opt_in_allows_rfc1918(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MLTK_SERVER_WEBHOOK_ALLOW_PRIVATE", "yes")
    assert webhooks.allow_private_targets()
    assert all(webhooks.is_allowed_address(ip, allow_private=True) for ip in (
        "10.0.0.5", "192.168.1.1", "fd00::1",
    ))


@pytest.mark.parametrize("address", ["169.254.169.254", "127.0.0.1", "0.0.0.0", "fe80::1"])
def test_private_opt_in_still_refuses_special_targets(address: str) -> None:
    assert not webhooks.is_allowed_address(address, allow_private=True)


def test_deliver_json_checks_address_at_connect_time(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(webhooks, "validate_webhook_url", lambda url: True)
    calls = iter([[_info("8.8.8.8")], [_info("127.0.0.1")]])
    monkeypatch.setattr(webhooks.socket, "getaddrinfo", lambda *a, **k: next(calls))
    connect = MagicMock()
    monkeypatch.setattr(webhooks.socket, "create_connection", connect)
    assert not webhooks.deliver_json("http://example.test/", {})
    assert all("127.0.0.1" not in str(call) for call in connect.call_args_list)


def test_deliver_json_success_statuses(monkeypatch: pytest.MonkeyPatch) -> None:
    response = MagicMock()
    response.__enter__.return_value = response
    response.__exit__.return_value = False
    opener = MagicMock()
    opener.open.return_value = response
    monkeypatch.setattr(webhooks.urllib.request, "build_opener", lambda *a: opener)
    for status, expected in ((200, True), (204, True), (500, False)):
        response.status = status
        assert webhooks.deliver_json("http://example.test/", {}) is expected


def test_deliver_json_does_not_follow_redirects(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(webhooks.urllib.request, "build_opener", lambda *a: (_ for _ in ()).throw(
        webhooks.urllib.error.HTTPError("u", 302, "redirect", {}, None),
    ))
    assert not webhooks.deliver_json("http://example.test/", {})


def test_deliver_json_ignores_proxy_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.invalid")
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.invalid")
    opener = MagicMock()
    response = MagicMock(status=204)
    response.__enter__.return_value = response
    opener.open.return_value = response
    seen = []
    monkeypatch.setattr(
        webhooks.urllib.request, "build_opener", lambda *args: (seen.extend(args) or opener),
    )
    assert webhooks.deliver_json("http://example.test/", {})
    assert any(isinstance(arg, webhooks.urllib.request.ProxyHandler) for arg in seen)
