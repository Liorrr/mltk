"""Tests for webhook CRUD endpoints and should_fire logic."""
from __future__ import annotations

from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread
from unittest import mock

import pytest

pytest.importorskip("fastapi", reason="fastapi not installed — skipping webhook tests")
pytest.importorskip("httpx", reason="httpx not installed — TestClient requires it")

from fastapi.testclient import TestClient  # noqa: E402

from mltk.server.app import create_app  # noqa: E402
from mltk.server.auth import generate_api_key, hash_key  # noqa: E402
from mltk.server.webhooks import (  # noqa: E402
    WebhookConfig,
    _is_private_ip,
    send_webhook,
    should_fire,
    validate_webhook_url,
)

# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def client(tmp_path):
    """TestClient backed by an isolated temp database with a pre-registered API key.

    Yields a (TestClient, raw_key) tuple so webhook mutation tests can pass
    the required Bearer token.
    """
    db_file = str(tmp_path / "webhooks_test.db")
    application = create_app(db_path=db_file)
    raw_key = generate_api_key()
    application.state.storage.save_api_key(hash_key(raw_key), "wh-project")
    with TestClient(application) as c:
        yield c, raw_key


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_create_webhook(client):
    # SCENARIO: POST /api/webhooks with valid URL, events, and a valid Bearer key
    # WHY: webhook registration must persist and return an id
    # EXPECTED: HTTP 200, body contains webhook_id and status="created"
    c, raw_key = client
    resp = c.post(
        "/api/webhooks",
        json={"url": "http://example.com/hook", "events": ["on_failure"]},
        headers={"Authorization": f"Bearer {raw_key}"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "webhook_id" in body
    assert body["status"] == "created"
    assert isinstance(body["webhook_id"], int)


def test_list_webhooks(client):
    # SCENARIO: create two webhooks then call GET /api/webhooks
    # WHY: list endpoint must return all registered webhooks
    # EXPECTED: response contains exactly 2 items with correct fields
    c, raw_key = client
    auth = {"Authorization": f"Bearer {raw_key}"}
    c.post(
        "/api/webhooks",
        json={"url": "http://a.test/hook", "events": ["on_failure"]},
        headers=auth,
    )
    c.post(
        "/api/webhooks",
        json={"url": "http://b.test/hook", "events": ["on_success"], "project": "proj-x"},
        headers=auth,
    )
    resp = c.get("/api/webhooks")
    assert resp.status_code == 200, resp.text
    webhooks = resp.json()["webhooks"]
    assert len(webhooks) == 2
    urls = {wh["url"] for wh in webhooks}
    assert "http://a.test/hook" in urls
    assert "http://b.test/hook" in urls


def test_delete_webhook(client):
    # SCENARIO: create a webhook then delete it by id
    # WHY: deleted webhooks must not appear in subsequent list responses
    # EXPECTED: DELETE returns status="deleted"; GET returns empty list
    c, raw_key = client
    auth = {"Authorization": f"Bearer {raw_key}"}
    create_resp = c.post(
        "/api/webhooks",
        json={"url": "http://delete.test/hook", "events": ["on_failure"]},
        headers=auth,
    )
    wh_id = create_resp.json()["webhook_id"]

    del_resp = c.delete(f"/api/webhooks/{wh_id}", headers=auth)
    assert del_resp.status_code == 200, del_resp.text
    assert del_resp.json()["status"] == "deleted"

    list_resp = c.get("/api/webhooks")
    assert list_resp.json()["webhooks"] == []


def test_should_fire_on_failure():
    # SCENARIO: webhook subscribed to on_failure, run has failures
    # WHY: should_fire must return True only when the right event condition is met
    # EXPECTED: True for failed run, False for clean run
    config = WebhookConfig(id=1, url="http://x.test", events=["on_failure"])

    assert should_fire(config, {"failed": 3, "passed": 7}) is True
    assert should_fire(config, {"failed": 0, "passed": 10}) is False


def test_should_fire_on_success():
    # SCENARIO: webhook subscribed to on_success, run has no failures
    # WHY: on_success fires only when the run is clean
    # EXPECTED: True for clean run, False for failed run
    config = WebhookConfig(id=2, url="http://y.test", events=["on_success"])

    assert should_fire(config, {"failed": 0, "passed": 5}) is True
    assert should_fire(config, {"failed": 1, "passed": 4}) is False


def test_should_fire_no_matching_events():
    # SCENARIO: webhook has an event list that does not match the run state
    # WHY: unmatched subscriptions must return False — no false positives
    # EXPECTED: should_fire returns False regardless of run outcome
    config = WebhookConfig(id=3, url="http://z.test", events=["on_drift"])

    assert should_fire(config, {"failed": 0, "passed": 10}) is False
    assert should_fire(config, {"failed": 5, "passed": 5}) is False


def test_should_fire_empty_events():
    # SCENARIO: webhook has an empty events list
    # WHY: an empty subscription must never fire
    # EXPECTED: should_fire returns False for any run
    config = WebhookConfig(id=4, url="http://w.test", events=[])

    assert should_fire(config, {"failed": 0, "passed": 10}) is False
    assert should_fire(config, {"failed": 5, "passed": 5}) is False


# ---------------------------------------------------------------------------
# validate_webhook_url — URL safety checks
# ---------------------------------------------------------------------------


def test_validate_webhook_url_accepts_https():
    # SCENARIO: standard public HTTPS URL
    # WHY: HTTPS URLs are the expected common case for webhook targets
    # EXPECTED: validate_webhook_url returns True
    assert validate_webhook_url("https://hooks.slack.com/services/abc/def") is True


def test_validate_webhook_url_accepts_http():
    # SCENARIO: plain HTTP URL (e.g. internal dev environment — non-private IP)
    # WHY: HTTP targets may be used in controlled environments
    # EXPECTED: validate_webhook_url returns True for a public hostname
    assert validate_webhook_url("http://example.com/webhook") is True


def test_validate_webhook_url_rejects_localhost():
    # SCENARIO: URL targets localhost by name
    # WHY: localhost is a loopback address; dispatching to it enables SSRF
    # EXPECTED: validate_webhook_url returns False
    assert validate_webhook_url("http://localhost/hook") is False
    assert validate_webhook_url("http://localhost:8080/hook") is False


def test_validate_webhook_url_rejects_loopback_ip():
    # SCENARIO: URL targets 127.x.x.x IP
    # WHY: loopback IPs are equivalent to localhost — SSRF risk
    # EXPECTED: validate_webhook_url returns False
    assert validate_webhook_url("http://127.0.0.1/hook") is False
    assert validate_webhook_url("http://127.1.2.3/hook") is False


def test_validate_webhook_url_rejects_private_ip():
    # SCENARIO: URL targets RFC-1918 private IP ranges
    # WHY: private IPs can reach internal services — classic SSRF vector
    # EXPECTED: validate_webhook_url returns False for all private ranges
    assert validate_webhook_url("http://10.0.0.1/hook") is False
    assert validate_webhook_url("http://172.16.0.1/hook") is False
    assert validate_webhook_url("http://192.168.1.100/hook") is False


def test_validate_webhook_url_rejects_file_scheme():
    # SCENARIO: URL uses file:// scheme
    # WHY: file:// allows reading local filesystem — critical SSRF variant
    # EXPECTED: validate_webhook_url returns False
    assert validate_webhook_url("file:///etc/passwd") is False


def test_validate_webhook_url_rejects_missing_host():
    # SCENARIO: URL has no hostname
    # WHY: a URL without a host is malformed and cannot be safely dispatched to
    # EXPECTED: validate_webhook_url returns False
    assert validate_webhook_url("https:///no-host") is False
    assert validate_webhook_url("not-a-url") is False


def test_create_webhook_auth_required(client):
    # SCENARIO: POST /api/webhooks with no Authorization header
    # WHY: webhook creation must be protected — unauthenticated callers
    #      must not be able to register arbitrary webhook targets
    # EXPECTED: HTTP 401
    c, _ = client
    resp = c.post(
        "/api/webhooks",
        json={"url": "http://example.com/hook", "events": ["on_failure"]},
    )
    assert resp.status_code == 401, resp.text


def test_delete_webhook_auth_required(client):
    # SCENARIO: DELETE /api/webhooks/{id} with no Authorization header
    # WHY: deletion without auth would allow anyone to remove webhooks
    # EXPECTED: HTTP 401 (even if webhook_id is non-existent)
    c, _ = client
    resp = c.delete("/api/webhooks/1")
    assert resp.status_code == 401, resp.text


def test_delete_webhook_not_found(client):
    # SCENARIO: DELETE /api/webhooks/{id} where id does not exist
    # WHY: must return 404, not 500, for unknown ids
    # EXPECTED: HTTP 404
    c, raw_key = client
    resp = c.delete(
        "/api/webhooks/99999",
        headers={"Authorization": f"Bearer {raw_key}"},
    )
    assert resp.status_code == 404, resp.text


# ---------------------------------------------------------------------------
# P1-19: Webhook URL validation at creation (route-level)
# ---------------------------------------------------------------------------


def test_create_webhook_rejects_private_ip(client):
    # SCENARIO: POST /api/webhooks with a private-IP URL (e.g. 10.x.x.x)
    # WHY: the route must validate the URL BEFORE saving — SSRF mitigation
    # EXPECTED: HTTP 422 with a descriptive error; webhook is NOT persisted
    c, raw_key = client
    auth = {"Authorization": f"Bearer {raw_key}"}
    resp = c.post(
        "/api/webhooks",
        json={"url": "http://10.0.0.1/internal", "events": ["on_failure"]},
        headers=auth,
    )
    assert resp.status_code == 422, resp.text
    assert "Invalid webhook URL" in resp.json()["detail"]

    # Verify nothing was persisted
    list_resp = c.get("/api/webhooks")
    assert list_resp.json()["webhooks"] == []


def test_create_webhook_rejects_localhost(client):
    # SCENARIO: POST /api/webhooks with localhost URL
    # WHY: localhost is a loopback address — must be rejected at creation time
    # EXPECTED: HTTP 422
    c, raw_key = client
    auth = {"Authorization": f"Bearer {raw_key}"}
    resp = c.post(
        "/api/webhooks",
        json={"url": "http://localhost:9999/hook", "events": ["on_failure"]},
        headers=auth,
    )
    assert resp.status_code == 422, resp.text


def test_create_webhook_rejects_file_scheme(client):
    # SCENARIO: POST /api/webhooks with file:// URL
    # WHY: file scheme allows local filesystem access — critical SSRF
    # EXPECTED: HTTP 422
    c, raw_key = client
    auth = {"Authorization": f"Bearer {raw_key}"}
    resp = c.post(
        "/api/webhooks",
        json={"url": "file:///etc/passwd", "events": ["on_failure"]},
        headers=auth,
    )
    assert resp.status_code == 422, resp.text


# ---------------------------------------------------------------------------
# P1-22: Redirect not followed in send_webhook
# ---------------------------------------------------------------------------


def test_send_webhook_does_not_follow_redirect():
    # SCENARIO: webhook target returns a 302 redirect to a different URL
    # WHY: following redirects could bypass SSRF URL checks if the
    #      redirect target is an internal/private address
    # EXPECTED: send_webhook returns False (does not follow the redirect)

    class _RedirectHandler(BaseHTTPRequestHandler):
        """Minimal HTTP handler that always responds with 302."""

        def do_POST(self):  # noqa: N802
            self.send_response(302)
            self.send_header("Location", "http://127.0.0.1:1/internal")
            self.end_headers()

        def log_message(self, *args):  # noqa: ARG002
            pass  # suppress stdout noise in test output

    server = HTTPServer(("127.0.0.1", 0), _RedirectHandler)
    port = server.server_address[1]
    thread = Thread(target=server.handle_request, daemon=True)
    thread.start()

    try:
        # Patch validate_webhook_url AND _is_private_ip so we can actually
        # reach our local test server (on 127.0.0.1 — normally blocked by
        # both the URL validator and the DNS-rebinding defense).
        with mock.patch(
            "mltk.server.webhooks.validate_webhook_url", return_value=True,
        ), mock.patch(
            "mltk.server.webhooks._is_private_ip", return_value=False,
        ):
            result = send_webhook(
                f"http://127.0.0.1:{port}/hook",
                {"event": "test"},
            )
        # The _NoRedirectHandler raises on the 302, so send_webhook catches
        # the exception and returns False — redirect was NOT followed
        assert result is False
    finally:
        server.server_close()
        thread.join(timeout=5)


# ---------------------------------------------------------------------------
# P1-20: SSRF DNS rebinding defence
# ---------------------------------------------------------------------------


def test_is_private_ip_loopback():
    # SCENARIO: check _is_private_ip with loopback addresses
    # WHY: helper must correctly identify 127.x.x.x and ::1
    # EXPECTED: returns True
    assert _is_private_ip("127.0.0.1") is True
    assert _is_private_ip("127.255.255.255") is True
    assert _is_private_ip("::1") is True


def test_is_private_ip_rfc1918():
    # SCENARIO: check _is_private_ip with RFC-1918 addresses
    # WHY: must catch all three private ranges (10/8, 172.16/12, 192.168/16)
    # EXPECTED: returns True
    assert _is_private_ip("10.0.0.1") is True
    assert _is_private_ip("172.16.0.1") is True
    assert _is_private_ip("192.168.1.100") is True


def test_is_private_ip_public():
    # SCENARIO: check _is_private_ip with public addresses
    # WHY: public addresses must NOT be flagged as private
    # EXPECTED: returns False
    assert _is_private_ip("8.8.8.8") is False
    assert _is_private_ip("1.1.1.1") is False


def test_is_private_ip_invalid_input():
    # SCENARIO: pass a hostname string (not an IP) to _is_private_ip
    # WHY: must not raise; should return False for non-IP strings
    # EXPECTED: returns False
    assert _is_private_ip("example.com") is False
    assert _is_private_ip("not-an-ip") is False


def test_send_webhook_blocks_dns_rebinding():
    # SCENARIO: DNS resolves to a public IP during validate_webhook_url
    #           but resolves to a private IP (127.0.0.1) at dispatch time
    # WHY: this is a DNS rebinding attack — the attacker controls DNS
    #      and flips the record between validation and request time
    # EXPECTED: send_webhook returns False because the resolved IP is
    #           re-validated before the HTTP request is made

    # validate_webhook_url passes (attacker's DNS returned a safe IP)
    # but socket.getaddrinfo now returns a loopback address
    fake_addrinfo = [
        (2, 1, 6, "", ("127.0.0.1", 80)),  # AF_INET, SOCK_STREAM, TCP
    ]
    with mock.patch(
        "mltk.server.webhooks.validate_webhook_url", return_value=True,
    ), mock.patch(
        "mltk.server.webhooks.socket.getaddrinfo", return_value=fake_addrinfo,
    ):
        result = send_webhook(
            "http://attacker.example.com/steal",
            {"secret": "data"},
        )
    assert result is False


def test_send_webhook_blocks_dns_rebinding_private_10():
    # SCENARIO: DNS rebinding to a 10.x.x.x address
    # WHY: must also catch rebinding into RFC-1918 10/8 range
    # EXPECTED: send_webhook returns False
    fake_addrinfo = [
        (2, 1, 6, "", ("10.0.0.5", 80)),
    ]
    with mock.patch(
        "mltk.server.webhooks.validate_webhook_url", return_value=True,
    ), mock.patch(
        "mltk.server.webhooks.socket.getaddrinfo", return_value=fake_addrinfo,
    ):
        result = send_webhook(
            "http://attacker.example.com/steal",
            {"data": "exfil"},
        )
    assert result is False


def test_send_webhook_allows_public_resolved_ip():
    # SCENARIO: DNS resolves to a genuine public IP
    # WHY: must not block legitimate public webhook targets
    # EXPECTED: send_webhook proceeds past DNS check (may still fail on
    #           network I/O, but the DNS-rebinding guard must not reject it)
    fake_addrinfo = [
        (2, 1, 6, "", ("93.184.216.34", 443)),  # example.com
    ]
    with mock.patch(
        "mltk.server.webhooks.validate_webhook_url", return_value=True,
    ), mock.patch(
        "mltk.server.webhooks.socket.getaddrinfo", return_value=fake_addrinfo,
    ), mock.patch(
        "mltk.server.webhooks.urllib.request.build_opener",
    ) as mock_opener:
        # Simulate a successful 200 response
        mock_resp = mock.MagicMock()
        mock_resp.status = 200
        mock_resp.__enter__ = mock.MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = mock.MagicMock(return_value=False)
        mock_opener.return_value.open.return_value = mock_resp

        result = send_webhook(
            "https://hooks.example.com/webhook",
            {"event": "test"},
        )
    assert result is True


def test_send_webhook_blocks_dns_empty_addrinfo():
    # SCENARIO: socket.getaddrinfo returns an empty list
    # WHY: if DNS resolution yields nothing, we must not proceed
    # EXPECTED: send_webhook returns False
    with mock.patch(
        "mltk.server.webhooks.validate_webhook_url", return_value=True,
    ), mock.patch(
        "mltk.server.webhooks.socket.getaddrinfo", return_value=[],
    ):
        result = send_webhook(
            "http://vanished.example.com/hook",
            {"event": "test"},
        )
    assert result is False
