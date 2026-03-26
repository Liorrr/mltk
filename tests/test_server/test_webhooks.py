"""Tests for webhook CRUD endpoints and should_fire logic."""
from __future__ import annotations

import pytest

pytest.importorskip("fastapi", reason="fastapi not installed — skipping webhook tests")
pytest.importorskip("httpx", reason="httpx not installed — TestClient requires it")

from fastapi.testclient import TestClient  # noqa: E402

from mltk.server.app import create_app  # noqa: E402
from mltk.server.webhooks import WebhookConfig, should_fire  # noqa: E402

# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def client(tmp_path):
    """TestClient backed by an isolated temp database."""
    db_file = str(tmp_path / "webhooks_test.db")
    application = create_app(db_path=db_file)
    with TestClient(application) as c:
        yield c


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_create_webhook(client):
    # SCENARIO: POST /api/webhooks with valid URL and events
    # WHY: webhook registration must persist and return an id
    # EXPECTED: HTTP 200, body contains webhook_id and status="created"
    resp = client.post(
        "/api/webhooks",
        json={"url": "http://example.com/hook", "events": ["on_failure"]},
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
    client.post(
        "/api/webhooks",
        json={"url": "http://a.test/hook", "events": ["on_failure"]},
    )
    client.post(
        "/api/webhooks",
        json={"url": "http://b.test/hook", "events": ["on_success"], "project": "proj-x"},
    )
    resp = client.get("/api/webhooks")
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
    create_resp = client.post(
        "/api/webhooks",
        json={"url": "http://delete.test/hook", "events": ["on_failure"]},
    )
    wh_id = create_resp.json()["webhook_id"]

    del_resp = client.delete(f"/api/webhooks/{wh_id}")
    assert del_resp.status_code == 200, del_resp.text
    assert del_resp.json()["status"] == "deleted"

    list_resp = client.get("/api/webhooks")
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
