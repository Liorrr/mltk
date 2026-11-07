"""Access-control tests for the results server."""
from __future__ import annotations

import logging
import sqlite3

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from mltk.server import create_app  # noqa: E402
from mltk.server.auth import generate_api_key, hash_key  # noqa: E402
from mltk.server.storage import _MIGRATIONS, Storage  # noqa: E402

RESULTS = [{"name": "quality", "passed": True}]
READS = [
    "/api/runs",
    "/api/runs/1",
    "/api/trends/project-a",
    "/api/summary/project-a",
    "/api/compare?run_a=1&run_b=2",
    "/api/webhooks",
]


def _key(app, project: str = "default", role: str = "admin") -> str:
    raw = generate_api_key()
    app.state.storage.save_api_key(hash_key(raw), project, role=role)
    return raw


def _headers(raw: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {raw}"}


@pytest.fixture
def app_client(tmp_path):
    app = create_app(db_path=str(tmp_path / "t.db"))
    with TestClient(app) as client:
        yield app, client


@pytest.mark.parametrize("path", READS)
def test_reads_require_key(app_client, path: str):
    _, client = app_client
    assert client.get(path).status_code == 401


@pytest.mark.parametrize("path", READS)
def test_reads_work_with_reader_key(app_client, path: str):
    app, client = app_client
    raw = _key(app, role="reader")
    assert client.get(path, headers=_headers(raw)).status_code in {200, 404}


def test_health_and_metrics_stay_open(app_client):
    _, client = app_client
    for path in ("/api/health", "/api/health/live", "/metrics"):
        assert client.get(path).status_code != 401


def test_public_read_serves_reads_without_key(app_client, monkeypatch):
    app, client = app_client
    monkeypatch.setenv("MLTK_SERVER_PUBLIC_READ", "1")
    for path in READS:
        expected = 404 if "runs/1" in path or "compare" in path else 200
        assert client.get(path).status_code == expected
    response = client.post("/api/runs", json={"project": "default", "results": RESULTS})
    assert response.status_code == 401


def test_public_read_still_rejects_bad_key(app_client, monkeypatch):
    _, client = app_client
    monkeypatch.setenv("MLTK_SERVER_PUBLIC_READ", "true")
    assert client.get("/api/runs", headers=_headers("mltk_invalid")).status_code == 401


def test_reader_cannot_submit_run(app_client):
    app, client = app_client
    raw = _key(app, role="reader")
    response = client.post("/api/runs", json={"results": RESULTS}, headers=_headers(raw))
    assert response.status_code == 403


def test_writer_can_submit_run(app_client):
    app, client = app_client
    raw = _key(app, role="writer")
    response = client.post("/api/runs", json={"results": RESULTS}, headers=_headers(raw))
    assert response.status_code == 200


def test_writer_cannot_manage_webhooks(app_client, monkeypatch):
    app, client = app_client
    monkeypatch.setattr("mltk.server.routes.validate_webhook_url", lambda _: True)
    raw = _key(app, role="writer")
    payload = {"url": "https://example.com/hook", "events": []}
    assert client.post("/api/webhooks", json=payload, headers=_headers(raw)).status_code == 403
    webhook_id = app.state.storage.save_webhook(payload["url"], [])
    assert client.delete(f"/api/webhooks/{webhook_id}", headers=_headers(raw)).status_code == 403


def test_admin_can_manage_webhooks(app_client, monkeypatch):
    app, client = app_client
    monkeypatch.setattr("mltk.server.routes.validate_webhook_url", lambda _: True)
    raw = _key(app, role="admin")
    response = client.post(
        "/api/webhooks", json={"url": "https://example.com/hook", "events": []},
        headers=_headers(raw),
    )
    assert response.status_code == 200
    webhook_id = response.json()["webhook_id"]
    assert client.delete(f"/api/webhooks/{webhook_id}", headers=_headers(raw)).status_code == 200


def test_role_less_key_acts_as_writer(app_client, caplog):
    app, client = app_client
    raw = generate_api_key()
    app.state.storage._conn.execute(
        "INSERT INTO api_keys (key_hash, project, created_at, role) VALUES (?, ?, ?, NULL)",
        (hash_key(raw), "default", "2026-01-01T00:00:00+00:00"),
    )
    app.state.storage._conn.commit()
    with caplog.at_level(logging.WARNING, logger="mltk.server.rbac"):
        first = client.post("/api/runs", json={"results": RESULTS}, headers=_headers(raw))
        second = client.post("/api/runs", json={"results": RESULTS}, headers=_headers(raw))
        assert first.status_code == 200
        assert second.status_code == 200
    matches = [r for r in caplog.records if "API key has no role" in r.message]
    assert len(matches) == 1
    assert client.post(
        "/api/webhooks", json={"url": "https://example.com/hook", "events": []},
        headers=_headers(raw),
    ).status_code == 403


def test_webhook_list_masks_urls(app_client):
    app, client = app_client
    raw = _key(app, role="reader")
    app.state.storage.save_webhook("https://hooks.slack.com/services/T0/B0/secret", [])
    response = client.get("/api/webhooks", headers=_headers(raw))
    assert response.status_code == 200
    assert "https://hooks.slack.com/..." in response.text
    assert "secret" not in response.text


def test_scoped_key_lists_only_own_runs(app_client):
    app, client = app_client
    app.state.storage.save_run("project-a", RESULTS)
    app.state.storage.save_run("project-b", RESULTS)
    raw = _key(app, "project-a", "reader")
    response = client.get("/api/runs", headers=_headers(raw))
    assert response.status_code == 200
    assert {run["project"] for run in response.json()["runs"]} == {"project-a"}


def test_scoped_key_other_project_run_is_404(app_client):
    app, client = app_client
    run_id = app.state.storage.save_run("project-b", RESULTS)
    raw = _key(app, "project-a", "reader")
    response = client.get(f"/api/runs/{run_id}", headers=_headers(raw))
    assert response.status_code == 404
    assert response.json()["detail"] == "Run not found"


def test_scoped_key_compare_other_project_is_404(app_client):
    app, client = app_client
    own = app.state.storage.save_run("project-a", RESULTS)
    other = app.state.storage.save_run("project-b", RESULTS)
    raw = _key(app, "project-a", "reader")
    response = client.get(f"/api/compare?run_a={own}&run_b={other}", headers=_headers(raw))
    assert response.status_code == 404
    # Same message as a run that does not exist, so scope is not revealed.
    assert response.json()["detail"] == f"Run {other} not found"


def test_scoped_key_trends_other_project_is_404(app_client):
    app, client = app_client
    raw = _key(app, "project-a", "reader")
    response = client.get("/api/trends/project-b", headers=_headers(raw))
    assert response.status_code == 404
    assert response.json()["detail"] == "Project not found"


def test_unscoped_key_sees_all_projects(app_client):
    app, client = app_client
    app.state.storage.save_run("project-a", RESULTS)
    app.state.storage.save_run("project-b", RESULTS)
    raw = _key(app, "default", "reader")
    response = client.get("/api/runs", headers=_headers(raw))
    assert {run["project"] for run in response.json()["runs"]} == {
        "project-a", "project-b"
    }


def test_scoped_key_webhook_create_other_project_403(app_client, monkeypatch):
    app, client = app_client
    monkeypatch.setattr("mltk.server.routes.validate_webhook_url", lambda _: True)
    raw = _key(app, "project-a", "admin")
    response = client.post(
        "/api/webhooks",
        json={"url": "https://example.com/hook", "events": [], "project": "project-b"},
        headers=_headers(raw),
    )
    assert response.status_code == 403


def test_scoped_key_webhook_create_global_403(app_client, monkeypatch):
    app, client = app_client
    monkeypatch.setattr("mltk.server.routes.validate_webhook_url", lambda _: True)
    raw = _key(app, "project-a", "admin")
    response = client.post(
        "/api/webhooks", json={"url": "https://example.com/hook", "events": []},
        headers=_headers(raw),
    )
    assert response.status_code == 403


def test_scoped_key_webhook_delete_other_is_404(app_client):
    app, client = app_client
    webhook_id = app.state.storage.save_webhook("https://example.com/hook", [], "project-b")
    raw = _key(app, "project-a", "admin")
    response = client.delete(f"/api/webhooks/{webhook_id}", headers=_headers(raw))
    assert response.status_code == 404
    assert response.json()["detail"] == "Webhook not found"


def test_scoped_key_lists_only_own_webhooks(app_client):
    app, client = app_client
    app.state.storage.save_webhook("https://example.com/a", [], "project-a")
    app.state.storage.save_webhook("https://example.com/b", [], "project-b")
    app.state.storage.save_webhook("https://example.com/global", [], None)
    raw = _key(app, "project-a", "reader")
    response = client.get("/api/webhooks", headers=_headers(raw))
    assert response.status_code == 200
    assert [item["project"] for item in response.json()["webhooks"]] == ["project-a"]


def test_migration_adds_role_column_and_keeps_existing_keys(tmp_path):
    db_path = tmp_path / "v1.db"
    conn = sqlite3.connect(db_path)
    for statement in _MIGRATIONS[0][2]:
        conn.execute(statement)
    conn.execute(
        "CREATE TABLE schema_versions (version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)"
    )
    conn.execute("INSERT INTO schema_versions VALUES (1, ?)", ("2026-01-01T00:00:00+00:00",))
    raw = generate_api_key()
    conn.execute(
        "INSERT INTO api_keys (key_hash, project, created_at) VALUES (?, ?, ?)",
        (hash_key(raw), "legacy", "2026-01-01T00:00:00+00:00"),
    )
    conn.commit()
    conn.close()
    storage = Storage(str(db_path))
    assert storage.verify_api_key(hash_key(raw)) == "legacy"
    assert storage.get_key_role(hash_key(raw)) is None


def test_save_api_key_rejects_unknown_role(tmp_path):
    storage = Storage(str(tmp_path / "roles.db"))
    with pytest.raises(ValueError, match="role"):
        storage.save_api_key("a" * 64, "project", role="operator")
