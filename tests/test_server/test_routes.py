"""Tests for mltk server API routes via FastAPI TestClient."""
from __future__ import annotations

import pytest

pytest.importorskip("fastapi", reason="fastapi not installed — skipping server route tests")
pytest.importorskip("httpx", reason="httpx not installed — TestClient requires it")

from fastapi.testclient import TestClient  # noqa: E402

from mltk.server.app import create_app  # noqa: E402
from mltk.server.auth import generate_api_key, hash_key  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def client(tmp_path):
    """TestClient backed by a temp-file database with a pre-registered API key.

    Returns a (TestClient, raw_key) tuple so tests that call POST /api/runs
    can supply the required Bearer token.

    # SCENARIO: isolated client for every test
    # WHY: prevents state leakage between route tests
    # EXPECTED: each test starts with an empty database and one valid API key
    """
    db_file = str(tmp_path / "routes_test.db")
    app = create_app(db_path=db_file)
    raw_key = generate_api_key()
    # Use "default" scope so the key works for any project name in tests
    app.state.storage.save_api_key(hash_key(raw_key), "default")
    with TestClient(app) as c:
        yield c, raw_key


SAMPLE_RESULTS = [
    {"name": "test_schema", "passed": True, "severity": "info",
     "message": "schema ok", "details": {}, "duration_ms": 8.0},
    {"name": "test_drift", "passed": False, "severity": "error",
     "message": "drift detected", "details": {"psi": 0.15}, "duration_ms": 42.0},
]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_health(client):
    # SCENARIO: call the health endpoint on a freshly started server
    # WHY: health check must always return 200 so load-balancers can probe it
    # EXPECTED: HTTP 200 with status "ok" and service name
    c, _ = client
    resp = c.get("/api/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["service"] == "mltk-server"


def test_submit_run(client):
    # SCENARIO: POST a valid run payload to /api/runs with a valid Bearer key
    # WHY: core ingest path must persist data and return a usable run_id
    # EXPECTED: HTTP 200, run_id is a positive integer, status is "saved"
    c, raw_key = client
    payload = {"project": "myproject", "results": SAMPLE_RESULTS}
    resp = c.post("/api/runs", json=payload, headers={"Authorization": f"Bearer {raw_key}"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "saved"
    assert isinstance(body["run_id"], int)
    assert body["run_id"] > 0


def test_list_runs(client):
    # SCENARIO: submit two runs then fetch the run list
    # WHY: GET /api/runs must reflect all submitted runs
    # EXPECTED: response contains a "runs" list with at least the two submitted runs
    c, raw_key = client
    auth = {"Authorization": f"Bearer {raw_key}"}
    c.post("/api/runs", json={"project": "proj", "results": SAMPLE_RESULTS[:1]}, headers=auth)
    c.post("/api/runs", json={"project": "proj", "results": SAMPLE_RESULTS}, headers=auth)

    resp = c.get("/api/runs")
    assert resp.status_code == 200
    body = resp.json()
    assert "runs" in body
    assert len(body["runs"]) >= 2


def test_get_run(client):
    # SCENARIO: submit a run then retrieve it by id
    # WHY: GET /api/runs/{id} must return full run details including results
    # EXPECTED: project matches, total/passed counts correct, results list present
    c, raw_key = client
    auth = {"Authorization": f"Bearer {raw_key}"}
    post_resp = c.post(
        "/api/runs",
        json={"project": "detail_proj", "results": SAMPLE_RESULTS},
        headers=auth,
    )
    run_id = post_resp.json()["run_id"]

    resp = c.get(f"/api/runs/{run_id}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["project"] == "detail_proj"
    assert body["total"] == 2
    assert body["passed"] == 1
    assert body["failed"] == 1
    assert len(body["results"]) == 2


def test_get_run_not_found(client):
    # SCENARIO: request a run id that does not exist
    # WHY: API must return 404 rather than 500 for unknown ids
    # EXPECTED: HTTP 404 response
    c, _ = client
    resp = c.get("/api/runs/99999")
    assert resp.status_code == 404


def test_trends(client):
    # SCENARIO: submit several runs for a project then fetch trends
    # WHY: trend endpoint must aggregate scores over time for charting
    # EXPECTED: trends list length matches submitted runs, each point has score + timestamp
    c, raw_key = client
    auth = {"Authorization": f"Bearer {raw_key}"}
    for _ in range(3):
        c.post("/api/runs", json={"project": "trend_proj", "results": SAMPLE_RESULTS}, headers=auth)

    resp = c.get("/api/trends/trend_proj")
    assert resp.status_code == 200
    body = resp.json()
    assert body["project"] == "trend_proj"
    assert "trends" in body
    assert len(body["trends"]) == 3
    for point in body["trends"]:
        assert "score" in point
        assert "timestamp" in point


def test_list_runs_project_filter(client):
    # SCENARIO: submit runs for two projects then filter the list
    # WHY: query param ?project= must narrow results to the specified project
    # EXPECTED: only runs matching the requested project are returned
    c, raw_key = client
    auth = {"Authorization": f"Bearer {raw_key}"}
    c.post("/api/runs", json={"project": "alpha", "results": SAMPLE_RESULTS[:1]}, headers=auth)
    c.post("/api/runs", json={"project": "beta", "results": SAMPLE_RESULTS}, headers=auth)

    resp = c.get("/api/runs?project=alpha")
    assert resp.status_code == 200
    body = resp.json()
    for run in body["runs"]:
        assert run["project"] == "alpha"
