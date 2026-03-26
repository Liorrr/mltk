"""Tests for API key authentication on the mltk server."""
from __future__ import annotations

import pytest

pytest.importorskip("fastapi", reason="fastapi not installed — skipping auth tests")
pytest.importorskip("httpx", reason="httpx not installed — TestClient requires it")

from fastapi.testclient import TestClient  # noqa: E402

from mltk.server.app import create_app  # noqa: E402
from mltk.server.auth import generate_api_key, hash_key  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_RESULTS = [
    {"name": "test_accuracy", "passed": True, "severity": "info",
     "message": "accuracy OK", "details": {}, "duration_ms": 10.0},
]


@pytest.fixture
def app_with_key(tmp_path):
    """App + TestClient + a pre-registered API key."""
    db_file = str(tmp_path / "auth_test.db")
    application = create_app(db_path=db_file)
    raw_key = generate_api_key("test-project")
    application.state.storage.save_api_key(hash_key(raw_key), "test-project")
    with TestClient(application) as client:
        yield client, raw_key


@pytest.fixture
def client(tmp_path):
    """Bare TestClient with no registered keys."""
    db_file = str(tmp_path / "auth_nokey.db")
    application = create_app(db_path=db_file)
    with TestClient(application) as c:
        yield c


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_generate_key_format():
    # SCENARIO: generate a new API key
    # WHY: keys must be identifiable as mltk keys and URL-safe
    # EXPECTED: raw key starts with "mltk_"
    key = generate_api_key("any-project")
    assert key.startswith("mltk_"), f"Expected 'mltk_' prefix, got: {key}"
    assert len(key) > 10, "Key is suspiciously short"


def test_hash_key_deterministic():
    # SCENARIO: hash the same raw key twice
    # WHY: SHA-256 must be deterministic for lookup to work
    # EXPECTED: both hashes are identical
    raw = "mltk_somesecretvalue"
    assert hash_key(raw) == hash_key(raw)
    assert len(hash_key(raw)) == 64  # SHA-256 hex = 64 chars


def test_submit_with_valid_key(app_with_key):
    # SCENARIO: POST /api/runs with a valid Bearer token
    # WHY: authenticated requests must succeed with run_id in response
    # EXPECTED: HTTP 200 and {"run_id": ..., "status": "saved"}
    client, raw_key = app_with_key
    resp = client.post(
        "/api/runs",
        json={"project": "test-project", "results": SAMPLE_RESULTS},
        headers={"Authorization": f"Bearer {raw_key}"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "run_id" in body
    assert body["status"] == "saved"


def test_submit_without_key(client):
    # SCENARIO: POST /api/runs with no Authorization header
    # WHY: unauthenticated requests must be rejected at the auth layer
    # EXPECTED: HTTP 401
    resp = client.post(
        "/api/runs",
        json={"project": "default", "results": SAMPLE_RESULTS},
    )
    assert resp.status_code == 401, resp.text


def test_submit_with_invalid_key(client):
    # SCENARIO: POST /api/runs with a fabricated Bearer token not in DB
    # WHY: unknown keys must not grant access even if format looks valid
    # EXPECTED: HTTP 401
    resp = client.post(
        "/api/runs",
        json={"project": "default", "results": SAMPLE_RESULTS},
        headers={"Authorization": "Bearer mltk_thiskeyisnotregistered"},
    )
    assert resp.status_code == 401, resp.text
