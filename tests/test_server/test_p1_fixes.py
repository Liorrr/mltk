"""Tests for P1-23, P1-24, P1-27, P1-28 server hardening fixes."""
from __future__ import annotations

import json
import logging

import pytest

pytest.importorskip("fastapi", reason="fastapi not installed — skipping server tests")
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

    Returns a (TestClient, raw_key) tuple.
    """
    db_file = str(tmp_path / "p1_test.db")
    app = create_app(db_path=db_file)
    raw_key = generate_api_key()
    app.state.storage.save_api_key(hash_key(raw_key), "default")
    with TestClient(app) as c:
        yield c, raw_key


VALID_RESULTS = [
    {"name": "test_a", "passed": True, "severity": "info",
     "message": "ok", "details": {}, "duration_ms": 5.0},
    {"name": "test_b", "passed": False, "severity": "error",
     "message": "drift", "details": {"psi": 0.2}, "duration_ms": 20.0},
]


# ---------------------------------------------------------------------------
# P1-23: Request body size limit
# ---------------------------------------------------------------------------

class TestRequestBodyLimit:
    """Verify the 10 MB request body size limit middleware."""

    def test_oversized_request_returns_413(self, client):
        # SCENARIO: send a request with Content-Length > 10MB
        # WHY: servers must reject excessively large payloads to prevent OOM
        # EXPECTED: HTTP 413 with descriptive error
        c, raw_key = client
        resp = c.post(
            "/api/runs",
            content=b"x" * 100,  # small body, but header says huge
            headers={
                "Authorization": f"Bearer {raw_key}",
                "Content-Type": "application/json",
                "Content-Length": str(11_000_000),  # > 10MB
            },
        )
        assert resp.status_code == 413
        assert "too large" in resp.json()["detail"].lower()

    def test_normal_request_passes(self, client):
        # SCENARIO: send a normally-sized request
        # WHY: middleware must not block legitimate payloads
        # EXPECTED: HTTP 200 (or 422 if payload is invalid, but NOT 413)
        c, raw_key = client
        payload = {"project": "test", "results": VALID_RESULTS}
        resp = c.post(
            "/api/runs",
            json=payload,
            headers={"Authorization": f"Bearer {raw_key}"},
        )
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# P1-24: Typed result models for API input
# ---------------------------------------------------------------------------

class TestTypedResultModels:
    """Verify that ResultItem model validates and rejects bad payloads."""

    def test_valid_typed_results(self, client):
        # SCENARIO: submit well-formed results matching ResultItem schema
        # WHY: typed models should accept valid data without regressions
        # EXPECTED: HTTP 200, run saved
        c, raw_key = client
        payload = {"project": "typed", "results": VALID_RESULTS}
        resp = c.post(
            "/api/runs",
            json=payload,
            headers={"Authorization": f"Bearer {raw_key}"},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "saved"

    def test_missing_required_fields_rejected(self, client):
        # SCENARIO: submit a result missing the required 'name' field
        # WHY: typed model requires 'name' and 'passed' — must reject partial data
        # EXPECTED: HTTP 422 validation error
        c, raw_key = client
        payload = {
            "project": "typed",
            "results": [{"passed": True}],  # missing 'name'
        }
        resp = c.post(
            "/api/runs",
            json=payload,
            headers={"Authorization": f"Bearer {raw_key}"},
        )
        assert resp.status_code == 422

    def test_wrong_type_rejected(self, client):
        # SCENARIO: submit a result with 'passed' as a string instead of bool
        # WHY: strict typing catches accidental data format issues early
        # EXPECTED: HTTP 422 validation error
        c, raw_key = client
        payload = {
            "project": "typed",
            "results": [{"name": "t", "passed": "not_a_bool"}],
        }
        resp = c.post(
            "/api/runs",
            json=payload,
            headers={"Authorization": f"Bearer {raw_key}"},
        )
        # Pydantic coerces some values; "not_a_bool" should fail
        assert resp.status_code == 422

    def test_results_limit_exceeded(self, client):
        # SCENARIO: submit more than 10,000 results in a single run
        # WHY: validator caps at 10,000 to prevent abuse
        # EXPECTED: HTTP 422 with message about maximum
        c, raw_key = client
        huge_results = [{"name": f"t{i}", "passed": True} for i in range(10_001)]
        payload = {"project": "abuse", "results": huge_results}
        resp = c.post(
            "/api/runs",
            json=payload,
            headers={"Authorization": f"Bearer {raw_key}"},
        )
        assert resp.status_code == 422
        assert "10,000" in resp.text or "10000" in resp.text

    def test_defaults_applied(self, client):
        # SCENARIO: submit minimal result (only name + passed) and verify defaults
        # WHY: ResultItem defaults (severity=info, message="", etc.) must apply
        # EXPECTED: saved run contains default values
        c, raw_key = client
        payload = {
            "project": "defaults",
            "results": [{"name": "minimal", "passed": True}],
        }
        resp = c.post(
            "/api/runs",
            json=payload,
            headers={"Authorization": f"Bearer {raw_key}"},
        )
        assert resp.status_code == 200
        run_id = resp.json()["run_id"]

        detail = c.get(f"/api/runs/{run_id}")
        assert detail.status_code == 200
        result = detail.json()["results"][0]
        assert result["severity"] == "info"
        assert result["message"] == ""
        assert result["duration_ms"] == 0.0


# ---------------------------------------------------------------------------
# P1-27: Liveness vs readiness probe separation
# ---------------------------------------------------------------------------

class TestHealthProbes:
    """Verify liveness and readiness probes."""

    def test_liveness_returns_200(self, client):
        # SCENARIO: call /api/health/live on a running server
        # WHY: liveness probe must always succeed if process is alive
        # EXPECTED: HTTP 200 with status "alive"
        c, _ = client
        resp = c.get("/api/health/live")
        assert resp.status_code == 200
        assert resp.json()["status"] == "alive"

    def test_readiness_returns_200_when_db_works(self, client):
        # SCENARIO: call /api/health/ready with a working database
        # WHY: readiness confirms traffic can be served (DB accessible)
        # EXPECTED: HTTP 200 with status "ready"
        c, _ = client
        resp = c.get("/api/health/ready")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ready"

    def test_readiness_returns_503_when_db_broken(self, tmp_path):
        # SCENARIO: call /api/health/ready when database is inaccessible
        # WHY: readiness must return 503 so orchestrators stop routing traffic
        # EXPECTED: HTTP 503
        db_file = str(tmp_path / "ready_test.db")
        app = create_app(db_path=db_file)
        with TestClient(app) as c:
            # Break the storage by closing the connection
            app.state.storage.close()
            resp = c.get("/api/health/ready")
            assert resp.status_code == 503

    def test_original_health_still_works(self, client):
        # SCENARIO: call the original /api/health endpoint
        # WHY: backward compatibility — existing probes must not break
        # EXPECTED: HTTP 200 with status "ok"
        c, _ = client
        resp = c.get("/api/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


# ---------------------------------------------------------------------------
# P1-28: Structured JSON logging
# ---------------------------------------------------------------------------

class TestJsonLogging:
    """Verify structured JSON logging when MLTK_LOG_FORMAT=json."""

    def test_json_logging_format(self, tmp_path, monkeypatch):
        # SCENARIO: set MLTK_LOG_FORMAT=json and emit a log record
        # WHY: production environments need machine-parseable log lines
        # EXPECTED: log output is valid JSON with timestamp, level, message, logger
        monkeypatch.setenv("MLTK_LOG_FORMAT", "json")

        from mltk.server.logging_config import JsonFormatter

        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="mltk.test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="test message",
            args=(),
            exc_info=None,
        )
        output = formatter.format(record)
        parsed = json.loads(output)
        assert parsed["level"] == "INFO"
        assert parsed["message"] == "test message"
        assert parsed["logger"] == "mltk.test"
        assert "timestamp" in parsed

    def test_setup_logging_json_mode(self, monkeypatch):
        # SCENARIO: call setup_logging with MLTK_LOG_FORMAT=json
        # WHY: setup_logging must configure root logger with JsonFormatter
        # EXPECTED: root logger has a handler with JsonFormatter
        monkeypatch.setenv("MLTK_LOG_FORMAT", "json")

        from mltk.server.logging_config import JsonFormatter, setup_logging

        # Save original handlers to restore after test
        original_handlers = logging.root.handlers[:]
        original_level = logging.root.level
        try:
            setup_logging()
            assert any(
                isinstance(h.formatter, JsonFormatter)
                for h in logging.root.handlers
            )
        finally:
            logging.root.handlers = original_handlers
            logging.root.level = original_level

    def test_setup_logging_text_mode_no_json_formatter(self, monkeypatch):
        # SCENARIO: call setup_logging with default text mode
        # WHY: text mode should NOT install JsonFormatter
        # EXPECTED: no JsonFormatter on root logger
        monkeypatch.setenv("MLTK_LOG_FORMAT", "text")

        from mltk.server.logging_config import JsonFormatter, setup_logging

        original_handlers = logging.root.handlers[:]
        original_level = logging.root.level
        try:
            setup_logging()
            json_handlers = [
                h for h in logging.root.handlers
                if isinstance(h.formatter, JsonFormatter)
            ]
            assert len(json_handlers) == 0
        finally:
            logging.root.handlers = original_handlers
            logging.root.level = original_level
