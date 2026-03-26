"""Tests for the run comparison module and /api/compare endpoint."""
from __future__ import annotations

import pytest

pytest.importorskip("fastapi", reason="fastapi not installed — skipping comparison tests")
pytest.importorskip("httpx", reason="httpx not installed — TestClient requires it")

from fastapi.testclient import TestClient  # noqa: E402

from mltk.server.app import create_app  # noqa: E402
from mltk.server.auth import generate_api_key, hash_key  # noqa: E402
from mltk.server.comparison import compare_runs  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _result(name: str, passed: bool, sev: str = "info", msg: str = "ok") -> dict:
    return {
        "name": name, "passed": passed, "severity": sev,
        "message": msg, "details": {}, "duration_ms": 5.0,
    }


def _make_run(results: list[dict], score: float = 0.0) -> dict:  # type: ignore[type-arg]
    """Build a minimal run dict compatible with compare_runs."""
    return {"score": score, "results": results}


def _simple_result(name: str, passed: bool) -> dict:
    return {"name": name, "passed": passed}


# ---------------------------------------------------------------------------
# Unit tests — compare_runs pure function
# ---------------------------------------------------------------------------


def test_compare_identical():
    # SCENARIO: two runs with exactly the same passing tests
    # WHY: no regressions means all change lists should be empty
    # EXPECTED: new_failures=[], fixed=[], still_passing has all tests, score_change=0
    run_a = _make_run([_result("test_a", True), _result("test_b", True)], score=100.0)
    run_b = _make_run([_result("test_a", True), _result("test_b", True)], score=100.0)

    diff = compare_runs(run_a, run_b)

    assert diff["new_failures"] == []
    assert diff["fixed"] == []
    assert diff["still_failing"] == []
    assert sorted(diff["still_passing"]) == ["test_a", "test_b"]
    assert diff["new_tests"] == []
    assert diff["removed_tests"] == []
    assert diff["score_change"] == 0.0


def test_compare_new_failure():
    # SCENARIO: test_b passed in run A but fails in run B (regression)
    # WHY: must correctly identify regressions so CI can surface them
    # EXPECTED: new_failures=["test_b"], score_change is negative
    run_a = _make_run(
        [_result("test_a", True), _result("test_b", True)],
        score=100.0,
    )
    run_b = _make_run(
        [_result("test_a", True), _result("test_b", False)],
        score=50.0,
    )

    diff = compare_runs(run_a, run_b)

    assert diff["new_failures"] == ["test_b"]
    assert diff["fixed"] == []
    assert diff["still_passing"] == ["test_a"]
    assert diff["score_change"] == -50.0


def test_compare_fixed():
    # SCENARIO: test_a failed in run A but passes in run B
    # WHY: must surface fixes so teams know which issues are resolved
    # EXPECTED: fixed=["test_a"], score_change is positive
    run_a = _make_run(
        [_result("test_a", False), _result("test_b", True)],
        score=50.0,
    )
    run_b = _make_run(
        [_result("test_a", True), _result("test_b", True)],
        score=100.0,
    )

    diff = compare_runs(run_a, run_b)

    assert diff["fixed"] == ["test_a"]
    assert diff["new_failures"] == []
    assert diff["still_passing"] == ["test_b"]
    assert diff["score_change"] == 50.0


def test_compare_new_test():
    # SCENARIO: run B contains test_c which did not exist in run A
    # WHY: new tests must be reported as new_tests, not as regressions
    # EXPECTED: new_tests=["test_c"], no new_failures or fixed
    run_a = _make_run([_result("test_a", True)], score=100.0)
    run_b = _make_run([_result("test_a", True), _result("test_c", True)], score=100.0)

    diff = compare_runs(run_a, run_b)

    assert diff["new_tests"] == ["test_c"]
    assert diff["removed_tests"] == []
    assert diff["new_failures"] == []
    assert diff["fixed"] == []


def test_compare_removed_test():
    # SCENARIO: test_b exists in run A but is absent from run B
    # WHY: removed tests are structurally significant and must be reported
    # EXPECTED: removed_tests=["test_b"]
    run_a = _make_run([_result("test_a", True), _result("test_b", True)], score=100.0)
    run_b = _make_run([_result("test_a", True)], score=100.0)

    diff = compare_runs(run_a, run_b)

    assert diff["removed_tests"] == ["test_b"]
    assert diff["new_tests"] == []


# ---------------------------------------------------------------------------
# Integration tests — /api/compare endpoint
# ---------------------------------------------------------------------------


@pytest.fixture
def client_with_key(tmp_path):
    """TestClient + pre-registered API key for integration tests."""
    db_file = str(tmp_path / "compare_test.db")
    application = create_app(db_path=db_file)
    raw_key = generate_api_key("cmp-project")
    application.state.storage.save_api_key(hash_key(raw_key), "cmp-project")
    with TestClient(application) as c:
        yield c, raw_key


def _submit(client: TestClient, raw_key: str, results: list[dict]) -> int:  # type: ignore[type-arg]
    """Helper: submit a run and return its run_id."""
    resp = client.post(
        "/api/runs",
        json={"project": "cmp-project", "results": results},
        headers={"Authorization": f"Bearer {raw_key}"},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()["run_id"]


def test_compare_endpoint_returns_diff(client_with_key):
    # SCENARIO: submit two runs then call GET /api/compare?run_a=1&run_b=2
    # WHY: endpoint must wire compare_runs and return structured diff
    # EXPECTED: HTTP 200, diff contains new_failures with the regressed test
    client, raw_key = client_with_key

    run_a_id = _submit(client, raw_key, [
        _result("test_schema", True),
        _result("test_drift", True),
    ])
    run_b_id = _submit(client, raw_key, [
        _result("test_schema", True),
        _result("test_drift", False, "critical", "drift!"),
    ])

    resp = client.get(f"/api/compare?run_a={run_a_id}&run_b={run_b_id}")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["run_a"] == run_a_id
    assert body["run_b"] == run_b_id
    assert "test_drift" in body["diff"]["new_failures"]


def test_compare_endpoint_404_on_missing_run(client_with_key):
    # SCENARIO: request comparison with a non-existent run_b id
    # WHY: missing runs must return 404 with a clear error, not a 500
    # EXPECTED: HTTP 404
    client, raw_key = client_with_key

    run_a_id = _submit(client, raw_key, [
        _result("test_x", True),
    ])

    resp = client.get(f"/api/compare?run_a={run_a_id}&run_b=99999")
    assert resp.status_code == 404, resp.text
