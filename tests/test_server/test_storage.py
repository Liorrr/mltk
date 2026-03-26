"""Tests for mltk.server.storage — SQLite persistence layer."""
from __future__ import annotations

import pytest

from mltk.server.storage import Storage

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_RESULTS = [
    {
        "name": "test_a", "passed": True, "severity": "info",
        "message": "ok", "details": {}, "duration_ms": 10.0,
    },
    {
        "name": "test_b", "passed": False, "severity": "error",
        "message": "fail", "details": {"score": 0.4}, "duration_ms": 20.0,
    },
    {
        "name": "test_c", "passed": True, "severity": "info",
        "message": "ok", "details": {}, "duration_ms": 5.0,
    },
]


@pytest.fixture
def storage(tmp_path):
    """In-memory-like storage backed by a temp file."""
    # SCENARIO: fresh storage for every test
    # WHY: avoid cross-test state bleed
    # EXPECTED: each test gets a clean database
    db_file = str(tmp_path / "test.db")
    return Storage(db_path=db_file)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_save_and_get_run(storage):
    # SCENARIO: save a run and retrieve it by id
    # WHY: core round-trip must be lossless
    # EXPECTED: run dict matches what was saved, results count matches
    run_id = storage.save_run("myproject", SAMPLE_RESULTS)
    assert isinstance(run_id, int)
    assert run_id > 0

    run = storage.get_run(run_id)
    assert run is not None
    assert run["project"] == "myproject"
    assert run["total"] == 3
    assert run["passed"] == 2
    assert run["failed"] == 1
    assert len(run["results"]) == 3

    names = {r["name"] for r in run["results"]}
    assert names == {"test_a", "test_b", "test_c"}

    # details JSON round-trips correctly
    test_b = next(r for r in run["results"] if r["name"] == "test_b")
    assert test_b["details"] == {"score": 0.4}
    assert test_b["passed"] is False


def test_list_runs(storage):
    # SCENARIO: save multiple runs then list them
    # WHY: list_runs must return all saved runs in reverse-chronological order
    # EXPECTED: most-recently saved run appears first
    id1 = storage.save_run("proj", SAMPLE_RESULTS[:1])
    id2 = storage.save_run("proj", SAMPLE_RESULTS[:2])
    id3 = storage.save_run("proj", SAMPLE_RESULTS)

    runs = storage.get_runs()
    assert len(runs) >= 3
    ids = [r["id"] for r in runs]
    # most recent first
    assert ids.index(id3) < ids.index(id2) < ids.index(id1)


def test_trends(storage):
    # SCENARIO: save three runs for a project then fetch trends
    # WHY: trends must surface score progression in chronological order
    # EXPECTED: trend list length == number of runs, scores are floats
    for i in range(3):
        results = [{"name": f"t{j}", "passed": j <= i, "severity": "info",
                    "message": "", "details": {}, "duration_ms": 1.0}
                   for j in range(3)]
        storage.save_run("trend_proj", results)

    trends = storage.get_trends("trend_proj")
    assert len(trends) == 3
    for point in trends:
        assert "score" in point
        assert "timestamp" in point
        assert isinstance(point["score"], float)

    # chronological order: ids should be ascending
    ids = [p["id"] for p in trends]
    assert ids == sorted(ids)


def test_empty_db(storage):
    # SCENARIO: query a freshly created database
    # WHY: callers must handle zero-results gracefully
    # EXPECTED: get_runs returns [] and get_run returns None
    assert storage.get_runs() == []
    assert storage.get_run(999) is None
    assert storage.get_trends("nosuchproject") == []


def test_project_filter(storage):
    # SCENARIO: save runs for two different projects then filter by project
    # WHY: list_runs(project=X) must not return runs from other projects
    # EXPECTED: each project only sees its own runs
    storage.save_run("alpha", SAMPLE_RESULTS[:1])
    storage.save_run("alpha", SAMPLE_RESULTS[:1])
    storage.save_run("beta", SAMPLE_RESULTS[:2])

    alpha_runs = storage.get_runs(project="alpha")
    beta_runs = storage.get_runs(project="beta")

    assert len(alpha_runs) == 2
    assert len(beta_runs) == 1
    assert all(r["project"] == "alpha" for r in alpha_runs)
    assert all(r["project"] == "beta" for r in beta_runs)


def test_run_not_found(storage):
    # SCENARIO: request a run id that was never saved
    # WHY: get_run must return None rather than raising an exception
    # EXPECTED: None returned for unknown id
    result = storage.get_run(99999)
    assert result is None


def test_score_calculation(storage):
    # SCENARIO: save a fully-passing and a fully-failing run
    # WHY: score field must reflect pass-rate accurately
    # EXPECTED: 100% pass → score 100.0, 0% pass → score 0.0
    all_pass = [{"name": "t", "passed": True, "severity": "info",
                 "message": "", "details": {}, "duration_ms": 1.0}]
    all_fail = [{"name": "t", "passed": False, "severity": "error",
                 "message": "", "details": {}, "duration_ms": 1.0}]

    id_pass = storage.save_run("score_proj", all_pass)
    id_fail = storage.save_run("score_proj", all_fail)

    run_pass = storage.get_run(id_pass)
    run_fail = storage.get_run(id_fail)

    assert run_pass["score"] == pytest.approx(100.0)
    assert run_fail["score"] == pytest.approx(0.0)
