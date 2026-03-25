"""Tests for mltk.testing.selection — smart test selection."""
from __future__ import annotations

import pytest

from mltk.testing.selection import build_test_map, select_affected_tests

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_project(tmp_path):
    """Create a minimal fake project with src and test trees."""
    src = tmp_path / "src" / "mypkg"
    src.mkdir(parents=True)

    tests = tmp_path / "tests"
    tests.mkdir()

    # Source modules
    (src / "__init__.py").write_text("", encoding="utf-8")
    (src / "drift.py").write_text("x = 1\n", encoding="utf-8")
    (src / "schema.py").write_text("y = 2\n", encoding="utf-8")

    # Test that imports drift
    drift_test = tests / "test_drift.py"
    drift_test.write_text(
        "from mypkg.drift import x\n\ndef test_x():\n    assert x == 1\n",
        encoding="utf-8",
    )

    # Test that imports schema
    schema_test = tests / "test_schema.py"
    schema_test.write_text(
        "from mypkg import schema\n\ndef test_y():\n    assert schema.y == 2\n",
        encoding="utf-8",
    )

    return {
        "src": tmp_path / "src",
        "tests": tests,
        "drift_src": str(src / "drift.py"),
        "schema_src": str(src / "schema.py"),
        "drift_test": str(drift_test),
        "schema_test": str(schema_test),
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_build_test_map(fake_project):
    # SCENARIO: project with two source files each imported by one test file
    # WHY: map must correctly link source modules to their test files
    # EXPECTED: drift.py -> [test_drift.py], schema.py -> [test_schema.py]
    mapping = build_test_map(fake_project["tests"], fake_project["src"])

    drift_key = fake_project["drift_src"]
    schema_key = fake_project["schema_src"]

    assert drift_key in mapping
    assert fake_project["drift_test"] in mapping[drift_key]

    assert schema_key in mapping
    assert fake_project["schema_test"] in mapping[schema_key]


def test_select_affected(fake_project):
    # SCENARIO: drift.py changes; test_drift.py must be selected
    # WHY: changed source files must trigger their dependent tests
    # EXPECTED: test_drift.py is in the affected list
    mapping = build_test_map(fake_project["tests"], fake_project["src"])
    affected = select_affected_tests([fake_project["drift_src"]], mapping)

    assert fake_project["drift_test"] in affected


def test_no_affected(fake_project):
    # SCENARIO: an unrelated file changes that no test imports
    # WHY: selection must return empty list when no tests are linked
    # EXPECTED: []
    mapping = build_test_map(fake_project["tests"], fake_project["src"])
    unrelated = str(fake_project["src"] / "mypkg" / "utils.py")
    affected = select_affected_tests([unrelated], mapping)

    assert affected == []


def test_empty_map():
    # SCENARIO: empty test_map passed to select_affected_tests
    # WHY: must handle empty input gracefully without errors
    # EXPECTED: empty list returned
    affected = select_affected_tests(["some/file.py"], {})
    assert affected == []


def test_build_test_map_empty_dirs(tmp_path):
    # SCENARIO: test_dir and src_dir exist but contain no Python files
    # WHY: empty project must not crash, just return empty map
    # EXPECTED: empty dict returned
    src = tmp_path / "src"
    src.mkdir()
    tests = tmp_path / "tests"
    tests.mkdir()

    mapping = build_test_map(tests, src)
    assert mapping == {}


def test_select_affected_deduplication(fake_project):
    # SCENARIO: same source file listed twice in changed_files
    # WHY: test file must appear only once in the result
    # EXPECTED: no duplicates in affected list
    mapping = build_test_map(fake_project["tests"], fake_project["src"])
    affected = select_affected_tests(
        [fake_project["drift_src"], fake_project["drift_src"]],
        mapping,
    )

    assert affected.count(fake_project["drift_test"]) == 1
