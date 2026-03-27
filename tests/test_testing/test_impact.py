"""Tests for mltk.testing.impact -- test impact analysis and coverage assertions.

Impact analysis determines which tests to run after a source change by
building a dependency graph from Python imports.  These tests verify:

- Direct impact: changing a source file flags the test that imports it.
- Transitive impact: if test_A imports module_B which imports module_C,
  changing module_C still flags test_A.
- Coverage assertion: detects when impacted tests were skipped in CI.
- Edge cases: empty inputs, no test files, no changed files.
"""
from __future__ import annotations

import pytest

from mltk.core.assertion import MltkAssertionError
from mltk.testing.impact import analyze_impact, assert_impact_coverage


def _create_src_file(base, rel_path, content):
    """Helper: create a Python file at base/rel_path with given content."""
    path = base / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _setup_project(tmp_path):
    """Create a minimal project layout for impact analysis tests.

    Layout::

        tmp_path/
            src/
                mypkg/
                    __init__.py
                    core.py          # no imports
                    utils.py         # imports core
                    unrelated.py     # no imports
            tests/
                test_core.py         # imports mypkg.core
                test_utils.py        # imports mypkg.utils
                test_unrelated.py    # imports mypkg.unrelated
    """
    # Source files
    _create_src_file(tmp_path, "src/mypkg/__init__.py", "")
    _create_src_file(tmp_path, "src/mypkg/core.py", "VALUE = 42\n")
    _create_src_file(
        tmp_path, "src/mypkg/utils.py",
        "from mypkg import core\ndef helper(): return core.VALUE\n",
    )
    _create_src_file(tmp_path, "src/mypkg/unrelated.py", "X = 1\n")

    # Test files
    _create_src_file(
        tmp_path, "tests/test_core.py",
        "from mypkg.core import VALUE\ndef test_value(): assert VALUE == 42\n",
    )
    _create_src_file(
        tmp_path, "tests/test_utils.py",
        "from mypkg.utils import helper\ndef test_helper(): assert helper() == 42\n",
    )
    _create_src_file(
        tmp_path, "tests/test_unrelated.py",
        "from mypkg.unrelated import X\ndef test_x(): assert X == 1\n",
    )

    return tmp_path


# ---------------------------------------------------------------------------
# analyze_impact tests
# ---------------------------------------------------------------------------


class TestAnalyzeImpact:
    """Tests for the analyze_impact function."""

    def test_direct_dependency(self, tmp_path):
        # SCENARIO: Change core.py -> test_core.py should be flagged
        # WHY: Direct imports must be detected; this is the base case
        # EXPECTED: test_core.py in result
        project = _setup_project(tmp_path)
        changed = [str(project / "src" / "mypkg" / "core.py")]

        result = analyze_impact(changed, project_root=str(project), test_dir="tests")

        test_names = [str(r).replace("\\", "/") for r in result]
        assert any("test_core" in t for t in test_names)

    def test_unrelated_change_returns_empty(self, tmp_path):
        # SCENARIO: Change a file that no test imports
        # WHY: Impact analysis must not over-select; unrelated changes = no tests
        # EXPECTED: empty result (or only truly related tests)
        project = _setup_project(tmp_path)
        # Create a source file that nothing imports
        _create_src_file(project, "src/mypkg/orphan.py", "ORPHAN = True\n")
        changed = [str(project / "src" / "mypkg" / "orphan.py")]

        result = analyze_impact(changed, project_root=str(project), test_dir="tests")

        assert result == []

    def test_transitive_dependency(self, tmp_path):
        # SCENARIO: utils.py imports core.py; change core.py -> test_utils.py flagged
        # WHY: Transitive deps are the main value-add over naive filename matching.
        #   If core.py breaks, utils.py (which depends on it) is also affected,
        #   so test_utils.py must run.
        # EXPECTED: test_utils.py in result
        project = _setup_project(tmp_path)
        changed = [str(project / "src" / "mypkg" / "core.py")]

        result = analyze_impact(changed, project_root=str(project), test_dir="tests")

        test_names = [str(r).replace("\\", "/") for r in result]
        assert any("test_utils" in t for t in test_names)

    def test_no_changed_files(self, tmp_path):
        # SCENARIO: Empty changed_files list
        # WHY: Edge case -- nothing changed means nothing to test
        # EXPECTED: empty list, no crash
        project = _setup_project(tmp_path)

        result = analyze_impact([], project_root=str(project), test_dir="tests")

        assert result == []

    def test_no_test_directory(self, tmp_path):
        # SCENARIO: test_dir does not exist
        # WHY: Edge case -- new project with no tests yet should not crash
        # EXPECTED: empty list
        _create_src_file(tmp_path, "src/mypkg/core.py", "X = 1\n")
        changed = [str(tmp_path / "src" / "mypkg" / "core.py")]

        result = analyze_impact(
            changed, project_root=str(tmp_path), test_dir="nonexistent_tests"
        )

        assert result == []


# ---------------------------------------------------------------------------
# assert_impact_coverage tests
# ---------------------------------------------------------------------------


class TestAssertImpactCoverage:
    """Tests for the assert_impact_coverage assertion."""

    def test_all_impacted_tests_executed(self, tmp_path):
        # SCENARIO: All impacted tests were actually run
        # WHY: This is the happy path -- CI executed everything it should
        # EXPECTED: result.passed is True
        project = _setup_project(tmp_path)
        changed = [str(project / "src" / "mypkg" / "unrelated.py")]

        # First figure out what's impacted, then claim we ran it
        impacted = analyze_impact(changed, str(project), "tests")
        result = assert_impact_coverage(
            changed, executed_tests=impacted,
            project_root=str(project), test_dir="tests",
        )

        assert result.passed is True
        assert result.details["missing_tests"] == []

    def test_missing_impacted_test_fails(self, tmp_path):
        # SCENARIO: Changed core.py but did NOT run test_core.py
        # WHY: This is the coverage gap that the assertion is designed to catch.
        #   A CI misconfiguration silently skipped an impacted test.
        # EXPECTED: MltkAssertionError raised, missing_tests is non-empty
        project = _setup_project(tmp_path)
        changed = [str(project / "src" / "mypkg" / "core.py")]

        with pytest.raises(MltkAssertionError) as exc_info:
            assert_impact_coverage(
                changed, executed_tests=[],
                project_root=str(project), test_dir="tests",
            )

        assert exc_info.value.result.passed is False
        assert len(exc_info.value.result.details["missing_tests"]) > 0

    def test_result_has_duration(self, tmp_path):
        # SCENARIO: The @timed_assertion decorator populates duration_ms
        # WHY: All timed assertions must report wall-clock time
        # EXPECTED: duration_ms >= 0
        project = _setup_project(tmp_path)

        result = assert_impact_coverage(
            [], executed_tests=[],
            project_root=str(project), test_dir="tests",
        )

        assert result.duration_ms >= 0.0
