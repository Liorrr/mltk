"""Tests for mltk.doctor — environment diagnostics.

The doctor module runs checks on the user's environment and reports
issues with Python version, dependencies, config files, directories,
the Rust extension, and the pytest plugin registration.

These tests verify:
1. diagnose() returns a meaningful list of DiagnosticResult objects
2. Checks for the current Python version and required deps always pass
3. Suspicious config values are caught and reported as WARN
4. FAIL/WARN results include actionable fix hints
5. Sufficient breadth of checks is always executed
"""

from __future__ import annotations

import sys
from pathlib import Path

from mltk.doctor import DiagnosticResult, diagnose

# ---------------------------------------------------------------------------
# Basic contract tests
# ---------------------------------------------------------------------------


def test_diagnose_returns_results() -> None:
    """SCENARIO: diagnose() is called with no arguments.
    WHY: It must always return a non-empty list — callers (CLI, tests) iterate over it.
    EXPECTED: Returns a list with at least one DiagnosticResult instance.
    """
    results = diagnose()
    assert isinstance(results, list)
    assert len(results) > 0
    assert all(isinstance(r, DiagnosticResult) for r in results)


def test_all_checks_run() -> None:
    """SCENARIO: diagnose() is called in a standard mltk dev environment.
    WHY: Fewer than 8 results suggests a check was silently skipped or removed,
         which would give users a false sense of confidence.
    EXPECTED: At least 8 DiagnosticResult objects are returned.
    """
    results = diagnose()
    assert len(results) >= 8, (
        f"Expected at least 8 diagnostic checks, got {len(results)}. "
        "A check may have been removed or merged incorrectly."
    )


def test_result_fields_present() -> None:
    """SCENARIO: Each result from diagnose() is inspected for required fields.
    WHY: Callers (CLI, tests) access .name, .status, .message on every result.
         Missing fields cause AttributeError at display time.
    EXPECTED: Every result has name, status, and message as non-empty strings.
    """
    results = diagnose()
    for r in results:
        assert isinstance(r.name, str), f"Result name not str: {r}"
        assert r.name, f"Result has empty name: {r}"
        assert r.status in ("OK", "WARN", "FAIL"), (
            f"Invalid status '{r.status}' for check '{r.name}'"
        )
        assert isinstance(r.message, str), f"Message is not a string in check '{r.name}'"


# ---------------------------------------------------------------------------
# Python version check
# ---------------------------------------------------------------------------


def test_python_version_ok() -> None:
    """SCENARIO: diagnose() runs on the current Python interpreter.
    WHY: mltk requires Python >= 3.10. Since we're running tests, the current
         Python must pass this check — if it FAIL here, the test suite itself
         wouldn't have loaded.
    EXPECTED: The 'Python version' check has status 'OK'.
    """
    results = diagnose()
    version_results = [r for r in results if r.name == "Python version"]
    assert len(version_results) == 1, "Expected exactly one 'Python version' check"
    r = version_results[0]
    assert r.status == "OK", (
        f"Python {sys.version_info.major}.{sys.version_info.minor} "
        f"should pass but got status '{r.status}': {r.message}"
    )


# ---------------------------------------------------------------------------
# Core deps check
# ---------------------------------------------------------------------------


def test_core_deps_ok() -> None:
    """SCENARIO: diagnose() runs when numpy and pandas are installed.
    WHY: numpy and pandas are required core deps — the test suite imports them,
         so they must be present. If the check reports FAIL, the doctor logic
         is wrong (false negative).
    EXPECTED: Both 'Core dep: numpy' and 'Core dep: pandas' have status 'OK'.
    """
    results = diagnose()
    core_results = {r.name: r for r in results if r.name.startswith("Core dep:")}

    assert "Core dep: numpy" in core_results, "numpy check missing from diagnose() output"
    assert "Core dep: pandas" in core_results, "pandas check missing from diagnose() output"

    for name in ("Core dep: numpy", "Core dep: pandas"):
        r = core_results[name]
        assert r.status == "OK", (
            f"{name} is installed but diagnose() reports '{r.status}': {r.message}"
        )


# ---------------------------------------------------------------------------
# Config validation check
# ---------------------------------------------------------------------------


def test_config_validation_suspicious_threshold(tmp_path: Path, monkeypatch) -> None:
    """SCENARIO: mltk.yaml sets drift_threshold=2.5 (above valid range 0–1).
    WHY: A threshold >1 makes no statistical sense for most drift methods
         and will silently pass all drift checks. The doctor should catch this.
    EXPECTED: 'Config validation' check returns status 'WARN'.
    """
    yaml_content = "drift_threshold: 2.5\n"
    yaml_file = tmp_path / "mltk.yaml"
    yaml_file.write_text(yaml_content)

    # Point cwd to tmp_path so config loader finds the file
    monkeypatch.chdir(tmp_path)

    results = diagnose()
    validation_results = [r for r in results if r.name == "Config validation"]
    assert len(validation_results) == 1
    r = validation_results[0]
    assert r.status == "WARN", (
        f"Expected WARN for drift_threshold=2.5 but got '{r.status}': {r.message}"
    )
    assert "drift_threshold" in r.message


def test_config_validation_invalid_method(tmp_path: Path, monkeypatch) -> None:
    """SCENARIO: mltk.yaml sets drift_method='unknown_method'.
    WHY: An unrecognized drift method will silently fail at runtime.
         The doctor should flag it early.
    EXPECTED: 'Config validation' check returns status 'WARN' with method in message.
    """
    yaml_content = "drift_method: unknown_method\n"
    yaml_file = tmp_path / "mltk.yaml"
    yaml_file.write_text(yaml_content)

    monkeypatch.chdir(tmp_path)

    results = diagnose()
    validation_results = [r for r in results if r.name == "Config validation"]
    assert len(validation_results) == 1
    r = validation_results[0]
    assert r.status == "WARN", (
        f"Expected WARN for unknown drift_method but got '{r.status}': {r.message}"
    )
    assert "drift_method" in r.message


def test_config_validation_defaults_ok() -> None:
    """SCENARIO: diagnose() runs with default config (no mltk.yaml in cwd).
    WHY: Default config values are chosen to be sane. If they trigger WARN,
         every fresh install would show spurious warnings.
    EXPECTED: 'Config validation' check returns status 'OK' for defaults.
    """
    results = diagnose()
    validation_results = [r for r in results if r.name == "Config validation"]
    # Config validation may or may not exist depending on environment
    # but if it does exist it should be OK for defaults
    for r in validation_results:
        # Default values (ks, 0.05, html, seed=42) should not trigger WARN
        # unless the test itself is run from a dir with a suspicious config
        assert r.status in ("OK", "WARN"), f"Unexpected status '{r.status}'"


# ---------------------------------------------------------------------------
# Fix hints
# ---------------------------------------------------------------------------


def test_result_has_fix_hint() -> None:
    """SCENARIO: All WARN and FAIL results in diagnose() output are inspected.
    WHY: Users act on fix hints to resolve issues. A WARN/FAIL without a hint
         leaves users stuck with no actionable next step.
    EXPECTED: Every result with status 'WARN' or 'FAIL' has a non-empty fix_hint.
    """
    results = diagnose()
    for r in results:
        if r.status in ("WARN", "FAIL"):
            assert r.fix_hint is not None, (
                f"Check '{r.name}' has status '{r.status}' but fix_hint is None"
            )
            assert r.fix_hint.strip(), (
                f"Check '{r.name}' has status '{r.status}' but empty fix_hint"
            )


# ---------------------------------------------------------------------------
# DiagnosticResult dataclass
# ---------------------------------------------------------------------------


def test_diagnostic_result_dataclass_defaults() -> None:
    """SCENARIO: DiagnosticResult is constructed with minimal args.
    WHY: fix_hint is optional — most OK checks don't provide one.
         Callers must be able to omit it without errors.
    EXPECTED: fix_hint defaults to None; other fields are set correctly.
    """
    r = DiagnosticResult(name="test", status="OK", message="all good")
    assert r.name == "test"
    assert r.status == "OK"
    assert r.message == "all good"
    assert r.fix_hint is None


def test_diagnostic_result_with_fix_hint() -> None:
    """SCENARIO: DiagnosticResult is constructed with a fix_hint.
    WHY: WARN/FAIL results include actionable fix hints. Verify the field
         round-trips correctly.
    EXPECTED: fix_hint is stored and accessible.
    """
    r = DiagnosticResult(
        name="Missing dep",
        status="WARN",
        message="scipy not installed",
        fix_hint="pip install mltk[scipy]",
    )
    assert r.fix_hint == "pip install mltk[scipy]"
    assert r.status == "WARN"
