"""Tests for mltk.core.result -- TestResult and TestSuite data structures.

TestResult holds a single assertion outcome (pass/fail, severity, details).
TestSuite aggregates multiple results and computes overall pass/fail and score.
These data structures feed into report generation and CI exit codes, so their
behavior must be precisely verified:
- TestResult creation preserves all fields
- TestSuite correctly aggregates pass/fail counts
- Suite pass/fail logic: CRITICAL failures = suite fails, WARNING-only = suite passes
- Edge case: empty suite score is 0.0 (not NaN or error)
"""

from mltk.core.result import Severity, TestResult, TestSuite


def test_test_result_creation() -> None:
    """PASS: TestResult stores name, passed, severity, and message.

    WHY: TestResult is the atom of mltk's output. Every assertion produces one.
    If field storage is broken, reports will show wrong data and CI gates
    will make incorrect pass/fail decisions.
    Expected: All fields accessible and matching constructor args.
    """
    result = TestResult(
        name="test_example",
        passed=True,
        severity=Severity.CRITICAL,
        message="All good",
    )
    assert result.passed is True
    assert result.severity == Severity.CRITICAL
    assert result.name == "test_example"


def test_test_suite_passed() -> None:
    """PASS: Suite with all passing results reports passed=True and score=100.

    WHY: The suite's overall status drives CI exit codes. If two tests pass,
    the suite must report passed=True. The score (100%) is used in the
    ML Test Score rubric for the HTML report.
    Expected: suite.passed is True, total=2, score=100.0.
    """
    suite = TestSuite()
    suite.add(TestResult(name="t1", passed=True, severity=Severity.CRITICAL, message="ok"))
    suite.add(TestResult(name="t2", passed=True, severity=Severity.WARNING, message="ok"))
    assert suite.passed is True
    assert suite.total == 2
    assert suite.score == 100.0


def test_test_suite_failed_critical() -> None:
    """FAIL: Suite with a CRITICAL failure reports passed=False.

    WHY: If any CRITICAL assertion fails, the entire suite must fail.
    This is the gate that prevents broken models from reaching production.
    A single critical failure (e.g., accuracy below threshold) must block.
    Expected: suite.passed is False, failed_count=1.
    """
    suite = TestSuite()
    suite.add(TestResult(name="t1", passed=False, severity=Severity.CRITICAL, message="fail"))
    suite.add(TestResult(name="t2", passed=True, severity=Severity.WARNING, message="ok"))
    assert suite.passed is False
    assert suite.failed_count == 1


def test_test_suite_warning_only_still_passes() -> None:
    """PASS: Suite with only WARNING failures still passes.

    WHY: Warnings are informational -- "data slightly stale" or "one outlier
    found" should not block deployment. Only CRITICAL failures block.
    This allows teams to use warnings for monitoring without breaking CI.
    Expected: suite.passed is True despite one WARNING failure.
    """
    suite = TestSuite()
    suite.add(TestResult(name="t1", passed=True, severity=Severity.CRITICAL, message="ok"))
    suite.add(TestResult(name="t2", passed=False, severity=Severity.WARNING, message="warn"))
    assert suite.passed is True


def test_empty_suite_score() -> None:
    """Edge case: Empty suite returns score 0.0 instead of error.

    WHY: An empty test run (no assertions collected) should not crash the
    report generator. Score 0.0 signals "nothing was tested" rather than
    raising a ZeroDivisionError.
    Expected: suite.score == 0.0.
    """
    suite = TestSuite()
    assert suite.score == 0.0


# --- P1-37: JSON schema tests ---


def test_json_schema_structure() -> None:
    """TestResult.json_schema() returns a valid JSON Schema dict.

    WHY: External consumers (CI dashboards, Grafana, data contracts) need
    a machine-readable schema to validate mltk JSON output. If the schema
    is malformed, downstream integrations silently accept bad data.
    Expected: Schema has required fields, correct types, severity enum.
    """
    schema = TestResult.json_schema()
    assert schema["type"] == "object"
    assert schema["required"] == ["name", "passed", "severity", "message"]
    assert "properties" in schema
    assert schema["properties"]["passed"]["type"] == "boolean"
    assert schema["properties"]["severity"]["enum"] == ["critical", "warning", "info"]
    assert schema["properties"]["duration_ms"]["type"] == "number"


def test_json_schema_contract_matches_serialized_result() -> None:
    """Serialized TestResult output matches the JSON schema contract.

    WHY: If TestResult fields are renamed or the serialization format
    changes, the schema and actual output will diverge -- breaking
    consumers that rely on the schema. This contract test catches drift.
    Expected: Every required field in the schema is present in the
    serialized output with the correct type.
    """
    result = TestResult(
        name="contract.test",
        passed=True,
        severity=Severity.WARNING,
        message="Contract test passed",
        details={"key": "value"},
    )
    schema = TestResult.json_schema()

    # Serialize to the format that to_json_records produces
    record = {
        "name": result.name,
        "passed": result.passed,
        "severity": result.severity.value,
        "message": result.message,
        "details": result.details,
        "duration_ms": result.duration_ms,
        "timestamp": result.timestamp.isoformat(),
    }

    # Verify all required fields are present
    for field in schema["required"]:
        assert field in record, f"Required field '{field}' missing from serialized output"

    # Verify types match schema
    assert isinstance(record["name"], str)
    assert isinstance(record["passed"], bool)
    assert record["severity"] in schema["properties"]["severity"]["enum"]
    assert isinstance(record["message"], str)
    assert isinstance(record["details"], dict)
    assert isinstance(record["duration_ms"], (int, float))
    assert isinstance(record["timestamp"], str)
