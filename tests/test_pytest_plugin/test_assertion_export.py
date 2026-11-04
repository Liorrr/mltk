"""End-to-end tests for pytest assertion export and report warnings."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import mltk
from mltk.pytest_plugin.plugin import MltkReportCollector


def _run_pytest(tmp_path: Path, source: str, *extra: str) -> subprocess.CompletedProcess[str]:
    test_file = tmp_path / "test_sample.py"
    test_file.write_text(source, encoding="utf-8")
    env = {
        **os.environ,
        "PYTHONPATH": str(Path(mltk.__file__).resolve().parents[1])
        + os.pathsep
        + os.environ.get("PYTHONPATH", ""),
    }
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(test_file),
            "-q",
            "-p",
            "no:cacheprovider",
            *extra,
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )


def _exported(
    tmp_path: Path, source: str, *, returncode: int = 0
) -> list[dict[str, object]]:
    output = tmp_path / "results.json"
    completed = _run_pytest(tmp_path, source, "--mltk-export-json", str(output))
    assert completed.returncode == returncode, completed.stdout + completed.stderr
    return json.loads(output.read_text(encoding="utf-8"))


def _record_for(records: list[dict[str, object]], test_name: str) -> dict[str, object]:
    return next(record for record in records if str(record["name"]).endswith(test_name))


def test_passing_assertion_exports_details_and_warning(tmp_path: Path) -> None:
    records = _exported(
        tmp_path,
        """
from mltk.core.assertion import assert_true
from mltk.core.result import Severity

def test_pass():
    assert_true(True, "acc.check", "ok", accuracy=0.93)

def test_warning():
    assert_true(False, "warn.check", "warning", severity=Severity.WARNING)
""",
    )
    passed = _record_for(records, "test_pass")
    warning = _record_for(records, "test_warning")
    assert passed["assertions"] == [
        {
            "name": "acc.check",
            "passed": True,
            "severity": "critical",
            "message": "ok",
            "details": {"accuracy": 0.93},
            "duration_ms": passed["assertions"][0]["duration_ms"],
            "caught": False,
        }
    ]
    assert warning["assertions"][0]["passed"] is False
    assert warning["assertions"][0]["severity"] == "warning"
    assert warning["assertions"][0]["caught"] is False


def test_caught_critical_keeps_plain_test_headline(tmp_path: Path) -> None:
    records = _exported(
        tmp_path,
        """
import pytest
from mltk.core.assertion import MltkAssertionError, assert_true

def test_caught():
    with pytest.raises(MltkAssertionError):
        assert_true(False, "caught.check", "boom")
""",
    )
    record = _record_for(records, "test_caught")
    assertion = record["assertions"][0]
    assert assertion["name"] == "caught.check"
    assert assertion["caught"] is True
    assert record["passed"] is True
    assert record["severity"] == "info"
    assert record["message"] == ""
    assert record["details"] == {}


def test_multiple_fixture_and_isolated_assertions_export_in_order(tmp_path: Path) -> None:
    records = _exported(
        tmp_path,
        """
import pytest
from mltk.core.assertion import assert_true

@pytest.fixture
def checked():
    assert_true(True, "fixture.check", "fixture")

def test_a(checked):
    assert_true(True, "a.first", "first")
    assert_true(True, "a.second", "second")

def test_b():
    assert_true(True, "b.only", "only")
""",
    )
    test_a = _record_for(records, "test_a")
    test_b = _record_for(records, "test_b")
    assert [item["name"] for item in test_a["assertions"]] == [
        "fixture.check",
        "a.first",
        "a.second",
    ]
    assert [item["name"] for item in test_b["assertions"]] == ["b.only"]


def test_critical_failure_is_deduped_and_is_headline(tmp_path: Path) -> None:
    records = _exported(
        tmp_path,
        """
from mltk.core.assertion import assert_true

def test_failure():
    assert_true(False, "critical.check", "failed", value=4)
""",
        returncode=1,
    )
    record = _record_for(records, "test_failure")
    assert len(record["assertions"]) == 1
    assert record["assertions"][0]["name"] == "critical.check"
    assert record["assertions"][0]["caught"] is False
    assert record["message"] == "failed"


def test_plain_test_has_empty_assertions_and_old_defaults(tmp_path: Path) -> None:
    records = _exported(tmp_path, "def test_plain():\n    pass\n")
    record = _record_for(records, "test_plain")
    assert record["assertions"] == []
    assert record["severity"] == "info"
    assert record["message"] == ""
    assert record["details"] == {}


def test_assertion_export_handles_string_severity_and_partial_result() -> None:
    assertion = SimpleNamespace(
        name="partial",
        passed=False,
        severity="critical",
        details={"value": 1},
    )
    collector = MltkReportCollector()
    collector.add(
        "tests::partial",
        "passed",
        0.1,
        ml_results=[assertion],  # type: ignore[list-item]
    )

    exported = collector.to_json_records()[0]["assertions"][0]
    assert exported["name"] == "partial"
    assert exported["severity"] == "critical"
    assert exported["duration_ms"] == 0.0
    assert exported["caught"] is True


def test_recording_is_inactive_without_flag_and_active_with_export(tmp_path: Path) -> None:
    source = """
from mltk.core import recorder

def test_recording_state():
    assert recorder._current.get() IS_ACTIVE
"""
    inactive = _run_pytest(tmp_path, source.replace("IS_ACTIVE", "is None"))
    assert inactive.returncode == 0, inactive.stdout + inactive.stderr

    active = _run_pytest(
        tmp_path,
        source.replace("IS_ACTIVE", "is not None"),
        "--mltk-export-json",
        str(tmp_path / "active.json"),
    )
    assert active.returncode == 0, active.stdout + active.stderr


def test_shared_async_fixture_assertions_are_isolated_per_test(tmp_path: Path) -> None:
    pytest.importorskip("anyio")
    records = _exported(
        tmp_path,
        """
import pytest
from mltk.core.assertion import assert_true

pytestmark = pytest.mark.anyio

@pytest.fixture(scope="module")
def anyio_backend():
    return "asyncio"

@pytest.fixture(scope="module")
async def shared():
    yield 1

async def test_one(shared):
    assert_true(True, "async.one", "ok")

async def test_two(shared):
    assert_true(True, "async.two", "ok")

async def test_three(shared):
    assert_true(True, "async.three", "ok")
""",
    )
    for test_name, assertion_name in (
        ("test_one", "async.one"),
        ("test_two", "async.two"),
        ("test_three", "async.three"),
    ):
        record = _record_for(records, test_name)
        assert [item["name"] for item in record["assertions"]] == [assertion_name]


def test_server_only_does_not_arm_recorder(tmp_path: Path) -> None:
    source = """
from mltk.core import recorder

def test_server_recording_state():
    assert recorder._current.get() is None
"""
    completed = _run_pytest(
        tmp_path,
        source,
        "--mltk-server",
        "http://127.0.0.1:9",
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_makereport_deduplicates_large_result_list_in_order() -> None:
    """Regression pin: the linear id()-based dedup keeps order and identity semantics.

    The previous quadratic dedup gave the same result (only slowly), so this pins
    behaviour across the refactor rather than failing on the old code.
    """
    from mltk.core import recorder
    from mltk.core.result import Severity, TestResult
    from mltk.pytest_plugin.plugin import pytest_runtest_makereport

    class Config:
        _mltk_collector = MltkReportCollector()

    item = type("Item", (), {"config": Config(), "nodeid": "tests::large"})()
    call = type("Call", (), {"when": "call", "excinfo": None, "duration": 0.1})()
    results = [
        TestResult(
            name=f"result.{index}",
            passed=True,
            severity=Severity.INFO,
            message="ok",
        )
        for index in range(20_000)
    ]
    token = recorder.arm()
    try:
        for result in results:
            recorder.record(result)
        pytest_runtest_makereport(item, call)
    finally:
        recorder.disarm(token)

    recorded = item.config._mltk_collector.results[0]["ml_results"]
    assert len(recorded) == 20_000
    assert [result.name for result in recorded] == [result.name for result in results]


def test_server_payload_excludes_assertions_without_mutating_records(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mltk.core.result import Severity, TestResult
    from mltk.pytest_plugin.plugin import _push_to_server

    collector = MltkReportCollector()
    assertion = TestResult(
        name="server.check",
        passed=True,
        severity=Severity.INFO,
        message="ok",
        details={"value": 1},
    )
    collector.add(
        "tests::server",
        "passed",
        0.1,
        ml_result=assertion,
        ml_results=[assertion],
    )
    captured: dict[str, object] = {}

    class Response:
        def __enter__(self) -> Response:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return b'{"run_id": "run"}'

    def fake_urlopen(request: object, timeout: int) -> Response:
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return Response()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    class Session:
        class config:
            class pluginmanager:
                @staticmethod
                def get_plugin(name: str) -> None:
                    return None

    _push_to_server(collector, "http://server", Session())

    pushed_record = captured["payload"]["results"][0]
    assert "assertions" not in pushed_record
    assert "assertions" in collector.to_json_records()[0]


class _Writer:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def sep(self, marker: str, title: str | None = None) -> None:
        pass

    def line(self, text: str = "") -> None:
        self.lines.append(text)


class _Reporter:
    def __init__(self) -> None:
        self._tw = _Writer()


class _PluginManager:
    def __init__(self, reporter: _Reporter) -> None:
        self.reporter = reporter

    def get_plugin(self, name: str) -> object | None:
        return self.reporter if name == "terminalreporter" else None


class _Session:
    def __init__(self, collector: MltkReportCollector, reporter: _Reporter) -> None:
        self.config = type("Config", (), {})()
        self.config._mltk_collector = collector
        self.config.pluginmanager = _PluginManager(reporter)

    def getoption(self, name: str, default: object = None) -> object:
        return default


def _html_session(collector: MltkReportCollector) -> tuple[_Session, _Reporter]:
    reporter = _Reporter()
    session = _Session(collector, reporter)
    session.config.getoption = lambda name, default=None: (
        True if name == "--mltk-report" else default
    )
    return session, reporter


def test_html_import_error_prints_install_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    from mltk.pytest_plugin.plugin import pytest_sessionfinish

    cause = ModuleNotFoundError("missing", name="jinja2")

    def fail(*args: object, **kwargs: object) -> None:
        raise ImportError("report dependency missing") from cause

    monkeypatch.setattr("mltk.report.generator.generate_report", fail)
    session, reporter = _html_session(MltkReportCollector())
    session.config._mltk_collector.add("tests::test_x", "passed", 0.1)

    pytest_sessionfinish(session, 0)

    output = "\n".join(reporter._tw.lines)
    assert "jinja2" in output
    assert "pip install mlspec[report]" in output


def test_html_error_prints_warning_without_propagating(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mltk.pytest_plugin.plugin import pytest_sessionfinish

    def fail(*args: object, **kwargs: object) -> None:
        raise RuntimeError("disk full")

    monkeypatch.setattr("mltk.report.generator.generate_report", fail)
    session, reporter = _html_session(MltkReportCollector())
    session.config._mltk_collector.add("tests::test_x", "passed", 0.1)

    pytest_sessionfinish(session, 0)

    assert "HTML report failed: disk full" in "\n".join(reporter._tw.lines)
