"""Regression tests for report generation without the optional server dependency."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

import mltk


def _child_env() -> dict[str, str]:
    """Use this checkout's source tree in a fresh interpreter."""
    source_root = str(Path(mltk.__file__).resolve().parents[1])
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [source_root, env.get("PYTHONPATH", "")],
    )
    return env


class TestReportWithoutFastapi:
    """Verify report imports and the pytest option do not require fastapi."""

    def test_imports_succeed_without_fastapi_and_app_stays_lazy(self, tmp_path: Path) -> None:
        """IMPORTS: Report, compliance, and server submodules import without fastapi.

        WHY: The HTML report is an independent optional feature and must not silently
        disappear merely because the server extra is not installed.
        """
        code = (
            "import sys; "
            "sys.modules['fastapi'] = None; "
            "import mltk.report; "
            "import mltk.compliance; "
            "from mltk.server import metrics; "
            "import mltk.server; "
            "print('mltk.server.app' in sys.modules); "
            "assert 'mltk.server.app' not in sys.modules"
        )
        completed = subprocess.run(
            [sys.executable, "-c", code],
            cwd=tmp_path,
            env=_child_env(),
            capture_output=True,
            text=True,
            check=False,
        )

        assert completed.returncode == 0, completed.stderr
        assert completed.stdout.strip().endswith("False")

    def test_pytest_report_option_writes_html_without_fastapi(self, tmp_path: Path) -> None:
        """CLI: The pytest report option writes HTML in a fastapi-blocked process.

        WHY: This is the user-facing failure mode caused by eager server imports;
        a passing test run must still leave a report artifact behind.
        """
        pytest.importorskip("jinja2")
        test_file = tmp_path / "test_tiny.py"
        test_file.write_text("def test_tiny():\n    assert True\n", encoding="utf-8")
        code = (
            "import sys; "
            "sys.modules['fastapi'] = None; "
            "import pytest; "
            f"sys.exit(pytest.main([{str(test_file)!r}, '-q', '-p', 'no:cacheprovider', "
            "'--mltk-report']))"
        )

        completed = subprocess.run(
            [sys.executable, "-c", code],
            cwd=tmp_path,
            env=_child_env(),
            capture_output=True,
            text=True,
            check=False,
        )

        assert completed.returncode == 0, completed.stdout + completed.stderr
        assert list((tmp_path / "mltk-reports").glob("report-*.html"))


class TestServerLazyAttribute:
    """Verify the optional application remains available when fastapi is installed."""

    def test_create_app_is_callable_and_unknown_attribute_fails(self) -> None:
        """API: create_app remains lazily importable and unknown names fail normally.

        WHY: PEP 562 must preserve the public import while avoiding accidental imports
        of server.app for unrelated server submodules.
        """
        pytest.importorskip("fastapi")
        code = (
            "import sys; "
            "import mltk.server; "
            "assert 'mltk.server.app' not in sys.modules, 'eager app import'; "
            "from mltk.server import create_app; "
            "assert callable(create_app); "
            "assert 'mltk.server.app' in sys.modules; "
            "assert not hasattr(mltk.server, 'nope')"
        )
        completed = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            env=_child_env(),
            text=True,
        )

        assert completed.returncode == 0, completed.stdout + completed.stderr
