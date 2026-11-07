"""Tests for generated-code environment and working-directory isolation."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from mltk.core.assertion import MltkAssertionError
from mltk.domains.codegen import assert_code_executes, assert_code_passes_tests


def test_secret_env_var_not_visible_to_child(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MLTK_TEST_SECRET_TOKEN", "hidden")

    result = assert_code_executes(
        "import os\nprint(os.environ.get('MLTK_TEST_SECRET_TOKEN'))"
    )

    assert result.passed
    assert result.details["stdout"].strip() == "None"


def test_allowlisted_env_and_stdlib_imports_still_work() -> None:
    result = assert_code_executes(
        "import json, math, os\n"
        "print(bool(os.environ.get('PATH')))\n"
        "print(json.dumps({'value': math.sqrt(16)}))"
    )

    assert result.passed
    assert result.details["stdout"].splitlines() == ["True", '{"value": 4.0}']


def test_env_kwarg_adds_variables() -> None:
    result = assert_code_executes(
        "import os\nprint(os.environ['MLTK_TEST_EXTRA'])",
        env={"MLTK_TEST_EXTRA": "provided"},
    )

    assert result.passed
    assert result.details["stdout"].strip() == "provided"


def test_inherit_env_passes_full_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MLTK_TEST_SECRET_TOKEN", "visible")

    result = assert_code_executes(
        "import os\nprint(os.environ.get('MLTK_TEST_SECRET_TOKEN'))",
        inherit_env=True,
    )

    assert result.passed
    assert result.details["stdout"].strip() == "visible"


def test_child_runs_in_fresh_temp_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    result = assert_code_executes(
        "import os\n"
        "print(os.getcwd())\n"
        "open('relative-output.txt', 'w', encoding='utf-8').close()"
    )

    assert result.passed
    child_cwd = Path(result.details["stdout"].strip())
    assert child_cwd != tmp_path
    assert child_cwd.name.startswith("mltk_codegen_")
    assert not (tmp_path / "relative-output.txt").exists()


def test_temp_dir_removed_after_run() -> None:
    result = assert_code_executes("import os\nprint(os.getcwd())")

    assert result.passed
    assert not os.path.exists(result.details["stdout"].strip())


def test_timeout_still_reported() -> None:
    with pytest.raises(MltkAssertionError) as exc_info:
        assert_code_executes("while True: pass", timeout_seconds=0.2)

    assert "timed out" in str(exc_info.value).lower()


def test_assert_code_passes_tests_uses_same_isolation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MLTK_TEST_SECRET_TOKEN", "hidden")
    code = "import os\nvalue = os.environ.get('MLTK_TEST_SECRET_TOKEN')"
    tests = (
        "assert value is None\n"
        "assert os.path.basename(os.getcwd()).startswith('mltk_codegen_')"
    )

    result = assert_code_passes_tests(code, tests)

    assert result.passed


def test_user_site_variables_reach_the_child(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    monkeypatch.setenv("PYTHONUSERBASE", str(tmp_path / "userbase"))
    monkeypatch.setenv("MLTK_TEST_SECRET_TOKEN", "hidden")

    result = assert_code_executes(
        "import os\n"
        "print(os.environ.get('PYTHONUSERBASE'))\n"
        "print(os.environ.get('MLTK_TEST_SECRET_TOKEN'))"
    )

    assert result.passed
    assert result.details["stdout"].splitlines() == [str(tmp_path / "userbase"), "None"]


def test_extra_replaces_differently_cased_key_when_names_are_case_insensitive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mltk.domains.codegen import _build_child_env

    monkeypatch.setenv("PATH", "inherited")
    child = _build_child_env(env={"Path": "extra"}, inherit_env=False, case_insensitive=True)
    assert [k for k in child if k.upper() == "PATH"] == ["Path"]
    assert child["Path"] == "extra"


def test_extra_keeps_differently_cased_key_when_names_are_case_sensitive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mltk.domains.codegen import _build_child_env

    monkeypatch.setenv("PATH", "inherited")
    child = _build_child_env(env={"Path": "extra"}, inherit_env=False, case_insensitive=False)
    assert child["Path"] == "extra"
    assert child["PATH"] == "inherited"
