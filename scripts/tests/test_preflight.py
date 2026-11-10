"""Regression tests for pre-PR validation and stale CI evidence rejection."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import preflight  # noqa: E402


def ci_run(**changes):
    result = {
        "databaseId": 10, "headSha": "candidate", "headBranch": "feat/example", "event": "push",
        "workflowName": "CI", "status": "completed", "conclusion": "success",
        "url": "https://example.test/10",
    }
    result.update(changes)
    return result


def green_jobs():
    return [{"name": name, "status": "completed", "conclusion": "success"}
            for name in preflight.REQUIRED_JOBS]


@pytest.mark.parametrize("changes", [
    {"headSha": "old"}, {"headBranch": "other"}, {"event": "pull_request"},
    {"workflowName": "Release"}, {"status": "in_progress"}, {"conclusion": "failure"},
    {"conclusion": "cancelled"}, {"conclusion": "skipped"},
])
def test_unqualified_ci_cannot_open_gate(changes):
    """Wrong or incomplete CI evidence must never qualify a commit."""
    with pytest.raises(RuntimeError):
        preflight.select_run([ci_run(**changes)], "candidate", "feat/example")


def test_latest_failure_overrides_previous_success():
    with pytest.raises(RuntimeError, match="Latest branch CI"):
        preflight.select_run(
            [ci_run(), ci_run(databaseId=11, conclusion="failure")], "candidate", "feat/example",
        )


def test_exact_green_run_qualifies():
    assert preflight.select_run([ci_run()], "candidate", "feat/example")["databaseId"] == 10


def test_empty_run_list_fails():
    with pytest.raises(RuntimeError, match="No branch CI"):
        preflight.select_run([], "candidate", "feat/example")


def test_green_workflow_with_skipped_job_fails():
    jobs = green_jobs()
    jobs[0]["conclusion"] = "skipped"
    with pytest.raises(RuntimeError, match="not successful"):
        preflight.verify_jobs(jobs)


def test_missing_matrix_cell_fails():
    with pytest.raises(RuntimeError, match="missing"):
        preflight.verify_jobs(green_jobs()[1:])


def test_all_required_jobs_pass():
    preflight.verify_jobs(green_jobs())


def test_environment_overrides_do_not_hide_failures(monkeypatch, tmp_path):
    for name in ("PYTHONPATH", "PYTHONHOME", "PYTEST_ADDOPTS", "PYTEST_PLUGINS",
                 "PYTEST_DISABLE_PLUGIN_AUTOLOAD", "COVERAGE_PROCESS_START", "COVERAGE_RCFILE",
                 "PYTHONOPTIMIZE"):
        monkeypatch.setenv(name, "override")
    env = preflight.clean_env(tmp_path / "python")
    assert all(env.get(name) != "override" for name in (
        "PYTHONPATH", "PYTHONHOME", "PYTEST_ADDOPTS", "PYTEST_PLUGINS",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD", "COVERAGE_PROCESS_START",
        "COVERAGE_RCFILE", "PYTHONOPTIMIZE",
    ))
    assert env["PATH"].split(os.pathsep)[0] == str(tmp_path)


def test_console_entrypoint_and_failure_stop(monkeypatch, tmp_path):
    """A collection failure stops coverage/tests instead of being ignored."""
    executable = tmp_path / ("pytest.exe" if os.name == "nt" else "pytest")
    executable.touch()
    calls = []

    def run(args, **kwargs):
        calls.append(args)
        if "--collect-only" in args:
            raise subprocess.CalledProcessError(2, args)

    monkeypatch.setattr(preflight, "run", run)
    with pytest.raises(subprocess.CalledProcessError):
        preflight.job("test", tmp_path / "python")
    assert calls[-1] == [str(executable), "--collect-only", "-q"]
    assert not any("--cov=mltk" in args for args in calls)


def test_setup_python_script_location(monkeypatch, tmp_path):
    """Windows setup-python keeps console scripts beside, not with, python.exe."""
    scripts = tmp_path / "Scripts"
    monkeypatch.setattr(preflight.sysconfig, "get_path", lambda name: str(scripts))
    assert preflight.scripts_directory(Path(sys.executable)) == scripts


def test_local_isolates_minimal_checks_and_runs_lint_before_full_suite(monkeypatch):
    calls = []
    monkeypatch.setattr(preflight.sys, "version_info", (3, 12))
    monkeypatch.setattr(preflight.shutil, "which", lambda name: "cargo")
    monkeypatch.setattr(preflight, "run", lambda *a, **k: None)
    monkeypatch.setattr(preflight, "job", lambda name, python, **kw: calls.append(
        (name, python, kw.get("install", False)),
    ))
    preflight.local()
    assert [name for name, _, _ in calls] == [
        "lint", "claim-regression", "docs-freshness", "test", "rust",
    ]
    assert calls[1][1] == calls[2][1]
    assert len({calls[i][1] for i in (0, 1, 3)}) == 3
    assert calls[0][2] is True
    assert calls[1][2] is True
    assert calls[3][2] is True
    assert {name for name, _, _ in calls} == set(preflight.COMMANDS)


@pytest.mark.skipif(os.name != "nt", reason="Windows denies deletion of open files")
def test_cleanup_lock_does_not_turn_success_into_failure(monkeypatch, tmp_path):
    import tempfile

    handles = []
    real_temporary = tempfile.TemporaryDirectory
    monkeypatch.setattr(preflight.sys, "version_info", (3, 12))
    monkeypatch.setattr(preflight.shutil, "which", lambda name: "cargo")
    monkeypatch.setattr(preflight, "run", lambda *a, **k: None)
    monkeypatch.setattr(preflight.tempfile, "TemporaryDirectory", lambda **kwargs:
                        real_temporary(dir=tmp_path, **kwargs))

    def job(name, python, **kwargs):
        if name == "lint":
            python.parent.mkdir(parents=True)
            handles.append((python.parent / "locked.bin").open("wb"))

    monkeypatch.setattr(preflight, "job", job)
    try:
        assert preflight.main(["local"]) == 0
    finally:
        for handle in handles:
            handle.close()


def mock_checkout(monkeypatch, *, dirty="", pushed="candidate", ancestor=True):
    def capture(args):
        if args[:3] == ["git", "rev-parse", "--show-toplevel"]:
            return str(preflight.ROOT)
        if args[:2] == ["git", "status"]:
            return dirty
        if args[:2] == ["git", "symbolic-ref"]:
            return "feat/example"
        if args[:3] == ["git", "rev-parse", "HEAD"]:
            return "candidate"
        if args[:3] == ["git", "remote", "get-url"]:
            return "https://github.com/example/project.git"
        if args[:3] == ["gh", "repo", "view"]:
            return json.dumps({"nameWithOwner": "example/project"})
        if args[:2] == ["git", "ls-remote"]:
            return f"base\trefs/heads/master\n{pushed}\trefs/heads/feat/example"
        if args[:3] == ["gh", "run", "list"]:
            return json.dumps([ci_run()])
        if args[:3] == ["gh", "run", "view"]:
            return json.dumps({"headSha": "candidate", "jobs": green_jobs()})
        raise AssertionError(args)

    monkeypatch.setattr(preflight, "capture", capture)
    monkeypatch.setattr(preflight.subprocess, "run",
                        lambda *a, **k: SimpleNamespace(returncode=0 if ancestor else 1))
    return capture


@pytest.mark.parametrize(("changes", "message"), [
    ({"dirty": " M file.py"}, "dirty"),
    ({"pushed": "old"}, "Remote branch differs"),
    ({"ancestor": False}, "Incorporate current"),
])
def test_unready_checkout_fails(monkeypatch, changes, message):
    mock_checkout(monkeypatch, **changes)
    with pytest.raises(RuntimeError, match=message):
        preflight.ready()


def test_ready_prints_exact_evidence(monkeypatch, capsys):
    mock_checkout(monkeypatch)
    preflight.ready()
    output = capsys.readouterr().out
    assert "PR ready: candidate" in output
    assert "https://example.test/10" in output


def test_moving_base_invalidates_evidence(monkeypatch):
    original = mock_checkout(monkeypatch)
    calls = 0

    def capture(args):
        nonlocal calls
        result = original(args)
        if args[:2] == ["git", "ls-remote"]:
            calls += 1
            if calls == 2:
                return result.replace("base", "new-base")
        return result

    monkeypatch.setattr(preflight, "capture", capture)
    with pytest.raises(RuntimeError, match="Remote branch or master changed"):
        preflight.ready()


def test_cli_failure_is_nonzero(monkeypatch):
    def fail():
        raise RuntimeError("blocked")

    monkeypatch.setattr(preflight, "ready", fail)
    assert preflight.main(["ready"]) == 1


def test_open_never_creates_pr_when_gate_fails(monkeypatch, tmp_path):
    body = tmp_path / "body.md"
    body.write_text("Changes", encoding="utf-8")
    mock_checkout(monkeypatch, pushed="old")
    calls = []
    monkeypatch.setattr(preflight, "run", lambda *a, **k: calls.append(a))
    with pytest.raises(RuntimeError, match="Remote branch differs"):
        preflight.open_pr("Example", body)
    assert calls == []


def test_open_uses_verified_branch_and_repository(monkeypatch, tmp_path):
    body = tmp_path / "body.md"
    body.write_text("Changes", encoding="utf-8")
    mock_checkout(monkeypatch)
    calls = []
    bodies = []

    def run(args, **kwargs):
        calls.append(args)
        bodies.append(Path(args[-1]).read_text(encoding="utf-8"))

    monkeypatch.setattr(preflight, "run", run)
    preflight.open_pr("Example", body)
    assert len(calls) == 1
    assert calls[0][:-1] == [
        "gh", "pr", "create", "--repo", "example/project", "--base", "master",
        "--head", "feat/example", "--title", "Example", "--body-file",
    ]
    assert "Commit: `candidate`" in bodies[0]
    assert "https://example.test/10" in bodies[0]
    assert body.read_text(encoding="utf-8") == "Changes"


def test_job_install_refuses_global_environment(monkeypatch):
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    monkeypatch.setattr(preflight.sys, "prefix", preflight.sys.base_prefix)
    calls = []
    monkeypatch.setattr(preflight, "job", lambda *a, **k: calls.append(a))
    assert preflight.main(["job", "test", "--install"]) == 1
    assert calls == []


def test_ci_uses_shared_jobs_and_keeps_matrix():
    """Prevent workflow edits from silently dropping local/CI parity."""
    workflow = (preflight.ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    config = yaml.load(workflow, Loader=yaml.BaseLoader)
    for name in preflight.COMMANDS:
        steps = config["jobs"][name]["steps"]
        command = f"python scripts/preflight.py job {name}"
        if name != "rust":
            command += " --install"
        assert any(step.get("run") == command for step in steps)
    assert config["on"]["push"]["branches"] == ["**"]
    matrix = config["jobs"]["test"]["strategy"]["matrix"]
    assert matrix == {"os": ["ubuntu-latest", "macos-latest", "windows-latest"],
                      "python": ["3.10", "3.12", "3.13"]}
    expected = {name for name, job in config["jobs"].items() if "strategy" not in job}
    expected |= {f"test ({os_name}, {version})" for os_name in matrix["os"]
                 for version in matrix["python"]}
    expected |= {f"rust ({os_name})"
                 for os_name in config["jobs"]["rust"]["strategy"]["matrix"]["os"]}
    assert expected == preflight.REQUIRED_JOBS
    assert any("--cov-fail-under=60" in command for command in preflight.COMMANDS["test"])
    assert "pip-audit --strict" in workflow
    assert "cargo audit" in workflow
