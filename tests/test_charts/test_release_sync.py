"""Release tooling keeps the chart in step with the package version."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[2]
CHART_YAML = ROOT / "charts" / "mltk" / "Chart.yaml"


def _load(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def bump() -> ModuleType:
    return _load("_mltk_bump_under_test", ROOT / "scripts" / "bump.py")


@pytest.fixture
def smoke() -> ModuleType:
    return _load("_mltk_kind_smoke_under_test", ROOT / "charts" / "mltk" / "kind_smoke.py")


def test_bump_sets_chart_app_version_only(bump: ModuleType) -> None:
    text = CHART_YAML.read_text(encoding="utf-8")
    bumped = bump._bump_chart_app_version(text, "9.9.9")
    assert 'appVersion: "9.9.9"' in bumped
    # The chart's own release number is independent of the package version.
    chart_version = next(
        line for line in text.splitlines() if line.startswith("version:")
    )
    assert chart_version in bumped.splitlines()


def test_release_dry_run_touches_chart_and_prose_targets(
    bump: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Avoid the pytest --collect-only subprocess; counts are irrelevant here.
    monkeypatch.setattr(bump, "get_live_counts", lambda *a, **k: _zero_counts(bump))
    printed: list[str] = []
    monkeypatch.setattr(bump, "_print_utf8", printed.append)
    writes: list[Path] = []
    monkeypatch.setattr(Path, "write_text", lambda self, *a, **k: writes.append(self))

    assert bump.cmd_release("9.9.9", dry_run=True) == 0

    out = "".join(printed)
    assert "+++ b/charts/mltk/Chart.yaml" in out
    assert '+appVersion: "9.9.9"' in out
    # Prose targets outside COUNT_TARGETS were never walked before.
    assert "+++ b/charts/mltk/README.md" in out
    assert "+++ b/docs/guides/container-deployment.md" in out
    assert writes == []


def _zero_counts(bump: ModuleType) -> object:
    return bump.LiveCounts(
        assertions=0, cli=0, cli_summary="", mcp=0, scanners=0, tests=0,
        version="0.0.0",
    )


def test_release_keeps_backlog_header_bump_through_count_refresh(
    bump: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    # BACKLOG is both a header-bump and a count target. The count step used to
    # re-read the file from disk and overwrite the header bump.
    monkeypatch.setattr(bump, "get_live_counts", lambda *a, **k: _zero_counts(bump))
    monkeypatch.setattr(bump, "_git_add", lambda files: None)
    written: dict[Path, str] = {}
    monkeypatch.setattr(
        Path, "write_text", lambda self, data, *a, **k: written.__setitem__(self, data)
    )

    assert bump.cmd_release("9.9.9", dry_run=False) == 0

    backlog = written[ROOT / "BACKLOG.md"]
    header = next(line for line in backlog.splitlines() if line.startswith("## DONE (S0-S"))
    assert header.endswith("— v9.9.9")
    assert 'appVersion: "9.9.9"' in written[ROOT / "charts" / "mltk" / "Chart.yaml"]


def test_test_count_collects_against_this_checkout(
    bump: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    # In a git worktree the editable install points at another checkout; the
    # collection subprocess must import this checkout's src/ first.
    seen: dict[str, object] = {}

    class _Done:
        stdout = "7 tests collected in 0.01s"
        stderr = ""

    def _run(cmd: list[str], **kwargs: object) -> _Done:
        seen.update(kwargs)
        return _Done()

    monkeypatch.setattr(bump.subprocess, "run", _run)
    monkeypatch.setenv("PYTHONPATH", "elsewhere")

    assert bump.get_test_count() == 7
    env = seen["env"]
    assert isinstance(env, dict)
    parts = env["PYTHONPATH"].split(bump.os.pathsep)
    assert parts == [str(ROOT / "src"), "elsewhere"]


def test_kind_smoke_preloads_between_create_and_install(
    smoke: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[list[str]] = []
    monkeypatch.setenv("MLTK_KIND_SMOKE", "1")
    monkeypatch.setattr(smoke, "PRELOAD", "ghcr.io/liorrr/mltk:v9.9.9")
    monkeypatch.setattr(smoke.shutil, "which", lambda name: name)
    monkeypatch.setattr(
        smoke.subprocess, "run", lambda cmd, **kwargs: calls.append(cmd)
    )

    smoke.main()

    verbs = [" ".join(cmd[:2]) for cmd in calls]
    assert verbs == [
        "kind create",
        "kind load",
        "helm install",
        "kubectl wait",
        "kind delete",
    ]
    load = calls[1]
    assert load[2:] == ["docker-image", "ghcr.io/liorrr/mltk:v9.9.9", "--name", "mltk-smoke"]


def test_kind_smoke_skips_preload_when_unset(
    smoke: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[list[str]] = []
    monkeypatch.setenv("MLTK_KIND_SMOKE", "1")
    monkeypatch.setattr(smoke, "PRELOAD", "")
    monkeypatch.setattr(smoke.shutil, "which", lambda name: name)
    monkeypatch.setattr(
        smoke.subprocess, "run", lambda cmd, **kwargs: calls.append(cmd)
    )

    smoke.main()

    assert all(cmd[:2] != ["kind", "load"] for cmd in calls)
