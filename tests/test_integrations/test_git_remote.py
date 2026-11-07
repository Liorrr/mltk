"""Tests for Git remote URL parsing and target selection."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from mltk.integrations.git_remote import (
    find_push_remote,
    parse_remote_url,
    redact_remote_url,
)

requires_git = pytest.mark.skipif(shutil.which("git") is None, reason="git is not on PATH")


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://github.com/Owner/Repo.git/", ("github.com", "owner/repo")),
        ("HTTP://GitHub.com/Owner/Repo", ("github.com", "owner/repo")),
        ("ssh://git@github.com:443/Owner/Repo.git", ("github.com", "owner/repo")),
        ("git@github.com:Owner/Repo.git", ("github.com", "owner/repo")),
        ("GIT@GitHub.com:Owner/Repo.GIT", ("github.com", "owner/repo")),
        ("git://GitHub.com/Owner/Repo/", ("github.com", "owner/repo")),
        ("https://user:token@github.com/Owner/Repo.git", ("github.com", "owner/repo")),
        ("github.com:Owner/Repo.git", ("github.com", "owner/repo")),
        ("deploy@github.com:Owner/Repo", ("github.com", "owner/repo")),
        ("git@github-work:Owner/Repo.git", ("github-work", "owner/repo")),
        ("ssh://git@ssh.github.com:443/Owner/Repo.git", ("github.com", "owner/repo")),
    ],
)
def test_parse_remote_url_forms(url: str, expected: tuple[str, str]) -> None:
    assert parse_remote_url(url) == expected


@pytest.mark.parametrize("url", [
    "", "github.com", "https://github.com/", "not a url",
    "ssh.github.com:443/Owner/Repo.git", r"C:\\repos\\owner\\repo", "C:/repos/repo",
])
def test_parse_remote_url_rejects_garbage(url: str) -> None:
    assert parse_remote_url(url) is None


def test_redact_remote_url_strips_token() -> None:
    url = "https://user:tok@github.com/me/repo.git"
    assert redact_remote_url(url) == "https://github.com/me/repo.git"


def _git(*args: str, cwd: Path) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


@requires_git
def test_find_push_remote_picks_matching_remote(tmp_path: Path) -> None:
    _git("init", cwd=tmp_path)
    _git("remote", "add", "origin", "https://github.com/me/fork.git", cwd=tmp_path)
    _git("remote", "add", "upstream", "git@github.com:Owner/Repo.git", cwd=tmp_path)

    assert find_push_remote(tmp_path, "owner/repo") == "upstream"


@requires_git
def test_find_push_remote_prefers_origin_when_several_match(tmp_path: Path) -> None:
    _git("init", cwd=tmp_path)
    _git("remote", "add", "upstream", "git@github.com:Owner/Repo.git", cwd=tmp_path)
    _git("remote", "add", "origin", "https://github.com/owner/repo.git", cwd=tmp_path)

    assert find_push_remote(tmp_path, "owner/repo") == "origin"


@requires_git
def test_find_push_remote_no_match_raises_with_redacted_urls(tmp_path: Path) -> None:
    _git("init", cwd=tmp_path)
    _git("remote", "add", "origin", "https://user:tok@github.com/other/repo.git", cwd=tmp_path)

    with pytest.raises(RuntimeError, match=r"No git remote") as exc_info:
        find_push_remote(tmp_path, "owner/repo")

    assert "tok" not in str(exc_info.value)
    assert "origin=https://github.com/other/repo.git" in str(exc_info.value)


_TARGET = "https://github.com/owner/repo.git"
_OTHER = "https://github.com/someone/other.git"
_SSH_G_OUTPUT = "user git\nhostname github.com\nport 22\n"


def test_redact_remote_url_strips_password_containing_at() -> None:
    url = "https://user:p@ss@github.com/me/repo.git"
    assert redact_remote_url(url) == "https://github.com/me/repo.git"


@requires_git
def test_find_push_remote_uses_the_push_url(tmp_path: Path) -> None:
    _git("init", cwd=tmp_path)
    _git("remote", "add", "origin", "https://github.com/owner/repo.git", cwd=tmp_path)
    _git("remote", "set-url", "--push", "origin", _OTHER, cwd=tmp_path)

    with pytest.raises(RuntimeError, match=r"No git remote"):
        find_push_remote(tmp_path, "owner/repo")


@requires_git
def test_find_push_remote_matches_on_push_url_when_fetch_differs(tmp_path: Path) -> None:
    _git("init", cwd=tmp_path)
    _git("remote", "add", "origin", "https://github.com/someone/mirror.git", cwd=tmp_path)
    _git("remote", "set-url", "--push", "origin", "https://github.com/owner/repo.git", cwd=tmp_path)

    assert find_push_remote(tmp_path, "owner/repo") == "origin"


@requires_git
def test_find_push_remote_needs_every_push_url_to_match(tmp_path: Path) -> None:
    _git("init", cwd=tmp_path)
    _git("remote", "add", "origin", "https://github.com/owner/repo.git", cwd=tmp_path)
    _git("remote", "set-url", "--add", "--push", "origin", _TARGET, cwd=tmp_path)
    _git("remote", "set-url", "--add", "--push", "origin", _OTHER, cwd=tmp_path)

    with pytest.raises(RuntimeError, match=r"No git remote"):
        find_push_remote(tmp_path, "owner/repo")


@requires_git
def test_find_push_remote_resolves_ssh_host_alias(tmp_path: Path) -> None:
    _git("init", cwd=tmp_path)
    _git("remote", "add", "origin", "git@github-work:Owner/Repo.git", cwd=tmp_path)

    aliases = {"github-work": "github.com"}
    assert find_push_remote(
        tmp_path, "owner/repo", resolve_ssh_host=lambda h: aliases.get(h, h),
    ) == "origin"


@requires_git
def test_find_push_remote_refuses_alias_to_another_host(tmp_path: Path) -> None:
    _git("init", cwd=tmp_path)
    _git("remote", "add", "origin", "git@work:Owner/Repo.git", cwd=tmp_path)

    with pytest.raises(RuntimeError, match=r"No git remote"):
        find_push_remote(tmp_path, "owner/repo", resolve_ssh_host=lambda h: "gitlab.example")


@requires_git
def test_https_host_is_not_resolved_through_ssh_config(tmp_path: Path) -> None:
    _git("init", cwd=tmp_path)
    _git("remote", "add", "origin", "https://work/Owner/Repo.git", cwd=tmp_path)

    with pytest.raises(RuntimeError, match=r"No git remote"):
        find_push_remote(tmp_path, "owner/repo", resolve_ssh_host=lambda h: "github.com")


@requires_git
def test_find_push_remote_reports_git_errors_as_runtime_error(tmp_path: Path) -> None:
    not_a_repo = tmp_path / "plain"
    not_a_repo.mkdir()
    with pytest.raises(RuntimeError, match=r"Could not read git remotes"):
        find_push_remote(not_a_repo, "owner/repo")


def test_ssh_config_hostname_reads_ssh_g(monkeypatch: pytest.MonkeyPatch) -> None:
    from mltk.integrations import git_remote

    git_remote._ssh_config_hostname.cache_clear()
    monkeypatch.setattr(
        git_remote.subprocess, "run",
        lambda *a, **k: subprocess.CompletedProcess(a, 0, _SSH_G_OUTPUT, ""),
    )
    assert git_remote._ssh_config_hostname("github-work") == "github.com"
    git_remote._ssh_config_hostname.cache_clear()


def test_ssh_config_hostname_falls_back_to_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    from mltk.integrations import git_remote

    git_remote._ssh_config_hostname.cache_clear()

    def no_ssh(*args, **kwargs):
        raise FileNotFoundError("ssh")

    monkeypatch.setattr(git_remote.subprocess, "run", no_ssh)
    assert git_remote._ssh_config_hostname("github-work") == "github-work"
    git_remote._ssh_config_hostname.cache_clear()
