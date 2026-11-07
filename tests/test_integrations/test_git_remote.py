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
        ("ssh.github.com:443/Owner/Repo.git", ("github.com", "owner/repo")),
    ],
)
def test_parse_remote_url_forms(url: str, expected: tuple[str, str]) -> None:
    assert parse_remote_url(url) == expected


@pytest.mark.parametrize("url", ["", "github.com", "https://github.com/", "not a url"])
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
