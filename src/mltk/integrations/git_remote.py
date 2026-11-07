"""Helpers for selecting and safely displaying git remotes."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from urllib.parse import urlsplit

__all__ = ["find_push_remote", "parse_remote_url", "redact_remote_url"]


def _normalise_host(host: str) -> str:
    host = host.lower().rstrip(".")
    return "github.com" if host == "ssh.github.com" else host


def _parse_owner_repo(path: str) -> str | None:
    path = path.strip().strip("/")
    if path.lower().endswith(".git"):
        path = path[:-4].strip("/")
    parts = path.split("/")
    if len(parts) != 2 or not all(parts):
        return None
    return "/".join(parts).lower()


def parse_remote_url(url: str) -> tuple[str, str] | None:
    """Parse a git remote URL into its host and lower-case repository path.

    Args:
        url: A supported HTTP(S), SSH, git protocol, or SCP-style remote URL.

    Returns:
        A ``(host, "owner/repo")`` tuple, or ``None`` when the URL is not
        a supported repository URL.
    """
    value = url.strip()
    if not value:
        return None

    host: str
    path: str
    if re.match(r"^[a-z][a-z0-9+.-]*://", value, re.IGNORECASE):
        parsed = urlsplit(value)
        if not parsed.hostname or parsed.scheme.lower() not in {"http", "https", "ssh", "git"}:
            return None
        host = parsed.hostname
        path = parsed.path
    elif re.match(r"^git@", value, re.IGNORECASE):
        match = re.fullmatch(r"git@([^:/]+):(.+)", value, re.IGNORECASE)
        if not match:
            return None
        host, path = match.groups()
    else:
        match = re.fullmatch(r"([^/:]+):(\d+)(/.+)", value)
        if not match:
            return None
        host, _, path = match.groups()

    repo = _parse_owner_repo(path)
    if repo is None:
        return None
    return _normalise_host(host), repo


def redact_remote_url(url: str) -> str:
    """Remove userinfo from a scheme-style remote URL for safe error messages.

    Args:
        url: A remote URL that may contain credentials.

    Returns:
        The URL with any ``user[:password]@`` part removed.
    """
    return re.sub(
        r"^([a-z][a-z0-9+.-]*://)[^/@]+@", r"\1", url.strip(), flags=re.IGNORECASE,
    )


def find_push_remote(repo_root: Path, repo: str, host: str = "github.com") -> str:
    """Find a configured remote that points at the target repository.

    Args:
        repo_root: Root directory of the local git repository.
        repo: Target repository as ``owner/name``.
        host: Target host, defaulting to GitHub's public host.

    Returns:
        The matching remote name, preferring ``origin`` when applicable.

    Raises:
        RuntimeError: If no configured remote points at the target repository.
    """
    names_result = subprocess.run(
        ["git", "-C", str(repo_root), "remote"],
        check=True,
        capture_output=True,
        text=True,
    )
    target = (_normalise_host(host), repo.lower().strip("/"))
    found: list[tuple[str, str]] = []
    for name in names_result.stdout.splitlines():
        remote = name.strip()
        if not remote:
            continue
        url_result = subprocess.run(
            ["git", "-C", str(repo_root), "remote", "get-url", remote],
            check=True,
            capture_output=True,
            text=True,
        )
        url = url_result.stdout.strip().splitlines()[0] if url_result.stdout.strip() else ""
        found.append((remote, url))
        if parse_remote_url(url) == target:
            if remote == "origin":
                return remote

    for remote, url in found:
        if parse_remote_url(url) == target:
            return remote

    details = ", ".join(f"{name}={redact_remote_url(url)}" for name, url in found)
    raise RuntimeError(
        f"No git remote in {repo_root} points at {repo}; found: {details}"
    )
