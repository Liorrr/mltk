"""Helpers for selecting and safely displaying git remotes."""

from __future__ import annotations

import functools
import re
import subprocess
from collections.abc import Callable
from pathlib import Path
from urllib.parse import urlsplit

__all__ = ["find_push_remote", "parse_remote_url", "redact_remote_url"]

_SCHEME_RE = re.compile(r"^[a-z][a-z0-9+.-]*://", re.IGNORECASE)
# git's scp-like syntax: ``[user@]host:path``, with no slash before the colon.
_SCP_RE = re.compile(r"^(?:[^@/]+@)?([^@:/]+):(.+)$")


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


def _parse(url: str) -> tuple[str, str, bool] | None:
    """Return ``(host, owner/repo, uses_ssh)`` for a remote URL, or ``None``."""
    value = url.strip()
    if not value:
        return None

    if _SCHEME_RE.match(value):
        parsed = urlsplit(value)
        scheme = parsed.scheme.lower()
        if not parsed.hostname or scheme not in {"http", "https", "ssh", "git"}:
            return None
        host, path, uses_ssh = parsed.hostname, parsed.path, scheme == "ssh"
    else:
        match = _SCP_RE.fullmatch(value)
        if not match:
            return None
        host, path = match.groups()
        # Single-letter "hosts" are Windows drive letters (C:\repo): a local path.
        if len(host) == 1:
            return None
        uses_ssh = True

    repo = _parse_owner_repo(path)
    if repo is None:
        return None
    return _normalise_host(host), repo, uses_ssh


def parse_remote_url(url: str) -> tuple[str, str] | None:
    """Parse a git remote URL into its host and lower-case repository path.

    Args:
        url: An HTTP(S), SSH, git protocol, or scp-style (``[user@]host:path``)
            remote URL.

    Returns:
        A ``(host, "owner/repo")`` tuple, or ``None`` when the URL is not
        a supported repository URL. SSH host aliases are returned as written;
        :func:`find_push_remote` resolves them through the SSH configuration.
    """
    parsed = _parse(url)
    return None if parsed is None else parsed[:2]


@functools.lru_cache(maxsize=64)
def _ssh_config_hostname(alias: str) -> str:
    """Return the host an SSH alias connects to (``ssh -G``), or the alias itself."""
    try:
        result = subprocess.run(
            ["ssh", "-G", alias],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return alias
    if result.returncode != 0:
        return alias
    for line in result.stdout.splitlines():
        key, _, value = line.strip().partition(" ")
        if key.lower() == "hostname" and value.strip():
            return value.strip()
    return alias


def _points_at(
    url: str,
    target: tuple[str, str],
    resolve_ssh_host: Callable[[str], str],
) -> bool:
    parsed = _parse(url)
    if parsed is None:
        return False
    host, repo, uses_ssh = parsed
    if repo != target[1]:
        return False
    if host == target[0]:
        return True
    return uses_ssh and _normalise_host(resolve_ssh_host(host)) == target[0]


def redact_remote_url(url: str) -> str:
    """Remove userinfo from a scheme-style remote URL for safe error messages.

    Args:
        url: A remote URL that may contain credentials.

    Returns:
        The URL with any ``user[:password]@`` part removed, up to the last
        ``@`` before the path (a password may itself contain ``@``).
    """
    return re.sub(
        r"^([a-z][a-z0-9+.-]*://)[^/]*@", r"\1", url.strip(), flags=re.IGNORECASE,
    )


def _git_lines(repo_root: Path, *args: str) -> list[str]:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        stderr = getattr(exc, "stderr", "") or ""
        raise RuntimeError(
            f"Could not read git remotes in {repo_root}: {stderr.strip() or exc}"
        ) from exc
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def find_push_remote(
    repo_root: Path,
    repo: str,
    host: str = "github.com",
    *,
    resolve_ssh_host: Callable[[str], str] | None = None,
) -> str:
    """Find a configured remote whose push URLs all point at the target repository.

    The push URLs are the ones ``git push`` uses: ``pushurl`` when set, and
    ``insteadOf``/``pushInsteadOf`` rewrites applied. A remote with several
    push URLs matches only when every one of them points at *repo*.

    Args:
        repo_root: Root directory of the local git repository.
        repo: Target repository as ``owner/name``.
        host: Target host, defaulting to GitHub's public host.
        resolve_ssh_host: Maps an SSH host alias to the host it connects to.
            Defaults to reading the SSH configuration with ``ssh -G``.

    Returns:
        The matching remote name, preferring ``origin`` when applicable.

    Raises:
        RuntimeError: If git cannot be run, or no configured remote points
            at the target repository.
    """
    resolver = resolve_ssh_host or _ssh_config_hostname
    target = (_normalise_host(host), repo.lower().strip("/"))
    found: list[tuple[str, list[str]]] = []
    for remote in _git_lines(repo_root, "remote"):
        urls = _git_lines(repo_root, "remote", "get-url", "--push", "--all", remote)
        found.append((remote, urls))

    matching = [
        remote for remote, urls in found
        if urls and all(_points_at(url, target, resolver) for url in urls)
    ]
    if "origin" in matching:
        return "origin"
    if matching:
        return matching[0]

    details = ", ".join(
        f"{name}={' '.join(redact_remote_url(url) for url in urls)}" for name, urls in found
    )
    raise RuntimeError(
        f"No git remote in {repo_root} pushes to {repo}; found: {details}"
    )
