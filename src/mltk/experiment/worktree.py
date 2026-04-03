"""Git worktree management for sandboxed experiment execution.

Provides a context-manager that creates an isolated git worktree for
running experiments without affecting the main working tree.  This is
the lowest-level sandbox primitive -- higher-level modules compose it
with process isolation and resource limits.

Usage::

    from mltk.experiment.worktree import GitWorktree

    with GitWorktree(repo_root=Path(".")) as wt:
        wt.write_file("config.yaml", "epochs: 10")
        result = wt.run_in_worktree("python", "train.py")
        print(result.stdout)

Security model:
    All git and user commands use ``subprocess.run()`` with
    ``shell=False``.  No ``eval()`` or ``exec()`` is ever used.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path

__all__ = ["GitWorktree", "find_git_root", "git_available"]

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------

def git_available() -> bool:
    """Check whether the ``git`` CLI is available on PATH.

    Runs ``git --version`` in a subprocess.  Returns ``True`` if the
    command succeeds, ``False`` for any error (missing binary,
    permission denied, etc.).  Never raises.

    Returns:
        ``True`` if git is usable, ``False`` otherwise.
    """
    try:
        subprocess.run(
            ["git", "--version"],
            capture_output=True,
            text=True,
            timeout=10.0,
            shell=False,
        )
        return True
    except Exception:  # noqa: BLE001
        return False


def find_git_root(path: Path | None = None) -> Path:
    """Find the root directory of the git repository containing *path*.

    Runs ``git rev-parse --show-toplevel`` in *path* (or the current
    working directory if *path* is ``None``).

    Args:
        path: Directory to start the search from.  Defaults to
            the current working directory.

    Returns:
        Resolved :class:`~pathlib.Path` to the repository root.

    Raises:
        FileNotFoundError: If *path* is not inside a git repository.
        RuntimeError: If the ``git`` binary is not installed.
    """
    cwd = Path(path) if path is not None else None
    try:
        result = _run_git(
            "rev-parse", "--show-toplevel", cwd=cwd,
        )
        return Path(result.stdout.strip()).resolve()
    except FileNotFoundError:
        raise RuntimeError(
            "git is not installed or not on PATH"
        ) from None
    except subprocess.CalledProcessError:
        target = str(cwd) if cwd else "current directory"
        raise FileNotFoundError(
            f"Not a git repository: {target}"
        ) from None


def _run_git(
    *args: str,
    cwd: Path | None = None,
    timeout: float = 30.0,
) -> subprocess.CompletedProcess[str]:
    """Run a git command in a subprocess.

    Constructs ``["git", *args]`` and delegates to
    ``subprocess.run()`` with ``shell=False``.

    Args:
        *args: Git sub-command and arguments (e.g. ``"status"``,
            ``"-b"``).
        cwd: Working directory for the command.
        timeout: Maximum wall-clock seconds before killing.

    Returns:
        Completed process with captured stdout/stderr.

    Raises:
        subprocess.CalledProcessError: On non-zero exit code.
        subprocess.TimeoutExpired: If *timeout* is exceeded.
        FileNotFoundError: If the ``git`` binary is not found.
    """
    return subprocess.run(
        ["git", *args],
        capture_output=True,
        text=True,
        check=True,
        timeout=timeout,
        cwd=cwd,
        shell=False,
    )


# ------------------------------------------------------------------
# GitWorktree context manager
# ------------------------------------------------------------------

class GitWorktree:
    """Context manager for temporary git worktrees.

    Creates an isolated git worktree on ``__enter__`` and tears it
    down (worktree remove + branch delete + directory cleanup) on
    ``__exit__``.  Cleanup never raises -- failures are logged as
    warnings so that the original exception (if any) propagates
    cleanly.

    Args:
        repo_root: Path to the main repository.
        branch_name: Name for the worktree branch.  Auto-generated
            as ``mltk-sandbox-<hex>`` if omitted.
        base_ref: Git ref to base the worktree on (default ``HEAD``).
        timeout: Timeout in seconds for each git command.

    Example::

        with GitWorktree(Path("/repo"), base_ref="main") as wt:
            wt.write_file("experiment.py", code)
            result = wt.run_in_worktree("python", "experiment.py")
    """

    def __init__(
        self,
        repo_root: Path,
        branch_name: str | None = None,
        base_ref: str = "HEAD",
        timeout: float = 30.0,
    ) -> None:
        self._repo_root = repo_root.resolve()
        self._branch = (
            branch_name
            or f"mltk-sandbox-{uuid.uuid4().hex[:12]}"
        )
        self._base_ref = base_ref
        self._timeout = timeout
        self._worktree_path: Path | None = None

    # -- properties ---------------------------------------------------

    @property
    def path(self) -> Path:
        """Absolute path to the worktree directory.

        Raises:
            RuntimeError: If accessed before entering the context
                manager.
        """
        if self._worktree_path is None:
            raise RuntimeError(
                "GitWorktree has not been entered yet -- "
                "use it as a context manager"
            )
        return self._worktree_path

    @property
    def branch(self) -> str:
        """Name of the worktree branch."""
        return self._branch

    # -- context manager ----------------------------------------------

    def __enter__(self) -> GitWorktree:
        """Create the temporary worktree.

        1. Create a temp directory with ``tempfile.mkdtemp``.
        2. Run ``git worktree add <dir> -b <branch> <base_ref>``.
        3. Store the worktree path.

        Returns:
            *self* for use in ``with`` statements.
        """
        tmpdir = tempfile.mkdtemp(prefix="mltk-sandbox-")
        try:
            _run_git(
                "worktree", "add", tmpdir,
                "-b", self._branch, self._base_ref,
                cwd=self._repo_root,
                timeout=self._timeout,
            )
        except Exception:
            # Clean up the temp dir on failure
            shutil.rmtree(tmpdir, ignore_errors=True)
            raise
        self._worktree_path = Path(tmpdir)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Tear down the worktree, branch, and temp directory.

        Each cleanup step is wrapped in a try/except so that
        failures are logged as warnings and never mask the
        original exception.
        """
        wt_path = self._worktree_path
        if wt_path is None:
            return

        # Step 1: git worktree remove --force
        try:
            _run_git(
                "worktree", "remove", "--force", str(wt_path),
                cwd=self._repo_root,
                timeout=self._timeout,
            )
        except Exception:  # noqa: BLE001
            logger.warning(
                "Failed to remove worktree at %s", wt_path,
                exc_info=True,
            )

        # Step 2: git branch -D <branch>
        try:
            _run_git(
                "branch", "-D", self._branch,
                cwd=self._repo_root,
                timeout=self._timeout,
            )
        except Exception:  # noqa: BLE001
            logger.warning(
                "Failed to delete branch %s", self._branch,
                exc_info=True,
            )

        # Step 3: Fallback directory cleanup
        try:
            if wt_path.exists():
                shutil.rmtree(wt_path, ignore_errors=True)
        except Exception:  # noqa: BLE001
            logger.warning(
                "Failed to remove directory %s", wt_path,
                exc_info=True,
            )

        self._worktree_path = None

    # -- public helpers -----------------------------------------------

    def run_in_worktree(
        self,
        *cmd: str,
        timeout: float = 60.0,
        env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        """Run an arbitrary command inside the worktree.

        The command is executed with ``cwd`` set to the worktree
        directory.  If *env* is provided it is merged with
        ``os.environ`` (env entries override existing variables).

        Args:
            *cmd: Command and arguments (e.g. ``"python"``,
                ``"train.py"``).
            timeout: Maximum wall-clock seconds.
            env: Extra environment variables to set.

        Returns:
            Completed process with captured stdout/stderr.

        Raises:
            RuntimeError: If the context manager has not been
                entered.
            subprocess.CalledProcessError: On non-zero exit.
            subprocess.TimeoutExpired: If *timeout* is exceeded.
        """
        merged_env: dict[str, str] | None = None
        if env is not None:
            merged_env = {**os.environ, **env}
        return subprocess.run(
            list(cmd),
            capture_output=True,
            text=True,
            check=True,
            timeout=timeout,
            cwd=self.path,
            env=merged_env,
            shell=False,
        )

    def write_file(
        self,
        relative_path: str,
        content: str,
    ) -> Path:
        """Write a file inside the worktree.

        Creates parent directories as needed.

        Args:
            relative_path: Path relative to the worktree root
                (e.g. ``"src/config.yaml"``).
            content: File content to write.

        Returns:
            Absolute path to the written file.

        Raises:
            ValueError: If *relative_path* escapes the worktree
                (path traversal).
        """
        target = (self.path / relative_path).resolve()
        if not target.is_relative_to(self.path):
            raise ValueError(
                f"Path traversal detected: {relative_path!r} "
                f"escapes worktree {self.path}"
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return target
