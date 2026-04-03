"""Tests for mltk.experiment.worktree -- git worktree sandbox primitive.

All tests mock ``subprocess.run`` so no real git repo is needed.
Covers:

1.  git_available() returns True when git works.
2.  git_available() returns False when git is missing.
3.  git_available() returns False on timeout.
4.  find_git_root() returns correct path from git output.
5.  find_git_root() raises FileNotFoundError when not a repo.
6.  find_git_root() raises RuntimeError when git is not installed.
7.  _run_git() constructs correct command args.
8.  _run_git() raises CalledProcessError on non-zero exit.
9.  _run_git() raises TimeoutExpired on timeout.
10. GitWorktree.__enter__ calls git worktree add with correct args.
11. GitWorktree.__exit__ calls git worktree remove and branch -D.
12. GitWorktree.__exit__ logs warning on cleanup failure (no raise).
13. GitWorktree.path raises RuntimeError before __enter__.
14. GitWorktree.branch returns the generated branch name.
15. run_in_worktree() runs command with cwd=worktree path.
16. run_in_worktree() merges env with os.environ.
17. write_file() creates file with correct content.
18. write_file() creates parent directories.
19. __enter__ cleans up tmpdir if git worktree add fails.
20. __exit__ removes directory even if git commands fail.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from mltk.experiment.worktree import (
    GitWorktree,
    _run_git,
    find_git_root,
    git_available,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ok_result(stdout: str = "") -> subprocess.CompletedProcess[str]:
    """Build a successful CompletedProcess."""
    return subprocess.CompletedProcess(
        args=["git"], returncode=0, stdout=stdout, stderr="",
    )


# ---------------------------------------------------------------------------
# git_available
# ---------------------------------------------------------------------------

def test_git_available_true() -> None:
    """PASS: git_available returns True when git --version succeeds.

    WHY: The happy path -- git is installed and responds.
    Expected: True.
    """
    with patch("mltk.experiment.worktree.subprocess.run") as mock:
        mock.return_value = _ok_result("git version 2.40.0")
        assert git_available() is True
        mock.assert_called_once()
        args = mock.call_args
        assert args[0][0] == ["git", "--version"]
        assert args[1]["shell"] is False


def test_git_available_false_file_not_found() -> None:
    """FAIL: git_available returns False when git binary is missing.

    WHY: On systems without git, subprocess.run raises
    FileNotFoundError.  git_available must catch it silently.
    Expected: False.
    """
    with patch("mltk.experiment.worktree.subprocess.run") as mock:
        mock.side_effect = FileNotFoundError("git not found")
        assert git_available() is False


def test_git_available_false_on_timeout() -> None:
    """FAIL: git_available returns False on subprocess timeout.

    WHY: A hung git process should not block forever.
    Expected: False.
    """
    with patch("mltk.experiment.worktree.subprocess.run") as mock:
        mock.side_effect = subprocess.TimeoutExpired(
            cmd=["git", "--version"], timeout=10.0,
        )
        assert git_available() is False


# ---------------------------------------------------------------------------
# find_git_root
# ---------------------------------------------------------------------------

def test_find_git_root_returns_path() -> None:
    """PASS: find_git_root returns the resolved root from git output.

    WHY: Core functionality -- must correctly parse git rev-parse
    output and return a resolved Path.
    Expected: Path matching the stdout.
    """
    with patch("mltk.experiment.worktree.subprocess.run") as mock:
        mock.return_value = _ok_result("/home/user/repo\n")
        root = find_git_root(Path("/home/user/repo/sub"))
        assert root == Path("/home/user/repo").resolve()


def test_find_git_root_not_a_repo() -> None:
    """FAIL: find_git_root raises FileNotFoundError outside a repo.

    WHY: CalledProcessError from git rev-parse means the directory
    is not inside a git repository.
    Expected: FileNotFoundError with descriptive message.
    """
    with patch("mltk.experiment.worktree.subprocess.run") as mock:
        mock.side_effect = subprocess.CalledProcessError(
            returncode=128, cmd=["git", "rev-parse"],
        )
        with pytest.raises(FileNotFoundError, match="Not a git"):
            find_git_root(Path("/tmp/not-a-repo"))


def test_find_git_root_git_not_installed() -> None:
    """FAIL: find_git_root raises RuntimeError when git is missing.

    WHY: FileNotFoundError from subprocess means the git binary
    itself is not found.  This should become a RuntimeError.
    Expected: RuntimeError mentioning git not installed.
    """
    with patch("mltk.experiment.worktree.subprocess.run") as mock:
        mock.side_effect = FileNotFoundError("git not found")
        with pytest.raises(RuntimeError, match="not installed"):
            find_git_root()


# ---------------------------------------------------------------------------
# _run_git
# ---------------------------------------------------------------------------

def test_run_git_constructs_correct_args() -> None:
    """PASS: _run_git prefixes args with 'git' and passes kwargs.

    WHY: The helper is the foundation for all git operations.
    It must construct the right command list and pass shell=False.
    Expected: subprocess.run called with ["git", "status", "-b"].
    """
    with patch("mltk.experiment.worktree.subprocess.run") as mock:
        mock.return_value = _ok_result()
        _run_git("status", "-b", cwd=Path("/repo"), timeout=15.0)
        mock.assert_called_once_with(
            ["git", "status", "-b"],
            capture_output=True,
            text=True,
            check=True,
            timeout=15.0,
            cwd=Path("/repo"),
            shell=False,
        )


def test_run_git_raises_on_nonzero_exit() -> None:
    """FAIL: _run_git raises CalledProcessError on non-zero exit.

    WHY: check=True means subprocess.run raises automatically.
    We verify _run_git does not swallow it.
    Expected: CalledProcessError propagates.
    """
    with patch("mltk.experiment.worktree.subprocess.run") as mock:
        mock.side_effect = subprocess.CalledProcessError(
            returncode=1, cmd=["git", "status"],
        )
        with pytest.raises(subprocess.CalledProcessError):
            _run_git("status")


def test_run_git_raises_on_timeout() -> None:
    """FAIL: _run_git raises TimeoutExpired when git hangs.

    WHY: Timeout enforcement is critical for sandbox safety.
    Expected: TimeoutExpired propagates.
    """
    with patch("mltk.experiment.worktree.subprocess.run") as mock:
        mock.side_effect = subprocess.TimeoutExpired(
            cmd=["git", "status"], timeout=30.0,
        )
        with pytest.raises(subprocess.TimeoutExpired):
            _run_git("status")


# ---------------------------------------------------------------------------
# GitWorktree.__enter__ / __exit__
# ---------------------------------------------------------------------------

@patch("mltk.experiment.worktree.tempfile.mkdtemp")
@patch("mltk.experiment.worktree.subprocess.run")
def test_enter_calls_worktree_add(
    mock_run: MagicMock,
    mock_mkdtemp: MagicMock,
) -> None:
    """PASS: __enter__ creates tmpdir and calls git worktree add.

    WHY: The enter method must create a temp directory, then invoke
    git worktree add with the correct branch and base ref args.
    Expected: mkdtemp called, git worktree add called with all args.
    """
    mock_mkdtemp.return_value = "/tmp/mltk-sandbox-abc"
    mock_run.return_value = _ok_result()

    wt = GitWorktree(
        repo_root=Path("/repo"),
        branch_name="test-branch",
        base_ref="main",
        timeout=20.0,
    )
    result = wt.__enter__()

    assert result is wt
    assert wt._worktree_path == Path("/tmp/mltk-sandbox-abc")
    mock_mkdtemp.assert_called_once_with(prefix="mltk-sandbox-")
    mock_run.assert_called_once_with(
        [
            "git", "worktree", "add",
            "/tmp/mltk-sandbox-abc",
            "-b", "test-branch", "main",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=20.0,
        cwd=Path("/repo").resolve(),
        shell=False,
    )


@patch("mltk.experiment.worktree.shutil.rmtree")
@patch("mltk.experiment.worktree.subprocess.run")
def test_exit_calls_worktree_remove_and_branch_delete(
    mock_run: MagicMock,
    mock_rmtree: MagicMock,
    tmp_path: Path,
) -> None:
    """PASS: __exit__ removes worktree, deletes branch, cleans dir.

    WHY: Proper cleanup prevents worktree and branch accumulation.
    The exit must call git worktree remove --force and git branch -D.
    Expected: Two git commands called in order.
    """
    mock_run.return_value = _ok_result()

    wt = GitWorktree(
        repo_root=Path("/repo"), branch_name="cleanup-br",
    )
    # Simulate having entered the context manager
    wt_dir = tmp_path / "worktree"
    wt_dir.mkdir()
    wt._worktree_path = wt_dir

    wt.__exit__(None, None, None)

    assert mock_run.call_count == 2
    calls = mock_run.call_args_list

    # First call: git worktree remove --force <path>
    assert calls[0] == call(
        [
            "git", "worktree", "remove", "--force",
            str(wt_dir),
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=30.0,
        cwd=Path("/repo").resolve(),
        shell=False,
    )

    # Second call: git branch -D <branch>
    assert calls[1] == call(
        ["git", "branch", "-D", "cleanup-br"],
        capture_output=True,
        text=True,
        check=True,
        timeout=30.0,
        cwd=Path("/repo").resolve(),
        shell=False,
    )

    assert wt._worktree_path is None


@patch("mltk.experiment.worktree.shutil.rmtree")
@patch("mltk.experiment.worktree.subprocess.run")
def test_exit_logs_warning_on_cleanup_failure(
    mock_run: MagicMock,
    mock_rmtree: MagicMock,
    tmp_path: Path,
) -> None:
    """SAFE: __exit__ logs warnings but never raises on cleanup errors.

    WHY: Cleanup failures must not mask the original exception.
    Each cleanup step is independently wrapped in try/except.
    Expected: No exception raised despite all git commands failing.
    """
    mock_run.side_effect = subprocess.CalledProcessError(
        returncode=1, cmd=["git"],
    )

    wt = GitWorktree(
        repo_root=Path("/repo"), branch_name="fail-br",
    )
    wt_dir = tmp_path / "worktree"
    wt_dir.mkdir()
    wt._worktree_path = wt_dir

    # Should not raise even though git commands fail
    wt.__exit__(None, None, None)

    # Both git commands were attempted
    assert mock_run.call_count == 2
    # rmtree fallback was called because wt_dir still exists
    mock_rmtree.assert_called_once_with(wt_dir, ignore_errors=True)


@patch("mltk.experiment.worktree.shutil.rmtree")
@patch("mltk.experiment.worktree.tempfile.mkdtemp")
@patch("mltk.experiment.worktree.subprocess.run")
def test_enter_cleans_tmpdir_on_git_failure(
    mock_run: MagicMock,
    mock_mkdtemp: MagicMock,
    mock_rmtree: MagicMock,
) -> None:
    """SAFE: __enter__ removes tmpdir if git worktree add fails.

    WHY: If the git command fails after creating the temp directory,
    the directory must be cleaned up to avoid leaking disk space.
    Expected: rmtree called on the tmpdir, original error re-raised.
    """
    mock_mkdtemp.return_value = "/tmp/mltk-sandbox-fail"
    mock_run.side_effect = subprocess.CalledProcessError(
        returncode=128, cmd=["git", "worktree", "add"],
    )

    wt = GitWorktree(repo_root=Path("/repo"))
    with pytest.raises(subprocess.CalledProcessError):
        wt.__enter__()

    mock_rmtree.assert_called_once_with(
        "/tmp/mltk-sandbox-fail", ignore_errors=True,
    )
    assert wt._worktree_path is None


# ---------------------------------------------------------------------------
# GitWorktree.path / .branch
# ---------------------------------------------------------------------------

def test_path_raises_before_enter() -> None:
    """FAIL: Accessing .path before __enter__ raises RuntimeError.

    WHY: The worktree directory does not exist until __enter__ is
    called.  Accessing it early must fail with a clear message.
    Expected: RuntimeError mentioning context manager.
    """
    wt = GitWorktree(repo_root=Path("/repo"))
    with pytest.raises(RuntimeError, match="not been entered"):
        _ = wt.path


def test_branch_returns_generated_name() -> None:
    """PASS: .branch returns the branch name (auto or explicit).

    WHY: Callers may need the branch name for logging or further
    git operations.  It should be available immediately.
    Expected: auto-generated name starts with 'mltk-sandbox-'.
    """
    wt = GitWorktree(repo_root=Path("/repo"))
    assert wt.branch.startswith("mltk-sandbox-")
    assert len(wt.branch) == len("mltk-sandbox-") + 12


def test_branch_returns_explicit_name() -> None:
    """PASS: .branch returns the explicit name when provided.

    WHY: Users should be able to specify their own branch name.
    Expected: exact match with the provided name.
    """
    wt = GitWorktree(
        repo_root=Path("/repo"), branch_name="my-branch",
    )
    assert wt.branch == "my-branch"


# ---------------------------------------------------------------------------
# run_in_worktree
# ---------------------------------------------------------------------------

@patch("mltk.experiment.worktree.subprocess.run")
def test_run_in_worktree_uses_worktree_cwd(
    mock_run: MagicMock,
    tmp_path: Path,
) -> None:
    """PASS: run_in_worktree sets cwd to the worktree path.

    WHY: Commands must execute inside the isolated worktree, not
    the main repo.  cwd= is the mechanism for this.
    Expected: subprocess.run called with cwd=worktree_path.
    """
    mock_run.return_value = _ok_result("output")

    wt = GitWorktree(repo_root=Path("/repo"))
    wt._worktree_path = tmp_path

    result = wt.run_in_worktree("python", "train.py", timeout=45.0)

    assert result.stdout == "output"
    mock_run.assert_called_once_with(
        ["python", "train.py"],
        capture_output=True,
        text=True,
        check=True,
        timeout=45.0,
        cwd=tmp_path,
        env=None,
        shell=False,
    )


@patch("mltk.experiment.worktree.subprocess.run")
@patch.dict("os.environ", {"EXISTING": "val"}, clear=True)
def test_run_in_worktree_merges_env(
    mock_run: MagicMock,
    tmp_path: Path,
) -> None:
    """PASS: run_in_worktree merges env with os.environ.

    WHY: Users may need to set experiment-specific env vars
    (e.g. CUDA_VISIBLE_DEVICES) without losing system variables.
    Expected: merged env contains both system and user entries.
    """
    mock_run.return_value = _ok_result()

    wt = GitWorktree(repo_root=Path("/repo"))
    wt._worktree_path = tmp_path

    wt.run_in_worktree(
        "echo", "hello",
        env={"MY_VAR": "123"},
    )

    called_env = mock_run.call_args[1]["env"]
    assert called_env["EXISTING"] == "val"
    assert called_env["MY_VAR"] == "123"


def test_run_in_worktree_raises_before_enter() -> None:
    """FAIL: run_in_worktree raises RuntimeError before __enter__.

    WHY: Without a worktree directory, there is no cwd to use.
    Expected: RuntimeError from .path property access.
    """
    wt = GitWorktree(repo_root=Path("/repo"))
    with pytest.raises(RuntimeError, match="not been entered"):
        wt.run_in_worktree("ls")


# ---------------------------------------------------------------------------
# write_file
# ---------------------------------------------------------------------------

def test_write_file_creates_content(tmp_path: Path) -> None:
    """PASS: write_file writes content and returns absolute path.

    WHY: The primary way to inject experiment code into the
    worktree.  Must write correct content and return the path.
    Expected: file exists with correct content.
    """
    wt = GitWorktree(repo_root=Path("/repo"))
    wt._worktree_path = tmp_path

    result_path = wt.write_file("config.yaml", "epochs: 10\n")

    assert result_path == tmp_path / "config.yaml"
    assert result_path.read_text(encoding="utf-8") == "epochs: 10\n"


def test_write_file_creates_parent_dirs(tmp_path: Path) -> None:
    """PASS: write_file creates intermediate directories.

    WHY: Experiment files may be nested (e.g. src/models/net.py).
    The method must create missing parent directories automatically.
    Expected: nested file written successfully.
    """
    wt = GitWorktree(repo_root=Path("/repo"))
    wt._worktree_path = tmp_path

    result_path = wt.write_file(
        "src/models/net.py", "import torch\n",
    )

    assert result_path == tmp_path / "src" / "models" / "net.py"
    assert result_path.exists()
    assert result_path.read_text(encoding="utf-8") == "import torch\n"


def test_write_file_raises_before_enter() -> None:
    """FAIL: write_file raises RuntimeError before __enter__.

    WHY: Without a worktree path, there is nowhere to write.
    Expected: RuntimeError from .path property access.
    """
    wt = GitWorktree(repo_root=Path("/repo"))
    with pytest.raises(RuntimeError, match="not been entered"):
        wt.write_file("foo.txt", "bar")
