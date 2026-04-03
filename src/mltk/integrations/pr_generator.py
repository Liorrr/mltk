"""Pull request generator — create GitHub PRs from scan findings + fixes.

Uses :class:`GitWorktree` to create an isolated branch with the fix applied,
pushes to remote, and creates a pull request via the GitHub REST API.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

if TYPE_CHECKING:
    from mltk.integrations.github_adapter import GitHubIssuesAdapter
    from mltk.scan.finding import FixSuggestion, ScanFinding

__all__ = ["PullRequestGenerator", "PullRequestResult", "render_pr_body"]


@dataclass
class PullRequestResult:
    """Result of a successfully created pull request.

    Attributes:
        url: The html_url of the created PR on GitHub.
        branch: The branch name that was pushed to remote.
        number: The PR number assigned by GitHub.
        draft: Whether the PR was created as a draft.
    """

    url: str
    branch: str
    number: int
    draft: bool


class PullRequestGenerator:
    """Create GitHub pull requests from scan findings and fix suggestions.

    Workflow:
    1. Generate an isolated branch name from the scanner name + random suffix.
    2. Create a :class:`~mltk.experiment.worktree.GitWorktree` for that branch.
    3. Write the fix code snippet into the worktree, commit, and push.
    4. Open a PR via the GitHub REST API.

    Args:
        github: A configured :class:`~mltk.integrations.github_adapter.GitHubIssuesAdapter`.
        repo_root: Path to the git repository root.  If ``None``, the root is
            located automatically from the current working directory.

    Example::

        from mltk.integrations.github_adapter import GitHubIssuesAdapter
        from mltk.integrations.pr_generator import PullRequestGenerator

        gh = GitHubIssuesAdapter("owner/repo", token="ghp_...")
        gen = PullRequestGenerator(gh)
        result = gen.create_pr(finding, fix)
        print(result.url)
    """

    def __init__(
        self,
        github: GitHubIssuesAdapter,
        repo_root: Path | None = None,
    ) -> None:
        self._github = github
        self._repo_root = repo_root  # resolved lazily inside create_pr

    def create_pr(
        self,
        finding: ScanFinding,
        fix: FixSuggestion,
        base_branch: str = "main",
        draft: bool = True,
        labels: list[str] | None = None,
        target_file: str = "_mltk_fix.py",
    ) -> PullRequestResult:
        """Create a pull request containing the fix for *finding*.

        Steps:
        1. Sanitise the scanner name and build a unique branch name.
        2. Create a temporary git worktree on that branch.
        3. Optionally write ``fix.code_snippet`` to *target_file*.
        4. Commit and push the branch inside the worktree context.
        5. Open a draft (or non-draft) PR via the GitHub API.

        Args:
            finding: The :class:`~mltk.scan.finding.ScanFinding` to fix.
            fix: The :class:`~mltk.scan.finding.FixSuggestion` to apply.
            base_branch: Target branch for the PR merge (default ``"main"``).
            draft: When ``True`` (default), the PR is created as a draft.
            labels: Optional list of label strings to attach to the PR.
            target_file: Relative path inside the worktree where
                ``fix.code_snippet`` is written (default ``"_mltk_fix.py"``).

        Returns:
            A :class:`PullRequestResult` with the PR URL, branch, number,
            and draft status.

        Raises:
            RuntimeError: If pushing the branch to remote fails.
        """
        if not fix.code_snippet:
            raise ValueError(
                "fix.code_snippet is empty — nothing to commit. "
                "Cannot create a PR without code changes."
            )

        from mltk.experiment.worktree import GitWorktree, find_git_root

        # Resolve repo root lazily
        repo_root: Path = self._repo_root if self._repo_root is not None else find_git_root()

        # Build a sanitised, unique branch name
        safe_scanner = re.sub(r"[^a-zA-Z0-9-]", "-", finding.scanner_name)
        branch = f"mltk/fix-{safe_scanner}-{uuid4().hex[:8]}"

        with GitWorktree(repo_root, branch_name=branch) as wt:
            # Write fix code
            wt.write_file(target_file, fix.code_snippet)

            # Stage only the fix file (not untracked worktree artifacts)
            wt.run_in_worktree("git", "add", "--", target_file)

            # Commit
            commit_msg = f"fix({finding.scanner_name}): {fix.title}"
            wt.run_in_worktree("git", "commit", "-m", commit_msg)

            # Push — must happen inside the context while the branch still exists
            try:
                wt.run_in_worktree("git", "push", "-u", "origin", branch)
            except subprocess.CalledProcessError as exc:
                raise RuntimeError(
                    f"Failed to push branch '{branch}' to remote: {exc.stderr.strip()}"
                ) from exc

        # Open the PR after the worktree is cleaned up (branch is already on remote)
        pr_data = self._github.create_pull_request(
            head=branch,
            base=base_branch,
            title=f"[mltk-autofix] {fix.title}",
            body=render_pr_body(finding, fix),
            draft=draft,
            labels=labels,
        )

        return PullRequestResult(
            url=pr_data.get("html_url", ""),
            branch=branch,
            number=int(pr_data.get("number", 0)),
            draft=bool(pr_data.get("draft", draft)),
        )


def render_pr_body(finding: ScanFinding, fix: FixSuggestion) -> str:
    """Render a structured Markdown body for a pull request.

    Produces sections for the finding details, fix description, and an
    optional fenced code block (omitted when ``fix.code_snippet`` is empty).

    Args:
        finding: The :class:`~mltk.scan.finding.ScanFinding` that triggered
            the fix.
        fix: The :class:`~mltk.scan.finding.FixSuggestion` being applied.

    Returns:
        Formatted Markdown string suitable for use as a PR body.
    """
    severity_label = finding.result.severity.value.upper()
    scanner = finding.scanner_name or "unknown"

    sections: list[str] = [
        "## Finding",
        f"- **Scanner:** {scanner}",
        f"- **Severity:** {severity_label}",
        f"- **Message:** {finding.result.message}",
        "",
        "## Fix",
        f"- **Title:** {fix.title}",
        f"- **Description:** {fix.description}",
        f"- **Confidence:** {fix.confidence}",
        f"- **Category:** {fix.category}",
    ]

    if fix.code_snippet:
        sections += [
            "",
            "## Code",
            "```python",
            fix.code_snippet,
            "```",
        ]

    sections += [
        "",
        "---",
        "*Auto-generated by mltk experiment runner. Review before merging.*",
    ]

    return "\n".join(sections)
