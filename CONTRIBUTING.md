# Contributing to mltk

Thank you for your interest in contributing to mltk! This guide will help you get started.

## Development Setup

```bash
# Clone the repository
git clone https://github.com/Liorrr/mltk.git
cd mltk

# Install in development mode with all dev dependencies
pip install -e ".[dev,scipy,sklearn,cli,report]"

# Install Rust toolchain (for optional acceleration)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Verify setup
pytest -q              # Run tests
ruff check src/ tests/ # Run linter
cargo test             # Run Rust tests (in rust/)
```

## Code Style

- **Python**: Follow PEP 8, enforced by `ruff`
- **Line length**: 100 characters
- **Type hints**: Required on all function signatures
- **Docstrings**: Required on all public functions with Args, Returns, Example sections
- **Imports**: Sorted by `ruff` (isort-compatible)

```bash
# Format code
ruff format src/ tests/

# Check linting
ruff check src/ tests/

# Auto-fix lint issues
ruff check --fix src/ tests/
```

## Writing Assertions

Every assertion function follows this pattern:

```python
from mltk.core.assertion import assert_true, timed_assertion
from mltk.core.result import Severity, TestResult

@timed_assertion
def assert_my_check(data, threshold=0.5) -> TestResult:
    """One-line description.

    Args:
        data: What to validate.
        threshold: Minimum required value.

    Returns:
        TestResult with check details.

    Example:
        >>> assert_my_check([1, 2, 3], threshold=0.5)
    """
    value = compute_something(data)
    passed = value >= threshold

    return assert_true(
        passed,
        name="module.my_check",
        message=f"Check: {value:.4f} {'>='>= if passed else '<'} {threshold}",
        severity=Severity.CRITICAL,
        value=value,
        threshold=threshold,
    )
```

Key rules:
- Use `@timed_assertion` decorator (adds `duration_ms`)
- Return `TestResult` via `assert_true()`
- CRITICAL severity raises `MltkAssertionError` on failure
- Include meaningful details in `**kwargs`

## Writing Tests

Every test follows this pattern:

```python
def test_check_passes(self) -> None:
    """PASS: Describe the passing scenario.

    WHY: Explain why this test matters in ML context.
    Expected: What the assertion should return.
    """
    result = assert_my_check(good_data, threshold=0.5)
    assert result.passed is True

def test_check_fails(self) -> None:
    """FAIL: Describe the failing scenario.

    WHY: Explain what ML bug this catches.
    Expected: MltkAssertionError raised.
    """
    with pytest.raises(MltkAssertionError):
        assert_my_check(bad_data, threshold=0.9)
```

## Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Write code + tests + docs
4. Run `python scripts/preflight.py local` with Python 3.12 (or CI's 3.10/3.13).
   This creates fresh lint, minimal, and full virtual environments, installs the same
   dependencies as CI, verifies the imported checkout, then runs lint, the slim
   regression, documentation freshness, console-pytest collection and coverage,
   scripts tests, and Rust format/clippy/tests. Rust must be installed. Checks
   stop on failure; nothing commits, stages, pushes, or opens a PR automatically.
5. Commit the final changes and push the feature branch. CI runs on branch pushes
   before a PR exists, including the complete OS/Python matrix and security jobs.
6. Wait for branch CI, then run `python scripts/preflight.py ready`. It requires
   a clean checkout, HEAD equal to the pushed branch, current remote master in
   its ancestry, and the latest CI run for that exact commit to have succeeded.
   If master has advanced, incorporate it and repeat validation. Any new commit
   needs fresh CI evidence. Run this check immediately before opening the PR.
7. Write a PR body file outside the checkout, then run
   `python scripts/preflight.py open --title "Your title"
   --body-file ../pr-body.md` (on one line). This reruns the gate and creates the
   PR only after it passes, using the verified origin repository and branch.
   It appends the verified commit, CI URL, and master revision automatically.
   PR CI still checks the merge result. Local success is not proof of other OSes,
   dependency-audit success, or future dependency resolution.

Steps 6 and 7 apply to branches pushed to this repository. When `origin` is a
fork, `ready` and `open` stop and say so, because the fork's master is not the
PR base. From a fork, run step 4, push to your fork, and open the PR against this
repository with `gh pr create` or the GitHub web page; PR CI checks the merge result.

Before running `ready` or `open`, install the GitHub CLI (`gh`) and authenticate
with `gh auth login` (or an appropriately scoped token). `ready` needs read
access to the origin repository and its Actions runs, plus Git access to origin.
`open` additionally needs permission to create pull requests in that repository.
The local checks do not require GitHub CLI access.

The `ready` command is a contributor process gate; it cannot prevent someone
from opening a PR directly through GitHub. Keep branch protection and required
PR checks enabled. The helper targets `origin` and stops when `origin` is a
fork (see above); fork PRs rely on this repository's PR checks instead.

### PR Checklist

- [ ] Local preflight passed; `ready` verified the final pushed commit (from a
      fork, `ready` does not apply and PR CI is the check).
- [ ] New functions have docstrings (Args, Returns, Example)
- [ ] New tests have scenario + WHY docstrings
- [ ] Doc page updated or created for new assertions
- [ ] CHANGELOG.md updated

Branch and PR runs are intentionally separate: the first checks the pushed
commit before PR creation, the second checks the merge result. This uses more CI
capacity on open PR updates; superseded runs for the same event/ref are cancelled.
The local `job --install` interface is intended for CI or an activated virtualenv;
use `local` for isolated developer checks. This workflow targets master-based PRs.

## Architecture Overview

```
src/mltk/
  core/       Base types (TestResult, MltkConfig, assertions)
  data/       Data quality (schema, drift, PII, labels, freshness)
  model/      Model quality (metrics, bias, regression, slicing)
  inference/  Performance (latency, throughput, contracts)
  pipeline/   Pipeline (reproducibility, E2E, checksum)
  monitor/    Production (degradation, SLA)
  domains/    Domain kits (cv/, nlp/, speech/, tabular/)
  report/     HTML report generation
  pytest_plugin/  pytest integration
  cli/        Typer CLI commands
  _rust.py    Rust acceleration bridge

rust/         PyO3 crate (KS test, PSI)
tests/        pytest test suite (204 tests)
docs/         MkDocs documentation
```

## Release Process

| Command | Purpose |
|---|---|
| `python scripts/bump.py refresh` | Update all count references from source (auto-run by pre-commit) |
| `python scripts/bump.py verify` | Check for drift, exit non-zero if found (CI gate) |
| `python scripts/bump.py release --dry-run X.Y.Z` | Preview all changes a release would make |
| `python scripts/bump.py release X.Y.Z` | Bump version, roll CHANGELOG, refresh counts |

### Pre-commit hook

Install once:
```bash
pip install pre-commit
pre-commit install
```

On every commit, the hook runs `bump.py refresh` and auto-stages any count corrections so they land in your commit.

### Shipping a release

1. Curate `CHANGELOG.md` `[Unreleased]` section manually.
2. `python scripts/bump.py release --dry-run X.Y.Z` — review diff.
3. `python scripts/bump.py release X.Y.Z` — writes files and git-adds them.
4. `git commit -m "chore: release vX.Y.Z"` then `git tag vX.Y.Z`.

## License

By contributing, you agree that your contributions will be licensed under the
[Elastic License 2.0](LICENSE) and that the licensor (Lior Cohen) retains the
right to offer commercial licenses for the software.
