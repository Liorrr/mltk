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
4. Run `pytest -q` and `ruff check src/ tests/`
5. Submit PR with description of changes

### PR Checklist

- [ ] All tests pass (`pytest -q`)
- [ ] Lint clean (`ruff check src/ tests/`)
- [ ] New functions have docstrings (Args, Returns, Example)
- [ ] New tests have scenario + WHY docstrings
- [ ] Doc page updated or created for new assertions
- [ ] CHANGELOG.md updated

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

By contributing, you agree that your contributions will be licensed under Apache-2.0.
