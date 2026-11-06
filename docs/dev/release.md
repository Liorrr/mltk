# Release Process

See `CONTRIBUTING.md` in the repo root for the full workflow.

## Quick reference

| Task | Command |
|------|---------|
| Refresh doc counts (pre-commit does this automatically) | `python scripts/bump.py refresh` |
| Check for count drift (CI gate) | `python scripts/bump.py verify` |
| Preview a version bump | `python scripts/bump.py release --dry-run <version>` |
| Bump version + roll CHANGELOG | `python scripts/bump.py release <version>` |
| Full release make target | `make bump-release VERSION=<version>` |
| Print a version's GitHub Release notes | `python scripts/bump.py notes <version>` |

## What gets updated

`bump release` updates all of the following in one operation and stages the changed files:
- `pyproject.toml`, `rust/Cargo.toml` — `version = "X.Y.Z"`
- `src/mltk/__init__.py` — `__version__` fallback literal
- `charts/mltk/Chart.yaml` — `appVersion`
- `CHANGELOG.md` — rolls `[Unreleased]` → `[X.Y.Z] — YYYY-MM-DD` and opens a new, empty `[Unreleased]`
- `BACKLOG.md` — the version on the current DONE header
- Assertion/test/scanner/MCP/CLI counts in the files listed in `COUNT_TARGETS` (`scripts/bump.py`): `README.md`, `CLAUDE.md`, `BACKLOG.md`, `docs/index.md`, `docs/roadmap.md` and a set of API pages
- The current-version mentions and image tags in the files listed in `PROSE_VERSION_TARGETS`: `README.md`, `CLAUDE.md`, `docs/index.md`, `docs/guides/demo-script.md`, `docs/guides/container-deployment.md` and `charts/mltk/README.md`

**`Since: vX.Y.Z` markers in API docs are intentionally preserved** — they are historical feature-introduction markers, not current-version claims.

### Manual edits

`bump release` does not touch these:
- `rust/Cargo.lock` — set the `mltk-rust` package's `version` to the new version.
- `BACKLOG.md` — add the release entry, and replace "(unreleased)" on the sprint headings it ships.

## Before tagging

1. Open a release PR with the bump. Its CI must pass on a run newer than master's head.
2. Rehearse the Release workflow locally. The workflow runs only on a tag push and publishes every time, so there is no CI dry run. Build from a clean export of the release commit, so local build artifacts cannot end up in the wheel:

    ```bash
    git archive <release-commit> | tar -x -C /tmp/mltk-release
    cd /tmp/mltk-release
    maturin build --release --out dist
    maturin sdist --out dist
    twine check dist/*
    ```

    Then repeat the `test-wheels` job: install the wheel into a fresh Python 3.12 venv together with `pytest scipy` only, check `python -c "import mltk; print(mltk.__version__)"`, and run `python -m pytest tests/test_rust_bridge.py -q` from the clean tree. Install the sdist into a fresh venv on another supported Python version too; that builds the Rust extension from source, the way users on other versions and Homebrew install it.
3. Merge the release PR with a merge commit, and wait for master CI to pass on the merge commit.

## Publishing

Push an annotated tag on the merge commit:

```bash
git tag -a vX.Y.Z <merge-commit> -m "vX.Y.Z — <highlights>"
git push origin vX.Y.Z
```

- **Release** (`.github/workflows/release.yml`, on the tag): builds CPython 3.12 wheels for Linux x86_64 and arm64, macOS arm64 and Windows, plus the sdist; runs `test-wheels` on Linux, macOS and Windows; then publishes to TestPyPI and to PyPI (`mlspec`). Its `release-notes` job fails the run before anything is published if `CHANGELOG.md` has no section with entries for the tag's version. After PyPI, `github-release` creates the GitHub Release with the output of `bump.py notes` (a re-run updates the notes). If that job fails, re-run it: Docker publishing starts only when the whole workflow succeeds.
- **Publish Docker images** (`.github/workflows/docker-publish.yml`) starts when Release completes. It waits until the version is on PyPI, builds `runtime-slim` and `runtime-full` for amd64 and arm64, pushes `ghcr.io/liorrr/mltk:vX.Y.Z`, `:latest`, `:vX.Y.Z-full` and `:full`, and mirrors them to Docker Hub as `liorcohenil/mltk`. Its runs are listed under the default branch, not the tag. The Docker Hub mirror job is `continue-on-error`, so check the registry rather than the job status.

## After publishing

- Check that `https://pypi.org/pypi/mlspec/X.Y.Z/json` (and the same path on `test.pypi.org`) lists the four wheels and the sdist, that `gh release view vX.Y.Z` shows the notes, and that the GHCR and Docker Hub `vX.Y.Z` digests equal `latest` (`vX.Y.Z-full` equals `full`).
- Bump the Homebrew tap (`Liorrr/homebrew-mltk`): set the formula's `url` and `sha256` to the PyPI sdist, through a pull request. On a push to `main`, `brew test-bot` builds and tests nothing; on a PR it builds, audits and tests the formula. Read the `brew test` step itself: a failed or skipped step still leaves the check green.
