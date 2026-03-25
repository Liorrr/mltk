"""mltk environment diagnostics — run `mltk doctor` to check your setup."""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import sys
from dataclasses import dataclass


@dataclass
class DiagnosticResult:
    """Result of a single diagnostic check."""

    name: str
    status: str  # "OK", "WARN", "FAIL"
    message: str
    fix_hint: str | None = None


def _check_python_version() -> DiagnosticResult:
    """Python version must be >= 3.10."""
    vi = sys.version_info
    version_str = f"{vi.major}.{vi.minor}.{vi.micro}"
    if vi >= (3, 10):
        return DiagnosticResult(
            name="Python version",
            status="OK",
            message=f"Python {version_str} (>=3.10 required)",
        )
    return DiagnosticResult(
        name="Python version",
        status="FAIL",
        message=f"Python {version_str} is below the required 3.10",
        fix_hint="Upgrade to Python 3.10 or later: https://python.org/downloads",
    )


def _check_core_deps() -> list[DiagnosticResult]:
    """numpy and pandas are required core dependencies."""
    results = []
    for dep in ("numpy", "pandas"):
        try:
            importlib.import_module(dep)
            results.append(
                DiagnosticResult(
                    name=f"Core dep: {dep}",
                    status="OK",
                    message=f"{dep} is installed",
                )
            )
        except ImportError:
            results.append(
                DiagnosticResult(
                    name=f"Core dep: {dep}",
                    status="FAIL",
                    message=f"{dep} is missing (required for mltk to function)",
                    fix_hint=f"pip install {dep}",
                )
            )
    return results


def _check_optional_deps() -> list[DiagnosticResult]:
    """Optional extras — WARN if missing, not FAIL."""
    optional: list[tuple[str, str]] = [
        ("scipy", "mltk[scipy]"),
        ("sklearn", "mltk[sklearn]"),
        ("typer", "mltk[cli]"),
        ("rich", "mltk[cli]"),
        ("plotly", "mltk[report]"),
        ("jinja2", "mltk[report]"),
        ("yaml", "mltk[yaml]"),
        ("nltk", "mltk[nlp]"),
        ("jiwer", "mltk[speech]"),
        ("cv2", "mltk[cv]"),
    ]
    results = []
    for module, extra in optional:
        try:
            importlib.import_module(module)
            results.append(
                DiagnosticResult(
                    name=f"Optional dep: {module}",
                    status="OK",
                    message=f"{module} is installed",
                )
            )
        except ImportError:
            results.append(
                DiagnosticResult(
                    name=f"Optional dep: {module}",
                    status="WARN",
                    message=f"{module} not installed — some features unavailable",
                    fix_hint=f"pip install {extra}",
                )
            )
    return results


def _check_config_file() -> DiagnosticResult:
    """Check that mltk.yaml or pyproject.toml exists in cwd."""
    import pathlib

    yaml_path = pathlib.Path("mltk.yaml")
    pyproject_path = pathlib.Path("pyproject.toml")

    if yaml_path.exists():
        return DiagnosticResult(
            name="Config file",
            status="OK",
            message=f"Found {yaml_path.resolve()}",
        )
    if pyproject_path.exists():
        return DiagnosticResult(
            name="Config file",
            status="OK",
            message=f"Found {pyproject_path.resolve()} (uses [tool.mltk] section)",
        )
    return DiagnosticResult(
        name="Config file",
        status="WARN",
        message="No mltk.yaml or pyproject.toml found in current directory",
        fix_hint="Run: mltk init  — or create mltk.yaml manually",
    )


def _check_report_dir() -> DiagnosticResult:
    """Check report_dir exists and is writable."""
    import pathlib

    from mltk.core.config import MltkConfig

    try:
        config = MltkConfig.load()
        report_dir = pathlib.Path(config.report_dir)
        if not report_dir.exists():
            return DiagnosticResult(
                name="Report directory",
                status="WARN",
                message=f"report_dir '{report_dir}' does not exist (will be created on first run)",
                fix_hint=f"Run: mkdir -p {report_dir}  — or run any mltk test to auto-create",
            )
        # Check writable by attempting to create a temp file
        test_file = report_dir / ".mltk_write_check"
        try:
            test_file.touch()
            test_file.unlink()
        except OSError:
            return DiagnosticResult(
                name="Report directory",
                status="WARN",
                message=f"report_dir '{report_dir}' exists but is not writable",
                fix_hint=f"Check permissions: chmod u+w {report_dir}",
            )
        return DiagnosticResult(
            name="Report directory",
            status="OK",
            message=f"report_dir '{report_dir}' exists and is writable",
        )
    except Exception as exc:  # noqa: BLE001
        return DiagnosticResult(
            name="Report directory",
            status="WARN",
            message=f"Could not check report_dir: {exc}",
            fix_hint="Verify report_dir in your mltk.yaml",
        )


def _check_baseline_dir() -> DiagnosticResult:
    """Check baseline_dir exists."""
    import pathlib

    from mltk.core.config import MltkConfig

    try:
        config = MltkConfig.load()
        baseline_dir = pathlib.Path(config.baseline_dir)
        if baseline_dir.exists():
            return DiagnosticResult(
                name="Baseline directory",
                status="OK",
                message=f"baseline_dir '{baseline_dir}' exists",
            )
        return DiagnosticResult(
            name="Baseline directory",
            status="WARN",
            message=f"baseline_dir '{baseline_dir}' does not exist",
            fix_hint=(
                f"Run: mkdir -p {baseline_dir}  — needed for drift baseline comparisons"
            ),
        )
    except Exception as exc:  # noqa: BLE001
        return DiagnosticResult(
            name="Baseline directory",
            status="WARN",
            message=f"Could not check baseline_dir: {exc}",
            fix_hint="Verify baseline_dir in your mltk.yaml",
        )


def _check_rust_extension() -> DiagnosticResult:
    """Check if the compiled Rust extension is available."""
    try:
        from mltk._mltk_rust import ks_test  # noqa: F401

        return DiagnosticResult(
            name="Rust extension",
            status="OK",
            message="mltk._mltk_rust is available (fast KS test enabled)",
        )
    except ImportError:
        return DiagnosticResult(
            name="Rust extension",
            status="WARN",
            message="Rust extension not available — using Python fallback (slower)",
            fix_hint=(
                "Install from source with Rust toolchain: "
                "pip install mltk --no-binary mltk  "
                "(requires cargo)"
            ),
        )


def _check_pytest_plugin() -> DiagnosticResult:
    """Check if the mltk pytest plugin is registered."""
    spec = importlib.util.find_spec("mltk.pytest_plugin.plugin")
    if spec is None:
        return DiagnosticResult(
            name="pytest plugin",
            status="WARN",
            message="mltk.pytest_plugin.plugin module not found",
            fix_hint="Reinstall mltk: pip install --force-reinstall mltk",
        )

    # Check entry point registration
    try:
        eps = importlib.metadata.entry_points(group="pytest11")
        ep_names = [ep.name for ep in eps]
        if "mltk" in ep_names:
            return DiagnosticResult(
                name="pytest plugin",
                status="OK",
                message="mltk pytest plugin registered via entry-points (pytest11)",
            )
        return DiagnosticResult(
            name="pytest plugin",
            status="WARN",
            message=(
                "mltk not found in pytest11 entry-points — "
                "plugin may not auto-load in pytest"
            ),
            fix_hint=(
                "Reinstall in editable mode: pip install -e .  "
                "or add 'mltk' to pytest plugins in pyproject.toml"
            ),
        )
    except Exception:  # noqa: BLE001
        return DiagnosticResult(
            name="pytest plugin",
            status="WARN",
            message="Could not verify pytest plugin entry-point registration",
            fix_hint="Ensure mltk is installed via pip (not just on PYTHONPATH)",
        )


def _check_config_values() -> DiagnosticResult:
    """Check for suspicious configuration values."""
    from mltk.core.config import MltkConfig

    try:
        config = MltkConfig.load()
    except Exception as exc:  # noqa: BLE001
        return DiagnosticResult(
            name="Config validation",
            status="WARN",
            message=f"Failed to load config for validation: {exc}",
            fix_hint="Check mltk.yaml syntax",
        )

    issues: list[str] = []

    if not (0.0 < config.drift_threshold <= 1.0):
        issues.append(
            f"drift_threshold={config.drift_threshold} is outside (0, 1] "
            "(typical range: 0.01–0.10)"
        )

    valid_drift_methods = {"ks", "psi", "chi2", "kl", "wasserstein", "jensen_shannon"}
    if config.drift_method not in valid_drift_methods:
        issues.append(
            f"drift_method='{config.drift_method}' is not a recognized method "
            f"(valid: {', '.join(sorted(valid_drift_methods))})"
        )

    valid_report_formats = {"html", "json", "markdown", "none"}
    if config.report_format not in valid_report_formats:
        issues.append(
            f"report_format='{config.report_format}' is not recognized "
            f"(valid: {', '.join(sorted(valid_report_formats))})"
        )

    if config.seed < 0:
        issues.append(f"seed={config.seed} is negative (use a non-negative integer)")

    if not config.pii_patterns:
        issues.append("pii_patterns is empty — PII detection will not scan any patterns")

    if issues:
        return DiagnosticResult(
            name="Config validation",
            status="WARN",
            message="Suspicious config values detected: " + "; ".join(issues),
            fix_hint="Review mltk.yaml — see docs for valid value ranges",
        )

    return DiagnosticResult(
        name="Config validation",
        status="OK",
        message="Config values are within expected ranges",
    )


def diagnose() -> list[DiagnosticResult]:
    """Run all diagnostic checks and return results.

    Returns a list of DiagnosticResult in the order they were checked.
    Status values:
    - "OK"   — check passed, no action needed
    - "WARN" — non-critical issue, mltk still works but with reduced functionality
    - "FAIL" — critical issue, mltk cannot function correctly

    Example::

        results = diagnose()
        for r in results:
            print(f"[{r.status}] {r.name}: {r.message}")
    """
    results: list[DiagnosticResult] = []

    # 1. Python version
    results.append(_check_python_version())

    # 2. Core deps (numpy, pandas)
    results.extend(_check_core_deps())

    # 3. Optional deps
    results.extend(_check_optional_deps())

    # 4. Config file presence
    results.append(_check_config_file())

    # 5. Report directory
    results.append(_check_report_dir())

    # 6. Baseline directory
    results.append(_check_baseline_dir())

    # 7. Rust extension
    results.append(_check_rust_extension())

    # 8. pytest plugin registration
    results.append(_check_pytest_plugin())

    # 9. Config value validation
    results.append(_check_config_values())

    return results
