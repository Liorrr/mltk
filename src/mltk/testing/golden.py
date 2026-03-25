"""Golden test sets — versioned baseline management for ML tests."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np

from mltk.core.assertion import assert_true, timed_assertion
from mltk.core.result import Severity, TestResult


def save_golden(
    data: dict | list | np.ndarray,
    path: str | Path,
    version: str | None = None,
) -> Path:
    """Save golden baseline data with version metadata.

    Wraps *data* in a JSON envelope::

        {
            "version": "<version>",
            "timestamp": "<ISO-8601>",
            "data": <data>
        }

    Numpy arrays are converted to nested lists for JSON serialisation.
    The parent directory is created if it does not exist.

    Args:
        data: Baseline data to persist.  Dict, list, or numpy array.
        path: Destination file path (``*.json``).
        version: Optional version tag.  Defaults to ``"1.0.0"``.

    Returns:
        Resolved :class:`~pathlib.Path` of the written file.

    Example:
        >>> import numpy as np
        >>> from mltk.testing.golden import save_golden
        >>> p = save_golden({"accuracy": 0.95}, "/tmp/baseline.json", version="1.0.0")
    """
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)

    serialisable_data: dict | list
    if isinstance(data, np.ndarray):
        serialisable_data = data.tolist()
    else:
        serialisable_data = data

    envelope = {
        "version": version or "1.0.0",
        "timestamp": datetime.now().isoformat(),
        "data": serialisable_data,
    }

    dest.write_text(json.dumps(envelope, indent=2), encoding="utf-8")
    return dest.resolve()


def load_golden(path: str | Path) -> dict:
    """Load a golden baseline file.

    Args:
        path: Path to a previously saved golden JSON file.

    Returns:
        Dict with keys ``"version"``, ``"timestamp"``, and ``"data"``.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If the file is not a valid golden envelope.
    """
    src = Path(path)
    if not src.exists():
        raise FileNotFoundError(f"Golden file not found: {src}")

    raw = json.loads(src.read_text(encoding="utf-8"))
    if "data" not in raw:
        raise ValueError(f"File does not look like a golden baseline: {src}")
    return raw


def _max_numeric_diff(
    current: dict | list | np.ndarray,
    golden_data: dict | list,
) -> float:
    """Recursively compute max absolute numeric difference."""
    if isinstance(current, np.ndarray):
        current = current.tolist()

    if isinstance(current, dict) and isinstance(golden_data, dict):
        if not golden_data:
            return 0.0
        diffs = []
        for key in golden_data:
            if key in current:
                diffs.append(_max_numeric_diff(current[key], golden_data[key]))
        return max(diffs) if diffs else 0.0

    if isinstance(current, list) and isinstance(golden_data, list):
        if not golden_data:
            return 0.0
        diffs = []
        for c_val, g_val in zip(current, golden_data, strict=False):
            diffs.append(_max_numeric_diff(c_val, g_val))
        return max(diffs) if diffs else 0.0

    try:
        return abs(float(current) - float(golden_data))
    except (TypeError, ValueError):
        return 0.0 if current == golden_data else 1.0


@timed_assertion
def assert_matches_golden(
    current: dict | list | np.ndarray,
    golden_path: str | Path,
    tolerance: float = 0.01,
) -> TestResult:
    """Assert that *current* data matches the saved golden baseline.

    For numeric data the maximum absolute element-wise difference must be
    ``<= tolerance``.  Works recursively on dicts and lists.

    Args:
        current: Data produced by the system under test.
        golden_path: Path to the golden file written by :func:`save_golden`.
        tolerance: Maximum allowed numeric deviation.  Default ``0.01``.

    Returns:
        :class:`~mltk.core.result.TestResult` — passes when within tolerance.

    Raises:
        :class:`~mltk.core.assertion.MltkAssertionError`: On mismatch with
            CRITICAL severity.
        FileNotFoundError: If the golden file does not exist.

    Example:
        >>> from mltk.testing.golden import save_golden, assert_matches_golden
        >>> save_golden({"f1": 0.88}, "/tmp/baseline.json")
        PosixPath('/tmp/baseline.json')
        >>> assert_matches_golden({"f1": 0.881}, "/tmp/baseline.json", tolerance=0.01)
    """
    envelope = load_golden(golden_path)
    golden_data = envelope["data"]

    max_diff = _max_numeric_diff(current, golden_data)
    passed = max_diff <= tolerance

    return assert_true(
        passed,
        name="golden.matches",
        message=(
            f"Data matches golden baseline (max_diff={max_diff:.6f} <= {tolerance})"
            if passed
            else f"Data deviates from golden baseline (max_diff={max_diff:.6f} > {tolerance})"
        ),
        severity=Severity.CRITICAL,
        max_diff=round(max_diff, 6),
        tolerance=tolerance,
        golden_version=envelope.get("version", "unknown"),
    )
