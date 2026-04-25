"""Discovery of the Trivy binary on the host.

The ``mltk.container`` module shells out to Trivy for vulnerability
scanning. This helper locates the binary in three places: the system
``PATH`` (preferred, covers manual installs and package managers),
the ``trivy-py`` Python package (which bundles a binary under
``site-packages/trivy/bin/``), and finally raises an ``ImportError``
with a clear install hint when none is found.

Keeping discovery in one place means every call site fails with the
same actionable message instead of cryptic subprocess errors.
"""

from __future__ import annotations

import shutil
from pathlib import Path

__all__ = ["find_trivy_binary"]


def find_trivy_binary() -> str:
    """Return the path to an available Trivy binary.

    Returns:
        Absolute filesystem path to the Trivy executable.

    Raises:
        ImportError: If no Trivy binary can be located on ``PATH`` or
            inside an installed ``trivy-py`` package.
    """
    found = shutil.which("trivy")
    if found:
        return found

    try:
        import trivy as trivy_py  # noqa: PLC0415
    except ImportError:
        trivy_py = None

    if trivy_py is not None and getattr(trivy_py, "__file__", None):
        bin_dir = Path(trivy_py.__file__).parent / "bin"
        for candidate in ("trivy", "trivy.exe"):
            bin_path = bin_dir / candidate
            if bin_path.exists():
                return str(bin_path)

    raise ImportError(
        "Trivy binary not found. Install via: pip install mltk[container] "
        "or install Trivy separately: "
        "https://trivy.dev/latest/getting-started/installation/"
    )
