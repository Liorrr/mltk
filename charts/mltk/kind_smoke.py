"""Optional KinD smoke for the mltk Helm chart.

Refuses to run unless MLTK_KIND_SMOKE=1. Not a default CI job.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys

CLUSTER = "mltk-smoke"
RELEASE = "mltk"
CHART_DIR = os.path.dirname(os.path.abspath(__file__))


def _need(binary: str) -> str:
    path = shutil.which(binary)
    if path is None:
        raise SystemExit(f"kind smoke requires {binary} on PATH")
    return path


def main() -> None:
    if os.environ.get("MLTK_KIND_SMOKE") != "1":
        raise SystemExit(
            "Refusing to create a cluster. Set MLTK_KIND_SMOKE=1 to run."
        )
    kind = _need("kind")
    helm = _need("helm")
    kubectl = _need("kubectl")
    subprocess.run([kind, "create", "cluster", "--name", CLUSTER], check=True)
    try:
        subprocess.run(
            [helm, "install", RELEASE, CHART_DIR, "--wait", "--timeout", "120s"],
            check=True,
        )
        subprocess.run(
            [
                kubectl,
                "wait",
                "--for=condition=available",
                f"deployment/{RELEASE}",
                "--timeout=120s",
            ],
            check=True,
        )
    finally:
        subprocess.run([kind, "delete", "cluster", "--name", CLUSTER], check=False)


if __name__ == "__main__":
    main()
    sys.exit(0)
