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
TIMEOUT = os.environ.get("MLTK_KIND_SMOKE_TIMEOUT", "15m")
# Optional image already pulled on the host, e.g. ghcr.io/liorrr/mltk:v0.13.0.
# The cluster is created and deleted on every run, so the load happens here,
# after `kind create` and before `helm install`.
PRELOAD = os.environ.get("MLTK_KIND_SMOKE_PRELOAD", "")


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
        if PRELOAD:
            subprocess.run(
                [kind, "load", "docker-image", PRELOAD, "--name", CLUSTER],
                check=True,
            )
        subprocess.run(
            [helm, "install", RELEASE, CHART_DIR, "--wait", "--timeout", TIMEOUT],
            check=True,
        )
        subprocess.run(
            [
                kubectl,
                "wait",
                "--for=condition=available",
                f"deployment/{RELEASE}",
                f"--timeout={TIMEOUT}",
            ],
            check=True,
        )
    finally:
        subprocess.run([kind, "delete", "cluster", "--name", CLUSTER], check=False)


if __name__ == "__main__":
    main()
    sys.exit(0)
