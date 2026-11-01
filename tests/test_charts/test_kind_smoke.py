"""KinD smoke is opt-in and must not create a cluster from pytest."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SMOKE = ROOT / "charts" / "mltk" / "kind_smoke.py"


def test_kind_smoke_refuses_without_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MLTK_KIND_SMOKE", raising=False)
    proc = subprocess.run(
        [sys.executable, str(SMOKE)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode != 0
    combined = proc.stdout + proc.stderr
    assert "MLTK_KIND_SMOKE=1" in combined
    assert "kind create" not in combined


def test_kind_smoke_is_not_default_ci() -> None:
    assert os.environ.get("MLTK_KIND_SMOKE", "") != "1"
