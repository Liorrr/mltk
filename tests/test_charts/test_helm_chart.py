"""Static tests for charts/mltk — no cluster required."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
CHART = ROOT / "charts" / "mltk"


def test_chart_metadata() -> None:
    text = (CHART / "Chart.yaml").read_text(encoding="utf-8")
    assert "apiVersion: v2" in text
    assert "name: mltk" in text
    assert 'appVersion: "0.13.0"' in text


def test_default_single_replica() -> None:
    values = (CHART / "values.yaml").read_text(encoding="utf-8")
    assert "replicaCount: 1" in values
    assert "ghcr.io/liorrr/mltk" in values


def test_deployment_uses_real_health_paths() -> None:
    deploy = (CHART / "templates" / "deployment.yaml").read_text(encoding="utf-8")
    assert "/api/health/live" in deploy
    assert "/api/health/ready" in deploy
    assert "--db" in deploy
    assert "server" in deploy
    # The devops-guide sample used /api/health (ok) and mltk-server:latest (stale).
    assert "mltk-server:latest" not in deploy


def test_ingress_disabled_by_default() -> None:
    values = (CHART / "values.yaml").read_text(encoding="utf-8")
    assert "enabled: false" in values
    ingress = (CHART / "templates" / "ingress.yaml").read_text(encoding="utf-8")
    assert ".Values.ingress.enabled" in ingress


def test_helm_template_if_available() -> None:
    helm = shutil.which("helm")
    if helm is None:
        pytest.skip("helm CLI not installed")
    rendered = subprocess.check_output(
        [helm, "template", "mltk", str(CHART)],
        text=True,
    )
    assert "kind: Deployment" in rendered
    assert "/api/health/live" in rendered
    assert "/api/health/ready" in rendered
    assert "ghcr.io/liorrr/mltk:latest" in rendered
    assert "kind: Service" in rendered
    assert "kind: PersistentVolumeClaim" in rendered
    assert "kind: Ingress" not in rendered
