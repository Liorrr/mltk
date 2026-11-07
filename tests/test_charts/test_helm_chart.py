"""Static tests for charts/mltk — no cluster required."""

from __future__ import annotations

import re
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
    assert "appVersion:" in text


def test_chart_app_version_matches_pyproject() -> None:
    chart = (CHART / "Chart.yaml").read_text(encoding="utf-8")
    chart_version = next(
        line.split(":", 1)[1].strip().strip('"')
        for line in chart.splitlines()
        if line.startswith("appVersion:")
    )
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    project_version = re.search(
        r"^version\s*=\s*[\"']([^\"']+)[\"']$", pyproject, re.MULTILINE
    ).group(1)
    assert chart_version == project_version


def test_default_single_replica() -> None:
    values = (CHART / "values.yaml").read_text(encoding="utf-8")
    assert "replicaCount: 1" in values
    assert "ghcr.io/liorrr/mltk" in values


def test_deployment_has_safe_rollout_and_security_defaults() -> None:
    deploy = (CHART / "templates" / "deployment.yaml").read_text(encoding="utf-8")
    values = (CHART / "values.yaml").read_text(encoding="utf-8")
    assert "type: Recreate" in deploy
    assert "automountServiceAccountToken: false" in deploy
    assert "runAsNonRoot" in values
    assert "drop" in values


def test_deployment_derives_mount_and_image_tag() -> None:
    deploy = (CHART / "templates" / "deployment.yaml").read_text(encoding="utf-8")
    values = (CHART / "values.yaml").read_text(encoding="utf-8")
    assert "mountPath: {{ dir .Values.server.dbPath | quote }}" in deploy
    assert "{{- if not (isAbs .Values.server.dbPath) }}" in deploy
    assert (
        'fail "server.dbPath must be an absolute path; '
        'its directory is the PVC mount point"'
    ) in deploy
    assert 'default (printf "v%s" .Chart.AppVersion)' in deploy
    assert "tag: \"\"" in values


def test_deployment_uses_real_health_paths() -> None:
    deploy = (CHART / "templates" / "deployment.yaml").read_text(encoding="utf-8")
    assert "/api/health/live" in deploy
    assert "/api/health/ready" in deploy
    assert "--db" in deploy
    assert "server" in deploy
    # The devops-guide sample used /api/health (ok) and mltk-server:latest (stale).
    assert "mltk-server:latest" not in deploy


def test_extra_env_rendered() -> None:
    values = (CHART / "values.yaml").read_text(encoding="utf-8")
    deploy = (CHART / "templates" / "deployment.yaml").read_text(encoding="utf-8")
    assert "extraEnv: []" in values
    assert "toYaml . | nindent 12" in deploy


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
    app_version = next(
        line.split(":", 1)[1].strip().strip('"')
        for line in (CHART / "Chart.yaml").read_text(encoding="utf-8").splitlines()
        if line.startswith("appVersion:")
    )
    assert f"ghcr.io/liorrr/mltk:v{app_version}" in rendered
    assert "kind: Service" in rendered
    assert "kind: PersistentVolumeClaim" in rendered
    assert "kind: Ingress" not in rendered
