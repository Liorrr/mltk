"""Contract tests for the CI dependency audit."""
from __future__ import annotations

import re
import shlex
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import preflight  # noqa: E402


def security_workflow() -> str:
    workflow = (preflight.ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    config = yaml.load(workflow, Loader=yaml.BaseLoader)
    return "\n".join(step.get("run", "") for step in config["jobs"]["security"]["steps"])


def test_security_audit_matches_ci_install_extras():
    workflow = security_workflow()
    install = re.search(r"pip install -e\s+([\"'])(\.[^\"']+)[\"']\s+onnx", workflow)
    expected = re.search(r"^\.\[[^]]+\]$", preflight.CI_INSTALL[1])
    assert install is not None
    assert expected is not None
    assert install.group(2) == expected.group(0)


def test_security_audit_is_strict_and_uses_frozen_requirements():
    lines = [shlex.split(line) for line in security_workflow().splitlines()]
    audit_python = "$RUNNER_TEMP/audit-venv/bin/python"
    freeze = next(tokens for tokens in lines if tokens[:3] == [audit_python, "-m", "pip"]
                  and "freeze" in tokens)
    freeze_redirect = freeze.index(">")
    requirements_path = freeze[freeze_redirect + 1]
    assert freeze[3:5] == ["freeze", "--exclude-editable"]
    assert requirements_path == "$RUNNER_TEMP/mlspec-audit-requirements.txt"

    audit = next(tokens for tokens in lines if tokens and tokens[0] == "pip-audit")
    assert "--strict" in audit
    requirements_option = audit.index("-r")
    assert audit[requirements_option + 1] == requirements_path


def test_pip_audit_is_not_installed_into_audited_venv():
    workflow = security_workflow()
    install_lines = [line for line in workflow.splitlines() if "pip install pip-audit" in line]
    assert install_lines
    assert all("audit-venv" not in line for line in install_lines)


def test_security_audit_does_not_ignore_vulnerabilities():
    assert "--ignore-vuln" not in security_workflow()
