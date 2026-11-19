"""Contract tests for the CI dependency audit."""
from __future__ import annotations

import shlex
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import preflight  # noqa: E402


def security_config() -> dict:
    workflow = (preflight.ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    return yaml.load(workflow, Loader=yaml.BaseLoader)


def security_workflow() -> str:
    return "\n".join(step.get("run", "")
                       for step in security_config()["jobs"]["security"]["steps"])


def matrix_pythons() -> list[str]:
    return security_config()["jobs"]["test"]["strategy"]["matrix"]["python"]


def security_commands() -> list[list[str]]:
    return [shlex.split(line) for line in security_workflow().splitlines() if line.strip()]


def test_security_audit_matches_ci_install_extras():
    commands = security_commands()
    for version in matrix_pythons():
        audit_python = f"$RUNNER_TEMP/audit-venv-{version}/bin/python"
        install = next(command for command in commands if command[:3] == [
            audit_python, "-m", "pip",
        ] and "install" in command)
        assert install == [audit_python, "-m", "pip", "install", *preflight.CI_INSTALL]


def test_security_audit_is_strict_and_uses_frozen_requirements():
    commands = security_commands()
    for version in matrix_pythons():
        audit_python = f"$RUNNER_TEMP/audit-venv-{version}/bin/python"
        requirements_path = f"$RUNNER_TEMP/mlspec-audit-{version}.txt"
        freeze = next(command for command in commands if command[:3] == [
            audit_python, "-m", "pip",
        ] and "freeze" in command)
        assert freeze == [
            audit_python, "-m", "pip", "freeze", "--exclude-editable", ">",
            requirements_path,
        ]

        audit = next(command for command in commands if command and command[0] == "pip-audit"
                     and requirements_path in command)
        assert "--strict" in audit
        requirements_option = audit.index("-r")
        assert audit[requirements_option + 1] == requirements_path


def test_security_audit_sets_up_and_audits_every_supported_python():
    setup = next(step for step in security_config()["jobs"]["security"]["steps"]
                 if step.get("uses") == "actions/setup-python@v5")
    assert setup["with"]["python-version"].split() == matrix_pythons()
    commands = security_commands()
    for version in matrix_pythons():
        assert ["python" + version, "-m", "venv",
                f"$RUNNER_TEMP/audit-venv-{version}"] in commands


def test_pip_audit_is_not_installed_into_audited_venv():
    workflow = security_workflow()
    install_lines = [line for line in workflow.splitlines() if "pip install pip-audit" in line]
    assert install_lines
    assert all("audit-venv" not in line for line in install_lines)


def test_security_audit_does_not_ignore_vulnerabilities():
    assert "--ignore-vuln" not in security_workflow()


def _extras(spec: str) -> set[str]:
    return set(spec[spec.index("[") + 1:spec.index("]")].split(","))


def test_ci_install_includes_the_nlp_extra():
    # ROUGE, BLEU and chrF tests skip without it; the audit installs the same set.
    [editable] = [item for item in preflight.CI_INSTALL if item.startswith(".[")]
    assert {"dev", "server", "nlp"} <= _extras(editable)


def test_dev_extra_installs_httpx_for_the_server_tests():
    # FastAPI's TestClient needs httpx; without it the server API tests skip.
    import re

    import tomllib

    pyproject = tomllib.loads((preflight.ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dev = pyproject["project"]["optional-dependencies"]["dev"]
    assert any(re.match(r"httpx\b", requirement) for requirement in dev)
