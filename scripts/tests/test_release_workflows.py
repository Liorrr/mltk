"""Contract tests for release build, test, and rehearsal workflows."""

from __future__ import annotations

import ast
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = ROOT / ".github" / "workflows"
SMOKE_TEST = ROOT / "tests" / "release" / "test_installed_package.py"


def smoke_version_env_name() -> str:
    tree = ast.parse(SMOKE_TEST.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            if any(
                isinstance(target, ast.Name)
                and target.id == "INSTALLED_PACKAGE_VERSION_ENV"
                for target in node.targets
            ):
                value = ast.literal_eval(node.value)
                assert isinstance(value, str)
                return value
    raise AssertionError("smoke test version environment constant not found")


def load_workflow(name: str) -> dict:
    return yaml.load((WORKFLOWS / name).read_text(encoding="utf-8"), Loader=yaml.BaseLoader)


def all_steps(workflow: dict):
    return [step for job in workflow["jobs"].values() for step in job.get("steps", [])]


def named_step(workflow: dict, job_name: str, step_name: str) -> dict:
    return next(
        step
        for step in workflow["jobs"][job_name]["steps"]
        if step.get("name") == step_name
    )


def granted_write_scopes(workflow: dict) -> set[str]:
    """Return permission scopes granted write access by a parsed workflow."""
    scopes: set[str] = set()

    def collect(permissions: object) -> None:
        if isinstance(permissions, dict):
            scopes.update(scope for scope, access in permissions.items() if access == "write")
        elif permissions == "write-all":
            scopes.add("*")

    collect(workflow.get("permissions"))
    for job in workflow.get("jobs", {}).values():
        collect(job.get("permissions"))
    return scopes


def test_release_keeps_tag_trigger_and_publish_jobs():
    workflow = load_workflow("release.yml")
    assert workflow["on"] == {"push": {"tags": ["v*"]}}
    assert workflow["jobs"]["build-and-test"]["uses"] == "./.github/workflows/build-and-test.yml"
    assert {"test-pypi", "pypi", "github-release"} <= set(workflow["jobs"])
    assert granted_write_scopes(workflow) == {"id-token", "contents"}
    assert granted_write_scopes(
        {"jobs": {"test-pypi": workflow["jobs"]["test-pypi"]}}
    ) == {"id-token"}
    assert granted_write_scopes({"jobs": {"pypi": workflow["jobs"]["pypi"]}}) == {"id-token"}
    assert granted_write_scopes(
        {"jobs": {"github-release": workflow["jobs"]["github-release"]}}
    ) == {"contents"}
    assert all(
        granted_write_scopes({"jobs": {name: job}}) == set()
        for name, job in workflow["jobs"].items()
        if name not in {"test-pypi", "pypi", "github-release"}
    )


def test_build_and_test_is_reusable_and_tests_each_wheel():
    workflow = load_workflow("build-and-test.yml")
    assert set(workflow["on"]) == {"workflow_call"}
    wheels = workflow["jobs"]["wheels"]
    names = {entry["name"] for entry in wheels["strategy"]["matrix"]["include"]}
    assert names == {"linux-x64", "linux-arm64", "macos", "windows"}
    artifacts = {
        step["with"]["name"]
        for step in wheels["steps"]
        if "upload-artifact" in step.get("uses", "")
    }
    assert artifacts == {"wheels-${{ matrix.name }}"}
    sdist_artifacts = {
        step["with"]["name"]
        for step in workflow["jobs"]["sdist"]["steps"]
        if "upload-artifact" in step.get("uses", "")
    }
    assert sdist_artifacts == {"sdist"}
    assert granted_write_scopes(workflow) == set()
    pip_lines = [step.get("run", "") for step in all_steps(workflow)]
    assert not any(
        "dist/*.tar.gz" in line or "pip install" in line and "||" in line
        for line in pip_lines
    )
    wheel_test = workflow["jobs"]["test-wheels"]
    test_names = {entry["name"] for entry in wheel_test["strategy"]["matrix"]["include"]}
    assert test_names == names
    install = next(
        step["run"]
        for step in wheel_test["steps"]
        if step.get("name") == "Install local wheel"
    )
    assert "--no-index" in install
    assert "--only-binary" in install
    assert "pip install pytest" in install
    assert wheel_test["strategy"]["fail-fast"] == "false"
    version_env = smoke_version_env_name()
    wheel_smoke = named_step(workflow, "test-wheels", "Run installed-package smoke tests")
    sdist_smoke = named_step(workflow, "test-sdist", "Run installed-package smoke tests")
    expected_version = "${{ env.MLTK_TEST_VERSION }}"
    assert wheel_smoke["env"].get(version_env) == expected_version
    assert sdist_smoke["env"].get(version_env) == expected_version
    sdist_install = named_step(workflow, "test-sdist", "Install local sdist")["run"]
    assert 'pip install "dist/mlspec-${VERSION}.tar.gz"' in sdist_install
    assert "--find-links" not in sdist_install
    assert "pip install pytest" in sdist_install
    assert "dtolnay/rust-toolchain" not in str(workflow)
    assert "cargo --version" in str(workflow)
    assert "rustc --version" in str(workflow)
    for workflow_name in ("build-and-test.yml", "release-rehearsal.yml"):
        for step in all_steps(load_workflow(workflow_name)):
            action = step.get("uses")
            if action:
                assert action.startswith(("actions/", "PyO3/maturin-action"))
    assert all(job in workflow["jobs"] for job in ("test-wheels", "test-sdist"))


def test_rehearsal_only_calls_build_workflow_without_publish_permissions():
    workflow = load_workflow("release-rehearsal.yml")
    assert set(workflow["on"]) == {"pull_request", "workflow_dispatch"}
    assert workflow["on"]["pull_request"]["paths"] == [
        ".github/workflows/release.yml",
        ".github/workflows/build-and-test.yml",
        ".github/workflows/release-rehearsal.yml",
        "pyproject.toml",
        "rust/**",
        "src/mltk/_rust.py",
        "MANIFEST.in",
        "tests/release/**",
    ]
    assert workflow["permissions"] == {"contents": "read"}
    assert workflow["jobs"]["build-and-test"]["uses"] == "./.github/workflows/build-and-test.yml"
    assert granted_write_scopes(workflow) == set()
    text = str(workflow)
    assert "pypa/gh-action-pypi-publish" not in text


def test_docker_publish_still_listens_to_release_only():
    workflow = load_workflow("docker-publish.yml")
    assert workflow["on"]["workflow_run"]["workflows"] == ["Release"]
