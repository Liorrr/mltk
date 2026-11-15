"""Smoke tests that run against the package installed from a release artifact."""

from __future__ import annotations

import os
from importlib import metadata
from pathlib import Path

import pytest

if not os.environ.get("MLTK_INSTALLED_PACKAGE_VERSION"):
    pytest.skip(
        "release artifact smoke tests require an installed-package version",
        allow_module_level=True,
    )


def test_installed_package_is_external_and_has_expected_version():
    import mltk

    version = os.environ["MLTK_INSTALLED_PACKAGE_VERSION"]
    repo_root = Path(os.environ["MLTK_REPO_ROOT"]).resolve()
    assert repo_root not in Path(mltk.__file__).resolve().parents
    assert metadata.version("mlspec") == version
    assert mltk.__version__ == version


def test_native_rust_bridge_is_loaded():
    from mltk import _rust

    assert _rust.RUST_AVAILABLE is True
    statistic, p_value = _rust.ks_test([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert isinstance(statistic, float)
    assert isinstance(p_value, float)


def test_pytest_plugin_is_registered(pytestconfig):
    assert pytestconfig.pluginmanager.hasplugin("mltk")


def test_identical_parity_runs_pass():
    from mltk.parity import compare_runs
    from mltk.parity.models import Item, LayerSpec, Manifest, Run, Trial

    def make_run(run_id: str) -> Run:
        return Run(
            run_id=run_id,
            label=run_id,
            items=[Item("item-1", "all", [Trial("1", values={"passed": True})])],
        )

    report = compare_runs(
        make_run("base"),
        make_run("candidate"),
        layers=[LayerSpec("passed")],
        manifest=Manifest(items={"item-1": "all"}, trial_ids=("1",)),
    )
    assert report.status == "PASS"
