"""Tests for MCP path confinement."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from ._helpers import assert_error, assert_ok, call_tool


@pytest.fixture(autouse=True)
def _default_roots(monkeypatch):
    """Use the default root (the cwd) unless a test sets roots itself.

    The package conftest adds pytest's base temp dir as a root so other MCP
    tests can use ``tmp_path``. That dir contains ``tmp_path.parent``, which
    would make every "outside the root" case below pass for the wrong reason.
    """
    monkeypatch.delenv("MLTK_MCP_ROOTS", raising=False)
    monkeypatch.delenv("MLTK_MCP_UNRESTRICTED", raising=False)


def confine_path(raw):
    # Imported lazily: importing mltk.mcp runs the package __init__, which
    # needs the mock ``mcp`` modules that the conftest fixture injects.
    from mltk.mcp._paths import confine_path as _confine

    return _confine(raw)


def _outside_error():
    from mltk.mcp._paths import PathOutsideRootError

    return PathOutsideRootError


def test_path_inside_root_allowed(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    child = tmp_path / "x.py"
    assert confine_path("x.py") == child


def test_root_itself_allowed(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert confine_path(".") == tmp_path


def test_parent_traversal_refused(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(_outside_error()):
        confine_path("../x")


def test_absolute_outside_refused(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(_outside_error()):
        confine_path(str(tmp_path.parent / "x"))


def test_sibling_prefix_refused(tmp_path, monkeypatch):
    root = tmp_path / "proj"
    root.mkdir()
    monkeypatch.setenv("MLTK_MCP_ROOTS", str(root))
    with pytest.raises(_outside_error()):
        confine_path(str(tmp_path / "proj_sibling" / "f.py"))


def test_symlink_escape_refused(tmp_path, monkeypatch):
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    link = root / "link"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable")
    monkeypatch.setenv("MLTK_MCP_ROOTS", str(root))
    with pytest.raises(_outside_error()):
        confine_path(str(link / "file.py"))


def test_multiple_roots_from_env(tmp_path, monkeypatch):
    first, second = tmp_path / "one", tmp_path / "two"
    first.mkdir()
    second.mkdir()
    monkeypatch.setenv("MLTK_MCP_ROOTS", f"{first}{__import__('os').pathsep}{second}")
    assert confine_path(str(second / "x")) == second / "x"


def test_default_root_is_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MLTK_MCP_ROOTS", raising=False)
    assert confine_path("file.csv") == tmp_path / "file.csv"


def test_unrestricted_env_skips_check(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MLTK_MCP_UNRESTRICTED", "true")
    assert confine_path(str(tmp_path.parent / "outside")) == tmp_path.parent / "outside"


def test_mltk_test_refuses_file_outside_root(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with patch("subprocess.run") as run:
        result = call_tool("mltk_test", suite_path=str(tmp_path.parent / "x.py"))
    assert_error(result)
    run.assert_not_called()


def test_mltk_test_runs_file_inside_root(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "test_x.py"
    path.write_text("def test_x(): pass\n", encoding="utf-8")
    proc = MagicMock(returncode=0, stdout="1 passed\n", stderr="")
    with patch("subprocess.run", return_value=proc) as run:
        result = call_tool("mltk_test", suite_path=str(path))
    assert_ok(result)
    run.assert_called_once()


def test_mltk_scan_refuses_outside_root(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = call_tool("mltk_scan", path=str(tmp_path.parent / "x.py"))
    assert_error(result)


def test_mltk_scan_inside_root_works(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "model.py"
    path.write_text("x = 1\n", encoding="utf-8")
    result = call_tool("mltk_scan", path=str(path))
    assert_ok(result)
    assert result["scan_performed"] is False


def test_mltk_eval_refuses_outside_root(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = call_tool("mltk_eval", dataset_path=str(tmp_path.parent / "data.csv"))
    assert_error(result)


def test_mltk_import_refuses_output_path_outside_root(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with patch("mltk.importer.loader.DatasetImporter.load") as load:
        result = call_tool(
            "mltk_import", source=str(tmp_path / "data.csv"),
            output_path=str(tmp_path.parent / "generated.py"),
        )
    assert_error(result)
    load.assert_not_called()


def test_mltk_import_hub_id_not_confined(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    imported = MagicMock()
    imported.mapping.columns_with_role.return_value = ["input"]
    imported.mapping.preview.return_value = "preview"
    imported.to_eval_dataset.return_value.sample_count = 1
    task = MagicMock(value="qa")
    with (
        patch("mltk.importer.loader.DatasetImporter.load", return_value=imported) as load,
        patch("mltk.importer.classify.classify_task", return_value=task),
        patch("mltk.importer.suite_gen.build_suite", return_value=[]),
        patch("mltk.importer.codegen.generate_pytest", return_value="code"),
    ):
        result = call_tool("mltk_import", source="org/name")
    assert_ok(result)
    load.assert_called_once_with("org/name")


def _import_patches(load_return):
    task = MagicMock(value="qa")
    return (
        patch("mltk.importer.loader.DatasetImporter.load", return_value=load_return),
        patch("mltk.importer.classify.classify_task", return_value=task),
        patch("mltk.importer.suite_gen.build_suite", return_value=[]),
        patch("mltk.importer.codegen.generate_pytest", return_value="code"),
    )


def _imported_stub():
    imported = MagicMock()
    imported.mapping.columns_with_role.return_value = ["input"]
    imported.mapping.preview.return_value = "preview"
    imported.to_eval_dataset.return_value.sample_count = 1
    return imported


def test_mltk_import_refuses_local_directory_outside_root(tmp_path, monkeypatch):
    root = tmp_path / "proj"
    outside = tmp_path / "elsewhere"
    root.mkdir()
    outside.mkdir()
    monkeypatch.chdir(root)
    monkeypatch.setenv("MLTK_MCP_ROOTS", str(root))
    p_load, p_cls, p_suite, p_code = _import_patches(_imported_stub())
    with p_load as load, p_cls, p_suite, p_code:
        result = call_tool("mltk_import", source=str(outside))
    assert_error(result)
    load.assert_not_called()


def test_mltk_import_accepts_local_directory_inside_root(tmp_path, monkeypatch):
    root = tmp_path / "proj"
    data_dir = root / "data"
    data_dir.mkdir(parents=True)
    monkeypatch.chdir(root)
    monkeypatch.setenv("MLTK_MCP_ROOTS", str(root))
    p_load, p_cls, p_suite, p_code = _import_patches(_imported_stub())
    with p_load as load, p_cls, p_suite, p_code:
        result = call_tool("mltk_import", source=str(data_dir))
    assert_ok(result)
    load.assert_called_once_with(str(data_dir.resolve()))


def test_roots_and_paths_expand_user_home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    (home / "proj").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setenv("MLTK_MCP_ROOTS", "~/proj")
    assert confine_path("~/proj/data.csv") == (home / "proj" / "data.csv").resolve()
    with pytest.raises(_outside_error()):
        confine_path("~/other/data.csv")


def test_mltk_import_scaffold_keeps_relative_paths(tmp_path, monkeypatch):
    pytest.importorskip("pandas")
    root = tmp_path / "proj"
    (root / "data").mkdir(parents=True)
    (root / "data" / "qa.csv").write_text(
        "question,answer\nWhat is 2+2?,4\nCapital of France?,Paris\n", encoding="utf-8",
    )
    (root / "data" / "gold.csv").write_text(
        "question,expected\nWhat is 2+2?,4\nCapital of France?,Paris\n", encoding="utf-8",
    )
    monkeypatch.chdir(root)
    monkeypatch.setenv("MLTK_MCP_ROOTS", str(root))

    result = call_tool(
        "mltk_import", source="data/qa.csv", input_column="question",
        target_column="answer", golden_path="data/gold.csv",
        golden_target_column="expected", golden_key="input", golden_key_column="question",
    )

    assert_ok(result)
    code = result["generated_code"]
    assert "data/qa.csv" in code
    assert "data/gold.csv" in code
    assert str(root) not in code
    assert str(root.resolve()) not in code
