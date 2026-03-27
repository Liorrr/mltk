"""Tests for role-based access control (RBAC) on the mltk server.

This test suite verifies the three-tier role hierarchy (admin > writer >
reader), the ``check_permission`` function, and the ``require_role``
FastAPI dependency that integrates with the existing auth layer.
"""
from __future__ import annotations

import pytest

pytest.importorskip("fastapi", reason="fastapi not installed — skipping RBAC tests")
pytest.importorskip("httpx", reason="httpx not installed — TestClient requires it")

from fastapi import Depends, FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from mltk.server.auth import generate_api_key, hash_key  # noqa: E402
from mltk.server.rbac import (  # noqa: E402
    ROLE_HIERARCHY,
    Role,
    check_permission,
    require_role,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_app_with_role(tmp_path, role: str):
    """Create a minimal FastAPI app with one protected endpoint.

    A fake ``storage`` object is attached to ``app.state`` that mimics
    the real ``Storage`` class just enough for the RBAC dependency to
    work: ``verify_api_key`` maps a key hash to a project, and
    ``get_key_role`` maps a key hash to a role.
    """
    app = FastAPI()
    raw_key = generate_api_key()
    key_hash = hash_key(raw_key)

    class _FakeStorage:
        """Minimal storage stub for RBAC testing."""

        def verify_api_key(self, kh: str) -> str | None:
            return "test-project" if kh == key_hash else None

        def get_key_role(self, kh: str) -> str | None:
            return role if kh == key_hash else None

    app.state.storage = _FakeStorage()

    @app.get("/protected")
    async def protected(
        _role_ok: bool = Depends(require_role("writer")),  # noqa: B008
    ):
        return {"ok": True}

    @app.get("/admin-only")
    async def admin_only(
        _role_ok: bool = Depends(require_role("admin")),  # noqa: B008
    ):
        return {"ok": True}

    @app.get("/read-only")
    async def read_only(
        _role_ok: bool = Depends(require_role("reader")),  # noqa: B008
    ):
        return {"ok": True}

    client = TestClient(app)
    return client, raw_key


# ---------------------------------------------------------------------------
# Tests — check_permission
# ---------------------------------------------------------------------------


def test_admin_has_all_permissions():
    # SCENARIO: admin role checked against every possible required role
    # WHY: admin is the top of the hierarchy and must satisfy all roles
    # EXPECTED: True for admin, writer, and reader
    assert check_permission("admin", "admin") is True
    assert check_permission("admin", "writer") is True
    assert check_permission("admin", "reader") is True


def test_writer_can_write_and_read():
    # SCENARIO: writer role checked against writer and reader
    # WHY: writers need to push results (write) and view dashboards (read)
    # EXPECTED: True for writer and reader, False for admin
    assert check_permission("writer", "writer") is True
    assert check_permission("writer", "reader") is True
    assert check_permission("writer", "admin") is False


def test_reader_is_read_only():
    # SCENARIO: reader role checked against all roles
    # WHY: a reader key must never be able to write or administer
    # EXPECTED: True only for reader
    assert check_permission("reader", "reader") is True
    assert check_permission("reader", "writer") is False
    assert check_permission("reader", "admin") is False


def test_unknown_role_always_denied():
    # SCENARIO: a role string that does not exist in the hierarchy
    # WHY: typos or injection attacks must fail closed, never granting access
    # EXPECTED: False for every required role
    assert check_permission("superuser", "reader") is False
    assert check_permission("", "reader") is False
    assert check_permission("ADMIN", "admin") is False  # case-sensitive


# ---------------------------------------------------------------------------
# Tests — Role class and hierarchy
# ---------------------------------------------------------------------------


def test_role_constants_match_hierarchy_keys():
    # SCENARIO: the Role class constants and ROLE_HIERARCHY keys
    # WHY: if they diverge, permission checks silently break
    # EXPECTED: every Role constant is a key in ROLE_HIERARCHY
    for r in (Role.ADMIN, Role.WRITER, Role.READER):
        assert r in ROLE_HIERARCHY, f"Role {r!r} missing from ROLE_HIERARCHY"


def test_hierarchy_is_strict_superset():
    # SCENARIO: admin's allowed set vs writer's vs reader's
    # WHY: the hierarchy must be strictly ordered — admin > writer > reader
    # EXPECTED: admin's set is a strict superset of writer's, which is a
    #           strict superset of reader's
    admin_set = ROLE_HIERARCHY["admin"]
    writer_set = ROLE_HIERARCHY["writer"]
    reader_set = ROLE_HIERARCHY["reader"]
    assert writer_set < admin_set, "writer set must be a strict subset of admin"
    assert reader_set < writer_set, "reader set must be a strict subset of writer"


# ---------------------------------------------------------------------------
# Tests — require_role FastAPI dependency
# ---------------------------------------------------------------------------


def test_require_role_grants_access(tmp_path):
    # SCENARIO: a writer key hits a writer-protected endpoint
    # WHY: the dependency must allow access when the role satisfies the requirement
    # EXPECTED: HTTP 200
    client, raw_key = _make_app_with_role(tmp_path, "writer")
    resp = client.get(
        "/protected",
        headers={"Authorization": f"Bearer {raw_key}"},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"ok": True}


def test_require_role_denies_insufficient_role(tmp_path):
    # SCENARIO: a reader key hits a writer-protected endpoint
    # WHY: readers must not be able to perform write operations
    # EXPECTED: HTTP 403 with a clear error message mentioning both roles
    client, raw_key = _make_app_with_role(tmp_path, "reader")
    resp = client.get(
        "/protected",
        headers={"Authorization": f"Bearer {raw_key}"},
    )
    assert resp.status_code == 403, resp.text
    detail = resp.json()["detail"]
    assert "reader" in detail
    assert "writer" in detail
