"""Role-based access control (RBAC) for the mltk server.

This module defines a simple but effective three-tier role system for
controlling who can do what on the mltk server API.  It is designed to
integrate with the existing API-key authentication layer in ``auth.py``
and the FastAPI dependency-injection pattern used in ``routes.py``.

Concepts
--------
**Role hierarchy** means a higher role automatically inherits the
permissions of every role below it.  An ``admin`` implicitly has
``writer`` *and* ``reader`` access, so you never need to grant multiple
roles to a single API key.

**Principle of least privilege** means CI/CD pipelines that only need to
push results should be ``writer`` keys, and dashboard viewers should be
``reader`` keys.  If a ``reader`` key leaks, the attacker cannot modify
any data.

Integration
-----------
``require_role`` returns a FastAPI dependency that can be composed with
the existing ``require_api_key`` dependency::

    @router.post("/api/runs")
    async def submit_run(
        project: str = Depends(require_api_key),
        _role_ok: bool = Depends(require_role("writer")),
    ):
        ...

The role is looked up from the ``api_keys`` table via a ``role`` column.
Until that migration lands, the module treats any key without an explicit
role as ``admin`` for backward compatibility.
"""

from __future__ import annotations

from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from mltk.server.auth import hash_key

_bearer_scheme = HTTPBearer(auto_error=False)


# ---------------------------------------------------------------------------
# Role definitions
# ---------------------------------------------------------------------------


class Role:
    """Role-based access control roles for the mltk server.

    WHY RBAC for ML testing:
      In a team environment, not everyone should have the same permissions.
      Data scientists need to READ test results and trends.  CI/CD pipelines
      need to WRITE results.  Only admins should manage API keys and webhooks.
      Without RBAC, a leaked CI token gives full admin access -- a single
      compromised token becomes a catastrophic breach.

    The three roles form a strict hierarchy:

    * ``admin``  -- full access: manage keys, webhooks, read/write results
    * ``writer`` -- write results, read everything (no key management)
    * ``reader`` -- read-only: view results, trends, dashboards

    Examples
    --------
    >>> Role.ADMIN
    'admin'
    >>> Role.WRITER
    'writer'
    """

    ADMIN: str = "admin"
    WRITER: str = "writer"
    READER: str = "reader"

    ALL: frozenset[str] = frozenset({"admin", "writer", "reader"})


# ---------------------------------------------------------------------------
# Hierarchy and permission checking
# ---------------------------------------------------------------------------

ROLE_HIERARCHY: dict[str, set[str]] = {
    "admin": {"admin", "writer", "reader"},
    "writer": {"writer", "reader"},
    "reader": {"reader"},
}
"""Maps each role to the set of roles it satisfies.

``admin`` satisfies every role because it is the top of the hierarchy.
``writer`` satisfies ``writer`` and ``reader`` (but not ``admin``).
``reader`` only satisfies itself.

This data structure makes permission checks O(1) -- a single set lookup
instead of walking a linked list or tree.
"""


def check_permission(user_role: str, required_role: str) -> bool:
    """Check if *user_role* has at least *required_role* permission.

    The check uses ``ROLE_HIERARCHY`` to determine whether the user's
    assigned role satisfies the required role.  Unknown roles are always
    denied to prevent privilege escalation through typos or injection.

    Parameters
    ----------
    user_role:
        The role string stored on the user's API key (e.g. ``"writer"``).
    required_role:
        The minimum role needed for the operation (e.g. ``"reader"``).

    Returns
    -------
    bool
        ``True`` if access should be granted, ``False`` otherwise.

    Examples
    --------
    >>> check_permission("admin", "reader")
    True
    >>> check_permission("reader", "writer")
    False
    >>> check_permission("unknown", "reader")
    False
    """
    allowed = ROLE_HIERARCHY.get(user_role)
    if allowed is None:
        # Unknown roles are always denied -- fail closed.
        return False
    return required_role in allowed


# ---------------------------------------------------------------------------
# FastAPI dependency
# ---------------------------------------------------------------------------


def _lookup_role_for_key(storage: object, key_hash: str) -> str:
    """Resolve the role for an API key hash from storage.

    This is intentionally lenient: if the storage layer does not yet have
    a ``get_key_role`` method (i.e. the role-column migration has not
    landed), the key is treated as ``admin`` for backward compatibility.

    Parameters
    ----------
    storage:
        The ``Storage`` instance attached to the running application.
    key_hash:
        SHA-256 hex digest of the raw API key.

    Returns
    -------
    str
        The role string (``admin``, ``writer``, or ``reader``).
    """
    getter = getattr(storage, "get_key_role", None)
    if getter is not None:
        role = getter(key_hash)
        if role is not None:
            return role
    # Fallback: legacy keys without an explicit role are treated as admin.
    return Role.ADMIN


def require_role(required_role: str):
    """Return a FastAPI dependency that enforces *required_role*.

    The returned async function extracts the Bearer token from the
    request, hashes it, looks up the corresponding role in storage, and
    raises ``HTTPException(403)`` if the role is insufficient.

    Parameters
    ----------
    required_role:
        The minimum role needed (one of ``admin``, ``writer``, ``reader``).

    Returns
    -------
    Callable
        An async dependency function compatible with ``Depends()``.

    Raises
    ------
    HTTPException(401)
        If no valid API key is provided.
    HTTPException(403)
        If the key's role does not satisfy *required_role*.

    Examples
    --------
    Use as a FastAPI dependency::

        @router.get("/api/results")
        async def list_results(
            _role: bool = Depends(require_role("reader")),
        ):
            ...
    """

    async def _dependency(
        request: Request,
        credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),  # noqa: B008
    ) -> bool:
        # --- Step 1: require a Bearer token ---
        if credentials is None:
            raise HTTPException(status_code=401, detail="API key required")

        raw_key = credentials.credentials
        key_hash = hash_key(raw_key)

        # --- Step 2: verify the key exists ---
        storage = request.app.state.storage
        project = storage.verify_api_key(key_hash)
        if project is None:
            raise HTTPException(status_code=401, detail="Invalid API key")

        # --- Step 3: check the role hierarchy ---
        user_role = _lookup_role_for_key(storage, key_hash)
        if not check_permission(user_role, required_role):
            raise HTTPException(
                status_code=403,
                detail=(
                    f"Insufficient permissions: role '{user_role}' does not "
                    f"satisfy required role '{required_role}'"
                ),
            )
        return True

    return _dependency
