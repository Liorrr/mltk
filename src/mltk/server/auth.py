"""API key authentication for mltk server."""
from __future__ import annotations

import hashlib
import secrets

from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

_bearer_scheme = HTTPBearer(auto_error=False)


def generate_api_key(project: str) -> str:  # noqa: ARG001
    """Generate a new API key for a project. Returns the raw key (store securely)."""
    raw_key = f"mltk_{secrets.token_urlsafe(32)}"
    return raw_key


def hash_key(raw_key: str) -> str:
    """Return a SHA-256 hex digest of the raw API key."""
    return hashlib.sha256(raw_key.encode()).hexdigest()


async def require_api_key(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),  # noqa: B008
) -> str:
    """FastAPI dependency — verify Bearer token, return project name."""
    if credentials is None:
        raise HTTPException(status_code=401, detail="API key required")
    storage = request.app.state.storage
    project = storage.verify_api_key(hash_key(credentials.credentials))
    if project is None:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return project
