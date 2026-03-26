"""FastAPI application for the mltk server platform."""
from __future__ import annotations

from fastapi import FastAPI

from mltk.server.routes import router
from mltk.server.static import static_router
from mltk.server.storage import Storage


def create_app(db_path: str = "mltk_server.db") -> FastAPI:
    """Create and configure the mltk server FastAPI application.

    Args:
        db_path: Path to the SQLite database file. Created on first run.

    Returns:
        Configured FastAPI application ready to be served with uvicorn.
    """
    app = FastAPI(
        title="mltk Server",
        version="0.6.0",
        description="Self-hosted ML test results platform — SonarQube for ML testing.",
    )
    app.state.storage = Storage(db_path)
    app.include_router(static_router)  # dashboard at /
    app.include_router(router, prefix="/api")
    return app
