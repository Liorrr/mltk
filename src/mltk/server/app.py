"""FastAPI application for the mltk server platform."""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from starlette.requests import Request
from starlette.responses import Response

from mltk.server.logging_config import setup_logging
from mltk.server.routes import router
from mltk.server.static import static_router
from mltk.server.storage import Storage

# Maximum allowed request body size: 10 MB
MAX_REQUEST_BODY_BYTES = 10_485_760


def create_app(db_path: str = "mltk_server.db") -> FastAPI:
    """Create and configure the mltk server FastAPI application.

    Args:
        db_path: Path to the SQLite database file. Created on first run.

    Returns:
        Configured FastAPI application ready to be served with uvicorn.
    """
    setup_logging()

    app = FastAPI(
        title="mltk Server",
        version="0.6.0",
        description="Self-hosted ML test results platform — SonarQube for ML testing.",
    )

    # --- Middleware: reject oversized request bodies (P1-23) ---------------
    @app.middleware("http")
    async def limit_request_size(request: Request, call_next) -> Response:  # type: ignore[type-arg]
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > MAX_REQUEST_BODY_BYTES:
            return JSONResponse(
                {"detail": "Request body too large (max 10MB)"},
                status_code=413,
            )
        return await call_next(request)

    app.state.storage = Storage(db_path)
    app.include_router(static_router)  # dashboard at /
    app.include_router(router, prefix="/api")
    return app
