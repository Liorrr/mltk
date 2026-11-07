"""FastAPI application for the mltk server platform."""
from __future__ import annotations

import logging

from fastapi import FastAPI
from starlette.responses import Response

# Single-source version — mltk.__version__ resolves the installed
# distribution (mlspec, legacy mltk) with a source-tree fallback.
from mltk import __version__ as _VERSION
from mltk.server.limits import RequestSizeLimitMiddleware
from mltk.server.logging_config import setup_logging
from mltk.server.routes import router
from mltk.server.static import static_router
from mltk.server.storage import Storage

# Maximum allowed request body size: 10 MB
MAX_REQUEST_BODY_BYTES = 10_485_760
_LOGGER = logging.getLogger(__name__)


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
        version=_VERSION,
        description="Self-hosted ML test results platform — SonarQube for ML testing.",
    )

    app.add_middleware(RequestSizeLimitMiddleware, max_bytes=MAX_REQUEST_BODY_BYTES)

    from mltk.server.auth import public_read_enabled  # noqa: PLC0415

    if public_read_enabled():
        _LOGGER.warning(
            "MLTK_SERVER_PUBLIC_READ is set: read endpoints are served without an API key."
        )

    app.state.storage = Storage(db_path)
    app.include_router(static_router)  # dashboard at /
    app.include_router(router, prefix="/api")

    from mltk.server.metrics import metrics_response  # noqa: PLC0415

    @app.get("/metrics", include_in_schema=False)
    def get_metrics() -> Response:
        """Expose Prometheus metrics. Returns 404 if mlspec[metrics] not installed."""
        result = metrics_response()
        if result is None:
            return Response(
                status_code=404,
                content="Metrics disabled. Install: pip install mlspec[metrics]",
            )
        body, content_type = result
        return Response(content=body, media_type=content_type)

    return app
