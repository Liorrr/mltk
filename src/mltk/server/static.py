"""Serve the dashboard HTML at the root URL."""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

static_router = APIRouter()


@static_router.get("/", response_class=HTMLResponse)
async def dashboard() -> HTMLResponse:
    """Serve the mltk dashboard at the root URL."""
    html_path = Path(__file__).parent / "dashboard.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))
