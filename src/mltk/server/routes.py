"""API routes for the mltk server."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

router = APIRouter()


class SubmitRunRequest(BaseModel):
    """Payload for submitting a completed test run."""

    project: str = "default"
    results: list[dict]  # type: ignore[type-arg]


@router.get("/health")
async def health() -> dict:  # type: ignore[type-arg]
    """Health check endpoint."""
    return {"status": "ok", "service": "mltk-server"}


@router.post("/runs")
async def submit_run(req: SubmitRunRequest, request: Request) -> dict:  # type: ignore[type-arg]
    """Submit test results from a test run."""
    storage = request.app.state.storage
    run_id = storage.save_run(req.project, req.results)
    return {"run_id": run_id, "status": "saved"}


@router.get("/runs")
async def list_runs(
    request: Request,
    project: str | None = None,
    limit: int = 50,
) -> dict:  # type: ignore[type-arg]
    """List recent test runs."""
    storage = request.app.state.storage
    return {"runs": storage.get_runs(project, limit)}


@router.get("/runs/{run_id}")
async def get_run(run_id: int, request: Request) -> dict:  # type: ignore[type-arg]
    """Get details of a specific run."""
    storage = request.app.state.storage
    run = storage.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return run


@router.get("/trends/{project}")
async def get_trends(
    project: str,
    request: Request,
    limit: int = 20,
) -> dict:  # type: ignore[type-arg]
    """Get score trends for a project."""
    storage = request.app.state.storage
    return {"project": project, "trends": storage.get_trends(project, limit)}
