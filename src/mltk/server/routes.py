"""API routes for the mltk server."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from mltk.server.auth import require_api_key
from mltk.server.comparison import compare_runs
from mltk.server.webhooks import send_webhook, should_fire

router = APIRouter()


class SubmitRunRequest(BaseModel):
    """Payload for submitting a completed test run."""

    project: str = "default"
    results: list[dict]  # type: ignore[type-arg]


class WebhookCreateRequest(BaseModel):
    """Payload for registering a webhook."""

    url: str
    events: list[str]
    project: str | None = None


@router.get("/health")
async def health() -> dict:  # type: ignore[type-arg]
    """Health check endpoint."""
    return {"status": "ok", "service": "mltk-server"}


@router.post("/runs")
async def submit_run(
    req: SubmitRunRequest,
    request: Request,
    _project: str = Depends(require_api_key),
) -> dict:  # type: ignore[type-arg]
    """Submit test results from a test run. Requires Bearer API key."""
    storage = request.app.state.storage
    run_id = storage.save_run(req.project, req.results)

    # Build run summary for webhook evaluation
    total = len(req.results)
    passed = sum(1 for r in req.results if r.get("passed", False))
    failed = total - passed
    run_summary = {"run_id": run_id, "project": req.project, "passed": passed, "failed": failed}

    # Fire matching webhooks (best-effort, non-blocking)
    webhooks = storage.get_webhooks(project=req.project)
    for wh in webhooks:
        if should_fire(wh, run_summary):
            payload = {
                "event": "on_failure" if failed > 0 else "on_success",
                "run_id": run_id,
                "project": req.project,
                "passed": passed,
                "failed": failed,
                "total": total,
            }
            send_webhook(wh.url, payload)

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


# ------------------------------------------------------------------
# Webhook CRUD endpoints
# ------------------------------------------------------------------

@router.post("/webhooks")
async def create_webhook(
    req: WebhookCreateRequest,
    request: Request,
    _project: str = Depends(require_api_key),
) -> dict:  # type: ignore[type-arg]
    """Register a new webhook. Requires Bearer API key."""
    storage = request.app.state.storage
    webhook_id = storage.save_webhook(req.url, req.events, req.project)
    return {"webhook_id": webhook_id, "status": "created"}


@router.get("/webhooks")
async def list_webhooks(
    request: Request,
    project: str | None = None,
) -> dict:  # type: ignore[type-arg]
    """List registered webhooks, optionally filtered by project."""
    storage = request.app.state.storage
    webhooks = storage.get_webhooks(project)
    return {
        "webhooks": [
            {"id": wh.id, "url": wh.url, "events": wh.events, "project": wh.project}
            for wh in webhooks
        ]
    }


@router.delete("/webhooks/{webhook_id}")
async def delete_webhook(
    webhook_id: int,
    request: Request,
    _project: str = Depends(require_api_key),
) -> dict:  # type: ignore[type-arg]
    """Remove a webhook by id. Requires Bearer API key."""
    storage = request.app.state.storage
    deleted = storage.delete_webhook(webhook_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Webhook not found")
    return {"status": "deleted"}


# ------------------------------------------------------------------
# Run comparison endpoint
# ------------------------------------------------------------------

@router.get("/compare")
async def compare(
    request: Request,
    run_a: int,
    run_b: int,
) -> dict:  # type: ignore[type-arg]
    """Compare two test runs and return a structured diff."""
    storage = request.app.state.storage
    run_data_a = storage.get_run(run_a)
    run_data_b = storage.get_run(run_b)
    if run_data_a is None:
        raise HTTPException(status_code=404, detail=f"Run {run_a} not found")
    if run_data_b is None:
        raise HTTPException(status_code=404, detail=f"Run {run_b} not found")
    diff = compare_runs(run_data_a, run_data_b)
    return {"run_a": run_a, "run_b": run_b, "diff": diff}
