"""API routes for the mltk server."""
from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from pydantic import BaseModel, field_validator
from starlette.concurrency import run_in_threadpool

from mltk.report.summarizer import summarize_test_history
from mltk.server.comparison import compare_runs
from mltk.server.rbac import KeyPrincipal, Role, require_role
from mltk.server.redact import mask_webhook_url
from mltk.server.webhooks import send_webhook, should_fire, validate_webhook_url

router = APIRouter()

# Route dependencies, built once. ``reader`` routes return None for the
# principal when public read is enabled and no key is sent.
_READER = Depends(require_role(Role.READER))
_WRITER = Depends(require_role(Role.WRITER))
_ADMIN = Depends(require_role(Role.ADMIN))


class ResultItem(BaseModel):
    """A single test result within a run submission."""

    name: str
    passed: bool
    severity: str = "info"
    message: str = ""
    details: dict = {}  # type: ignore[type-arg]
    duration_ms: float = 0.0


class SubmitRunRequest(BaseModel):
    """Payload for submitting a completed test run."""

    project: str = "default"
    results: list[ResultItem]

    @field_validator("results")
    @classmethod
    def limit_results(cls, v: list[ResultItem]) -> list[ResultItem]:
        if len(v) > 10_000:
            raise ValueError("Maximum 10,000 results per run")
        return v


class WebhookCreateRequest(BaseModel):
    """Payload for registering a webhook."""

    url: str
    events: list[str]
    project: str | None = None


@router.get("/health")
async def health() -> dict:  # type: ignore[type-arg]
    """Health check endpoint."""
    return {"status": "ok", "service": "mltk-server"}


@router.get("/health/live")
async def health_live() -> dict:  # type: ignore[type-arg]
    """Liveness probe — confirms the process is alive.

    Use for Kubernetes ``livenessProbe``. This always returns 200 if
    the server process is running, regardless of database state.
    """
    return {"status": "alive"}


@router.get("/health/ready")
async def health_ready(request: Request) -> dict:  # type: ignore[type-arg]
    """Readiness probe — confirms the server can serve traffic.

    Use for Kubernetes ``readinessProbe``. Verifies the database is
    accessible by executing a lightweight query. Returns 503 if the
    database is unreachable.
    """
    storage = request.app.state.storage
    try:
        storage.get_runs(limit=1)
        return {"status": "ready"}
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Database not ready") from exc


@router.post("/runs")
async def submit_run(
    req: SubmitRunRequest,
    request: Request,
    background_tasks: BackgroundTasks,
    principal: KeyPrincipal = _WRITER,
) -> dict:  # type: ignore[type-arg]
    """Submit test results from a test run. Requires Bearer API key.

    If the API key is scoped to a specific project (not "default" or empty),
    the request's ``project`` must match the key's project.
    """
    # Enforce project scope: scoped keys can only write to their own project
    if principal.scoped and req.project != principal.project:
        raise HTTPException(
            status_code=403,
            detail=f"API key is scoped to project '{principal.project}', "
            f"cannot write to '{req.project}'",
        )
    storage = request.app.state.storage
    results_dicts = [r.model_dump() for r in req.results]
    run_id = storage.save_run(req.project, results_dicts)

    # Build run summary for webhook evaluation
    total = len(req.results)
    passed = sum(1 for r in req.results if r.passed)
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
            background_tasks.add_task(send_webhook, wh.url, payload)

    return {"run_id": run_id, "status": "saved"}


@router.get("/runs")
async def list_runs(
    request: Request,
    project: str | None = None,
    limit: int = 50,
    principal: KeyPrincipal | None = _READER,
) -> dict:  # type: ignore[type-arg]
    """List recent test runs."""
    if principal is not None and principal.scoped:
        if project is not None and project != principal.project:
            raise HTTPException(status_code=404, detail="Project not found")
        project = principal.project
    storage = request.app.state.storage
    return {"runs": storage.get_runs(project, limit)}


@router.get("/runs/{run_id}")
async def get_run(
    run_id: int, request: Request,
    principal: KeyPrincipal | None = _READER,
) -> dict:  # type: ignore[type-arg]
    """Get details of a specific run."""
    storage = request.app.state.storage
    run = storage.get_run(run_id)
    if (
        not run
        or (principal is not None and principal.scoped and run["project"] != principal.project)
    ):
        raise HTTPException(status_code=404, detail="Run not found")
    return run


@router.get("/trends/{project}")
async def get_trends(
    project: str,
    request: Request,
    limit: int = 20,
    principal: KeyPrincipal | None = _READER,
) -> dict:  # type: ignore[type-arg]
    """Get score trends for a project."""
    if principal is not None and principal.scoped and project != principal.project:
        raise HTTPException(status_code=404, detail="Project not found")
    storage = request.app.state.storage
    return {"project": project, "trends": storage.get_trends(project, limit)}


# ------------------------------------------------------------------
# Resource summarization endpoint
# ------------------------------------------------------------------

@router.get("/summary/{project}")
async def get_summary(
    project: str,
    request: Request,
    limit: int = 20,
    principal: KeyPrincipal | None = _READER,
) -> dict:  # type: ignore[type-arg]
    """Get test history summary for a project.

    Fetches the most recent *limit* runs (with full per-test results)
    and produces trend analysis, common failures, flaky tests, and
    actionable recommendations.
    """
    if principal is not None and principal.scoped and project != principal.project:
        raise HTTPException(status_code=404, detail="Project not found")
    storage = request.app.state.storage
    run_summaries = storage.get_runs(project, limit)

    # Enrich each run summary with its per-test results so the
    # summarizer can do failure/flakiness analysis.
    enriched_runs: list[dict] = []  # type: ignore[type-arg]
    for rs in run_summaries:
        full_run = storage.get_run(rs["id"])
        if full_run is not None:
            enriched_runs.append(full_run)

    summary = summarize_test_history(enriched_runs)
    return {"project": project, "summary": summary}


# ------------------------------------------------------------------
# Webhook CRUD endpoints
# ------------------------------------------------------------------

@router.post("/webhooks")
async def create_webhook(
    req: WebhookCreateRequest,
    request: Request,
    principal: KeyPrincipal = _ADMIN,
) -> dict:  # type: ignore[type-arg]
    """Register a new webhook. Requires Bearer API key.

    The URL is validated before saving — private/loopback addresses and
    non-HTTP schemes are rejected with HTTP 422.
    """
    if principal.scoped and req.project != principal.project:
        raise HTTPException(
            status_code=403,
            detail=f"API key is scoped to project '{principal.project}', "
            f"cannot manage project '{req.project or 'default'}'",
        )
    # The check resolves the hostname, so it runs in a worker thread.
    if not await run_in_threadpool(validate_webhook_url, req.url):
        raise HTTPException(
            status_code=422,
            detail="Invalid webhook URL: must be a public http(s) endpoint",
        )
    storage = request.app.state.storage
    webhook_id = storage.save_webhook(req.url, req.events, req.project)
    return {"webhook_id": webhook_id, "status": "created"}


@router.get("/webhooks")
async def list_webhooks(
    request: Request,
    project: str | None = None,
    principal: KeyPrincipal | None = _READER,
) -> dict:  # type: ignore[type-arg]
    """List registered webhooks, optionally filtered by project."""
    if principal is not None and principal.scoped:
        if project is not None and project != principal.project:
            raise HTTPException(status_code=404, detail="Project not found")
        project = principal.project
    storage = request.app.state.storage
    webhooks = storage.get_webhooks(project)
    if principal is not None and principal.scoped:
        webhooks = [wh for wh in webhooks if wh.project == principal.project]
    return {
        "webhooks": [
            {
                "id": wh.id,
                "url": mask_webhook_url(wh.url),
                "events": wh.events,
                "project": wh.project,
            }
            for wh in webhooks
        ]
    }


@router.delete("/webhooks/{webhook_id}")
async def delete_webhook(
    webhook_id: int,
    request: Request,
    principal: KeyPrincipal = _ADMIN,
) -> dict:  # type: ignore[type-arg]
    """Remove a webhook by id. Requires Bearer API key."""
    storage = request.app.state.storage
    webhook = storage.get_webhook(webhook_id)
    if webhook is None or (principal.scoped and webhook.project != principal.project):
        raise HTTPException(status_code=404, detail="Webhook not found")
    storage.delete_webhook(webhook_id)
    return {"status": "deleted"}


# ------------------------------------------------------------------
# Run comparison endpoint
# ------------------------------------------------------------------

@router.get("/compare")
async def compare(
    request: Request,
    run_a: int,
    run_b: int,
    principal: KeyPrincipal | None = _READER,
) -> dict:  # type: ignore[type-arg]
    """Compare two test runs and return a structured diff."""
    storage = request.app.state.storage
    run_data_a = storage.get_run(run_a)
    run_data_b = storage.get_run(run_b)
    if (
        run_data_a is None
        or (
            principal is not None
            and principal.scoped
            and run_data_a["project"] != principal.project
        )
    ):
        raise HTTPException(status_code=404, detail=f"Run {run_a} not found")
    if (
        run_data_b is None
        or (
            principal is not None
            and principal.scoped
            and run_data_b["project"] != principal.project
        )
    ):
        raise HTTPException(status_code=404, detail=f"Run {run_b} not found")
    diff = compare_runs(run_data_a, run_data_b)
    return {"run_a": run_a, "run_b": run_b, "diff": diff}
