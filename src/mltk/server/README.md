# server

## Overview
Self-hosted mltk results platform: FastAPI app with SQLite storage, API-key auth, RBAC, webhooks, metrics, a live portal and a dashboard (`dashboard.html`).

## Interface
- `app.py`: `create_app`
- `routes.py`: request models `ResultItem`, `SubmitRunRequest`, `WebhookCreateRequest`; handlers `submit_run`, `list_runs`, `get_run`, `get_trends`, `get_summary`, `compare`, `create_webhook` (`POST /webhooks`), `list_webhooks` (`GET /webhooks`), `delete_webhook` (`DELETE /webhooks/{webhook_id}`), health probes `health_live`, `health_ready`
- `storage.py`: `Storage` (SQLite)
- `auth.py`: `generate_api_key`, `hash_key`, `require_api_key`
- `rbac.py`: `Role`, `check_permission`, `require_role`
- `audit_log.py`: `AuditLogger`, `assert_audit_log_complete`
- `webhooks.py`: `WebhookConfig`, `validate_webhook_url`, `send_webhook`, `should_fire`
- `github_ci.py`: `format_pr_comment`, `post_pr_comment`, `create_check_run`
- `metrics.py`: `record_assertion`, `record_container_scan`, `metrics_response`
- `comparison.py`: `compare_runs`
- `scheduler.py`: `TestScheduler`
- `portal.py`: `create_portal_html`, `get_portal_data`
- `static.py`: `dashboard`
- `logging_config.py`: `JsonFormatter`, `setup_logging`

## Invariants
- Webhook URLs should pass `validate_webhook_url` before dispatch.
- Key hashing goes through `hash_key` in `auth.py`.

## Side Effects
- SQLite reads/writes, outbound HTTP (webhooks, GitHub), scheduled test runs, log output.

## Verification
- Run: `python -m pytest tests/test_server -q`
