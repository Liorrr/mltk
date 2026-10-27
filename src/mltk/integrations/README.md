# integrations

## Overview
Adapters to external trackers, observability tools, pipelines and notifiers. Most depend on optional third-party packages or network credentials.

## Interface
- Issue trackers: `IssueTrackerAdapter` (`adapter.py`), `JiraAdapter`, `LinearAdapter`, `AsanaAdapter`, `GitHubIssuesAdapter`; `TicketDecisionEngine` (`dedup.py`), `IssueLinker`, `render_ticket` (`templates.py`)
- GitHub: `GitHubAppAuth`, `verify_webhook_signature`, `create_check_run`, `format_check_run_output` (`github_app.py`); `PullRequestGenerator`, `PullRequestResult`, `render_pr_body` (`pr_generator.py`)
- Experiment/observability: `MlflowLogger`, `WandbLogger`, `LangfuseAdapter`, `PhoenixAdapter`, `register_phoenix`, `MltkTracer` (OTEL), `generate_grafana_dashboard`, `export_grafana_dashboard`, `generate_provisioning_yaml`
- Pipelines/data: `assert_kubeflow_pipeline_success`, `assert_kubeflow_step_outputs`, `assert_sagemaker_pipeline_success`, `assert_sagemaker_step_status`, `assert_dvc_file_tracked`, `assert_dvc_data_version`
- Traces: `assert_trace_quality`, `to_quality_dict` (`from_capture.py`)
- Notifications: `notify_slack`, `format_slack_message`

## Invariants
- Tracker adapters implement the vendor-agnostic `IssueTrackerAdapter` contract.
- Third-party SDKs are optional dependencies.

## Side Effects
- Network calls to third-party services; Grafana export writes files; DVC checks read local metadata.

## Verification
- Run: `python -m pytest tests/test_integrations -q`
