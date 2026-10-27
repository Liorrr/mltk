# mltk Blueprint

## System Overview
`mltk` is "pytest for ML": assertion libraries, a pytest plugin, a CLI, an MCP server and a results server for testing data, models, training, inference, pipelines and LLM systems. Every package below follows the same pattern: `assert_*` functions return `TestResult` objects built on `core` and raise `MltkAssertionError` when a check fails at `Severity.CRITICAL` (the default), and heavy dependencies are imported lazily.

## Local Interface & Invariants
- `__init__.py`: `__version__` (read from package metadata, falls back to a literal) and convenience re-exports: `assert_container_vulnerabilities`, `assert_no_secrets_in_image` (container); `assert_schema`, `assert_dtypes`, `assert_no_nulls`, `assert_range`, `assert_unique`, `assert_row_count`, `assert_freshness`, `assert_no_drift`, `assert_no_pii` (data); `assert_metric`, `assert_no_regression`, `assert_slice_performance`, `assert_calibration`, `assert_no_bias`, `assert_intersectional_fairness` (model).
- `doctor.py`: `DiagnosticResult`, `diagnose()` - environment health checks (Python version, dependencies, config, report dir, Rust extension, pytest plugin).
- `jupyter.py`: `display_report(suite)` - notebook rendering of a `TestSuite`.
- `_rust.py`: optional accelerated statistics (`ks_test`, `psi`, `kl_divergence`, `chi_squared`, `js_divergence`, `wasserstein`, `cosine_similarity`, `scan_pii_fast`, ...) backed by the compiled `_mltk_rust` extension when present.
- Invariant: `mltk.core` is the base layer; other packages depend on it, not the reverse.

## Subdirectories & System Boundaries
- chat: [./chat/README.md](./chat/README.md) - rule-based Q&A over saved results.
- cli: [./cli/README.md](./cli/README.md) - Typer CLI entry point.
- compliance: [./compliance/README.md](./compliance/README.md) - regulatory framework mappings and reports (`templates/` holds only `.j2` files).
- container: [./container/README.md](./container/README.md) - Trivy-based image scanning.
- contracts: [./contracts/README.md](./contracts/README.md) - YAML data contracts.
- core: [./core/README.md](./core/README.md) - types, config, suite runner, plugin registry.
- cost: [./cost/README.md](./cost/README.md) - LLM cost tracking and budgets.
- data: [./data/README.md](./data/README.md) - data quality assertions.
- domains: [./domains/README.md](./domains/README.md) - CV, LLM, multimodal, NLP, speech, tabular and other domain kits.
- eval: [./eval/README.md](./eval/README.md) - LLM eval tasks, solvers, scorers.
- experiment: [./experiment/README.md](./experiment/README.md) - hypothesis validation with git-worktree sandboxing.
- importer: [./importer/README.md](./importer/README.md) - dataset import and suite generation.
- inference: [./inference/README.md](./inference/README.md) - latency, throughput, API contract assertions.
- integrations: [./integrations/README.md](./integrations/README.md) - trackers, MLOps and observability adapters.
- mcp: [./mcp/README.md](./mcp/README.md) - FastMCP server for AI agents.
- model: [./model/README.md](./model/README.md) - model quality, fairness, robustness assertions.
- monitor: [./monitor/README.md](./monitor/README.md) - drift, SLA and endpoint monitoring.
- pipeline: [./pipeline/README.md](./pipeline/README.md) - pipeline and reproducibility assertions.
- pytest_plugin: [./pytest_plugin/README.md](./pytest_plugin/README.md) - markers, fixtures, report export.
- registry: [./registry/README.md](./registry/README.md) - local test-resource collections.
- report: [./report/README.md](./report/README.md) - HTML, JUnit and model-card generators (`templates/` holds only `.j2` files).
- scan: [./scan/README.md](./scan/README.md) - automated issue scanning engine.
- server: [./server/README.md](./server/README.md) - self-hosted FastAPI results platform.
- testdefs: [./testdefs/README.md](./testdefs/README.md) - declarative YAML test suites.
- testing: [./testing/README.md](./testing/README.md) - flaky detection, golden sets, test selection.
- trace: [./trace/README.md](./trace/README.md) - canonical capture records for agent runs.
- training: [./training/README.md](./training/README.md) - training-time bug detection.

## Global Invariants
- Assertions fail via `MltkAssertionError` or a failed `TestResult`; they do not mutate inputs.
- Optional third-party SDKs and the Rust extension are never required at import time.

## Global Side Effects
- Filesystem: reports, baselines, golden files, generated tests, registry (`~/.mltk/registry`).
- Network: `server`, `integrations`, `monitor` cloud modules, `pytest_plugin --mltk-server`.
- Subprocesses: pytest (`mcp`), trivy (`container`), git (`experiment`), nvidia-smi (`monitor`).

## System Verification
- Run: `python -m pytest tests/regression -q` (slim gate); full suite `python -m pytest tests/ -x -q`
