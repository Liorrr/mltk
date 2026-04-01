"""mltk CLI — powered by Typer."""

from __future__ import annotations

from pathlib import Path


def main() -> None:
    """Entry point for mltk CLI.

    Initializes the Typer application and registers all subcommands
    (version, init, scan, drift, score, doctor, test, compliance).
    Requires ``typer`` to be installed via the ``mltk[cli]`` extra.
    """
    try:
        import typer
    except ImportError as err:
        print(  # noqa: T201
            "Typer is required for the mltk CLI. "
            "Install it with: pip install mltk[cli]"
        )
        raise SystemExit(1) from err

    app = typer.Typer(
        name="mltk",
        help=(
            "ML Test Kit -- pytest for ML. "
            "Unified testing across the entire ML lifecycle: "
            "data quality, drift detection, model validation, "
            "compliance, and monitoring."
        ),
    )

    @app.command()
    def version() -> None:
        """Show mltk version."""
        from mltk import __version__

        print(f"mltk v{__version__}")  # noqa: T201

    @app.command()
    def init() -> None:
        """Scaffold mltk.yaml + example test file."""
        config_content = """\
# mltk configuration
drift_method: ks
drift_threshold: 0.05
report_dir: ./mltk-reports
seed: 42
"""
        test_content = """\
\"\"\"Example mltk test file.\"\"\"

import pandas as pd
import pytest

from mltk.data import assert_no_nulls, assert_row_count, assert_schema


@pytest.mark.ml_data
def test_data_quality():
    # Replace with your actual data path
    df = pd.DataFrame({"id": [1, 2, 3], "value": [1.0, 2.0, 3.0]})
    assert_schema(df, {"id": "int64", "value": "float64"})
    assert_no_nulls(df)
    assert_row_count(df, min_rows=1)
"""
        config_path = Path("mltk.yaml")
        test_dir = Path("tests")
        test_path = test_dir / "test_mltk_example.py"

        config_path.write_text(config_content)
        print(f"Created {config_path}")  # noqa: T201

        test_dir.mkdir(exist_ok=True)
        test_path.write_text(test_content)
        print(f"Created {test_path}")  # noqa: T201

    @app.command()
    def scan(
        path: str = typer.Argument(
            ..., help="Path to a CSV or Parquet file to scan",
        ),
    ) -> None:
        """Quick data quality scan on a CSV/Parquet file.

        Reads the file, reports shape, dtypes, and null values.

        Example::

            mltk scan data/features.csv
            mltk scan data/features.parquet
        """
        import pandas as pd

        p = Path(path)
        if not p.exists():
            print(f"File not found: {path}")  # noqa: T201
            raise typer.Exit(1)

        if p.suffix == ".parquet":
            df = pd.read_parquet(p)
        else:
            df = pd.read_csv(p)

        print(f"MLTK Data Scan: {path}")  # noqa: T201
        print(f"  Rows: {len(df):,} | Columns: {len(df.columns)}")  # noqa: T201
        print(f"  Columns: {list(df.columns)}")  # noqa: T201
        print(f"  Dtypes: {dict(df.dtypes)}")  # noqa: T201

        # Null check
        nulls = df.isnull().sum()
        null_cols = nulls[nulls > 0]
        if len(null_cols) == 0:
            print("  [PASS] No null values")  # noqa: T201
        else:
            print(f"  [WARN] Nulls found: {dict(null_cols)}")  # noqa: T201

        # Row count
        print(f"  [INFO] Row count: {len(df):,}")  # noqa: T201

    @app.command()
    def drift(
        reference: str = typer.Argument(
            ..., help="Path to the reference (baseline) CSV file",
        ),
        current: str = typer.Argument(
            ..., help="Path to the current CSV file to compare",
        ),
        method: str = typer.Option(
            "psi",
            help=(
                "Drift detection method: ks, psi, kl, chi2, "
                "js, wasserstein, or auto (default: psi)"
            ),
        ),
    ) -> None:
        """Compare two datasets for distribution drift.

        Compares numeric columns shared between reference and current
        files, reporting per-column drift statistics.

        Example::

            mltk drift train.csv production.csv --method ks
        """
        import pandas as pd

        ref_path = Path(reference)
        cur_path = Path(current)

        if not ref_path.exists():
            print(f"Reference file not found: {reference}")  # noqa: T201
            raise typer.Exit(1)
        if not cur_path.exists():
            print(f"Current file not found: {current}")  # noqa: T201
            raise typer.Exit(1)

        ref_df = pd.read_csv(ref_path)
        cur_df = pd.read_csv(cur_path)

        from mltk.data.drift import assert_no_drift

        print(f"MLTK Drift Analysis: {reference} vs {current}")  # noqa: T201
        print(f"  Method: {method}")  # noqa: T201
        print()  # noqa: T201

        # Compare numeric columns
        common_cols = [
            c for c in ref_df.columns
            if c in cur_df.columns and pd.api.types.is_numeric_dtype(ref_df[c])
        ]

        for col in common_cols:
            try:
                result = assert_no_drift(
                    ref_df[col], cur_df[col], method=method
                )
                status = "OK" if result.passed else "DRIFT"
                stat = result.details.get("statistic", 0)
                print(f"  {col:20s} | {stat:.4f} | {status}")  # noqa: T201
            except Exception:
                # Drift detected (exception raised)
                print(f"  {col:20s} | DRIFT DETECTED")  # noqa: T201

    @app.command()
    def score() -> None:
        """Show the ML Test Score rubric and how to generate scores.

        The ML Test Score uses Google's 28-test rubric to quantify
        testing maturity across data, model, infrastructure, and
        monitoring.
        """
        print("ML Test Score")  # noqa: T201
        print("Run: pytest --mltk-report to generate scores")  # noqa: T201
        print()  # noqa: T201
        print("Categories (Google 28-test rubric):")  # noqa: T201
        print("  Data:           schema, distribution, drift, freshness, PII, labels")  # noqa: T201
        print("  Model:          metrics, regression, slicing, calibration, bias, adversarial")  # noqa: T201
        print("  Infrastructure: reproducibility, pipeline, contract, latency, throughput")  # noqa: T201
        print("  Monitoring:     drift monitoring, degradation, SLA, alerts")  # noqa: T201

    @app.command()
    def doctor() -> None:
        """Diagnose ML testing environment.

        Runs a series of checks on your environment and reports any issues
        with dependencies, config files, directories, and plugin registration.
        """
        from mltk.doctor import diagnose

        results = diagnose()

        # Status symbol map
        symbols = {"OK": "[OK  ]", "WARN": "[WARN]", "FAIL": "[FAIL]"}

        print("mltk doctor — environment diagnostics")  # noqa: T201
        print("=" * 60)  # noqa: T201

        ok_count = 0
        warn_count = 0
        fail_count = 0

        for r in results:
            symbol = symbols.get(r.status, f"[{r.status}]")
            print(f"{symbol} {r.name}: {r.message}")  # noqa: T201
            if r.fix_hint:
                print(f"         -> {r.fix_hint}")  # noqa: T201
            if r.status == "OK":
                ok_count += 1
            elif r.status == "WARN":
                warn_count += 1
            elif r.status == "FAIL":
                fail_count += 1

        print("=" * 60)  # noqa: T201
        print(  # noqa: T201
            f"Summary: {ok_count} OK, {warn_count} warnings, {fail_count} failures"
        )

        if fail_count > 0:
            raise typer.Exit(1)

    @app.command()
    def test(yaml_path: str) -> None:
        """Run YAML-defined test suite.

        Loads test definitions from a YAML file and executes them against
        the configured data source. Prints pass/fail results to stdout.

        Example YAML::

            data_source: data/features.csv
            tests:
              - name: No nulls
                assertion: no_nulls
        """
        from mltk.testdefs import load_test_suite, run_test_suite

        p = Path(yaml_path)
        if not p.exists():
            print(f"Test definition file not found: {yaml_path}")  # noqa: T201
            raise typer.Exit(1)

        suite = load_test_suite(str(p))
        results = run_test_suite(suite)

        passed = sum(1 for r in results if r.passed)
        total = len(results)

        print(f"mltk test — {yaml_path}")  # noqa: T201
        print(f"Results: {passed}/{total} passed")  # noqa: T201
        print()  # noqa: T201

        for r in results:
            status = "PASS" if r.passed else "FAIL"
            print(f"  [{status}] {r.name}: {r.message}")  # noqa: T201

        if passed < total:
            raise typer.Exit(1)

    @app.command("model-card")
    def model_card_cmd(
        results_json: str,
        model_name: str = "AI Model",
        model_version: str = "1.0",
        output: str = "model-card.md",
    ) -> None:
        """Generate a Google Model Card from test results JSON.

        Reads a JSON file produced by ``--mltk-export-json`` and writes a
        Markdown model card covering metrics, fairness, calibration, robustness,
        data quality, and known limitations.

        Args:
            results_json: Path to JSON file with mltk test results.
            model_name: Display name of the model (default: "AI Model").
            model_version: Version string for the model (default: "1.0").
            output: Destination path for the Markdown file (default: model-card.md).
        """
        from mltk.report import generate_model_card

        p = Path(results_json)
        if not p.exists():
            print(f"Results file not found: {results_json}")  # noqa: T201
            raise typer.Exit(1)

        card_path = generate_model_card(
            results_path=p,
            model_name=model_name,
            model_version=model_version,
            output_path=output,
        )
        print(f"Model card generated: {card_path}")  # noqa: T201

    @app.command()
    def compliance(
        results_json: str,
        risk_level: str = "high",
        system_name: str = "AI System",
    ) -> None:
        """Generate EU AI Act compliance report.

        Reads test results from a JSON file (produced by --mltk-export-json),
        assesses compliance based on the specified risk level, and writes
        a markdown report to the configured report directory.

        Args:
            results_json: Path to JSON file with test results.
            risk_level: EU AI Act risk level — "high", "limited", or "minimal".
            system_name: Name of the AI system being assessed.
        """
        from mltk.compliance import generate_compliance_report

        p = Path(results_json)
        if not p.exists():
            print(f"Results file not found: {results_json}")  # noqa: T201
            raise typer.Exit(1)

        report_path = generate_compliance_report(
            results_path=str(p),
            risk_level=risk_level,
            system_name=system_name,
        )

        print(f"EU AI Act compliance report generated: {report_path}")  # noqa: T201

    # Contract subcommands
    contract_app = typer.Typer(name="contract", help="Data contract operations")

    @contract_app.command("init")
    def contract_init() -> None:
        """Scaffold an example data contract YAML file."""
        example = """\
name: my_dataset
version: "1.0"

columns:
  id:
    type: int64
    nullable: false
    unique: true
  value:
    type: float64
    nullable: false
    range: [0, 100]
  label:
    type: int64
    nullable: false

quality:
  min_rows: 100
"""
        Path("contract.yaml").write_text(example)
        print("Created contract.yaml")  # noqa: T201

    @contract_app.command("validate")
    def contract_validate(
        data: str,
        contract: str = typer.Option("contract.yaml", help="Contract YAML path"),
    ) -> None:
        """Validate a data file against a contract."""
        import pandas as pd

        from mltk.contracts import validate_data

        p = Path(data)
        if not p.exists():
            print(f"File not found: {data}")  # noqa: T201
            raise typer.Exit(1)

        df = pd.read_csv(p) if p.suffix != ".parquet" else pd.read_parquet(p)
        suite = validate_data(df, contract)

        print(f"Contract validation: {suite.passed_count}/{suite.total} passed")  # noqa: T201
        for r in suite.results:
            status = "PASS" if r.passed else "FAIL"
            print(f"  [{status}] {r.name}: {r.message}")  # noqa: T201

    @contract_app.command("generate-tests")
    def contract_generate(
        contract: str = typer.Argument("contract.yaml"),
        output: str = typer.Option("tests/test_contract_gen.py", help="Output test file"),
    ) -> None:
        """Generate pytest test file from a data contract."""
        from mltk.contracts.generator import generate_tests_from_contract

        cp = Path(contract)
        if not cp.exists():
            print(f"Contract not found: {contract}")  # noqa: T201
            raise typer.Exit(1)

        result = generate_tests_from_contract(cp, output)
        print(f"Generated: {result}")  # noqa: T201

    app.add_typer(contract_app)

    # Docs subcommands
    docs_app = typer.Typer(name="docs", help="Documentation server")

    @docs_app.command("serve")
    def docs_serve(
        port: int = typer.Option(8000, help="Port to serve on"),
        host: str = typer.Option("127.0.0.1", help="Host to bind to"),
    ) -> None:
        """Serve documentation locally with hot reload."""
        import os
        import subprocess
        import sys

        port = int(os.environ.get("MLTK_DOCS_PORT", str(port)))
        host = os.environ.get("MLTK_DOCS_HOST", host)

        mkdocs_yml = Path("mkdocs.yml")
        if not mkdocs_yml.exists():
            print(  # noqa: T201
                "mkdocs.yml not found in current directory. "
                "Run this command from the project root."
            )
            raise typer.Exit(1)

        print(f"Serving docs at http://{host}:{port}")  # noqa: T201
        print("Press Ctrl+C to stop")  # noqa: T201
        subprocess.run(  # noqa: S603
            [sys.executable, "-m", "mkdocs", "serve",
             "-a", f"{host}:{port}"],
            check=False,
        )

    @docs_app.command("build")
    def docs_build(
        output: str = typer.Option("site", help="Output directory"),
    ) -> None:
        """Build static HTML documentation."""
        import subprocess
        import sys

        mkdocs_yml = Path("mkdocs.yml")
        if not mkdocs_yml.exists():
            print(  # noqa: T201
                "mkdocs.yml not found in current directory. "
                "Run this command from the project root."
            )
            raise typer.Exit(1)

        result = subprocess.run(  # noqa: S603
            [sys.executable, "-m", "mkdocs", "build",
             "-d", output],
            check=False,
        )
        if result.returncode == 0:
            print(f"Docs built to: {output}/")  # noqa: T201
            print(  # noqa: T201
                f"Open: {Path(output) / 'index.html'}"
            )
        else:
            raise typer.Exit(1)

    @docs_app.command("open")
    def docs_open(
        port: int = typer.Option(8000, help="Port to serve on"),
    ) -> None:
        """Build docs, start a local server, and open in browser."""
        import http.server
        import os
        import subprocess
        import sys
        import threading
        import webbrowser

        port = int(os.environ.get("MLTK_DOCS_PORT", str(port)))

        # Build first
        mkdocs_yml = Path("mkdocs.yml")
        if not mkdocs_yml.exists():
            print(  # noqa: T201
                "mkdocs.yml not found in current directory. "
                "Run this command from the project root."
            )
            raise typer.Exit(1)

        print("Building docs...")  # noqa: T201
        build_result = subprocess.run(  # noqa: S603
            [sys.executable, "-m", "mkdocs", "build", "-d", "site"],
            check=False,
        )
        if build_result.returncode != 0:
            print(  # noqa: T201
                "Docs build failed. Check mkdocs output above for details."
            )
            raise typer.Exit(1)

        # Serve with Python's built-in HTTP server
        site_dir = Path("site")
        os.chdir(str(site_dir))

        handler = http.server.SimpleHTTPRequestHandler
        server = http.server.HTTPServer(("127.0.0.1", port), handler)

        url = f"http://127.0.0.1:{port}"
        print(f"Docs available at: {url}")  # noqa: T201
        print("Press Ctrl+C to stop")  # noqa: T201

        # Open browser after a short delay
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()

        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped.")  # noqa: T201
            server.shutdown()

    app.add_typer(docs_app)

    # Registry subcommands
    registry_app = typer.Typer(name="registry", help="Test resource registry")

    @registry_app.command("push")
    def registry_push(
        name: str,
        source: str = typer.Option(".", help="Source directory to push"),
        description: str = typer.Option("", help="Collection description"),
        version: str = typer.Option("1.0", help="Collection version"),
        tags: str = typer.Option("", help="Comma-separated tags"),
    ) -> None:
        """Push a directory of test files to the registry as a named collection."""
        from mltk.registry import save_collection

        tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []
        col_dir = save_collection(
            name,
            source_dir=source,
            description=description,
            version=version,
            tags=tag_list,
        )
        print(f"Saved collection '{name}' to {col_dir}")  # noqa: T201

    @registry_app.command("pull")
    def registry_pull(
        name: str,
        target: str = typer.Option(".", help="Target directory to restore into"),
    ) -> None:
        """Pull a named collection from the registry into a local directory."""
        from mltk.registry import load_collection

        try:
            loaded = load_collection(name, target_dir=target)
            print(f"Loaded collection '{name}' to {loaded}")  # noqa: T201
        except ValueError as exc:
            print(f"Error: {exc}")  # noqa: T201
            raise typer.Exit(1) from exc

    @registry_app.command("list")
    def registry_list() -> None:
        """List all collections in the registry."""
        from mltk.registry import list_collections

        manifests = list_collections()
        if not manifests:
            print("Registry is empty. Use 'mltk registry push <name>' to add a collection.")  # noqa: T201
            return

        print(f"{'NAME':<20} {'VERSION':<10} {'TAGS':<30} DESCRIPTION")  # noqa: T201
        print("-" * 80)  # noqa: T201
        for m in manifests:
            tags_str = ", ".join(m.tags) if m.tags else ""
            desc = m.description[:40] + "..." if len(m.description) > 40 else m.description
            print(f"{m.name:<20} {m.version:<10} {tags_str:<30} {desc}")  # noqa: T201

    app.add_typer(registry_app)

    # Notify subcommands
    notify_app = typer.Typer(name="notify", help="Send notifications")

    @notify_app.command("slack")
    def notify_slack_cmd(
        webhook_url: str = typer.Option(
            ...,
            envvar="MLTK_SLACK_WEBHOOK",
            help="Slack incoming webhook URL",
        ),
        results_json: str = typer.Option(
            None,
            help="Path to mltk results JSON file (from --mltk-export-json)",
        ),
        message: str = typer.Option(None, help="Custom plain-text message to send"),
    ) -> None:
        """Send test results (or a custom message) to Slack.

        Reads a JSON results file produced by ``--mltk-export-json`` and posts
        a formatted Block Kit summary to the configured Slack webhook.
        Use ``--message`` to send a standalone plain-text notification instead.

        Example::

            mltk notify slack --results-json mltk-reports/results.json
            mltk notify slack --message "Nightly ML tests passed"
        """
        import json as _json

        from mltk.core.result import Severity, TestResult, TestSuite
        from mltk.integrations.slack import notify_slack

        suite: TestSuite | None = None

        if results_json is not None:
            p = Path(results_json)
            if not p.exists():
                print(f"Results file not found: {results_json}")  # noqa: T201
                raise typer.Exit(1)

            raw = _json.loads(p.read_text())
            # Support both array format (from --mltk-export-json) and
            # dict format {"results": [...]} for forward compatibility
            items_list = raw if isinstance(raw, list) else raw.get("results", [])
            suite = TestSuite()
            for item in items_list:
                suite.add(
                    TestResult(
                        name=item.get("name", "unknown"),
                        passed=item.get("passed", False),
                        severity=Severity(item.get("severity", "critical")),
                        message=item.get("message", ""),
                        details=item.get("details", {}),
                        duration_ms=item.get("duration_ms", 0.0),
                    )
                )

        if suite is None and message is None:
            print(  # noqa: T201
                "Nothing to send. Provide --results-json, --message, or both. "
                "Example: mltk notify slack --message 'Tests passed'"
            )
            raise typer.Exit(1)

        ok = notify_slack(webhook_url=webhook_url, suite=suite, message=message)
        if ok:
            print("Slack notification sent")  # noqa: T201
        else:
            print("Failed to send Slack notification")  # noqa: T201
            raise typer.Exit(1)

    app.add_typer(notify_app)

    @app.command()
    def chat(
        results_json: str = typer.Option(
            None, "--results-json",
            help="Path to results JSON from --mltk-export-json",
        ),
    ) -> None:
        """Interactive Q&A about test results."""
        from mltk.chat import chat_repl

        chat_repl(results_json)

    @app.command()
    def server(
        host: str = typer.Option("127.0.0.1", help="Host"),
        port: int = typer.Option(8080, help="Port"),
        db: str = typer.Option("mltk_server.db", help="SQLite database path"),
    ) -> None:
        """Start the mltk server platform."""
        try:
            import uvicorn
        except ImportError as err:
            print(  # noqa: T201
                "uvicorn is required for the mltk server. "
                "Install it with: pip install mltk[server]"
            )
            raise typer.Exit(1) from err
        from mltk.server import create_app

        application = create_app(db_path=db)
        print(f"mltk server at http://{host}:{port}")  # noqa: T201
        uvicorn.run(application, host=host, port=port)

    @app.command("server-create-key")
    def server_create_key(
        project: str = typer.Option("default", help="Project name to associate with this key"),
        db: str = typer.Option("mltk_server.db", help="SQLite database path"),
    ) -> None:
        """Generate an API key for the mltk server.

        Writes the key hash to the database and prints the raw key to stdout.
        Store the printed key securely — it cannot be retrieved again.

        Example::

            mltk server-create-key --project my-project
        """
        from mltk.server.auth import generate_api_key, hash_key
        from mltk.server.storage import Storage

        raw_key = generate_api_key()
        key_hash = hash_key(raw_key)

        storage = Storage(db)
        storage.save_api_key(key_hash, project)

        print(f"API key created for project '{project}':")  # noqa: T201
        print(f"  {raw_key}")  # noqa: T201
        print("Store this key securely — it will not be shown again.")  # noqa: T201

    @app.command("fda-audit")
    def fda_audit(
        results_json: str,
        system_name: str = "AI System",
        operator: str = "QA Engineer",
        output: str = "fda-audit-trail.md",
    ) -> None:
        """Generate FDA 21 CFR Part 11 audit trail."""
        from mltk.compliance import generate_fda_audit_trail

        p = Path(results_json)
        if not p.exists():
            print(f"Results file not found: {results_json}")  # noqa: T201
            raise typer.Exit(1)

        out = generate_fda_audit_trail(
            results_path=str(p),
            system_name=system_name,
            operator=operator,
            output_path=output,
        )
        print(f"FDA audit trail generated: {out}")  # noqa: T201

    @app.command("compliance-pdf")
    def compliance_pdf(
        html_file: str,
        output: str = typer.Option(None, help="Output path"),
    ) -> None:
        """Convert HTML compliance report to print-ready PDF."""
        from mltk.compliance import export_compliance_pdf

        p = Path(html_file)
        if not p.exists():
            print(f"HTML file not found: {html_file}")  # noqa: T201
            raise typer.Exit(1)

        out = export_compliance_pdf(html_path=str(p), output_path=output)
        print(f"Compliance PDF exported: {out}")  # noqa: T201

    @app.command()
    def compliance_gap(
        results_json: str,
        framework: str = "all",
    ) -> None:
        """Run compliance gap analysis across frameworks.

        Reads test results from a JSON file (produced by --mltk-export-json)
        and identifies which compliance requirements lack test coverage.

        Supported frameworks: all, eu-ai-act, owasp, nist-rmf, iso-42001, fda, hipaa, sr-11-7.

        Args:
            results_json: Path to JSON file with mltk test results.
            framework: Framework to analyse — "all" runs every framework.
        """
        import json

        p = Path(results_json)
        if not p.exists():
            print(f"Results file not found: {results_json}")  # noqa: T201
            raise typer.Exit(1)

        raw = json.loads(p.read_text())
        results: list[dict] = (
            raw if isinstance(raw, list) else raw.get("results", [])
        )

        valid_frameworks = {
            "all", "eu-ai-act", "owasp", "nist-rmf",
            "iso-42001", "fda", "hipaa", "sr-11-7",
        }
        fw = framework.lower().strip()
        if fw not in valid_frameworks:
            print(  # noqa: T201
                f"Unknown framework: {framework!r}. "
                f"Valid options: {sorted(valid_frameworks)}"
            )
            raise typer.Exit(1)

        print("=== mltk Compliance Gap Analysis ===")  # noqa: T201
        print()  # noqa: T201

        if fw in ("all", "eu-ai-act"):
            _gap_eu_ai_act(results)

        if fw in ("all", "owasp"):
            _gap_owasp(results)

        if fw in ("all", "nist-rmf"):
            _gap_nist(results)

        if fw in ("all", "iso-42001"):
            _gap_iso(results)

        if fw in ("all", "fda"):
            _gap_fda(results)

    # ------------------------------------------------------------------
    # Gap analysis helpers (one per framework)
    # ------------------------------------------------------------------

    def _gap_symbols() -> tuple[str, str, str]:
        """Return (pass_sym, fail_sym, dash) safe for the current terminal encoding."""
        import sys

        encoding = getattr(sys.stdout, "encoding", "") or "ascii"
        try:
            "\u2713\u2717\u2014".encode(encoding)
            return ("\u2713", "\u2717", "\u2014")
        except (UnicodeEncodeError, LookupError):
            return ("[PASS]", "[MISS]", "--")

    def _gap_eu_ai_act(results: list[dict]) -> None:
        """Print EU AI Act gap analysis."""
        from mltk.compliance.eu_ai_act import (
            ARTICLE_META,
            find_gaps,
            map_results_to_articles,
        )

        ok, miss, dash = _gap_symbols()
        grouped = map_results_to_articles(results)
        gaps = find_gaps(results, "high")
        total = len(ARTICLE_META)
        covered = total - len(gaps)

        print("EU AI Act (high risk):")  # noqa: T201
        for meta in ARTICLE_META:
            article = meta["article"]
            title = meta["title"]
            tests = grouped.get(article, [])
            count = len(tests)
            sym = miss if article in gaps else ok
            print(f"  {sym} {article} {dash} {title} ({count} tests)")  # noqa: T201
        print(f"  Coverage: {covered}/{total} articles ({_pct(covered, total)}%)")  # noqa: T201
        print()  # noqa: T201

    def _gap_owasp(results: list[dict]) -> None:
        """Print OWASP LLM Top 10 gap analysis."""
        from mltk.compliance.owasp_llm import OWASP_LLM_IDS, owasp_llm_scan

        ok, miss, dash = _gap_symbols()
        scan = owasp_llm_scan(results)
        total = len(OWASP_LLM_IDS)
        covered = sum(1 for entry in scan.values() if entry["covered"])

        print("OWASP LLM Top 10:")  # noqa: T201
        for owasp_id in OWASP_LLM_IDS:
            entry = scan[owasp_id]
            title = entry["title"]
            count = len(entry["tests"])
            sym = ok if entry["covered"] else miss
            print(f"  {sym} {owasp_id} {dash} {title} ({count} tests)")  # noqa: T201
        print(f"  Coverage: {covered}/{total} categories ({_pct(covered, total)}%)")  # noqa: T201
        print()  # noqa: T201

    def _gap_nist(results: list[dict]) -> None:
        """Print NIST AI RMF gap analysis."""
        try:
            from mltk.compliance.nist_ai_rmf import (
                NIST_RMF_FUNCTION_IDS,
                NIST_RMF_FUNCTIONS,
                map_results_to_measures,
            )
            from mltk.compliance.nist_ai_rmf import (
                find_gaps as nist_find_gaps,
            )
        except ImportError:
            # Module being built by another agent -- show placeholder.
            print("NIST AI RMF:")  # noqa: T201
            print("  (module not yet installed)")  # noqa: T201
            print()  # noqa: T201
            return

        ok, miss, dash = _gap_symbols()
        gaps = nist_find_gaps(results)
        grouped = map_results_to_measures(results)
        total = len(NIST_RMF_FUNCTION_IDS)
        covered = total - len(gaps)

        print("NIST AI RMF:")  # noqa: T201
        for func_id in NIST_RMF_FUNCTION_IDS:
            title = NIST_RMF_FUNCTIONS[func_id]["title"]
            tests = grouped.get(func_id, [])
            count = len(tests)
            sym = miss if func_id in gaps else ok
            print(f"  {sym} {title} {dash} {count} tests")  # noqa: T201
        print(f"  Coverage: {covered}/{total} functions ({_pct(covered, total)}%)")  # noqa: T201
        print()  # noqa: T201

    def _gap_iso(results: list[dict]) -> None:
        """Print ISO 42001 gap analysis."""
        try:
            from mltk.compliance.iso_42001 import (
                ANNEX_A_CONTROLS,
                ANNEX_A_IDS,
                map_results_to_clauses,
            )
            from mltk.compliance.iso_42001 import (
                find_gaps as iso_find_gaps,
            )
        except ImportError:
            print("ISO 42001:")  # noqa: T201
            print("  (module not yet installed)")  # noqa: T201
            print()  # noqa: T201
            return

        ok, miss, dash = _gap_symbols()
        gaps = iso_find_gaps(results)
        grouped = map_results_to_clauses(results)
        total = len(ANNEX_A_IDS)
        covered = total - len(gaps)

        print("ISO 42001:")  # noqa: T201
        for clause_id in ANNEX_A_IDS:
            title = ANNEX_A_CONTROLS[clause_id]["title"]
            tests = grouped.get(clause_id, [])
            count = len(tests)
            sym = miss if clause_id in gaps else ok
            print(f"  {sym} {clause_id} {dash} {title} ({count} tests)")  # noqa: T201
        print(f"  Coverage: {covered}/{total} clauses ({_pct(covered, total)}%)")  # noqa: T201
        print()  # noqa: T201

    def _gap_fda(results: list[dict]) -> None:
        """Print FDA coverage check (simple prefix match)."""
        ok, miss, _dash = _gap_symbols()
        fda_prefixes = ("fda.", "pipeline.")
        fda_tests = [
            r for r in results
            if str(r.get("name", "")).startswith(fda_prefixes)
        ]
        count = len(fda_tests)
        sym = ok if count > 0 else miss

        print("FDA (21 CFR Part 11):")  # noqa: T201
        print(f"  {sym} Test coverage: {count} tests with fda.*/pipeline.* prefix")  # noqa: T201
        if count > 0:
            for t in fda_tests:
                passed_label = "PASS" if t.get("passed") else "FAIL"
                print(f"    [{passed_label}] {t.get('name', '?')}")  # noqa: T201
        else:
            print("  No FDA/pipeline tests found -- add tests with fda.* or pipeline.* names")  # noqa: T201
        print()  # noqa: T201

    def _pct(covered: int, total: int) -> int:
        """Return integer percentage, avoiding division by zero."""
        return round(covered * 100 / total) if total > 0 else 0

    @app.command()
    def grafana_export(
        output: str = "mltk-grafana-dashboard.json",
        datasource: str = "mltk-sqlite",
        title: str = "mltk Test Results",
    ) -> None:
        """Export a Grafana dashboard JSON for mltk metrics.

        Generates a 4-panel dashboard (pass/fail trend, duration heatmap,
        failure rate by module, latest run summary) that can be imported
        into any Grafana instance.

        Args:
            output: Output file path for the dashboard JSON.
            datasource: Grafana datasource name.
            title: Dashboard title.
        """
        from mltk.integrations.grafana import export_grafana_dashboard

        path = export_grafana_dashboard(
            output_path=output,
            datasource=datasource,
        )
        print(f"Grafana dashboard exported: {path}")  # noqa: T201

    @app.command("scan-model")
    def scan_model(
        model: str = typer.Option(
            ..., help="Path to model file",
        ),
        data: str = typer.Option(
            ..., help="Path to CSV/parquet data",
        ),
        target: str = typer.Option(
            ..., help="Target column name",
        ),
        sensitive: str = typer.Option(
            "",
            help="Comma-separated sensitive columns",
        ),
        output: str = typer.Option(
            "",
            help="Path for generated test file",
        ),
        junit_xml: str = typer.Option(
            "",
            help="Path for JUnit XML output",
        ),
        export_json: str = typer.Option(
            "",
            help=(
                "Export scan results as JSON"
                " to file path"
            ),
        ),
    ) -> None:
        """Scan a model for issues and generate tests.

        Loads a serialized model and a dataset, runs every
        applicable scanner (bias, slicing, calibration,
        robustness, leakage, data quality, drift, overfit),
        and prints a summary of findings with severity levels.

        Scanners that cannot run (e.g., CalibrationScanner
        when the model has no predict_proba) are skipped
        automatically.

        Optionally generates a self-contained pytest file
        (--output) and/or JUnit XML report (--junit-xml)
        from the findings.

        Exit codes::

            0  No findings (model looks clean)
            1  One or more findings detected
            2  Scan error (model/data could not be loaded)

        Example::

            mltk scan-model \\
                --model model.pkl \\
                --data test.csv \\
                --target label \\
                --sensitive age,gender \\
                --output tests/test_scan.py
        """
        import sys as _sys

        import pandas as pd

        from mltk.scan.loader import load_model as _load

        # -- Load model -----------------------------------
        model_path = Path(model)
        if not model_path.exists():
            print(  # noqa: T201
                f"Model file not found: {model}"
            )
            raise typer.Exit(2)

        try:
            loaded = _load(model_path)
        except (ValueError, ImportError, TypeError) as exc:
            print(f"Cannot load model: {exc}")  # noqa: T201
            raise typer.Exit(2) from exc

        # -- Load data ------------------------------------
        data_path = Path(data)
        if not data_path.exists():
            print(f"Data file not found: {data}")  # noqa: T201
            raise typer.Exit(2)

        try:
            if data_path.suffix == ".parquet":
                df = pd.read_parquet(data_path)
            else:
                df = pd.read_csv(data_path)
        except Exception as exc:  # noqa: BLE001
            print(f"Cannot read data: {exc}")  # noqa: T201
            raise typer.Exit(2) from exc

        if target not in df.columns:
            print(  # noqa: T201
                f"Target column '{target}' not found. "
                f"Available: {list(df.columns)}"
            )
            raise typer.Exit(2)

        # -- Prepare X, y --------------------------------
        y = df[target].values
        X = df.drop(columns=[target])

        # -- Parse sensitive columns ----------------------
        sensitive_cols: list[str] = []
        if sensitive:
            sensitive_cols = [
                c.strip()
                for c in sensitive.split(",")
                if c.strip()
            ]
            missing = [
                c for c in sensitive_cols
                if c not in X.columns
            ]
            if missing:
                print(  # noqa: T201
                    f"Sensitive columns not found "
                    f"in data: {missing}"
                )
                raise typer.Exit(2)

        # -- Run scan -------------------------------------
        try:
            from mltk.scan import scan as _scan

            report = _scan(
                loaded.predict_fn,
                X,
                y,
                sensitive_columns=sensitive_cols,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"Scan failed: {exc}")  # noqa: T201
            raise typer.Exit(2) from exc

        # -- Print summary --------------------------------
        print(report.summary())  # noqa: T201

        # -- Optional: write test file --------------------
        if output:
            try:
                report.to_test_file(output)
                print(  # noqa: T201
                    f"Test file written: {output}"
                )
            except Exception as exc:  # noqa: BLE001
                print(  # noqa: T201
                    f"Failed to write test file: {exc}"
                )

        # -- Optional: write JUnit XML --------------------
        if junit_xml:
            try:
                report.to_junit(junit_xml)
                print(  # noqa: T201
                    f"JUnit XML written: {junit_xml}"
                )
            except Exception as exc:  # noqa: BLE001
                print(  # noqa: T201
                    f"Failed to write JUnit XML: {exc}"
                )

        # -- Optional: write JSON --------------------------
        if export_json:
            try:
                report.to_json(export_json)
                print(  # noqa: T201
                    f"JSON written: {export_json}"
                )
            except Exception as exc:  # noqa: BLE001
                print(  # noqa: T201
                    "Failed to write JSON:"
                    f" {exc}"
                )

        _sys.exit(report.exit_code)

    @app.command("list")
    def list_assertions(
        filter_keyword: str = typer.Argument(
            "", help="Filter by name, module, or description",
        ),
        output_format: str = typer.Option(
            "table", "--format",
            help="Output format: 'table' (human-readable) or 'json' (machine-readable)",
        ),
    ) -> None:
        """List all available mltk assertions.

        Scans every mltk subpackage for ``assert_*`` functions and
        prints them grouped by category.  Use ``--format json`` for
        machine-readable output.

        Examples::

            mltk list
            mltk list drift
            mltk list --format json
        """
        import json as _json

        from mltk.cli._discovery import discover_assertions

        fmt = output_format.strip().lower()
        if fmt not in ("table", "json"):
            print(  # noqa: T201
                f"Unknown format: {output_format!r}. "
                "Use 'table' or 'json'."
            )
            raise typer.Exit(1)

        entries = discover_assertions(filter_keyword)
        total = sum(len(v) for v in entries.values())

        if fmt == "json":
            payload: dict = {"total": total, "modules": {}}
            for category, items in sorted(entries.items()):
                payload["modules"][category] = [
                    {
                        "name": e["name"],
                        "module": e["module"],
                        "doc": e["doc"],
                    }
                    for e in items
                ]
            print(_json.dumps(payload, indent=2))  # noqa: T201
            return

        # table format
        print(f"mltk assertions ({total} total)")  # noqa: T201
        print()  # noqa: T201
        for category, items in sorted(entries.items()):
            print(f"{category} ({len(items)}):")  # noqa: T201
            for e in items:
                name = e["name"]
                mod = e["module"]
                doc = e["doc"]
                print(  # noqa: T201
                    f"  {name:<40s} {mod:<30s} {doc}"
                )
            print()  # noqa: T201

    app()
