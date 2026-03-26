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
        print("CLI requires: pip install mltk[cli]")  # noqa: T201
        raise SystemExit(1) from err

    app = typer.Typer(
        name="mltk",
        help="ML Test Kit -- pytest for ML. Unified testing across the entire ML lifecycle.",
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
    def scan(path: str) -> None:
        """Quick data quality scan on a CSV/Parquet file."""
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
        reference: str,
        current: str,
        method: str = "psi",
    ) -> None:
        """Compare two datasets for distribution drift."""
        import pandas as pd

        ref_path = Path(reference)
        cur_path = Path(current)

        if not ref_path.exists() or not cur_path.exists():
            print("Both files must exist")  # noqa: T201
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
        """Show ML Test Score (run pytest first to generate results)."""
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
            print("mkdocs.yml not found in current directory")  # noqa: T201
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
            print("mkdocs.yml not found in current directory")  # noqa: T201
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
            print("mkdocs.yml not found in current directory")  # noqa: T201
            raise typer.Exit(1)

        print("Building docs...")  # noqa: T201
        build_result = subprocess.run(  # noqa: S603
            [sys.executable, "-m", "mkdocs", "build", "-d", "site"],
            check=False,
        )
        if build_result.returncode != 0:
            print("Build failed")  # noqa: T201
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
            suite = TestSuite()
            for item in raw.get("results", []):
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
            print("Provide --results-json or --message (or both)")  # noqa: T201
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
            print("Server requires: pip install mltk[server]")  # noqa: T201
            raise typer.Exit(1) from err
        from mltk.server import create_app

        application = create_app(db_path=db)
        print(f"mltk server at http://{host}:{port}")  # noqa: T201
        uvicorn.run(application, host=host, port=port)

    app()
