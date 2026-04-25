"""Container scanning CLI commands.

Provides ``mltk container scan`` -- a Typer sub-app that runs
vulnerability and secret scans on container images using the
:mod:`mltk.container` assertions (backed by Trivy).

Exit codes::

    0  All scans passed (no CVEs above thresholds, no secrets)
    1  One or more scans failed
    2  Scan error (Trivy missing, image unavailable, etc.)
"""
from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from typing import Annotated, Any

import typer

app = typer.Typer(
    name="container",
    help="Container image scanning commands.",
)


@app.callback()
def _container_callback() -> None:
    """Container image scanning commands."""


@app.command("scan")
def container_scan(
    image: Annotated[
        str,
        typer.Argument(
            help="Container image reference (e.g. alpine:3.18)",
        ),
    ],
    max_critical: Annotated[
        int,
        typer.Option(
            "--max-critical",
            help="Max allowed CRITICAL CVEs",
        ),
    ] = 0,
    max_high: Annotated[
        int,
        typer.Option(
            "--max-high",
            help="Max allowed HIGH CVEs",
        ),
    ] = 0,
    severity_floor: Annotated[
        str,
        typer.Option(
            "--severity-floor",
            help="Minimum severity to report",
        ),
    ] = "MEDIUM",
    junit_xml: Annotated[
        str | None,
        typer.Option(
            "--junit-xml",
            help="Write JUnit XML report to path",
        ),
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Output results as JSON",
        ),
    ] = False,
) -> None:
    """Scan a container image for vulnerabilities and secrets.

    Runs :func:`assert_container_vulnerabilities` and
    :func:`assert_no_secrets_in_image` against the given image.
    Both assertions are wrapped so their pass/fail state is
    surfaced as an exit code rather than an exception.
    """
    from rich.console import Console

    console = Console(stderr=True)

    try:
        from mltk.container.assertions import (
            assert_container_vulnerabilities,
            assert_no_secrets_in_image,
        )
    except ImportError:
        console.print(
            "[red]mltk[container] not installed. "
            "Run: pip install mltk[container][/red]"
        )
        raise typer.Exit(1) from None

    from mltk.core.assertion import MltkAssertionError  # noqa: PLC0415

    results: list[Any] = []

    try:
        vuln_result = assert_container_vulnerabilities(
            image,
            max_critical=max_critical,
            max_high=max_high,
            severity_floor=severity_floor,
        )
        results.append(vuln_result)
    except MltkAssertionError as exc:
        results.append(exc.result)
    except Exception as exc:  # noqa: BLE001
        console.print(
            f"[red]Vulnerability scan error: {exc}[/red]"
        )
        raise typer.Exit(2) from exc

    try:
        secret_result = assert_no_secrets_in_image(image)
        results.append(secret_result)
    except MltkAssertionError as exc:
        results.append(exc.result)
    except Exception as exc:  # noqa: BLE001
        console.print(
            f"[red]Secret scan error: {exc}[/red]"
        )
        raise typer.Exit(2) from exc

    all_passed = all(r.passed for r in results)

    # Record to Prometheus if available (no-op when mltk[metrics] not installed)
    try:
        from mltk.server.metrics import record_container_scan  # noqa: PLC0415
        vuln_details = next(
            (r.details for r in results if r.name == "container.vulnerabilities"), {}
        )
        record_container_scan(
            critical=int(vuln_details.get("critical_count", 0)),
            high=int(vuln_details.get("high_count", 0)),
            medium=int(vuln_details.get("medium_count", 0)),
        )
    except Exception:  # noqa: BLE001
        pass

    if json_output:
        output = {
            "image": image,
            "passed": all_passed,
            "results": [
                {
                    "name": r.name,
                    "passed": r.passed,
                    "message": r.message,
                    "details": r.details,
                }
                for r in results
            ],
        }
        print(json.dumps(output, indent=2))  # noqa: T201
    else:
        for r in results:
            status = (
                "[green]PASS[/green]"
                if r.passed
                else "[red]FAIL[/red]"
            )
            console.print(f"{status} {r.name}: {r.message}")

    if junit_xml:
        _write_junit_xml(results, junit_xml, image)

    raise typer.Exit(0 if all_passed else 1)


def _write_junit_xml(
    results: list[Any], path: str, image: str,
) -> None:
    """Write a JUnit XML report for container scan results.

    The report contains one ``<testsuite>`` with one
    ``<testcase>`` per assertion result. Failed assertions
    get a nested ``<failure>`` element whose ``message``
    attribute carries the assertion message.
    """
    failures = sum(1 for r in results if not r.passed)

    root = ET.Element("testsuites")
    suite = ET.SubElement(
        root,
        "testsuite",
        attrib={
            "name": "mltk.container",
            "image": image,
            "tests": str(len(results)),
            "failures": str(failures),
        },
    )
    for r in results:
        case = ET.SubElement(
            suite,
            "testcase",
            attrib={
                "name": r.name,
                "classname": "mltk.container",
            },
        )
        if not r.passed:
            ET.SubElement(
                case,
                "failure",
                attrib={"message": r.message},
            )

    tree = ET.ElementTree(root)
    tree.write(path, encoding="utf-8", xml_declaration=True)
