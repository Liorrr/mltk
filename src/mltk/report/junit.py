"""JUnit XML export -- converts mltk results to CI-compatible XML.

Jenkins, GitLab CI, Azure DevOps, CircleCI, and most CI/CD systems parse
JUnit XML natively.  They display test results in dashboards, track trends,
and gate deployments.  JSON is great for mltk internals but invisible to
CI dashboards -- this module bridges that gap.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


def _classname_from_name(name: str) -> str:
    """Derive a JUnit classname from a dotted test name.

    Takes the module-path prefix (everything before the last dot).
    Falls back to ``"mltk"`` when there is no dot.

    Examples:
        >>> _classname_from_name("data.schema.check")
        'mltk.data.schema'
        >>> _classname_from_name("simple_test")
        'mltk'
    """
    parts = name.rsplit(".", 1)
    if len(parts) == 2:
        return f"mltk.{parts[0]}"
    return "mltk"


def format_result_to_junit(result: dict[str, Any]) -> ET.Element:
    """Convert a single mltk result dict to a JUnit ``<testcase>`` element.

    Args:
        result: A dict with at least ``name`` and ``passed`` keys.
            Optional keys: ``duration_ms``, ``message``.

    Returns:
        An :class:`xml.etree.ElementTree.Element` representing a
        ``<testcase>``, with a ``<failure>`` child when the test failed.

    Example:
        >>> r = {"name": "data.schema.check", "passed": True,
        ...      "duration_ms": 50.0, "message": "ok"}
        >>> elem = format_result_to_junit(r)
        >>> elem.tag
        'testcase'
    """
    name = str(result.get("name", "unknown"))
    passed = result.get("passed", True)
    duration_ms = float(result.get("duration_ms", 0.0))
    message = str(result.get("message", ""))

    duration_s = duration_ms / 1000.0

    testcase = ET.Element("testcase")
    testcase.set("name", name)
    testcase.set("classname", _classname_from_name(name))
    testcase.set("time", f"{duration_s:.6f}")

    if not passed:
        failure = ET.SubElement(testcase, "failure")
        failure.set("message", message)
        failure.set("type", "MltkAssertionError")
        failure.text = message

    return testcase


def export_junit_xml(
    results: list[dict[str, Any]],
    output_path: str = "mltk-results.xml",
    suite_name: str = "mltk",
) -> str:
    """Export mltk results as a JUnit XML file.

    Produces a standards-compliant JUnit XML document that CI/CD systems
    (Jenkins, GitLab, Azure DevOps, CircleCI) can ingest for test
    reporting, trend tracking, and deployment gating.

    Args:
        results: List of result dicts (keys: name, passed, duration_ms,
            message).
        output_path: Destination file path for the XML.
        suite_name: Value for the ``<testsuite name="...">`` attribute.

    Returns:
        The absolute path to the written XML file as a string.

    Example:
        >>> results = [
        ...     {"name": "data.schema.check", "passed": True,
        ...      "duration_ms": 50.0, "message": "ok"},
        ...     {"name": "model.metric.accuracy", "passed": False,
        ...      "duration_ms": 120.0,
        ...      "message": "accuracy 0.75 < 0.80"},
        ... ]
        >>> path = export_junit_xml(results, "report.xml")
    """
    total = len(results)
    failures = sum(1 for r in results if not r.get("passed", True))
    total_duration_ms = sum(
        float(r.get("duration_ms", 0.0)) for r in results
    )
    total_duration_s = total_duration_ms / 1000.0

    testsuites = ET.Element("testsuites")
    testsuite = ET.SubElement(testsuites, "testsuite")
    testsuite.set("name", suite_name)
    testsuite.set("tests", str(total))
    testsuite.set("failures", str(failures))
    testsuite.set("errors", "0")
    testsuite.set("time", f"{total_duration_s:.3f}")

    for result in results:
        testcase = format_result_to_junit(result)
        testsuite.append(testcase)

    tree = ET.ElementTree(testsuites)
    ET.indent(tree, space="  ")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tree.write(
        str(out),
        encoding="unicode",
        xml_declaration=True,
    )

    return str(out.resolve())
