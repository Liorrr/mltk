"""Console output formatting -- rich terminal display for scan results.

Transforms a ScanReport into a human-readable summary for the
terminal.  The output is designed for quick triage: a box header
shows what was scanned, findings are listed by severity (CRITICAL
first), and a one-line summary tells the developer exactly what to
run next.

Two rendering modes are supported automatically:

- **Unicode** (default on modern terminals): box-drawing
  characters, severity icons (cross, warning sign, info).
- **ASCII** (fallback for legacy terminals): plain dashes and
  brackets, letter-based severity indicators.

The renderer detects which mode to use by inspecting
``sys.stdout.encoding`` -- no configuration needed.

Example::

    from mltk.scan.console import format_console_output

    text = format_console_output(report)
    print(text)

Produces output like::

    +-- mltk scan --------------------+
    | Model: classifier | 10000 ...   |
    +---------------------------------+

      [X] CRITICAL  Accuracy drops ...
      [!] WARNING   Model uncalibrated
      [i] INFO      Predictions ...

    Summary: 1 critical, 1 warning, 1 info
    -> Run: pytest tests/test_scan_results.py
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

__all__ = ["format_console_output"]

# ------------------------------------------------------------------
# Severity ordering and display config
# ------------------------------------------------------------------
_SEVERITY_ORDER = {"critical": 0, "warning": 1, "info": 2}

# Unicode glyphs
_UNICODE_ICONS: dict[str, str] = {
    "critical": "\u2717",  # ✗
    "warning": "\u26a0",   # ⚠
    "info": "\u2139",      # ℹ
}

# ASCII fallback
_ASCII_ICONS: dict[str, str] = {
    "critical": "[X]",
    "warning": "[!]",
    "info": "[i]",
}

# Severity labels padded to uniform width
_SEVERITY_LABELS: dict[str, str] = {
    "critical": "CRITICAL",
    "warning": "WARNING ",
    "info": "INFO    ",
}


# ------------------------------------------------------------------
# Encoding detection
# ------------------------------------------------------------------

def _supports_unicode() -> bool:
    """Return True if stdout can render unicode box-drawing.

    Checks ``sys.stdout.encoding`` for a UTF-capable codec.
    Falls back to ASCII when encoding is unknown or limited
    (e.g., ``cp1252``, ``ascii``).
    """
    try:
        encoding = getattr(
            sys.stdout, "encoding", None
        )
    except AttributeError:
        return False

    if encoding is None:
        return False

    upper = encoding.upper().replace("-", "")
    return "UTF" in upper


# ------------------------------------------------------------------
# Box drawing
# ------------------------------------------------------------------

def _box_unicode(lines: list[str], title: str) -> str:
    """Render a unicode box around *lines* with a *title*."""
    max_len = max(
        (len(line) for line in lines),
        default=0,
    )
    # Ensure minimum width for the title
    inner = max(max_len, len(title) + 2)

    top = (
        f"\u256d\u2500 {title} "
        + "\u2500" * (inner - len(title) - 1)
        + "\u256e"
    )
    bottom = (
        "\u2570"
        + "\u2500" * (inner + 2)
        + "\u256f"
    )

    rows: list[str] = [top]
    for line in lines:
        padded = line.ljust(inner)
        rows.append(f"\u2502 {padded} \u2502")
    rows.append(bottom)
    return "\n".join(rows)


def _box_ascii(lines: list[str], title: str) -> str:
    """Render an ASCII box around *lines* with a *title*."""
    max_len = max(
        (len(line) for line in lines),
        default=0,
    )
    inner = max(max_len, len(title) + 2)

    top = (
        "+-- "
        + title
        + " "
        + "-" * (inner - len(title) - 1)
        + "+"
    )
    bottom = "+" + "-" * (inner + 2) + "+"

    rows: list[str] = [top]
    for line in lines:
        padded = line.ljust(inner)
        rows.append(f"| {padded} |")
    rows.append(bottom)
    return "\n".join(rows)


# ------------------------------------------------------------------
# Header construction
# ------------------------------------------------------------------

def _format_number(n: int) -> str:
    """Format an integer with thousands separators.

    Uses comma grouping (e.g., 10000 becomes ``"10,000"``).
    """
    return f"{n:,}"


def _build_header_lines(report: Any) -> list[str]:
    """Extract header info from a ScanReport-like object.

    Accepts any object with the expected attributes so the
    module stays decoupled from the ScanReport dataclass
    (which may live in a different module or not yet exist).
    """
    lines: list[str] = []

    # Line 1: model type + sample count
    model_type = getattr(report, "model_type", "unknown")
    n_samples = getattr(report, "n_samples", 0)
    lines.append(
        f"Model: {model_type} | "
        f"{_format_number(n_samples)} samples"
    )

    # Line 2: feature breakdown
    n_features = getattr(report, "n_features", 0)
    if n_features > 0:
        lines.append(f"Features: {n_features}")

    # Line 3: scanners run / skipped
    scanners_run = getattr(report, "scanners_run", [])
    scanners_skipped = getattr(
        report, "scanners_skipped", []
    )
    total_scanners = (
        len(scanners_run) + len(scanners_skipped)
    )
    if total_scanners > 0:
        skip_note = ""
        if scanners_skipped:
            names = ", ".join(scanners_skipped)
            skip_note = f" ({names} skipped)"
        lines.append(
            f"Scanners: {len(scanners_run)}"
            f"/{total_scanners} run{skip_note}"
        )

    # Line 4: duration
    duration_ms = getattr(report, "duration_ms", 0.0)
    if duration_ms > 0:
        seconds = duration_ms / 1000.0
        lines.append(f"Duration: {seconds:.1f}s")

    return lines


# ------------------------------------------------------------------
# Findings formatting
# ------------------------------------------------------------------

def _format_finding(
    finding: Any,
    icons: dict[str, str],
) -> str:
    """Format a single finding as a one-line summary.

    Layout::

        <icon> <SEVERITY>  <message>  [<scanner>]
    """
    result = finding.result
    severity = result.severity.value
    icon = icons.get(severity, "?")
    label = _SEVERITY_LABELS.get(severity, severity)
    scanner = finding.scanner_name

    message = result.message
    # Truncate very long messages for terminal readability
    max_msg = 60
    if len(message) > max_msg:
        message = message[: max_msg - 3] + "..."

    scanner_tag = f"[{scanner}]" if scanner else ""
    return f"  {icon} {label}  {message}  {scanner_tag}"


def _count_by_severity(
    findings: list[Any],
) -> dict[str, int]:
    """Count findings per severity level."""
    counts: dict[str, int] = {
        "critical": 0,
        "warning": 0,
        "info": 0,
    }
    for f in findings:
        sev = f.result.severity.value
        if sev in counts:
            counts[sev] += 1
    return counts


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def format_console_output(
    report: Any,
    test_file: str = "tests/test_scan_results.py",
) -> str:
    """Format a ScanReport as rich console output.

    Produces a human-readable summary suitable for printing
    directly to the terminal.  The output has four sections:

    1. **Header box** -- model type, sample count, feature
       count, scanners run, and duration.
    2. **Findings list** -- one line per finding, sorted by
       severity (CRITICAL first, then WARNING, then INFO).
       Each line shows a severity icon, the message, and
       which scanner produced it.
    3. **Summary line** -- counts of critical, warnings, and
       info findings.
    4. **Next step** -- a ``pytest`` command the developer
       can copy-paste to run the generated tests.

    When no findings exist the output shows a congratulatory
    message instead of findings and summary.

    The renderer auto-detects unicode support by inspecting
    ``sys.stdout.encoding``.  On terminals that report a
    UTF-capable encoding, box-drawing characters and unicode
    severity icons are used.  Otherwise, plain ASCII
    equivalents are rendered.

    Args:
        report: A ScanReport (or any object with the same
            attributes: ``findings``, ``scanners_run``,
            ``scanners_skipped``, ``duration_ms``,
            ``model_type``, ``n_samples``, ``n_features``).
        test_file: Path shown in the "Run:" footer line.
            Defaults to ``"tests/test_scan_results.py"``.

    Returns:
        A multi-line string ready for ``print()``.

    Example::

        from mltk.scan.console import format_console_output

        output = format_console_output(report)
        print(output)
    """
    use_unicode = _supports_unicode()
    box_fn = _box_unicode if use_unicode else _box_ascii
    icons = (
        _UNICODE_ICONS if use_unicode else _ASCII_ICONS
    )
    arrow = "\u2192" if use_unicode else "->"

    # -- Header box --
    header_lines = _build_header_lines(report)
    sections: list[str] = [
        box_fn(header_lines, "mltk scan"),
    ]

    # -- Findings --
    findings = getattr(report, "findings", [])

    if not findings:
        sections.append("")
        sections.append(
            "  No issues found. "
            "Your model looks good!"
        )
        sections.append("")
        return "\n".join(sections)

    sorted_findings = sorted(
        findings,
        key=lambda f: _SEVERITY_ORDER.get(
            f.result.severity.value, 99
        ),
    )

    sections.append("")  # blank line after box
    for f in sorted_findings:
        sections.append(_format_finding(f, icons))

    # -- Summary --
    counts = _count_by_severity(findings)
    parts: list[str] = []
    if counts["critical"]:
        parts.append(
            f"{counts['critical']} critical"
        )
    if counts["warning"]:
        w = counts["warning"]
        label = "warning" if w == 1 else "warnings"
        parts.append(f"{w} {label}")
    if counts["info"]:
        parts.append(f"{counts['info']} info")

    summary = ", ".join(parts) if parts else "0 issues"

    sections.append("")
    sections.append(f"Summary: {summary}")
    sections.append(
        f"{arrow} Run: pytest {test_file}"
    )

    return "\n".join(sections)
