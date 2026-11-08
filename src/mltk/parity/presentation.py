"""Human-readable evidence and stable ordering for parity reports.

Formatting never computes or overrides a gate. The comparison engine owns status.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mltk.parity.models import ParityReport


def row_key(report: ParityReport, row: dict) -> tuple:
    """Order evidence by declared layer order, then stable identity."""
    layers = {layer["name"]: layer for layer in report.config.get("layers", [])}
    name = row.get("layer", "")
    order = row.get("order", layers.get(name, {}).get("order", 0))
    return order, name, row.get("stratum", ""), row.get("item_id", "")


def is_failing(rate: dict) -> bool:
    """Read the engine's gate, including explicit harness ledger requirements."""
    if rate.get("status") == "NOT_APPLICABLE" or rate.get("verdict") == "NOT_APPLICABLE":
        return False
    if rate.get("required") is False:
        return False
    if "required" not in rate and rate.get("role") == "diagnostic":
        return False
    return rate.get("gate_passed") is False or rate.get("status") in {
        "FAIL",
        "NOT_VERIFIED",
        "NOT_RUN",
    }


def ordered_rates(report: ParityReport, *, failing_first: bool = False) -> list[dict]:
    def rate_key(row: dict) -> tuple:
        order, name, stratum, _ = row_key(report, row)
        key = (order, stratum, name)
        return (not is_failing(row), *key) if failing_first else key

    return sorted(report.rates, key=rate_key)


def offending_rows(report: ParityReport) -> list[dict]:
    failing = {(r.get("layer"), r.get("stratum")) for r in report.rates if is_failing(r)}
    rows = [
        r
        for r in report.items
        if r.get("classification") not in {"UNCHANGED", "EXPECTED_DIFF", "SCORED_PAIR"}
        or r.get("execution_regression")
        or r.get("from_failure")
        or r.get("exclusion_reason")
        or r.get("group") == "Unconcluded"
        or r.get("status") in {"FAIL", "NOT_VERIFIED", "NOT_RUN"}
    ]
    return sorted(
        rows, key=lambda r: ((r.get("layer"), r.get("stratum")) not in failing, *row_key(report, r))
    )


def _number(value: Any) -> str:
    if value is None:
        return "--"
    return f"{value:.6g}" if isinstance(value, float) else str(value)


def finalize_report(report: ParityReport) -> ParityReport:
    """Set a self-contained message from engine verdicts; leave every gate intact."""
    parts = [f"Parity {report.mode}: {report.status}"]
    for rate in ordered_rates(report, failing_first=True):
        if not is_failing(rate):
            continue
        verdict = rate.get("verdict", rate.get("status", "--"))
        reasons = rate.get("reasons") or [rate.get("reason") or "UNSPECIFIED"]
        detail = (
            f"{rate.get('layer', '--')}@{rate.get('stratum', '--')}: "
            f"{verdict} ({', '.join(reasons)})"
        )
        if report.mode == "harness":
            detail += (
                f"; unlisted={rate.get('unlisted_count', 0)}, "
                f"stale={rate.get('stale_count', 0)}, "
                f"mismatch={rate.get('mismatch_count', 0)}, "
                f"n={_number(rate.get('n_compared', rate.get('n')))}"
            )
        else:
            detail += (
                f"; changed {_number(rate.get('changed_count'))}/"
                f"{_number(rate.get('n_state_compared', rate.get('n_compared')))}"
                f" vs floor={_number(rate.get('floor'))}, "
                f"MDC={_number(rate.get('effective_mdc', rate.get('mdc')))}"
            )
            if verdict == "DETECTION_NOT_DEMONSTRATED" or report.status == "NOT_VERIFIED":
                n_candidate = rate.get("n_state_compared", rate.get("n_compared"))
                n_required = rate.get("required_established", rate.get("n_required"))
                detail += (
                    f"; candidate n={_number(n_candidate)}"
                    f" vs required={_number(n_required)}"
                    f"; A/A n={_number(rate.get('n_valid_aa'))}"
                )
        parts.append(detail)
    if len(parts) == 1 and report.reasons:
        parts.append(", ".join(sorted(report.reasons)))
    ids = list(dict.fromkeys(r["item_id"] for r in offending_rows(report) if "item_id" in r))
    # Ledger failures can have no changed item row (a stale expected difference).
    for entry in report.config.get("expected_diff_audit", []):
        if entry.get("status") != "matched" and entry.get("item_id") not in ids:
            ids.append(entry.get("item_id"))
    lines = ["; ".join(parts)]
    if ids:
        lines.append("Offending item ids (first 10): " + ", ".join(str(i) for i in ids[:10]))
    lines.append(
        "Artifacts: "
        + (", ".join(report.artifacts.values()) if report.artifacts else "not written")
    )
    if report.status != "PASS":
        if report.mode == "harness":
            lines.append(
                "Hint: add expected-diff entries for intended changes; "
                "remove stale entries and correct mismatched entries."
            )
        elif report.mode == "product":
            lines.append(
                "Hint: supply passing control evidence for this A2 run; "
                "restore required coverage and inspect changed items."
            )
        else:
            lines.append("Hint: supply comparable eligible items and inspect changed recordings.")
    report.message = "\n".join(lines)
    return report
