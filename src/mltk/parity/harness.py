"""Offline replay parity with exact change accounting and item-paired TOST."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from .models import LayerSpec, Manifest, ParityReport, Run, validate_run, value_key

_PRIORITY = {"PASS": 0, "NOT_VERIFIED": 1, "FAIL": 2, "NOT_RUN": 3}


def _plain(value: Any) -> Any:
    """Copy JSON-compatible provenance; sets have an explicit, stable encoding."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    if isinstance(value, (set, frozenset)):
        value_key(value)
        return {"type": "id_set", "values": sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, Mapping) and all(isinstance(k, str) for k in value):
        return {k: _plain(value[k]) for k in sorted(value)}
    raise ValueError("Nonfinite or non-plain report data")


def _finite(value: Any) -> bool:
    try:
        return (not isinstance(value, bool) and isinstance(value, (float, int))
                and math.isfinite(value))
    except OverflowError:
        return False


def _same(left: Any, right: Any) -> bool:
    return value_key(left) == value_key(right)


def _key(entry: Mapping) -> tuple[str, str, str, str]:
    return entry["item_id"], entry["stratum"], entry["layer"], entry.get("trial_id", "")


def _matches(entry: Mapping, before: Any, after: Any) -> bool:
    if _same(before, after):
        return False
    if "from" in entry and not _same(entry["from"], before):
        return False
    if "to" in entry and not _same(entry["to"], after):
        return False
    if entry["direction"] == "changed":
        return True
    if not (_finite(before) and _finite(after)):
        return False
    return after > before if entry["direction"] == "increase" else after < before


def _ledger(expected_diff, recordings, layers):
    by_name = {layer.name: layer for layer in layers}
    ledger = {}
    for entry in expected_diff:
        if not isinstance(entry, Mapping):
            raise ValueError("expected_diff entries must be mappings")
        allowed = {"item_id", "stratum", "layer", "trial_id", "direction", "reason", "from", "to"}
        if set(entry) - allowed:
            raise ValueError("Unknown expected_diff fields")
        for name in ("item_id", "stratum", "layer", "direction", "reason"):
            if not isinstance(entry.get(name), str) or not entry[name].strip():
                raise ValueError(f"expected_diff requires nonempty {name}")
        if entry["item_id"] not in recordings.items:
            raise ValueError("expected_diff references unknown item")
        if entry["stratum"] != recordings.items[entry["item_id"]]:
            raise ValueError("expected_diff stratum mismatch")
        if entry["layer"] not in by_name:
            raise ValueError("expected_diff references unknown layer")
        layer = by_name[entry["layer"]]
        eligible = recordings.eligible_ids.get(layer.name, recordings.items)
        if entry["item_id"] not in eligible:
            raise ValueError("expected_diff references ineligible item")
        if "trial_id" in entry and entry["trial_id"] not in recordings.trial_ids:
            raise ValueError("expected_diff references unknown trial")
        if entry["direction"] not in {"increase", "decrease", "changed"}:
            raise ValueError("Invalid expected_diff direction")
        if layer.kind != "number" and not {"from", "to"} <= set(entry):
            raise ValueError("Nonnumeric expected_diff requires exact from/to states")
        if ("from" in entry) != ("to" in entry):
            raise ValueError("expected_diff from/to must be supplied together")
        for name in ("from", "to"):
            if name in entry:
                if value_key(entry[name])[0] != layer.kind:
                    raise ValueError("expected_diff state has wrong type")
                if layer.kind == "class" and entry[name] not in layer.vocabulary:
                    raise ValueError("expected_diff state outside vocabulary")
        key = _key(entry)
        if key in ledger:
            raise ValueError("Duplicate or conflicting expected_diff entry")
        ledger[key] = dict(entry)
    return dict(sorted(ledger.items()))


def _integrity(base, candidate, recordings, layers):
    if not isinstance(recordings, Manifest):
        raise ValueError("recordings must be a Manifest")
    if not isinstance(recordings.recording_hash, str) or not recordings.recording_hash.strip():
        raise ValueError("Manifest recording_hash is required")
    if not isinstance(recordings.items, dict) or not isinstance(layers, list):
        raise ValueError("Invalid manifest or layers")
    for run in (base, candidate):
        if not isinstance(run, Run):
            raise ValueError("Scores must be Run instances")
        validate_run(run, recordings, layers)
        if run.refs.get("recording_hash") != recordings.recording_hash:
            raise ValueError("Run recording_hash does not match manifest")
        if {item.item_id for item in run.items} != set(recordings.items):
            raise ValueError("Run item IDs do not exactly match manifest")
        for item in run.items:
            if {trial.trial_id for trial in item.trials} != set(recordings.trial_ids):
                raise ValueError(f"Trial IDs do not exactly match manifest: {item.item_id}")
            for trial in item.trials:
                if trial.status != "ok" or trial.error is not None:
                    raise ValueError(f"Failed/error record: {item.item_id}/{trial.trial_id}")
                if any(trial.values.get(layer.name) is None for layer in layers):
                    raise ValueError(f"Missing/null value: {item.item_id}/{trial.trial_id}")


def _state(state):
    # The reducer's frozen result is plain data, not an executable adapter.
    if isinstance(state, Mapping):
        return dict(state)
    from dataclasses import asdict

    return asdict(state)


def _layer_config(layer: LayerSpec, recordings: Manifest) -> dict:
    from .states import _policy

    minimum, threshold = _policy(layer, recordings)
    return _plain({
        **{k: v for k, v in vars(layer).items() if k != "oracle"},
        "vocabulary": sorted(layer.vocabulary),
        "effective_min_valid": minimum,
        "effective_failed_threshold": threshold,
    })


def _precondition(base: dict, candidate: dict) -> dict:
    evidence = {"base": base["pass_all_trials"], "candidate": candidate["pass_all_trials"]}
    if any(value is False for value in evidence.values()):
        status, reason = "FAIL", "precondition_failed"
    elif any(value is not True for value in evidence.values()):
        status, reason = "NOT_VERIFIED", "precondition_oracle_incomplete"
    else:
        status, reason = "PASS", None
    return {"status": status, "reason": reason, **evidence}


def _tost(differences, margin, alpha):
    missing = {
        "status": "NOT_VERIFIED", "pvalue": None, "mean_difference": None,
        "n": len(differences), "ci_low": None, "ci_high": None,
    }
    if margin is None:
        return {**missing, "reason": "missing_margin"}
    if not _finite(margin) or margin <= 0:
        return {**missing, "reason": "invalid_margin"}
    if not _finite(alpha) or not 0 < alpha < 0.5:
        return {**missing, "reason": "invalid_alpha"}
    if len(differences) < 2:
        return {**missing, "reason": "insufficient_pairs"}
    if not all(_finite(d) for d in differences):
        return {**missing, "reason": "nonfinite_difference"}
    try:
        from mltk.core.statistics import paired_tost

        result = paired_tost(differences, margin, alpha)
        required = {"status", "pvalue", "mean_difference", "n", "ci_low", "ci_high", "reason"}
        if not isinstance(result, dict) or not required <= result.keys():
            raise ValueError("Incomplete statistical result")
        if result["status"] not in {"PASS", "FAIL", "NOT_VERIFIED"}:
            raise ValueError("Invalid statistical status")
        if result["n"] != len(differences):
            raise ValueError("Statistical pair count mismatch")
        if result["status"] in {"PASS", "FAIL"}:
            numbers = ("pvalue", "mean_difference", "ci_low", "ci_high")
            if not all(_finite(result[k]) for k in numbers):
                raise ValueError("Nonfinite statistical result")
            if not 0 <= result["pvalue"] <= 1 or result["ci_low"] > result["ci_high"]:
                raise ValueError("Invalid statistical interval or probability")
        _plain(result)
        return result
    except ImportError:
        return {**missing, "reason": "statistics_unavailable"}
    except Exception:
        # A failed statistics backend cannot certify equivalence.
        return {**missing, "reason": "statistics_error"}


def compare_harness(
    base_scores: Run,
    candidate_scores: Run,
    *,
    recordings: Manifest,
    layers: list[LayerSpec],
    expected_diff=(),
    margins=None,
    alpha=0.05,
) -> ParityReport:
    """Compare stored scores without executing scoring or model inference.

    Item ledger entries cover a changed reduced value and only trial changes
    having that exact before/after pair. Trial entries cover one trial. A reduced
    change also counts as explained when *every* changed trial has its own
    matching entry. Item entries with unchanged reduced values are stale.
    Scored-layer ledger entries are audited but never remove TOST observations.
    Diagnostic rows are descriptive; verdict/precondition layers are required.
    Preconditions additionally require every scheduled trial on both sides to
    oracle-pass, independently of equality, expected differences, and TOST.
    """
    report = ParityReport(mode="harness", status="NOT_RUN")
    try:
        # Preserve usable provenance even when subsequent integrity checks fail.
        report.refs = _plain({
            "base": {"run_id": getattr(base_scores, "run_id", None),
                     "label": getattr(base_scores, "label", None),
                     "refs": getattr(base_scores, "refs", None)},
            "candidate": {"run_id": getattr(candidate_scores, "run_id", None),
                          "label": getattr(candidate_scores, "label", None),
                          "refs": getattr(candidate_scores, "refs", None)},
            "recording_hash": getattr(recordings, "recording_hash", None),
        })
        expected_diff = list(expected_diff)
        report.config["expected_diff"] = _plain(expected_diff)
        _integrity(base_scores, candidate_scores, recordings, layers)
        ledger = _ledger(expected_diff, recordings, layers)
        margins = {} if margins is None else margins
        if not isinstance(margins, Mapping):
            raise ValueError("margins must be a mapping")
        margin_config = []
        for key, margin in margins.items():
            if isinstance(key, str):
                name, stratum = key, None
            elif isinstance(key, tuple) and len(key) == 2 and all(isinstance(k, str) for k in key):
                name, stratum = key
            else:
                raise ValueError("Invalid margin key")
            if name not in {layer.name for layer in layers}:
                raise ValueError("Margin references unknown layer")
            if stratum is not None and stratum not in recordings.items.values():
                raise ValueError("Margin references unknown stratum")
            margin_config.append({"layer": name, "stratum": stratum,
                                  "margin": margin if _finite(margin) else None,
                                  "valid": _finite(margin) and margin > 0})
        report.config = {
            "alpha": alpha if _finite(alpha) else None,
            "margins": sorted(margin_config, key=lambda r: (r["layer"], r["stratum"] or "")),
            "layers": [_layer_config(layer, recordings)
                       for layer in sorted(layers, key=lambda layer: (layer.order, layer.name))],
            "manifest": _plain({
                "items": recordings.items, "trial_ids": sorted(recordings.trial_ids),
                "recording_hash": recordings.recording_hash,
                "eligible_ids": {k: sorted(v) for k, v in recordings.eligible_ids.items()},
            }),
            "expected_diff": [_plain(entry) for entry in ledger.values()],
            "ledger_coverage": "exact_item_transition_or_explicit_trials",
            "statistical_unit": "item", "difference": "candidate_minus_base",
        }
        return _compare(report, base_scores, candidate_scores, recordings, layers, ledger,
                        margins, alpha)
    except Exception as exc:
        # Public comparison is non-raising, including invalid caller oracle output.
        report.status = "NOT_RUN"
        report.reasons.append(f"invalid_input: {type(exc).__name__}: {exc}")
        return report


def _compare(report, base, candidate, recordings, layers, ledger, margins, alpha):
    try:
        from .states import reduce_item
    except ImportError:
        report.reasons.append("reducer_unavailable")
        return report

    base_items = {item.item_id: item for item in base.items}
    candidate_items = {item.item_id: item for item in candidate.items}
    audit = {key: {**_plain(entry), "status": "stale", "covers": []}
             for key, entry in ledger.items()}
    required_statuses = []
    for layer in sorted(layers, key=lambda layer: (layer.order, layer.name)):
        eligible = set(recordings.eligible_ids.get(layer.name, recordings.items))
        for stratum in sorted(set(recordings.items.values())):
            ids = sorted(i for i in eligible if recordings.items[i] == stratum)
            rows = []
            differences = []
            for item_id in ids:
                left, right = base_items[item_id], candidate_items[item_id]
                a = _state(reduce_item(left, layer, recordings))
                b = _state(reduce_item(right, layer, recordings))
                row = {
                    "item_id": item_id, "stratum": stratum, "layer": layer.name,
                    "base": _plain(a), "candidate": _plain(b),
                    "base_source_id": left.source_id, "candidate_source_id": right.source_id,
                    "base_metadata": _plain(left.metadata),
                    "candidate_metadata": _plain(right.metadata),
                    "status": "PASS", "reasons": [], "trial_changes": [],
                }
                valid = a["status"] == b["status"] == "VALID"
                item_changed = valid and not _same(a["value"], b["value"])
                key = item_id, stratum, layer.name, ""
                item_match = item_changed and key in ledger and _matches(
                    ledger[key], a["value"], b["value"])
                if item_match:
                    audit[key]["status"] = "matched"
                    audit[key]["covers"].append("item")
                elif key in ledger and item_changed:
                    audit[key]["status"] = "mismatch"
                lt = {trial.trial_id: trial for trial in left.trials}
                rt = {trial.trial_id: trial for trial in right.trials}
                all_trials_explicit = True
                for trial_id in sorted(recordings.trial_ids):
                    before, after = lt[trial_id].values[layer.name], rt[trial_id].values[layer.name]
                    if _same(before, after):
                        continue
                    trial_key = item_id, stratum, layer.name, trial_id
                    trial_match = trial_key in ledger and _matches(ledger[trial_key], before, after)
                    all_trials_explicit = all_trials_explicit and trial_match
                    coverage = None
                    if trial_match:
                        audit[trial_key]["status"] = "matched"
                        audit[trial_key]["covers"].append(trial_id)
                        coverage = "trial"
                    else:
                        if trial_key in ledger:
                            audit[trial_key]["status"] = "mismatch"
                        if item_match and _same(before, a["value"]) and _same(after, b["value"]):
                            audit[key]["covers"].append(trial_id)
                            coverage = "item"
                    row["trial_changes"].append({
                        "trial_id": trial_id, "from": _plain(before), "to": _plain(after),
                        "expected": coverage is not None, "coverage": coverage,
                    })
                if layer.reducer == "majority":
                    if any(not change["expected"] for change in row["trial_changes"]):
                        row["status"] = "FAIL"
                        row["reasons"].append("unlisted_trial_change")
                    explained = item_match or (row["trial_changes"] and all_trials_explicit)
                    if item_changed and not explained:
                        row["status"] = "FAIL"
                        row["reasons"].append("unlisted_item_change")
                if not valid:
                    if row["status"] != "FAIL":
                        row["status"] = "NOT_VERIFIED"
                    row["reasons"].append("unresolved_item_state")
                    row["group"] = "Unconcluded"
                elif layer.reducer == "mean":
                    difference = b["value"] - a["value"]
                    row["difference"] = difference if _finite(difference) else None
                    differences.append(difference)
                row["classification"] = (
                    "UNLISTED_CHANGE" if row["status"] == "FAIL" else
                    "INDETERMINATE" if not valid else
                    "SCORED_PAIR" if layer.reducer == "mean" else
                    "EXPECTED_DIFF" if row["trial_changes"] else "UNCHANGED"
                )
                row["reason"] = ("unresolved_item_state" if not valid else
                                 row["reasons"][0] if row["reasons"] else None)
                if layer.role == "precondition":
                    precondition = _precondition(a, b)
                    row["precondition"] = precondition
                    row["comparison_status"] = row["status"]
                    row["status"] = max(
                        (row["status"], precondition["status"]), key=_PRIORITY.get)
                    if precondition["reason"]:
                        row["reasons"].append(precondition["reason"])
                        if precondition["status"] == "FAIL":
                            row["classification"] = "PRECONDITION_FAILED"
                            row["reason"] = precondition["reason"]
                        else:
                            row["group"] = "Unconcluded"
                            if row["status"] != "FAIL":
                                row["classification"] = "PRECONDITION_NOT_VERIFIED"
                                row["reason"] = precondition["reason"]
                rows.append(row)
            if layer.reducer == "mean":
                margin = margins.get((layer.name, stratum), margins.get(layer.name))
                gate = _tost(differences, margin, alpha)
                if len(differences) != len(ids) and gate["status"] == "PASS":
                    gate.update(status="NOT_VERIFIED", reason="unresolved_item_state")
                gate["margin"] = margin if _finite(margin) else None
            else:
                gate = {"status": max((r.get("comparison_status", r["status"]) for r in rows),
                                      key=_PRIORITY.get)
                        if rows else "NOT_VERIFIED", "n": len(rows),
                        "reason": "exact_comparison" if rows else "no_comparable_items"}
                if rows and gate["status"] != "PASS":
                    gate["reason"] = ("unlisted_change" if gate["status"] == "FAIL"
                                      else "unresolved_item_state")
            if layer.role == "precondition" and rows:
                precondition = max((row["precondition"] for row in rows),
                                   key=lambda result: _PRIORITY[result["status"]])
                gate.update(comparison_status=gate["status"], comparison_reason=gate["reason"],
                            precondition_status=precondition["status"],
                            precondition_reason=precondition["reason"])
                if (precondition["status"] != "PASS"
                        and _PRIORITY[precondition["status"]] >= _PRIORITY[gate["status"]]):
                    gate.update(status=precondition["status"], reason=precondition["reason"])
            gate.update(layer=layer.name, stratum=stratum, required=layer.role != "diagnostic",
                        eligible=len(ids))
            report.rates.append(gate)
            report.items.extend(rows)
            if gate["required"]:
                required_statuses.append(gate["status"])
                if gate["status"] != "PASS":
                    report.reasons.append(f"{layer.name}/{stratum}: {gate['reason']}")
    report.config["expected_diff_audit"] = list(audit.values())
    for entry in audit.values():
        if entry["status"] != "matched":
            required_statuses.append("FAIL")
            report.reasons.append(
                f"expected_diff_{entry['status']}: {entry['item_id']}/{entry['stratum']}/"
                f"{entry['layer']}/{entry.get('trial_id', 'item')}"
            )
    if not any(rate["required"] and rate["eligible"] for rate in report.rates):
        required_statuses.append("NOT_VERIFIED")
        report.reasons.append("no_comparable_required_layers")
    report.status = max(required_statuses, key=_PRIORITY.get)
    return report
