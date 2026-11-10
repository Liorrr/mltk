"""Offline pair and noise-calibrated product parity over frozen populations."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import asdict
from fractions import Fraction

from mltk.core.statistics import (
    benjamini_hochberg,
    exact_mcnemar,
    paired_bootstrap,
    wilson_interval,
)
from mltk.parity.comparison import CHANGED_CLASSES, UNRESOLVED_CLASSES, classify_pair
from mltk.parity.models import LayerSpec, Manifest, ParityConfig, ParityReport, Run, validate_run
from mltk.parity.presentation import finalize_report
from mltk.parity.states import _policy, reduce_item


def _validate(runs: tuple[Run, ...], layers: list[LayerSpec], manifest: Manifest) -> None:
    if not layers:
        raise ValueError("At least one layer is required")
    for run in runs:
        validate_run(run, manifest, layers)
    for layer in layers:
        _policy(layer, manifest)


def _config(config: ParityConfig) -> None:
    for name in ("confidence", "quiet_fraction", "established_fraction"):
        value = getattr(config, name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0 <= value <= 1
        ):
            raise ValueError(f"Invalid {name}")
    if not 0 < config.confidence < 1 or config.established_fraction == 0:
        raise ValueError("Confidence must be in (0,1), establishment in (0,1]")
    if config.bh_q is not None and (
        isinstance(config.bh_q, bool)
        or not isinstance(config.bh_q, (int, float))
        or not math.isfinite(config.bh_q)
        or not 0 < config.bh_q < 1
    ):
        raise ValueError("bh_q must be in (0,1)")
    if type(config.bootstrap_resamples) is not int or config.bootstrap_resamples < 1:
        raise ValueError("bootstrap_resamples must be a positive integer")
    if config.bootstrap_seed is not None and type(config.bootstrap_seed) is not int:
        raise ValueError("bootstrap_seed must be an integer")
    if type(getattr(config, "fail_on_negative_flip", False)) is not bool:
        raise ValueError("fail_on_negative_flip must be bool")


def _effective_config(manifest, layers, config=None):
    result = asdict(config) if config else {}
    result["manifest"] = {
        "items": dict(sorted(manifest.items.items())),
        "trial_ids": sorted(manifest.trial_ids),
        "recording_hash": manifest.recording_hash,
        "eligible_ids": {name: sorted(ids) for name, ids in sorted(manifest.eligible_ids.items())},
    }
    result["layers"] = []
    for layer in sorted(layers, key=lambda layer: (layer.order, layer.name)):
        policy = {key: value for key, value in vars(layer).items() if key != "oracle"}
        policy["vocabulary"] = sorted(layer.vocabulary)
        policy["min_valid"], policy["failed_threshold"] = _policy(layer, manifest)
        result["layers"].append(policy)
    return result


def _refs(run):
    return {"run_id": run.run_id, "label": run.label, "refs": run.refs}


def _population(manifest: Manifest, layer: LayerSpec, stratum: str) -> list[str]:
    return sorted(
        i
        for i in manifest.eligible_ids.get(layer.name, manifest.items)
        if manifest.items[i] == stratum
    )


def _row(item_id, stratum, layer, base, candidate, base_item, candidate_item):
    row = classify_pair(
        base,
        candidate,
        base_source=base_item.source_id if base_item else None,
        candidate_source=candidate_item.source_id if candidate_item else None,
    )
    row.update(
        item_id=item_id,
        stratum=stratum,
        layer=layer.name,
        base=base,
        candidate=candidate,
        execution_regression=base["oracle"] is True and candidate["n_failed"] > 0,
    )
    return row


def _exclusion(row: dict) -> str | None:
    if row["base"]["status"] != "VALID":
        return "BASE_" + row["base"]["status"]
    if row["candidate"]["status"] != "VALID":
        return "CANDIDATE_" + row["candidate"]["status"]
    if row["source_changed"]:
        return "SOURCE_CHANGED"
    return None


def _statistics(rows: list[dict], confidence: float) -> dict:
    oracle_rows = [
        r for r in rows if r["base"]["oracle"] is not None and r["candidate"]["oracle"] is not None
    ]
    n = len(oracle_rows)
    b = sum(r["classification"] == "NEG_FLIP" for r in oracle_rows)
    c = sum(r["classification"] == "POS_FLIP" for r in oracle_rows)
    return {
        "n_compared": n,
        "n_state_compared": len(rows),
        "n_oracle_unknown": len(rows) - n,
        "b": b,
        "c": c,
        "mcnemar_p": exact_mcnemar(b, c),
        "nfr": b / n if n else None,
        "pfr": c / n if n else None,
        "nfr_interval": wilson_interval(b, n, confidence),
        "pfr_interval": wilson_interval(c, n, confidence),
        "oracle_denominator": "valid_source_comparable_oracle_known_items",
    }


def _base_rate(
    layer: LayerSpec,
    stratum: str,
    rows: list[dict],
    compared: list[dict],
    exclusions: Counter,
    confidence: float,
) -> dict:
    changed = sum(r["classification"] in CHANGED_CLASSES for r in compared)
    return dict(
        layer=layer.name,
        role=layer.role,
        required=layer.role != "diagnostic" and bool(rows),
        stratum=stratum,
        n_eligible=len(rows),
        n_excluded=sum(exclusions.values()),
        excluded_reasons=dict(sorted(exclusions.items())),
        noisy=None,
        n_unconcluded=sum(r["group"] == "Unconcluded" for r in rows),
        floor=None,
        quiet=None,
        established=None,
        changed_count=changed,
        changed_rate=changed / len(compared) if compared else None,
        mdc=None,
        from_failure_count=sum(r["from_failure"] for r in rows),
        execution_regression_count=sum(r["execution_regression"] for r in rows),
        negative_flip_count=sum(r["classification"] == "NEG_FLIP" for r in compared),
        candidate_agreement_counts=dict(
            sorted(
                Counter(f"{r['candidate']['x']}/{r['candidate']['k']}" for r in compared).items()
            )
        ),
        **_statistics(compared, confidence),
    )


def _upstream(rows: list[dict], layers: list[LayerSpec]) -> None:
    order = {layer.name: (layer.order, layer.name) for layer in layers}
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["item_id"]].append(row)
    for group in grouped.values():
        ordered = sorted(group, key=lambda r: order[r["layer"]])
        flipped = next(
            (r["layer"] for r in ordered if r["classification"] in {"NEG_FLIP", "POS_FLIP"}), None
        )
        changed = next(
            (r["layer"] for r in ordered if r["classification"] in CHANGED_CLASSES), None
        )
        for row in group:
            row.update(upstream_flipped_layer=flipped, upstream_changed_layer=changed)


def compare_runs(
    base: Run,
    candidate: Run,
    *,
    layers: list[LayerSpec],
    manifest: Manifest,
    config: ParityConfig | None = None,
) -> ParityReport:
    """Describe two stored runs without claiming A/A-calibrated sensitivity."""
    config = config or ParityConfig()
    _config(config)
    _validate((base, candidate), layers, manifest)
    maps = [{i.item_id: i for i in run.items} for run in (base, candidate)]
    rows, rates = [], []
    for layer in sorted(layers, key=lambda layer: (layer.order, layer.name)):
        for stratum in sorted(set(manifest.items.values())) or [""]:
            group, compared, excluded = [], [], Counter()
            for item_id in _population(manifest, layer, stratum):
                ai, bi = (mapping.get(item_id) for mapping in maps)
                a, b = (reduce_item(i, layer, manifest) for i in (ai, bi))
                row = _row(item_id, stratum, layer, a, b, ai, bi)
                row["exclusion_reason"] = _exclusion(row)
                group.append(row)
                if row["exclusion_reason"]:
                    excluded[row["exclusion_reason"]] += 1
                else:
                    compared.append(row)
            rate = _base_rate(layer, stratum, group, compared, excluded, config.confidence)
            passed = bool(group) and all(
                r["classification"] == "UNCHANGED"
                and not r["exclusion_reason"]
                and not r["execution_regression"]
                for r in group
            )
            if layer.role == "precondition":
                passed = passed and all(
                    r[side]["pass_all_trials"] is True
                    for r in group
                    for side in ("base", "candidate")
                )
            rate.update(
                verdict="UNCHANGED" if passed else "DIFFERENT_OR_UNCONCLUDED",
                reason=None if passed else "PAIR_NOT_IDENTICAL",
                gate_passed=passed,
            )
            if not group:
                rate.update(verdict="NOT_APPLICABLE", reason="NO_ELIGIBLE_ITEMS", gate_passed=True)
            elif not compared:
                rate.update(verdict="DETECTION_NOT_DEMONSTRATED", reason="NO_COMPARABLE_ITEMS")
            rate["reasons"] = [rate["reason"]] if rate["reason"] else []
            rates.append(rate)
            rows.extend(group)
    _upstream(rows, layers)
    required = [r for r in rates if r["required"]]
    passed = bool(required) and all(r["gate_passed"] for r in required)
    failed = any(
        (r["n_state_compared"] and not r["gate_passed"])
        or r["from_failure_count"]
        or r["execution_regression_count"]
        for r in required
    )
    return finalize_report(
        ParityReport(
            mode="pair",
            status="PASS" if passed else "FAIL" if failed else "NOT_VERIFIED",
            items=rows,
            rates=rates,
            refs={"base": _refs(base), "candidate": _refs(candidate)},
            config=_effective_config(manifest, layers, config),
            reasons=[]
            if passed
            else sorted({r["reason"] for r in required if r["reason"]})
            if required
            else ["NO_COMPARABLE_ITEMS"],
        )
    )


def compare_product(
    a1: Run,
    a2: Run,
    candidate: Run,
    *,
    layers: list[LayerSpec],
    manifest: Manifest,
    config: ParityConfig | None = None,
    control: dict[tuple[str, str], dict] | None = None,
) -> ParityReport:
    """Require coverage, quietness, preconditions and explicit control evidence."""
    if len({a1.run_id, a2.run_id, candidate.run_id}) != 3:
        raise ValueError("Product comparison requires three distinct run_id values")
    return _compare_product(
        a1, a2, candidate, layers=layers, manifest=manifest, config=config, control=control
    )


def _compare_product(
    a1, a2, candidate, *, layers, manifest, config=None, control=None, _control_evaluation=False
):
    config = config or ParityConfig()
    _validate((a1, a2, candidate), layers, manifest)
    _config(config)
    if control is not None:
        if not isinstance(control, dict):
            raise ValueError("Control evidence must map (layer, stratum) to evidence records")
        cells = {(layer.name, stratum) for layer in layers for stratum in manifest.items.values()}
        for key, evidence in control.items():
            if (
                not isinstance(key, tuple)
                or len(key) != 2
                or key not in cells
                or not isinstance(evidence, dict)
                or type(evidence.get("passed")) is not bool
                or type(evidence.get("seed")) is not int
                or type(evidence.get("n_items")) is not int
                or evidence["n_items"] < 0
                or not isinstance(evidence.get("changed_item_ids"), (list, tuple))
                or any(not isinstance(i, str) for i in evidence["changed_item_ids"])
                or len(set(evidence["changed_item_ids"])) != evidence["n_items"]
                or len(evidence["changed_item_ids"]) != evidence["n_items"]
                or not set(evidence["changed_item_ids"])
                <= set(manifest.eligible_ids.get(key[0], manifest.items))
            ):
                raise ValueError("Invalid control evidence record")
            if evidence.get("a2_run_id") != a2.run_id:
                raise ValueError("Control evidence A2 run_id differs from comparison A2")
            if evidence["passed"] and not any(
                manifest.items[i] == key[1] for i in evidence["changed_item_ids"]
            ):
                raise ValueError("Passed control evidence requires planted items in its stratum")
    maps = [{i.item_id: i for i in run.items} for run in (a1, a2, candidate)]
    rows, rates = [], []
    reduced = {}
    strata = sorted(set(manifest.items.values())) or [""]
    for layer in layers:
        for item_id in manifest.eligible_ids.get(layer.name, manifest.items):
            reduced[layer.name, item_id] = tuple(
                reduce_item(mapping.get(item_id), layer, manifest) for mapping in maps
            )
    preconditions = {}
    for stratum in strata:
        preconditions[stratum] = all(
            all(
                state["status"] == "VALID" and state["pass_all_trials"] is True
                for item_id in _population(manifest, layer, stratum)
                for state in reduced[layer.name, item_id]
            )
            for layer in layers
            if layer.role == "precondition"
        )
    for layer in sorted(layers, key=lambda layer: (layer.order, layer.name)):
        for stratum in strata:
            group, compared, exclusions = [], [], Counter()
            noisy = unresolved = valid_aa = stable = 0
            for item_id in _population(manifest, layer, stratum):
                ai, aai, bi = (mapping.get(item_id) for mapping in maps)
                a, aa, b = reduced[layer.name, item_id]
                row = _row(item_id, stratum, layer, a, b, ai, bi)
                aa_row = _row(item_id, stratum, layer, a, aa, ai, aai)
                aa_unresolved = aa_row["classification"] in UNRESOLVED_CLASSES
                unresolved += aa_unresolved
                noisy += not aa_unresolved and aa_row["classification"] != "UNCHANGED"
                valid_aa += a["status"] == aa["status"] == "VALID"
                is_stable = (
                    aa_row["classification"] == "UNCHANGED"
                    and a["status"] == aa["status"] == "VALID"
                )
                stable += is_stable
                known_sources = {
                    i.source_id for i in (ai, aai, bi) if i is not None and i.source_id is not None
                }
                row["source_changed"] = len(known_sources) > 1
                row["source_unknown"] = any(i is None or i.source_id is None for i in (ai, aai, bi))
                exclusion = "AA_" + aa_row["classification"] if not is_stable else _exclusion(row)
                if exclusion:
                    exclusions[exclusion] += 1
                else:
                    compared.append(row)
                row.update(
                    a1=a,
                    a2=aa,
                    aa_classification=aa_row["classification"],
                    aa_reason=aa_row["reason"],
                    stable=is_stable,
                    exclusion_reason=exclusion,
                    unconcluded=aa_unresolved or row["classification"] in UNRESOLVED_CLASSES,
                )
                if row["unconcluded"]:
                    row["group"] = "Unconcluded"
                    row["reason"] = exclusion
                group.append(row)
            rate = _base_rate(layer, stratum, group, compared, exclusions, config.confidence)
            eligible, d = len(group), len(compared)
            conservative = noisy + unresolved
            floor = wilson_interval(conservative, eligible, config.confidence)["ci_high"]
            allowed_noise = math.floor(Fraction(str(config.quiet_fraction)) * eligible)
            required_established = math.ceil(Fraction(str(config.established_fraction)) * eligible)
            quiet = bool(eligible) and conservative <= allowed_noise
            established = bool(eligible) and valid_aa >= required_established
            candidate_established = bool(eligible) and d >= required_established
            mdc = math.floor(floor * stable) + 1 if floor is not None else None
            effective_mdc = math.floor(floor * d) + 1 if floor is not None else None
            control_ok = (
                layer.role != "verdict"
                or _control_evaluation
                or (control or {}).get((layer.name, stratum), {}).get("passed") is True
            )
            prereq = preconditions[stratum]
            reasons = []
            if not prereq:
                reasons.append("PRECONDITION_NOT_PASSED")
            if not control_ok:
                reasons.append(
                    "CONTROL_MISSING"
                    if (layer.name, stratum) not in (control or {})
                    else "CONTROL_FAILED"
                )
            if not established:
                reasons.append("NOT_ESTABLISHED")
            if not candidate_established:
                reasons.insert(0, "CANDIDATE_NOT_ESTABLISHED")
            if not d:
                reasons.append("NO_COMPARABLE_CANDIDATE_ITEMS")
            if reasons:
                verdict = "DETECTION_NOT_DEMONSTRATED"
            elif rate["changed_rate"] > floor:
                verdict = "DETECTED"
                reasons.append("CHANGE_ABOVE_NOISE_FLOOR")
            else:
                verdict = "NO_CHANGE_ABOVE_MDC"
            if not quiet:
                reasons.append("NOT_QUIET")
            if rate["from_failure_count"] or rate["execution_regression_count"]:
                reasons.append("CANDIDATE_FAILURE_ON_PASSING_ITEM")
            if getattr(config, "fail_on_negative_flip", False) and rate["negative_flip_count"]:
                reasons.append("STRICT_NEGATIVE_FLIP")
            gate_passed = verdict == "NO_CHANGE_ABOVE_MDC" and not reasons
            rate.update(
                noisy=noisy,
                n_unconcluded=sum(r["unconcluded"] for r in group),
                aa_unresolved=unresolved,
                conservative_noise_count=conservative,
                unconcluded_reasons=dict(
                    sorted(
                        Counter(r["exclusion_reason"] for r in group if r["unconcluded"]).items()
                    )
                ),
                floor=floor,
                quiet=quiet,
                established=established,
                candidate_established=candidate_established,
                required_established=required_established,
                allowed_noise=allowed_noise,
                n_valid_aa=valid_aa,
                n_stable=stable,
                mdc=mdc,
                effective_mdc=effective_mdc,
                mdc_attainable=mdc is not None and mdc <= stable,
                effective_mdc_attainable=effective_mdc is not None and effective_mdc <= d,
                control_passed=control_ok,
                control_evidence=(control or {}).get((layer.name, stratum)),
                preconditions_passed=prereq,
                verdict=verdict,
                reason=reasons[0] if reasons else None,
                reasons=reasons,
                gate_passed=gate_passed,
            )
            if not eligible:
                rate.update(
                    verdict="NOT_APPLICABLE",
                    reason="NO_ELIGIBLE_ITEMS",
                    reasons=[],
                    gate_passed=True,
                )
            if config.bootstrap_seed is not None and layer.reducer == "mean":
                differences = [r["candidate"]["value"] - r["base"]["value"] for r in compared]
                if any(not math.isfinite(difference) for difference in differences):
                    rate["bootstrap"] = {
                        "status": "NOT_VERIFIED",
                        "reason": "NONFINITE_DIFFERENCE",
                        "ci_low": None,
                        "ci_high": None,
                        "mean_difference": None,
                        "ci_method": "paired_bootstrap_percentile",
                        "n": len(differences),
                        "seed": config.bootstrap_seed,
                        "resamples": config.bootstrap_resamples,
                        "confidence": config.confidence,
                    }
                else:
                    rate["bootstrap"] = paired_bootstrap(
                        differences,
                        seed=config.bootstrap_seed,
                        resamples=config.bootstrap_resamples,
                        confidence=config.confidence,
                    )
                rate["bootstrap_item_ids"] = [r["item_id"] for r in compared]
            rates.append(rate)
            rows.extend(group)
    if config.bh_q is not None:
        for stratum in strata:
            family = [
                r
                for r in rates
                if r["stratum"] == stratum and r["role"] == "verdict" and r["n_eligible"]
            ]
            adjusted = benjamini_hochberg([r["mcnemar_p"] for r in family])
            for rate, pvalue in zip(family, adjusted, strict=True):
                rate.update(
                    mcnemar_p_adjusted=pvalue,
                    bh_q=config.bh_q,
                    bh_rejected=pvalue <= config.bh_q,
                    bh_family=[r["layer"] for r in family],
                )
    _upstream(rows, layers)
    required = [r for r in rates if r["required"]]
    failed = any(
        r["verdict"] == "DETECTED"
        or r["from_failure_count"]
        or r["execution_regression_count"]
        or "STRICT_NEGATIVE_FLIP" in r["reasons"]
        for r in required
    )
    status = (
        "FAIL"
        if failed
        else "PASS"
        if required and all(r["gate_passed"] for r in required)
        else "NOT_VERIFIED"
    )
    return finalize_report(
        ParityReport(
            mode="product",
            status=status,
            items=rows,
            rates=rates,
            refs={"a1": _refs(a1), "a2": _refs(a2), "candidate": _refs(candidate)},
            config=_effective_config(manifest, layers, config),
            reasons=sorted({reason for r in required for reason in r["reasons"]})
            if required
            else ["NO_COMPARABLE_ITEMS"],
        )
    )
