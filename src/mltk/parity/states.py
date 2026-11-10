"""Reduce scheduled trials into exact typed states, retaining incomplete evidence."""

from __future__ import annotations

import math
from collections import Counter
from statistics import mean
from typing import Any

from mltk.parity.models import Item, LayerSpec, Manifest, value_key


def _policy(layer: LayerSpec, manifest: Manifest) -> tuple[int, int]:
    k = len(manifest.trial_ids)
    if not k or len(set(manifest.trial_ids)) != k:
        raise ValueError("Scheduled trial IDs must be nonempty and unique")
    if k != 3 and (layer.min_valid is None or layer.failed_threshold is None):
        raise ValueError("General-k reduction requires explicit min_valid and failed_threshold")
    minimum = 2 if layer.min_valid is None else layer.min_valid
    threshold = 2 if layer.failed_threshold is None else layer.failed_threshold
    if any(type(v) is not int or not 1 <= v <= k for v in (minimum, threshold)):
        raise ValueError("Reduction thresholds must be integers between 1 and k")
    if minimum + threshold <= k:
        raise ValueError("Valid and failed thresholds must not overlap")
    fraction = layer.majority_fraction
    if (
        isinstance(fraction, bool)
        or not isinstance(fraction, (int, float))
        or not math.isfinite(fraction)
        or not 0.5 <= fraction < 1
    ):
        raise ValueError("majority_fraction must be in [0.5, 1)")
    if layer.reducer not in {"majority", "mean"}:
        raise ValueError("Unknown reducer")
    if layer.reducer == "mean" and layer.kind != "number":
        raise ValueError("Mean reduction requires a number layer")
    return minimum, threshold


def _oracle(layer: LayerSpec, value: Any) -> bool | None:
    result = layer.oracle(value) if layer.oracle is not None else None
    if result is not None and type(result) is not bool:
        raise ValueError("Oracle must return bool or None")
    return result


def reduce_item(item: Item | None, layer: LayerSpec, manifest: Manifest) -> dict:
    """Missing records are gaps, never execution failures or fabricated votes."""
    minimum, threshold = _policy(layer, manifest)
    k = len(manifest.trial_ids)
    trials = {} if item is None else {t.trial_id: t for t in item.trials}
    if item is not None and (
        len(trials) != len(item.trials) or set(trials) - set(manifest.trial_ids)
    ):
        raise ValueError("Duplicate or unscheduled trial IDs")
    values, verdicts, errors, gaps = [], [], {}, {}
    n_failed = n_missing = 0
    for trial_id in sorted(manifest.trial_ids):
        trial = trials.get(trial_id)
        if trial is None:
            n_missing += 1
            gaps[trial_id] = "MISSING_TRIAL"
        elif trial.status == "failed":
            if not trial.error:
                raise ValueError("Failed trials require an error reason")
            n_failed += 1
            errors[trial_id] = trial.error
        elif trial.status != "ok":
            raise ValueError("Invalid trial status")
        elif trial.values.get(layer.name) is None:
            gaps[trial_id] = "MISSING_LAYER" if layer.name not in trial.values else "NULL_VALUE"
        else:
            value = trial.values[layer.name]
            if value_key(value)[0] != layer.kind:
                raise ValueError("Layer value has the wrong type")
            if layer.kind == "class" and value not in layer.vocabulary:
                raise ValueError("Class value outside declared vocabulary")
            values.append(value)
            verdicts.append(_oracle(layer, value))
    n_valid = len(values)
    pass_all = None
    if any(v is False for v in verdicts):
        pass_all = False
    elif n_valid == k and all(v is True for v in verdicts):
        pass_all = True
    result = {
        "status": "NOT_SCORABLE",
        "value": None,
        "oracle": None,
        "x": None,
        "k": k,
        "n_valid": n_valid,
        "n_failed": n_failed,
        "n_missing": n_missing,
        "n_unscorable": len(gaps) - n_missing,
        "agreement": None,
        "pass_all_trials": pass_all,
        "reason": "INSUFFICIENT_VALID_TRIALS",
        "trial_errors": errors,
        "trial_reasons": gaps,
        "pass_all_trials_reason": (
            None
            if pass_all is not None
            else "NO_ORACLE"
            if layer.oracle is None
            else "INCOMPLETE_ORACLE_EVIDENCE"
        ),
    }
    if n_valid < minimum:
        if n_failed >= threshold:
            result.update(status="FAILED", reason="FAILED_TRIAL_THRESHOLD")
        elif item is None:
            result["reason"] = "MISSING_ITEM"
        return result
    if layer.reducer == "mean":
        value, x = mean(values), n_valid
    else:
        counts = Counter(value_key(value) for value in values)
        key, x = counts.most_common(1)[0]
        if x <= layer.majority_fraction * n_valid:
            result.update(status="INDETERMINATE", reason="NO_STRICT_MAJORITY")
            return result
        value = next(value for value in values if value_key(value) == key)
    result.update(
        status="VALID", value=value, oracle=_oracle(layer, value), x=x, agreement=x / k, reason=None
    )
    return result
