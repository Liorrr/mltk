"""Explicit synthetic planted controls, verified through the ordinary engine."""

from __future__ import annotations

import random
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from mltk.parity.comparison import CHANGED_CLASSES
from mltk.parity.engine import _compare_product
from mltk.parity.models import LayerSpec, Manifest, ParityConfig, Run, value_key


@dataclass
class ControlFixture:
    run: Run
    changed_item_ids: tuple[str, ...]
    failed_trial_ids: tuple[tuple[str, str], ...]
    seed: int
    layer: str = ""
    changes: dict[str, list[dict[str, Any]]] = field(default_factory=dict)


def make_positive_control(
    run: Run,
    *,
    layer: LayerSpec | str,
    n_items: int,
    mutation: Callable[[Any], Any],
    seed: int,
    failed_trials: int = 0,
) -> ControlFixture:
    """Mutate all observed valid target values on exactly n_items seeded items.

    failed_trials is the total number of mutated trial records to mark failed,
    sampled without replacement. Failures are execution-wide, as in real runs.
    This creates evidence to test; it does not assert that detection succeeded.
    """
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    if any(type(n) is not int or n < 0 for n in (n_items, failed_trials)):
        raise ValueError("Control counts must be nonnegative integers")
    if not callable(mutation):
        raise ValueError("mutation must be callable")
    name = layer.name if isinstance(layer, LayerSpec) else layer
    if not isinstance(name, str) or not name:
        raise ValueError("layer must name a layer")
    if len({i.item_id for i in run.items}) != len(run.items):
        raise ValueError("Duplicate item IDs")
    clone = deepcopy(run)
    clone.run_id = f"{run.run_id}:control:{seed}"
    clone.label = f"{run.label} (synthetic positive control)"
    items = {i.item_id: i for i in clone.items}
    available = sorted(
        i.item_id
        for i in clone.items
        if any(t.status == "ok" and t.values.get(name) is not None for t in i.trials)
    )
    if n_items > len(available):
        raise ValueError("Requested more changed items than available")
    rng = random.Random(seed)
    selected = tuple(sorted(rng.sample(available, n_items)))
    trial_map, changes = {}, {}
    for item_id in selected:
        item = items[item_id]
        if len({t.trial_id for t in item.trials}) != len(item.trials):
            raise ValueError("Duplicate trial IDs")
        changes[item_id] = []
        for trial in sorted(item.trials, key=lambda t: t.trial_id):
            if trial.status != "ok" or trial.values.get(name) is None:
                continue
            before = deepcopy(trial.values[name])
            after = mutation(deepcopy(before))
            if value_key(after)[0] != value_key(before)[0] or after is None:
                raise ValueError("Mutation must preserve the layer type")
            if value_key(after) == value_key(before):
                raise ValueError("Mutation did not change a target value")
            if isinstance(layer, LayerSpec) and (
                value_key(after)[0] != layer.kind
                or layer.kind == "class"
                and after not in layer.vocabulary
            ):
                raise ValueError("Mutation violates the layer policy")
            trial.values[name] = deepcopy(after)
            trial_map[item_id, trial.trial_id] = trial
            changes[item_id].append({"trial_id": trial.trial_id, "before": before, "after": after})
    if failed_trials > len(trial_map):
        raise ValueError("Requested more failed trials than mutated trials")
    failed = tuple(sorted(rng.sample(sorted(trial_map), failed_trials)))
    for key in failed:
        trial_map[key].status = "failed"
        trial_map[key].error = "SYNTHETIC_CONTROL_FAILURE"
    return ControlFixture(clone, selected, failed, seed, name, changes)


def check_positive_control(
    base: Run,
    fixture: ControlFixture,
    *,
    layers: list[LayerSpec],
    manifest: Manifest,
    a2: Run | None = None,
    identity: Run | None = None,
    config: ParityConfig | None = None,
) -> dict[tuple[str, str], bool]:
    """Verify identity and every planted item, then return per-stratum evidence.

    a2 defaults to the base recording; supply an independent repeat for measured
    A/A noise. identity defaults to base. Only the requirement for already-known
    control evidence is bypassed internally; all other product gates still run.
    Invalid/mismatched control identities return false evidence, not a release pass.
    """
    names = {layer.name for layer in layers}
    targets = [layer for layer in layers if layer.name == fixture.layer]
    if not targets:
        return {
            (layer.name, stratum): False
            for layer in layers
            for stratum in sorted(set(manifest.items.values()))
        }
    evidence = {
        (layer.name, stratum): False
        for layer in targets
        for stratum in sorted(set(manifest.items.values()))
    }
    selected = set(fixture.changed_item_ids)
    if (
        fixture.layer not in names
        or not selected
        or len(selected) != len(fixture.changed_item_ids)
        or not selected <= set(manifest.eligible_ids.get(fixture.layer, manifest.items))
    ):
        return evidence
    identity = identity if identity is not None else base
    a2 = a2 if a2 is not None else base
    # Full recording identities, not merely counts, must match the declared manifest.
    for run in (base, a2, identity, fixture.run):
        if {i.item_id: i.stratum for i in run.items} != manifest.items:
            return evidence
        if any({t.trial_id for t in i.trials} != set(manifest.trial_ids) for i in run.items):
            return evidence
    base_items = {i.item_id: i for i in base.items}
    # A changed minority trial must not disappear behind an unchanged majority.
    try:
        for item in identity.items:
            original = base_items[item.item_id]
            if item.source_id != original.source_id:
                return evidence
            originals = {t.trial_id: t for t in original.trials}
            for trial in item.trials:
                previous = originals[trial.trial_id]
                if (trial.status, trial.error) != (previous.status, previous.error):
                    return evidence
                if any(
                    value_key(trial.values.get(name)) != value_key(previous.values.get(name))
                    for name in names
                ):
                    return evidence
    except (TypeError, ValueError):
        return evidence
    try:
        identity_report = _compare_product(
            base,
            a2,
            identity,
            layers=layers,
            manifest=manifest,
            config=config,
            _control_evaluation=True,
        )
        planted = _compare_product(
            base,
            a2,
            fixture.run,
            layers=layers,
            manifest=manifest,
            config=config,
            _control_evaluation=True,
        )
    except (TypeError, ValueError):
        return evidence
    # Identity must be unchanged across all requested layers, including diagnostics.
    if any(
        r["classification"] != "UNCHANGED" or r["exclusion_reason"] is not None
        for r in identity_report.items
    ):
        return evidence
    target_rows = [r for r in planted.items if r["layer"] == fixture.layer]
    actual = {r["item_id"] for r in target_rows if r["classification"] in CHANGED_CLASSES}
    if actual != selected:
        return evidence
    identity_rates = {(r["layer"], r["stratum"]): r for r in identity_report.rates}
    for rate in planted.rates:
        key = rate["layer"], rate["stratum"]
        if key in evidence:
            evidence[key] = (
                identity_rates[key]["gate_passed"]
                and rate["verdict"] == "DETECTED"
                and any(manifest.items[i] == key[1] for i in selected)
                and all(not r["exclusion_reason"] for r in target_rows if r["stratum"] == key[1])
            )
    return evidence
