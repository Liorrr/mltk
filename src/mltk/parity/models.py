"""Typed stored-result contracts for offline parity comparison."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

ENGINE_VERSION = "1.0"
SCHEMA_VERSION = "1"


@dataclass
class Trial:
    trial_id: str
    status: str = "ok"
    values: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass
class Item:
    item_id: str
    stratum: str
    trials: list[Trial] = field(default_factory=list)
    source_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Run:
    run_id: str
    label: str
    items: list[Item]
    refs: dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION


@dataclass
class Manifest:
    """Expected item strata and trial identities, independent of parsed runs."""

    items: dict[str, str]
    trial_ids: tuple[str, ...] = ("1", "2", "3")
    recording_hash: str | None = None
    eligible_ids: dict[str, tuple[str, ...]] = field(default_factory=dict)


@dataclass
class LayerSpec:
    name: str
    kind: str = "bool"
    role: str = "verdict"
    order: int = 0
    oracle: Callable[[Any], bool | None] | None = None
    oracle_id: str | None = None
    vocabulary: tuple[str, ...] = ()
    reducer: str = "majority"
    min_valid: int | None = None
    failed_threshold: int | None = None
    majority_fraction: float = 0.5


@dataclass
class ParityConfig:
    confidence: float = 0.95
    quiet_fraction: float = 0.1
    established_fraction: float = 0.8
    bh_q: float | None = None
    bootstrap_seed: int | None = None
    bootstrap_resamples: int = 2000
    fail_on_negative_flip: bool = False


@dataclass
class ParityReport:
    mode: str
    status: str
    items: list[dict[str, Any]] = field(default_factory=list)
    rates: list[dict[str, Any]] = field(default_factory=list)
    refs: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)
    engine_version: str = ENGINE_VERSION
    schema_version: str = SCHEMA_VERSION

    @property
    def passed(self) -> bool:
        return self.status == "PASS"


def value_key(value: Any) -> tuple:
    """Exact typed equality, with unordered sets and no nonfinite numbers."""
    if value is None:
        return ("null",)
    if isinstance(value, bool):
        return ("bool", value)
    if isinstance(value, (int, float)):
        if not math.isfinite(value):
            raise ValueError("Layer numbers must be finite")
        return ("number", value)
    if isinstance(value, str):
        return ("class", value)
    if isinstance(value, (set, frozenset)) and all(isinstance(v, str) for v in value):
        return ("id_set", tuple(sorted(value)))
    raise ValueError("Unsupported layer value type")


def validate_run(run: Run, manifest: Manifest, layers: list[LayerSpec]) -> None:
    """Validate observed records; missing expected records remain visible to reducers."""
    if run.schema_version != SCHEMA_VERSION:
        raise ValueError("Unsupported run schema_version")
    if not isinstance(run.run_id, str) or not run.run_id:
        raise ValueError("run_id must be a nonempty string")
    if not isinstance(run.label, str) or not run.label:
        raise ValueError("label must be a nonempty string")
    if not isinstance(run.refs, dict):
        raise ValueError("refs must be an object")
    if not manifest.trial_ids or len(set(manifest.trial_ids)) != len(manifest.trial_ids):
        raise ValueError("Manifest trial IDs must be nonempty and unique")
    if any(not isinstance(t, str) or not t for t in manifest.trial_ids):
        raise ValueError("Trial IDs must be nonempty strings")
    if any(not isinstance(k, str) or not k or not isinstance(v, str) or not v
           for k, v in manifest.items.items()):
        raise ValueError("Manifest item IDs and strata must be nonempty strings")
    by_name = {layer.name: layer for layer in layers}
    if len(by_name) != len(layers):
        raise ValueError("Duplicate layer names")
    for layer in layers:
        if (not isinstance(layer.name, str) or not layer.name
                or layer.kind not in {"bool", "class", "id_set", "number"}):
            raise ValueError("Invalid layer name or kind")
        if layer.role not in {"verdict", "diagnostic", "precondition"}:
            raise ValueError("Invalid layer role")
        if layer.reducer not in {"majority", "mean"}:
            raise ValueError("Invalid reducer")
        if layer.reducer == "mean" and layer.kind != "number":
            raise ValueError("Mean reduction requires a number layer")
        if layer.kind == "class" and not layer.vocabulary:
            raise ValueError("Class layers require a declared vocabulary")
        if (any(not isinstance(v, str) for v in layer.vocabulary)
                or len(set(layer.vocabulary)) != len(layer.vocabulary)):
            raise ValueError("Vocabulary must contain unique strings")
        if type(layer.order) is not int:
            raise ValueError("Layer order must be an integer")
        if layer.oracle is not None and (not callable(layer.oracle)
                                         or not isinstance(layer.oracle_id, str)
                                         or not layer.oracle_id):
            raise ValueError("An oracle requires a versioned oracle_id")
    for name, eligible in manifest.eligible_ids.items():
        if name not in by_name or len(set(eligible)) != len(eligible):
            raise ValueError("Invalid eligibility declaration")
        if set(eligible) - set(manifest.items):
            raise ValueError("Eligibility contains unknown item IDs")
    seen: set[str] = set()
    for item in run.items:
        if item.item_id in seen or item.item_id not in manifest.items:
            raise ValueError("Duplicate or unknown item ID")
        seen.add(item.item_id)
        if item.stratum != manifest.items[item.item_id]:
            raise ValueError("Item stratum disagrees with manifest")
        if item.source_id is not None and (
                not isinstance(item.source_id, str) or not item.source_id):
            raise ValueError("source_id must be a nonempty string or None")
        if not isinstance(item.metadata, dict):
            raise ValueError("metadata must be an object")
        trials: set[str] = set()
        for trial in item.trials:
            if trial.trial_id in trials or trial.trial_id not in manifest.trial_ids:
                raise ValueError("Duplicate or unknown trial ID")
            trials.add(trial.trial_id)
            if trial.status not in {"ok", "failed"}:
                raise ValueError("Invalid trial status")
            if trial.status == "failed" and (not isinstance(trial.error, str) or not trial.error):
                raise ValueError("Failed trials require an error reason")
            if trial.status == "ok" and trial.error is not None:
                raise ValueError("Successful trials cannot carry an error")
            if set(trial.values) - set(by_name):
                raise ValueError("Trial contains unknown layers")
            for name, value in trial.values.items():
                if value is None:
                    continue
                layer = by_name[name]
                if value_key(value)[0] != layer.kind:
                    raise ValueError("Layer value has the wrong type")
                if layer.kind == "class" and value not in layer.vocabulary:
                    raise ValueError("Class value outside declared vocabulary")
