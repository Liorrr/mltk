"""Strict, offline JSON/JSONL loading for parity recordings."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mltk.parity.models import Item, LayerSpec, Manifest, Run, Trial, validate_run, value_key


def _object(pairs: list[tuple[str, Any]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _parse(text: str) -> Any:
    def invalid(value: str) -> None:
        raise ValueError(f"Nonfinite JSON value: {value}")
    return json.loads(text, object_pairs_hook=_object, parse_constant=invalid)


def _fields(obj: Any, required: set[str], optional: set[str]) -> dict:
    if not isinstance(obj, dict) or required - obj.keys() or obj.keys() - required - optional:
        raise ValueError(f"Malformed record; required fields: {sorted(required)}")
    return obj


def _string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _value(value: Any) -> Any:
    if isinstance(value, dict):
        _fields(value, {"type", "values"}, set())
        values = value["values"]
        if (value["type"] != "id_set" or not isinstance(values, list)
                or any(not isinstance(v, str) for v in values)
                or len(set(values)) != len(values)):
            raise ValueError("Malformed tagged ID set")
        return frozenset(values)
    value_key(value)
    return value


def load_run(source: Run | dict | str | Path, *, manifest: Manifest | None = None,
             layers: list[LayerSpec] | None = None) -> Run:
    """Load a run; paths select UTF-8 JSON or JSONL by extension.

    JSONL starts with a run header containing ``expected_item_ids``,
    ``expected_count``, and ``expected_trial_ids``, followed by item objects.
    Its declaration must match the independent caller manifest when supplied.
    Missing scheduled records remain visible to comparison through that manifest.
    """
    declared_trials = None
    if isinstance(source, Run):
        run = source
    else:
        if isinstance(source, (str, Path)):
            path = Path(source)
            content = path.read_text(encoding="utf-8")
            if path.suffix.lower() == ".jsonl":
                records = [_parse(line) for line in content.splitlines() if line.strip()]
                if not records:
                    raise ValueError("Empty JSONL recording")
                if not isinstance(records[0], dict) or "items" in records[0]:
                    raise ValueError("JSONL requires a header object without items")
                header = dict(records[0])
                ids = header.pop("expected_item_ids", None)
                count = header.pop("expected_count", None)
                trials = header.pop("expected_trial_ids", None)
                if (not isinstance(ids, list) or any(not isinstance(i, str) or not i for i in ids)
                        or len(set(ids)) != len(ids) or type(count) is not int or count != len(ids)
                        or not isinstance(trials, list) or not trials
                        or any(not isinstance(t, str) or not t for t in trials)
                        or len(set(trials)) != len(trials)):
                    raise ValueError("JSONL requires valid expected IDs/count/trials")
                declared_trials = set(trials)
                if manifest is not None and (set(ids) != set(manifest.items)
                                             or set(trials) != set(manifest.trial_ids)):
                    raise ValueError("JSONL header disagrees with manifest")
                # Without an external population, reject incomplete files outright.
                observed = [r.get("item_id") for r in records[1:] if isinstance(r, dict)]
                if set(observed) - set(ids) or (manifest is None and set(observed) != set(ids)):
                    raise ValueError("JSONL records disagree with expected IDs")
                header["items"] = records[1:]
                source = header
            else:
                source = _parse(content)
        obj = _fields(source, {"schema_version", "run_id", "label", "refs", "items"}, set())
        if not isinstance(obj["items"], list) or not isinstance(obj["refs"], dict):
            raise ValueError("items must be a list and refs an object")
        items = []
        for raw_item in obj["items"]:
            item = _fields(raw_item, {"item_id", "stratum", "trials"}, {"source_id", "metadata"})
            if (not isinstance(item["trials"], list)
                    or not isinstance(item.get("metadata", {}), dict)):
                raise ValueError("Malformed trials or metadata")
            trials = []
            for raw_trial in item["trials"]:
                trial = _fields(raw_trial, {"trial_id", "status", "values"}, {"error"})
                if not isinstance(trial["values"], dict):
                    raise ValueError("Trial values must be an object")
                trials.append(Trial(_string(trial["trial_id"], "trial_id"), trial["status"],
                                    {k: _value(v) for k, v in trial["values"].items()},
                                    trial.get("error")))
            items.append(Item(_string(item["item_id"], "item_id"),
                              _string(item["stratum"], "stratum"), trials,
                              item.get("source_id"), item.get("metadata", {})))
        run = Run(_string(obj["run_id"], "run_id"), _string(obj["label"], "label"),
                  items, obj["refs"], obj["schema_version"])
    if run.schema_version != "1":
        raise ValueError("Unsupported run schema_version")
    _string(run.run_id, "run_id")
    _string(run.label, "label")
    if not isinstance(run.refs, dict):
        raise ValueError("refs must be an object")
    json.dumps(run.refs, allow_nan=False)
    seen = set()
    for item in run.items:
        _string(item.item_id, "item_id")
        _string(item.stratum, "stratum")
        if item.source_id is not None:
            _string(item.source_id, "source_id")
        if not isinstance(item.metadata, dict):
            raise ValueError("metadata must be an object")
        json.dumps(item.metadata, allow_nan=False)
        if item.item_id in seen:
            raise ValueError("Duplicate item ID")
        seen.add(item.item_id)
        seen_trials = set()
        for trial in item.trials:
            _string(trial.trial_id, "trial_id")
            if trial.trial_id in seen_trials:
                raise ValueError("Duplicate trial ID")
            seen_trials.add(trial.trial_id)
            if trial.status not in {"ok", "failed"}:
                raise ValueError("Invalid trial status")
            if trial.status == "failed" and (not isinstance(trial.error, str) or not trial.error):
                raise ValueError("Failed trials require an error reason")
            if trial.status == "ok" and trial.error is not None:
                raise ValueError("Successful trials cannot carry an error")
            for value in trial.values.values():
                value_key(value)
        if declared_trials is not None and (
                seen_trials - declared_trials
                or (manifest is None and seen_trials != declared_trials)):
            raise ValueError("JSONL trials disagree with declared trial IDs")
        if manifest is not None:
            if (manifest.items.get(item.item_id) != item.stratum
                    or seen_trials - set(manifest.trial_ids)):
                raise ValueError("Run disagrees with manifest")
    if layers is not None:
        if manifest is None:
            raise ValueError("Layer validation requires an independent manifest")
        validate_run(run, manifest, layers)
    return run
