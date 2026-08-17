"""Golden-set binding for imported datasets.

Bind a user-provided golden/reference file onto an imported
:class:`~mltk.eval.dataset.EvalDataset`, filling each sample's
``target`` from the golden data. Binding is done by a key-column join
(a sample-side key matched against a golden-side key column) or, when no
key is given, by row order.

Every bound sample is stamped with ``metadata["scoring"]``:

* ``"exact"`` -- the sample ends up with a concrete ``target`` (from the
  golden file or a pre-existing in-dataset golden column). Generated
  Tier-2 tests score these against the reference.
* ``"judge"`` -- no reference is available. Generated Tier-2 tests fall
  back to an LLM judge (opt-in) that scores the model answer
  reference-free.

The binding never scores anything itself -- there is no model output at
import time. It only partitions samples and selects which scorer the
generated pytest scaffold will use per sample.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mltk.eval._types import EvalSample
from mltk.eval.dataset import EvalDataset

SCORING_EXACT = "exact"
SCORING_JUDGE = "judge"

# Sentinel key meaning "join on the sample's own input text".
INPUT_KEY = "input"


@dataclass
class GoldenSpec:
    """Declarative golden-binding request carried through the pipeline.

    Used by the CLI and MCP tool to describe a golden binding and by
    :func:`~mltk.importer.codegen.generate_pytest` to emit a
    self-contained binding fixture into the generated scaffold.

    Args:
        path: Path to the golden/reference file (CSV/TSV/JSON/JSONL).
        target_column: Column in the golden file holding the reference
            answer.
        key: Sample-side join key -- ``"input"`` to match on the
            sample's input text, otherwise a ``metadata`` field name.
            ``None`` binds by row order.
        golden_key: Golden-side key column. Defaults to ``key`` when a
            key is given.
        judge: Whether the judge fallback for unmatched samples is
            enabled in the generated scaffold.
    """

    path: str
    target_column: str
    key: str | None = None
    golden_key: str | None = None
    judge: bool = False


@dataclass
class GoldenBindingReport:
    """Outcome of a :func:`bind_golden` call.

    Args:
        total: Number of samples in the dataset.
        matched: Number of samples that received a target from the
            golden file.
        unmatched: Sample indices with no concrete target after binding
            (these are stamped ``metadata["scoring"] == "judge"``).
        key: The sample-side join key used (``None`` for row order).
    """

    total: int
    matched: int
    unmatched: list[int]
    key: str | None

    @property
    def match_rate(self) -> float:
        """Fraction of samples that received a golden target."""
        if self.total == 0:
            return 0.0
        return self.matched / self.total

    def summary(self) -> str:
        """Return a one-line human-readable summary."""
        mode = "row-order" if self.key is None else f"key={self.key!r}"
        return (
            f"golden binding ({mode}): {self.matched}/{self.total} matched "
            f"({self.match_rate:.0%}), {len(self.unmatched)} judge-scored"
        )


def _stringify(value: Any) -> str | None:
    """Normalize a cell value to a non-empty string or ``None``.

    Missing, NaN, and empty/whitespace values become ``None`` -- mirrors
    the target-normalization rules in
    :meth:`ImportResult.to_eval_dataset`.
    """
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    text = str(value).strip()
    return text or None


def load_golden(path: str | Path) -> list[dict[str, Any]]:
    """Load a golden/reference file into a list of row dicts.

    Supported formats (by extension):

    * ``.csv`` / ``.tsv`` -- delimited text (stdlib ``csv``; values are
      strings).
    * ``.json`` -- a top-level JSON array of objects.
    * ``.jsonl`` / ``.ndjson`` -- one JSON object per line.
    * ``.parquet`` -- via pandas (optional dependency).

    Args:
        path: Path to the golden file.

    Returns:
        List of row dictionaries.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If the extension is unsupported or the JSON payload
            is not a list of objects.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Golden file not found: {p}")

    suffix = p.suffix.lower()

    if suffix in {".csv", ".tsv"}:
        import csv

        delimiter = "\t" if suffix == ".tsv" else ","
        # utf-8-sig strips a leading BOM (Excel exports one) and reads
        # plain UTF-8 unchanged, so the first header never carries ﻿.
        with p.open(newline="", encoding="utf-8-sig") as fh:
            return [dict(row) for row in csv.DictReader(fh, delimiter=delimiter)]

    if suffix in {".jsonl", ".ndjson"}:
        rows: list[dict[str, Any]] = []
        with p.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if not isinstance(obj, dict):
                    raise ValueError(
                        f"Each JSONL line must be an object, got {type(obj).__name__}"
                    )
                rows.append(obj)
        return rows

    if suffix == ".json":
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, list) or not all(
            isinstance(item, dict) for item in data
        ):
            raise ValueError(
                "JSON golden file must be a top-level array of objects."
            )
        return list(data)

    if suffix == ".parquet":
        try:
            import pandas as pd
        except ImportError as err:  # pragma: no cover - optional dep
            raise ValueError(
                "Reading .parquet golden files requires pandas. "
                "Install it with: pip install mlspec[importer]"
            ) from err
        return pd.read_parquet(p).to_dict("records")

    raise ValueError(
        f"Unsupported golden file format: {suffix!r}. "
        "Use .csv, .tsv, .json, .jsonl, or .parquet."
    )


def _sample_key(sample: EvalSample, key: str) -> str | None:
    """Return the join value for *sample* under the sample-side *key*."""
    if key == INPUT_KEY:
        return _stringify(sample.input)
    return _stringify(sample.metadata.get(key))


def bind_golden(
    dataset: EvalDataset,
    golden: list[dict[str, Any]],
    *,
    target_column: str,
    key: str | None = None,
    golden_key: str | None = None,
) -> tuple[EvalDataset, GoldenBindingReport]:
    """Bind a golden reference set onto *dataset*.

    Returns a **new** :class:`~mltk.eval.dataset.EvalDataset` whose
    samples have their ``target`` filled from the golden data and a
    ``metadata["scoring"]`` marker of ``"exact"`` or ``"judge"``. The
    original dataset is not mutated; the card (provenance) is preserved
    and the fingerprint is recomputed from the new sample content.

    Join semantics:

    * ``key is None`` -- bind by row order: ``golden[i]`` supplies the
      target for sample ``i``. Extra golden rows are ignored; missing
      rows leave the sample unbound.
    * ``key`` given -- build a lookup from the golden-side ``golden_key``
      column (defaults to ``key``) to the ``target_column`` value, then
      match each sample's key. Use ``key="input"`` to match on the
      sample input text; any other value is read from sample metadata.

    A golden value overrides a pre-existing target. When the golden file
    supplies no value, a pre-existing target is kept (still ``"exact"``);
    only samples that end up with no target at all are marked
    ``"judge"``.

    Args:
        dataset: The imported dataset to bind onto.
        golden: Golden rows (e.g. from :func:`load_golden`).
        target_column: Column in the golden rows holding the reference.
        key: Sample-side join key, or ``None`` for row order.
        golden_key: Golden-side key column (defaults to ``key``).

    Returns:
        ``(bound_dataset, report)``.

    Raises:
        ValueError: If ``target_column`` (or the resolved ``golden_key``)
            is absent from the golden rows.
    """
    if golden and target_column not in golden[0]:
        raise ValueError(
            f"target_column={target_column!r} not in golden columns "
            f"{list(golden[0])}."
        )

    lookup: dict[str, str | None] = {}
    if key is not None:
        gkey = golden_key or key
        if golden and gkey not in golden[0]:
            raise ValueError(
                f"golden_key={gkey!r} not in golden columns {list(golden[0])}."
            )
        for row in golden:
            k = _stringify(row.get(gkey))
            if k is not None and k not in lookup:
                lookup[k] = _stringify(row.get(target_column))

    new_samples: list[EvalSample] = []
    unmatched: list[int] = []
    matched = 0

    for idx, sample in enumerate(dataset.samples):
        golden_value: str | None = None
        if key is None:
            if idx < len(golden):
                golden_value = _stringify(golden[idx].get(target_column))
        else:
            sk = _sample_key(sample, key)
            if sk is not None:
                golden_value = lookup.get(sk)

        if golden_value is not None:
            matched += 1

        new_target = golden_value if golden_value is not None else sample.target
        scoring = SCORING_EXACT if new_target is not None else SCORING_JUDGE
        if scoring == SCORING_JUDGE:
            unmatched.append(idx)

        new_metadata = {**sample.metadata, "scoring": scoring}
        new_samples.append(
            EvalSample(
                input=sample.input,
                target=new_target,
                metadata=new_metadata,
            )
        )

    bound = EvalDataset(
        name=dataset.name,
        version=dataset.version,
        samples=new_samples,
        card=dataset.card,
    )
    report = GoldenBindingReport(
        total=len(dataset.samples),
        matched=matched,
        unmatched=unmatched,
        key=key,
    )
    return bound, report
