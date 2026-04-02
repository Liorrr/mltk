"""Versioned evaluation datasets with metadata cards and a local registry.

Provides ``EvalDataset`` for wrapping evaluation samples with version
info and SHA-256 fingerprints, ``DatasetCard`` for HuggingFace-style
metadata, and ``DatasetRegistry`` for filesystem-based storage with
semver-aware versioning.

Architecture::

    CSV / JSON / list[EvalSample]
        |
        v
    EvalDataset (versioned + fingerprinted)
        |
        v
    DatasetRegistry (~/.mltk/datasets/)
        |
        v
    EvalTask.run()

Assertion provided:

- ``assert_dataset_quality`` -- validates sample count, target
  coverage, duplicate rate, and category distribution.

Quick start::

    from mltk.eval.dataset import (
        EvalDataset,
        DatasetRegistry,
        assert_dataset_quality,
    )
    from mltk.eval import EvalSample

    ds = EvalDataset(
        name="my-qa",
        version="1.0.0",
        samples=[
            EvalSample("What is 2+2?", "4"),
            EvalSample("Capital of France?", "Paris"),
        ],
    )

    registry = DatasetRegistry()
    registry.save(ds)
    loaded = registry.load("my-qa")

    result = assert_dataset_quality(loaded, min_samples=1)
    assert result.passed
"""

from __future__ import annotations

import hashlib
import json
import os
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mltk.core.assertion import assert_true, timed_assertion
from mltk.core.result import Severity, TestResult
from mltk.eval._types import EvalSample

# -------------------------------------------------------------------
# Default registry directory -- overridable via MLTK_DATASET_DIR
# -------------------------------------------------------------------
_DEFAULT_DATASET_DIR = Path.home() / ".mltk" / "datasets"


def _resolve_dataset_dir(
    registry_dir: str | Path | None,
) -> Path:
    """Return the effective dataset registry directory.

    Priority: explicit argument > ``MLTK_DATASET_DIR`` env var
    > ``~/.mltk/datasets/``.

    Args:
        registry_dir: Explicit override path.

    Returns:
        Resolved ``Path`` for the dataset registry root.
    """
    if registry_dir is not None:
        return Path(registry_dir)
    env = os.environ.get("MLTK_DATASET_DIR")
    if env:
        return Path(env)
    return _DEFAULT_DATASET_DIR


def _parse_semver(version: str) -> tuple[int, ...]:
    """Split a semver string into a comparable int tuple.

    Args:
        version: Dot-separated version string
            (e.g., ``"1.2.3"``).

    Returns:
        Tuple of ints for comparison
            (e.g., ``(1, 2, 3)``).

    Example:
        >>> _parse_semver("2.1.0")
        (2, 1, 0)
    """
    parts: list[int] = []
    for segment in version.split("."):
        try:
            parts.append(int(segment))
        except ValueError:
            parts.append(0)
    return tuple(parts)


# ===================================================================
# DatasetCard
# ===================================================================


@dataclass
class DatasetCard:
    """Metadata card for an evaluation dataset.

    Inspired by HuggingFace dataset cards and the *Datasheets
    for Datasets* paper (Gebru et al., 2021). Every
    ``EvalDataset`` carries a card describing its purpose,
    provenance, and license.

    The ``created`` field is auto-populated with an ISO-8601
    UTC timestamp when left empty at construction time.

    Args:
        description: Free-text summary of the dataset.
        task: Task type (e.g., ``"qa"``,
            ``"summarization"``).
        source: Origin URL or description.
        license: SPDX identifier (e.g., ``"Apache-2.0"``).
        tags: Searchable labels
            (e.g., ``["english", "qa"]``).
        created: ISO-8601 timestamp (auto-filled on
            creation).
        author: Person or organisation that curated the
            data.

    Example:
        >>> card = DatasetCard(
        ...     description="Geography Q&A pairs",
        ...     task="qa",
        ...     license="CC-BY-4.0",
        ...     tags=["geography", "english"],
        ... )
        >>> len(card.created) > 0
        True
    """

    description: str = ""
    task: str = ""
    source: str = ""
    license: str = ""
    tags: list[str] = field(default_factory=list)
    created: str = ""
    author: str = ""

    def __post_init__(self) -> None:
        if not self.created:
            self.created = (
                datetime.now(timezone.utc).isoformat()
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the card to a JSON-compatible dict.

        Returns:
            Dictionary with all card fields.
        """
        return {
            "description": self.description,
            "task": self.task,
            "source": self.source,
            "license": self.license,
            "tags": list(self.tags),
            "created": self.created,
            "author": self.author,
        }

    @classmethod
    def from_dict(
        cls, data: dict[str, Any]
    ) -> DatasetCard:
        """Deserialize a card from a dict.

        Args:
            data: Dictionary with card fields.

        Returns:
            Reconstructed ``DatasetCard``.
        """
        return cls(
            description=data.get("description", ""),
            task=data.get("task", ""),
            source=data.get("source", ""),
            license=data.get("license", ""),
            tags=list(data.get("tags", [])),
            created=data.get("created", ""),
            author=data.get("author", ""),
        )


# ===================================================================
# EvalDataset
# ===================================================================


@dataclass
class EvalDataset:
    """Versioned evaluation dataset with metadata card.

    Wraps a list of ``EvalSample`` with version info, a
    SHA-256 fingerprint for integrity checking, and a
    ``DatasetCard`` for documentation. The fingerprint is
    computed automatically from sample content and verified
    when loading from the registry.

    Args:
        name: Unique dataset identifier
            (e.g., ``"geography-qa"``).
        version: Semver string (e.g., ``"1.0.0"``).
        samples: List of evaluation samples.
        card: Metadata card (auto-created if omitted).
        fingerprint: SHA-256 hex digest
            (auto-computed if empty).

    Example:
        >>> ds = EvalDataset(
        ...     name="qa-basic",
        ...     version="1.0.0",
        ...     samples=[
        ...         EvalSample("2+2?", "4"),
        ...         EvalSample("Capital of France?",
        ...             "Paris"),
        ...     ],
        ... )
        >>> ds.sample_count
        2
        >>> ds.target_coverage
        1.0
    """

    name: str
    version: str
    samples: list[EvalSample]
    card: DatasetCard = field(
        default_factory=DatasetCard
    )
    fingerprint: str = ""

    def __post_init__(self) -> None:
        if not self.fingerprint:
            self.fingerprint = (
                self._compute_fingerprint()
            )

    # -- fingerprint ------------------------------------------------

    def _compute_fingerprint(self) -> str:
        """SHA-256 of sorted sample content for integrity.

        Samples are sorted by ``input`` text to ensure the
        fingerprint is stable regardless of list order. Both
        ``input`` and ``target`` (when present) contribute
        to the hash.

        Returns:
            Full 64-character hex SHA-256 digest.
        """
        h = hashlib.sha256()
        for s in sorted(
            self.samples, key=lambda x: x.input
        ):
            h.update(s.input.encode("utf-8"))
            if s.target:
                h.update(s.target.encode("utf-8"))
        return h.hexdigest()

    # -- properties -------------------------------------------------

    @property
    def sample_count(self) -> int:
        """Number of samples in the dataset.

        Returns:
            Length of the samples list.
        """
        return len(self.samples)

    @property
    def target_coverage(self) -> float:
        """Fraction of samples with a non-None target.

        Returns:
            Float in [0.0, 1.0]. Returns 0.0 for empty
            datasets.

        Example:
            >>> ds = EvalDataset("x", "1.0.0", [
            ...     EvalSample("a", "b"),
            ...     EvalSample("c"),
            ... ])
            >>> ds.target_coverage
            0.5
        """
        if not self.samples:
            return 0.0
        with_target = sum(
            1
            for s in self.samples
            if s.target is not None
        )
        return with_target / len(self.samples)

    @property
    def categories(self) -> dict[str, int]:
        """Distribution of ``metadata["category"]`` values.

        Only samples whose metadata contains a
        ``"category"`` key are counted. Returns an empty
        dict if no sample has the key.

        Returns:
            Mapping from category name to count.
        """
        counts: dict[str, int] = {}
        for s in self.samples:
            cat = s.metadata.get("category")
            if cat is not None:
                cat_str = str(cat)
                counts[cat_str] = (
                    counts.get(cat_str, 0) + 1
                )
        return counts

    # -- conversion -------------------------------------------------

    def to_sample_list(self) -> list[EvalSample]:
        """Return a copy of samples for ``EvalTask``.

        Returns:
            Shallow copy of the internal samples list.
        """
        return list(self.samples)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the dataset to a JSON-compatible dict.

        The returned dict can be written to disk with
        ``json.dumps`` and loaded back with ``from_dict``.

        Returns:
            Dictionary with all dataset fields including
            samples and fingerprint.

        Example:
            >>> d = ds.to_dict()
            >>> d["name"]
            'qa-basic'
        """
        return {
            "name": self.name,
            "version": self.version,
            "samples": [
                {
                    "input": s.input,
                    "target": s.target,
                    "metadata": dict(s.metadata),
                }
                for s in self.samples
            ],
            "card": self.card.to_dict(),
            "fingerprint": self.fingerprint,
        }

    @classmethod
    def from_dict(
        cls, data: dict[str, Any]
    ) -> EvalDataset:
        """Deserialize a dataset from a dict.

        Handles both full dicts (with card and fingerprint)
        and minimal dicts (name + version + samples).

        Args:
            data: Dictionary with dataset fields.

        Returns:
            Reconstructed ``EvalDataset``.

        Example:
            >>> ds2 = EvalDataset.from_dict(ds.to_dict())
            >>> ds2.name == ds.name
            True
        """
        samples = [
            EvalSample(
                input=s["input"],
                target=s.get("target"),
                metadata=dict(s.get("metadata", {})),
            )
            for s in data.get("samples", [])
        ]
        card_data = data.get("card")
        card = (
            DatasetCard.from_dict(card_data)
            if card_data
            else DatasetCard()
        )
        return cls(
            name=data["name"],
            version=data["version"],
            samples=samples,
            card=card,
            fingerprint=data.get("fingerprint", ""),
        )

    # -- file loaders -----------------------------------------------

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        name: str,
        version: str,
        *,
        input_column: str = "input",
        target_column: str = "target",
        delimiter: str = ",",
        encoding: str = "utf-8",
        **card_kwargs: Any,
    ) -> EvalDataset:
        """Load an evaluation dataset from a CSV file.

        Reads columns for input and target, treating all
        other columns as sample metadata.

        Args:
            path: Path to the CSV file.
            name: Dataset name.
            version: Semver version string.
            input_column: Column name for sample input
                (default ``"input"``).
            target_column: Column name for sample target
                (default ``"target"``).
            delimiter: Field delimiter (default ``","``).
            encoding: File encoding (default ``"utf-8"``).
            **card_kwargs: Passed to ``DatasetCard``.

        Returns:
            Constructed ``EvalDataset``.

        Raises:
            FileNotFoundError: If the CSV file does not
                exist.
            ValueError: If the input column is missing.

        Example:
            >>> ds = EvalDataset.from_csv(
            ...     "data/qa.csv", "qa", "1.0.0",
            ...     description="QA pairs",
            ... )
        """
        import csv

        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(
                f"CSV file not found: {file_path}"
            )

        samples: list[EvalSample] = []
        with open(
            file_path, newline="", encoding=encoding
        ) as fh:
            reader = csv.DictReader(
                fh, delimiter=delimiter
            )
            if (
                reader.fieldnames is not None
                and input_column not in reader.fieldnames
            ):
                raise ValueError(
                    f"Input column '{input_column}' not "
                    f"found. Available: "
                    f"{reader.fieldnames}"
                )
            for row in reader:
                inp = row.get(input_column, "")
                tgt = row.get(target_column)
                meta = {
                    k: v
                    for k, v in row.items()
                    if k
                    not in (
                        input_column,
                        target_column,
                    )
                }
                samples.append(
                    EvalSample(
                        input=inp,
                        target=tgt,
                        metadata=meta,
                    )
                )

        card = DatasetCard(**card_kwargs)
        return cls(
            name=name,
            version=version,
            samples=samples,
            card=card,
        )

    @classmethod
    def from_json(
        cls,
        path: str | Path,
        name: str,
        version: str,
        *,
        input_key: str = "input",
        target_key: str = "target",
        encoding: str = "utf-8",
        **card_kwargs: Any,
    ) -> EvalDataset:
        """Load an evaluation dataset from a JSON file.

        Expects either a JSON array of objects or an object
        with a ``"samples"`` key containing the array.

        Args:
            path: Path to the JSON file.
            name: Dataset name.
            version: Semver version string.
            input_key: Key for sample input
                (default ``"input"``).
            target_key: Key for sample target
                (default ``"target"``).
            encoding: File encoding (default ``"utf-8"``).
            **card_kwargs: Passed to ``DatasetCard``.

        Returns:
            Constructed ``EvalDataset``.

        Raises:
            FileNotFoundError: If the JSON file does not
                exist.

        Example:
            >>> ds = EvalDataset.from_json(
            ...     "data/qa.json", "qa", "1.0.0",
            ... )
        """
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(
                f"JSON file not found: {file_path}"
            )

        with open(file_path, encoding=encoding) as fh:
            raw = json.load(fh)

        # Accept [{"input":...}] and {"samples":[...]}
        if isinstance(raw, dict):
            records = raw.get("samples", [])
        else:
            records = raw

        samples: list[EvalSample] = []
        for rec in records:
            inp = rec.get(input_key, "")
            tgt = rec.get(target_key)
            meta = {
                k: v
                for k, v in rec.items()
                if k not in (input_key, target_key)
            }
            samples.append(
                EvalSample(
                    input=inp,
                    target=tgt,
                    metadata=meta,
                )
            )

        card = DatasetCard(**card_kwargs)
        return cls(
            name=name,
            version=version,
            samples=samples,
            card=card,
        )


# ===================================================================
# DatasetDiff
# ===================================================================


@dataclass
class DatasetDiff:
    """Comparison between two dataset versions.

    Produced by ``DatasetRegistry.diff()`` to summarise what
    changed between two versions of the same dataset. Sample
    lists allow callers to inspect the actual changes.

    Args:
        old_version: Version string of the baseline dataset.
        new_version: Version string of the updated dataset.
        added_samples: Samples in new but not in old.
        removed_samples: Samples in old but not in new.
        unchanged_samples: Samples present in both.
        schema_changes: Metadata key differences between
            versions (e.g., a new key added to metadata).
        suggested_bump: Recommended semver bump type --
            ``"major"``, ``"minor"``, or ``"patch"``.

    Example:
        >>> len(diff.added_samples)
        10
        >>> diff.suggested_bump
        'minor'
    """

    old_version: str
    new_version: str
    added_samples: list[EvalSample]
    removed_samples: list[EvalSample]
    unchanged_samples: list[EvalSample]
    schema_changes: list[str]
    suggested_bump: str


# ===================================================================
# DatasetInfo
# ===================================================================


@dataclass
class DatasetInfo:
    """Summary info for a registered dataset (for listing).

    Returned by ``DatasetRegistry.list()`` to give a quick
    overview without loading all samples into memory.

    Args:
        name: Dataset identifier.
        versions: All registered version strings.
        latest_version: Highest version by semver.
        sample_count: Number of samples in the latest
            version.
        card: Metadata card from the latest version.

    Example:
        >>> info.name
        'geography-qa'
        >>> info.latest_version
        '2.1.0'
    """

    name: str
    versions: list[str]
    latest_version: str
    sample_count: int
    card: DatasetCard


# ===================================================================
# DatasetRegistry
# ===================================================================


class DatasetRegistry:
    """Local filesystem registry for versioned eval datasets.

    Stores datasets under a directory tree with this layout::

        {registry_dir}/{name}/{version}/
            dataset.json    -- serialized EvalDataset
            card.json       -- DatasetCard metadata
            fingerprint.txt -- SHA-256 hex digest

    The default registry lives at ``~/.mltk/datasets/`` and
    can be overridden with the ``MLTK_DATASET_DIR`` env var.

    Args:
        registry_dir: Override path for the registry root.
            Falls back to ``MLTK_DATASET_DIR`` env var, then
            ``~/.mltk/datasets/``.

    Example:
        >>> reg = DatasetRegistry("/tmp/my-datasets")
        >>> reg.save(dataset)
        >>> loaded = reg.load("qa")
    """

    def __init__(
        self,
        registry_dir: str | Path | None = None,
    ) -> None:
        self._root = _resolve_dataset_dir(registry_dir)

    @property
    def root(self) -> Path:
        """The resolved registry root directory.

        Returns:
            Path to the top-level registry directory.
        """
        return self._root

    # -- save / load ------------------------------------------------

    def save(self, dataset: EvalDataset) -> Path:
        """Save a dataset to the registry.

        Creates the version directory and writes three files:
        ``dataset.json``, ``card.json``, and
        ``fingerprint.txt``.

        Args:
            dataset: The ``EvalDataset`` to persist.

        Returns:
            Path to the ``dataset.json`` file.

        Raises:
            ValueError: If name + version already exists.

        Example:
            >>> path = registry.save(ds)
            >>> path.exists()
            True
        """
        version_dir = (
            self._root / dataset.name / dataset.version
        )
        if version_dir.exists():
            raise ValueError(
                f"Dataset '{dataset.name}' version "
                f"'{dataset.version}' already exists at "
                f"{version_dir}"
            )

        version_dir.mkdir(parents=True, exist_ok=True)

        # dataset.json -- full serialization
        dataset_path = version_dir / "dataset.json"
        dataset_path.write_text(
            json.dumps(dataset.to_dict(), indent=2),
            encoding="utf-8",
        )

        # card.json -- metadata only
        card_path = version_dir / "card.json"
        card_path.write_text(
            json.dumps(
                dataset.card.to_dict(), indent=2
            ),
            encoding="utf-8",
        )

        # fingerprint.txt -- integrity check
        fp_path = version_dir / "fingerprint.txt"
        fp_path.write_text(
            dataset.fingerprint, encoding="utf-8"
        )

        return dataset_path

    def load(
        self,
        name: str,
        version: str | None = None,
    ) -> EvalDataset:
        """Load a dataset from the registry.

        If *version* is ``None``, the latest version is
        loaded (determined by semver sorting). The
        fingerprint is verified on load -- a mismatch
        produces a warning but does not raise.

        Args:
            name: Dataset identifier.
            version: Specific version string, or ``None``
                for the latest.

        Returns:
            The deserialized ``EvalDataset``.

        Raises:
            ValueError: If dataset or version not found.

        Example:
            >>> ds = registry.load("qa")
            >>> ds = registry.load("qa", version="1.0.0")
        """
        if version is None:
            versions = self.versions(name)
            if not versions:
                raise ValueError(
                    f"Dataset '{name}' not found in "
                    f"registry at {self._root}"
                )
            version = versions[-1]  # latest by semver

        version_dir = self._root / name / version
        dataset_path = version_dir / "dataset.json"
        if not dataset_path.exists():
            raise ValueError(
                f"Dataset '{name}' version '{version}' "
                f"not found at {version_dir}"
            )

        data = json.loads(
            dataset_path.read_text(encoding="utf-8")
        )
        dataset = EvalDataset.from_dict(data)

        # Verify fingerprint (warn, don't fail).
        # Compare both the JSON-embedded fingerprint and
        # the sidecar fingerprint.txt against a freshly
        # computed value to detect tampering.
        computed_fp = dataset._compute_fingerprint()

        # Check JSON-embedded fingerprint
        json_fp = data.get("fingerprint", "")
        if json_fp and json_fp != computed_fp:
            warnings.warn(
                f"Fingerprint mismatch for "
                f"'{name}' v{version}: "
                f"stored={json_fp}, "
                f"computed={computed_fp}. "
                f"Data may have been modified.",
                stacklevel=2,
            )

        # Check sidecar fingerprint.txt
        fp_path = version_dir / "fingerprint.txt"
        if fp_path.exists():
            stored_fp = fp_path.read_text(
                encoding="utf-8"
            ).strip()
            if (
                stored_fp
                and stored_fp != computed_fp
            ):
                warnings.warn(
                    f"Fingerprint mismatch for "
                    f"'{name}' v{version}: "
                    f"stored={stored_fp}, "
                    f"computed={computed_fp}. "
                    f"Data may have been modified.",
                    stacklevel=2,
                )

        return dataset

    # -- query ------------------------------------------------------

    def list(self) -> list[DatasetInfo]:
        """List all registered datasets with versions.

        Reads each dataset's latest version to populate the
        ``DatasetInfo`` summary. Datasets are sorted by name.

        Returns:
            List of ``DatasetInfo``, one per dataset name.

        Example:
            >>> for info in registry.list():
            ...     print(info.name, info.latest_version)
        """
        if not self._root.exists():
            return []

        infos: list[DatasetInfo] = []
        for entry in sorted(self._root.iterdir()):
            if not entry.is_dir():
                continue
            name = entry.name
            versions = self.versions(name)
            if not versions:
                continue

            latest = versions[-1]
            card = DatasetCard()
            sample_count = 0

            card_path = entry / latest / "card.json"
            if card_path.exists():
                try:
                    card_data = json.loads(
                        card_path.read_text(
                            encoding="utf-8"
                        )
                    )
                    card = DatasetCard.from_dict(
                        card_data
                    )
                except (
                    json.JSONDecodeError,
                    KeyError,
                ):
                    pass

            ds_path = (
                entry / latest / "dataset.json"
            )
            if ds_path.exists():
                try:
                    ds_data = json.loads(
                        ds_path.read_text(
                            encoding="utf-8"
                        )
                    )
                    sample_count = len(
                        ds_data.get("samples", [])
                    )
                except (
                    json.JSONDecodeError,
                    KeyError,
                ):
                    pass

            infos.append(
                DatasetInfo(
                    name=name,
                    versions=versions,
                    latest_version=latest,
                    sample_count=sample_count,
                    card=card,
                )
            )

        return infos

    def versions(self, name: str) -> list[str]:
        """List all versions of a dataset, sorted by semver.

        Args:
            name: Dataset identifier.

        Returns:
            Sorted list of version strings (ascending).
            Empty list if the dataset does not exist.

        Example:
            >>> registry.versions("qa")
            ['1.0.0', '1.1.0', '2.0.0']
        """
        dataset_dir = self._root / name
        if not dataset_dir.exists():
            return []

        found: list[str] = []
        for entry in dataset_dir.iterdir():
            if not entry.is_dir():
                continue
            if (entry / "dataset.json").exists():
                found.append(entry.name)

        return sorted(found, key=_parse_semver)

    def exists(
        self,
        name: str,
        version: str | None = None,
    ) -> bool:
        """Check if a dataset (and optionally version) exists.

        Args:
            name: Dataset identifier.
            version: If provided, check for this specific
                version.

        Returns:
            True if the dataset (and version, if given)
            exists.

        Example:
            >>> registry.exists("qa")
            True
            >>> registry.exists("qa", "1.0.0")
            True
        """
        if version is None:
            return len(self.versions(name)) > 0

        version_dir = self._root / name / version
        return (version_dir / "dataset.json").exists()

    def diff(
        self,
        name: str,
        old_version: str,
        new_version: str,
    ) -> DatasetDiff:
        """Compare two versions of a dataset.

        Loads both versions and compares samples by their
        ``input`` text. Also detects metadata key
        differences and suggests a semver bump type.

        Bump logic:

        - ``"major"`` -- samples were removed or schema
          keys changed.
        - ``"minor"`` -- samples were added (none removed,
          no schema changes).
        - ``"patch"`` -- identical sample sets (only
          metadata values may differ).

        Args:
            name: Dataset identifier.
            old_version: Baseline version string.
            new_version: Updated version string.

        Returns:
            ``DatasetDiff`` summarising the changes.

        Example:
            >>> diff = registry.diff(
            ...     "qa", "1.0.0", "2.0.0",
            ... )
            >>> len(diff.added_samples)
            5
        """
        old_ds = self.load(name, old_version)
        new_ds = self.load(name, new_version)

        old_by_input = {
            s.input: s for s in old_ds.samples
        }
        new_by_input = {
            s.input: s for s in new_ds.samples
        }

        old_inputs = set(old_by_input.keys())
        new_inputs = set(new_by_input.keys())

        added = [
            new_by_input[inp]
            for inp in sorted(new_inputs - old_inputs)
        ]
        removed = [
            old_by_input[inp]
            for inp in sorted(old_inputs - new_inputs)
        ]
        unchanged = [
            new_by_input[inp]
            for inp in sorted(old_inputs & new_inputs)
        ]

        # Detect metadata schema changes
        old_keys: set[str] = set()
        for s in old_ds.samples:
            old_keys.update(s.metadata.keys())

        new_keys: set[str] = set()
        for s in new_ds.samples:
            new_keys.update(s.metadata.keys())

        schema_changes: list[str] = []
        for k in sorted(new_keys - old_keys):
            schema_changes.append(f"added key: {k}")
        for k in sorted(old_keys - new_keys):
            schema_changes.append(
                f"removed key: {k}"
            )

        # Suggest bump type
        if removed or schema_changes:
            suggested = "major"
        elif added:
            suggested = "minor"
        else:
            suggested = "patch"

        return DatasetDiff(
            old_version=old_version,
            new_version=new_version,
            added_samples=added,
            removed_samples=removed,
            unchanged_samples=unchanged,
            schema_changes=schema_changes,
            suggested_bump=suggested,
        )

    def delete(
        self, name: str, version: str
    ) -> bool:
        """Delete a specific version of a dataset.

        Removes the version directory and its contents. If
        the dataset has no remaining versions, the dataset
        directory is also removed.

        Args:
            name: Dataset identifier.
            version: Version string to delete.

        Returns:
            True if the version was found and deleted,
            False if it did not exist.

        Example:
            >>> registry.delete("qa", "1.0.0")
            True
        """
        import shutil

        version_dir = self._root / name / version
        if not version_dir.exists():
            return False

        shutil.rmtree(version_dir)

        # Clean up empty parent directory
        dataset_dir = self._root / name
        if dataset_dir.exists() and not any(
            dataset_dir.iterdir()
        ):
            dataset_dir.rmdir()

        return True


# ===================================================================
# Assertion
# ===================================================================


@timed_assertion
def assert_dataset_quality(
    dataset: EvalDataset,
    *,
    min_samples: int = 50,
    min_target_coverage: float = 0.9,
    max_duplicate_rate: float = 0.01,
    min_categories: int | None = None,
) -> TestResult:
    """Assert evaluation dataset meets quality standards.

    Validates sample count, target coverage, duplicate rate,
    and (optionally) category diversity. Useful as a gate
    before running expensive evaluation pipelines.

    Assertion name: ``eval.dataset.quality``

    Args:
        dataset: The ``EvalDataset`` to validate.
        min_samples: Minimum required number of samples
            (default 50).
        min_target_coverage: Minimum fraction of samples
            with a non-None target (default 0.9).
        max_duplicate_rate: Maximum allowed fraction of
            duplicate inputs (default 0.01 = 1%).
        min_categories: If set, minimum number of distinct
            ``metadata["category"]`` values required.

    Returns:
        ``TestResult`` with pass/fail status, detailed
        metrics, and timing information.

    Example:
        >>> result = assert_dataset_quality(
        ...     dataset, min_samples=100,
        ... )
        >>> assert result.passed
    """
    issues: list[str] = []

    # Check sample count
    count = dataset.sample_count
    if count < min_samples:
        issues.append(
            f"sample_count={count} < "
            f"min_samples={min_samples}"
        )

    # Check target coverage
    coverage = dataset.target_coverage
    if coverage < min_target_coverage:
        issues.append(
            f"target_coverage={coverage:.3f} < "
            f"min_target_coverage="
            f"{min_target_coverage}"
        )

    # Check duplicate rate
    inputs = [s.input for s in dataset.samples]
    unique_count = len(set(inputs))
    total = len(inputs)
    if total > 0:
        dup_rate = 1.0 - (unique_count / total)
    else:
        dup_rate = 0.0
    if dup_rate > max_duplicate_rate:
        issues.append(
            f"duplicate_rate={dup_rate:.3f} > "
            f"max_duplicate_rate="
            f"{max_duplicate_rate}"
        )

    # Check category diversity
    cats = dataset.categories
    if min_categories is not None:
        cat_count = len(cats)
        if cat_count < min_categories:
            issues.append(
                f"categories={cat_count} < "
                f"min_categories={min_categories}"
            )

    passed = len(issues) == 0
    message = (
        f"Dataset '{dataset.name}' "
        f"v{dataset.version} passed all quality "
        f"checks ({count} samples, "
        f"{coverage:.0%} coverage)"
        if passed
        else (
            f"Dataset '{dataset.name}' "
            f"v{dataset.version} failed quality "
            f"checks: " + "; ".join(issues)
        )
    )

    return assert_true(
        passed,
        name="eval.dataset.quality",
        message=message,
        severity=Severity.CRITICAL,
        sample_count=count,
        target_coverage=round(coverage, 4),
        duplicate_rate=round(dup_rate, 4),
        categories=len(cats),
        issues=issues,
    )


# ===================================================================
# Exports
# ===================================================================

__all__ = [
    "DatasetCard",
    "EvalDataset",
    "DatasetDiff",
    "DatasetInfo",
    "DatasetRegistry",
    "assert_dataset_quality",
]
