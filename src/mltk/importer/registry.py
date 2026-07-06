"""Quality-gated registration of imported datasets.

Wraps :class:`~mltk.eval.dataset.DatasetRegistry` with an
``assert_dataset_quality`` gate that must pass before an imported
dataset is written to the local registry (``~/.mltk/datasets/`` by
default, or ``MLTK_DATASET_DIR``). The gate is *blocking*: a dataset
that fails the quality check is never saved.

Gate defaults are import-oriented and lenient on shape (small and
unlabeled eval sets are legitimate); the meaningful default guard is the
duplicate rate -- a mostly-duplicate dataset signals a broken import.
Callers can tighten any threshold. Provenance stamped in
``DatasetCard.source`` by ``to_eval_dataset`` is preserved as-is.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mltk.eval.dataset import EvalDataset

# Import-oriented gate defaults. Sample count and target coverage are
# intentionally permissive; duplicate rate is the load-bearing guard.
DEFAULT_MIN_SAMPLES = 1
DEFAULT_MIN_TARGET_COVERAGE = 0.0
DEFAULT_MAX_DUPLICATE_RATE = 0.5
DEFAULT_MIN_CATEGORIES: int | None = None


@dataclass
class RegistrationResult:
    """Outcome of a :func:`register_dataset` call.

    Args:
        saved: Whether the dataset was written to the registry.
        quality_passed: Whether the ``assert_dataset_quality`` gate
            passed.
        quality_detail: Assertion details (sample_count, coverage,
            duplicate_rate, categories, issues).
        reason: Human-readable explanation of the outcome.
        name: Dataset name.
        version: Dataset version.
        path: Path to the written ``dataset.json`` (``None`` when not
            saved).
    """

    saved: bool
    quality_passed: bool
    quality_detail: dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    name: str = ""
    version: str = ""
    path: str | None = None


def register_dataset(
    dataset: EvalDataset,
    *,
    registry_dir: str | Path | None = None,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    min_target_coverage: float = DEFAULT_MIN_TARGET_COVERAGE,
    max_duplicate_rate: float = DEFAULT_MAX_DUPLICATE_RATE,
    min_categories: int | None = DEFAULT_MIN_CATEGORIES,
    overwrite: bool = False,
) -> RegistrationResult:
    """Run the quality gate, then save *dataset* to the registry if it passes.

    Args:
        dataset: The imported dataset to register.
        registry_dir: Override the registry root (else ``MLTK_DATASET_DIR``
            then ``~/.mltk/datasets/``).
        min_samples: Minimum required sample count.
        min_target_coverage: Minimum fraction of samples with a target.
        max_duplicate_rate: Maximum allowed duplicate-input fraction.
        min_categories: Minimum distinct ``metadata["category"]`` values,
            or ``None`` to skip the check.
        overwrite: If ``True``, replace an existing ``name/version``
            instead of refusing. Defaults to ``False`` (non-destructive).

    Returns:
        A :class:`RegistrationResult`. ``saved`` is ``False`` when the
        gate fails or when the version already exists and ``overwrite``
        is ``False``.
    """
    from mltk.core.assertion import MltkAssertionError
    from mltk.eval.dataset import DatasetRegistry, assert_dataset_quality

    try:
        result = assert_dataset_quality(
            dataset,
            min_samples=min_samples,
            min_target_coverage=min_target_coverage,
            max_duplicate_rate=max_duplicate_rate,
            min_categories=min_categories,
        )
    except MltkAssertionError as err:
        result = err.result

    detail = dict(result.details)

    if not result.passed:
        return RegistrationResult(
            saved=False,
            quality_passed=False,
            quality_detail=detail,
            reason=result.message,
            name=dataset.name,
            version=dataset.version,
        )

    registry = DatasetRegistry(registry_dir)
    if registry.exists(dataset.name, dataset.version):
        if not overwrite:
            return RegistrationResult(
                saved=False,
                quality_passed=True,
                quality_detail=detail,
                reason=(
                    f"Dataset '{dataset.name}' v{dataset.version} already "
                    "exists. Pass overwrite=True (or bump the version) to "
                    "replace it."
                ),
                name=dataset.name,
                version=dataset.version,
            )
        registry.delete(dataset.name, dataset.version)

    path = registry.save(dataset)
    return RegistrationResult(
        saved=True,
        quality_passed=True,
        quality_detail=detail,
        reason=(
            f"Registered '{dataset.name}' v{dataset.version} "
            f"({dataset.sample_count} samples)."
        ),
        name=dataset.name,
        version=dataset.version,
        path=str(path),
    )
