"""Feature, model, and sensitive-column detection for mltk scan.

Auto-discovers column types (numeric vs categorical), model type
(classifier vs regressor), and likely sensitive/protected attributes.
Used by ScanEngine to build ScanContext without manual configuration.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Callable

import numpy as np
import pandas as pd

__all__ = [
    "detect_feature_types",
    "detect_model_type",
    "detect_sensitive_columns",
]

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# Sensitive-column keyword set
# -------------------------------------------------------------------

_SENSITIVE_KEYWORDS: frozenset[str] = frozenset({
    "gender",
    "sex",
    "race",
    "ethnicity",
    "age",
    "religion",
    "disability",
    "zipcode",
    "zip_code",
    "zip",
    "marital",
    "nationality",
    "native",
    "color",
    "orientation",
    "veteran",
    "pregnant",
    "income",
})


# -------------------------------------------------------------------
# Feature-type detection
# -------------------------------------------------------------------

def detect_feature_types(
    X: pd.DataFrame,
    categorical_threshold: int = 20,
) -> tuple[list[str], list[str]]:
    """Auto-detect numeric vs categorical columns.

    A column is **categorical** when:
      - its dtype is ``object`` or ``category``, OR
      - its number of unique values is <= *categorical_threshold*.

    Everything else is treated as numeric.

    Args:
        X: Feature DataFrame to inspect.
        categorical_threshold: Max unique values for a numeric
            column to still be considered categorical.

    Returns:
        ``(numeric_columns, categorical_columns)`` -- two lists
        of column names.  Every column in *X* appears in exactly
        one of the two lists.
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError(
            f"X must be a pandas DataFrame, got {type(X).__name__}"
        )
    if categorical_threshold < 0:
        raise ValueError(
            "categorical_threshold must be >= 0, "
            f"got {categorical_threshold}"
        )

    numeric: list[str] = []
    categorical: list[str] = []

    for col in X.columns:
        dtype = X[col].dtype

        # Object / explicit category → always categorical
        if dtype == "object" or dtype.name == "category":
            categorical.append(str(col))
            continue

        # Low-cardinality numeric → treat as categorical
        n_unique = X[col].nunique(dropna=True)
        if n_unique <= categorical_threshold:
            categorical.append(str(col))
        else:
            numeric.append(str(col))

    logger.debug(
        "Detected %d numeric, %d categorical columns",
        len(numeric),
        len(categorical),
    )
    return numeric, categorical


# -------------------------------------------------------------------
# Model-type detection
# -------------------------------------------------------------------

_PROBE_SIZE: int = 32


def detect_model_type(
    model_fn: Callable[..., np.ndarray],
    X_sample: pd.DataFrame | np.ndarray,
) -> str:
    """Detect whether *model_fn* is a classifier or regressor.

    Detection strategy (in order):

    1. **Duck-typing** -- check for ``predict_proba``,
       ``classes_``, or ``is_classifier()`` on the underlying
       object if *model_fn* exposes one.
    2. **Output probe** -- call *model_fn* on up to 32 rows,
       inspect the dtype, value range, and shape of the output.
    3. **Fallback** -- return ``"unknown"`` if neither heuristic
       is conclusive.

    Args:
        model_fn: A callable that accepts feature data and returns
            predictions (``np.ndarray`` or similar).
        X_sample: A small sample of feature data used for probing.

    Returns:
        ``"classifier"``, ``"regressor"``, or ``"unknown"``.
    """
    # -- Step 1: duck-typing on the callable/object ----------------
    obj = getattr(model_fn, "__self__", model_fn)

    if hasattr(obj, "predict_proba"):
        logger.debug("detect_model_type: predict_proba found")
        return "classifier"

    if hasattr(obj, "classes_"):
        logger.debug("detect_model_type: classes_ found")
        return "classifier"

    if callable(getattr(obj, "is_classifier", None)):
        try:
            if obj.is_classifier():
                logger.debug(
                    "detect_model_type: is_classifier() True"
                )
                return "classifier"
        except Exception:  # noqa: BLE001
            pass

    # -- Step 2: output probe on a small slice ---------------------
    try:
        sample = _safe_sample(X_sample, _PROBE_SIZE)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            preds = np.asarray(model_fn(sample))
    except Exception:  # noqa: BLE001
        logger.debug(
            "detect_model_type: probe failed, returning unknown"
        )
        return "unknown"

    return _classify_output(preds)


def _safe_sample(
    X: pd.DataFrame | np.ndarray,
    n: int,
) -> pd.DataFrame | np.ndarray:
    """Return at most *n* rows from *X*."""
    if isinstance(X, pd.DataFrame):
        return X.head(n)
    return X[:n]


def _classify_output(preds: np.ndarray) -> str:
    """Heuristic classification of model output."""
    if preds.size == 0:
        return "unknown"

    flat = preds.ravel()

    # Integer dtype → likely classifier
    if np.issubdtype(flat.dtype, np.integer):
        logger.debug("detect_model_type: integer output")
        return "classifier"

    # Boolean dtype → classifier
    if np.issubdtype(flat.dtype, np.bool_):
        logger.debug("detect_model_type: boolean output")
        return "classifier"

    # String / object dtype → classifier
    if flat.dtype.kind in ("U", "S", "O"):
        logger.debug("detect_model_type: string/object output")
        return "classifier"

    # Floating-point checks
    if np.issubdtype(flat.dtype, np.floating):
        unique_vals = np.unique(flat[~np.isnan(flat)])

        # Two unique values close to 0/1 → binary classifier
        if len(unique_vals) <= 2:
            logger.debug(
                "detect_model_type: <=2 unique floats"
            )
            return "classifier"

        # Few unique values relative to sample size →
        # likely multi-class classifier
        ratio = len(unique_vals) / max(len(flat), 1)
        if ratio < 0.05 and len(unique_vals) <= 20:
            logger.debug(
                "detect_model_type: low cardinality floats "
                "(ratio=%.3f, unique=%d)",
                ratio,
                len(unique_vals),
            )
            return "classifier"

        # Continuous spread → regressor
        logger.debug("detect_model_type: continuous floats")
        return "regressor"

    return "unknown"


# -------------------------------------------------------------------
# Sensitive-column detection
# -------------------------------------------------------------------

def detect_sensitive_columns(
    X: pd.DataFrame,
) -> list[str]:
    """Auto-detect likely sensitive / protected-attribute columns.

    Matching is case-insensitive and substring-based: a column
    named ``applicant_gender`` matches keyword ``gender``.

    The keyword set covers common US and EU protected classes:
    gender, sex, race, ethnicity, age, religion, disability,
    zipcode, marital status, nationality, income, color,
    orientation, veteran status, and pregnancy.

    Args:
        X: Feature DataFrame whose column names are inspected.

    Returns:
        Sorted list of column names that match at least one
        sensitive keyword.  Empty list when nothing matches.
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError(
            f"X must be a pandas DataFrame, got {type(X).__name__}"
        )

    matches: list[str] = []
    for col in X.columns:
        col_lower = str(col).lower().replace("-", "_")
        for kw in _SENSITIVE_KEYWORDS:
            if kw in col_lower:
                matches.append(str(col))
                break

    matches.sort()
    logger.debug(
        "Detected %d sensitive columns: %s",
        len(matches),
        matches,
    )
    return matches
