"""Model loading for mltk scan CLI.

Loads serialized models from disk and normalizes them into a
uniform ``predict_fn(X) -> np.ndarray`` callable.  Supports
pickle/joblib, ONNX, PyTorch, and Keras/HDF5 formats.

**Security warning**: pickle and joblib files can execute
arbitrary code during deserialization.  Only load models that
you trust.

All format-specific dependencies are optional.  When a required
library is missing, a clear ``ImportError`` is raised with
install instructions.
"""

from __future__ import annotations

import logging
import sys
import warnings
from collections.abc import Callable
from pathlib import Path

import numpy as np

__all__ = [
    "load_model",
    "LoadedModel",
]

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# Extension → format mapping
# -------------------------------------------------------------------

_EXT_FORMAT_MAP: dict[str, str] = {
    ".pkl": "pickle",
    ".pickle": "pickle",
    ".joblib": "joblib",
    ".onnx": "onnx",
    ".pt": "torch",
    ".pth": "torch",
    ".h5": "keras",
    ".hdf5": "keras",
    ".keras": "keras",
}


# -------------------------------------------------------------------
# Result container
# -------------------------------------------------------------------

class LoadedModel:
    """Normalized model loaded from disk.

    Attributes:
        predict_fn: ``(X: np.ndarray) -> np.ndarray`` callable
            that returns class labels or regression values.
        predict_proba_fn: Optional ``(X: np.ndarray) -> np.ndarray``
            callable that returns class probabilities.  ``None``
            when the model does not support probability output.
        raw: The underlying model object before wrapping.
        format: Format string (``"pickle"``, ``"onnx"``, etc.).
        path: Absolute path the model was loaded from.
    """

    __slots__ = (
        "predict_fn",
        "predict_proba_fn",
        "raw",
        "format",
        "path",
    )

    def __init__(
        self,
        predict_fn: Callable[..., np.ndarray],
        predict_proba_fn: Callable[..., np.ndarray] | None,
        raw: object,
        fmt: str,
        path: Path,
    ) -> None:
        self.predict_fn = predict_fn
        self.predict_proba_fn = predict_proba_fn
        self.raw = raw
        self.format = fmt
        self.path = path

    def __repr__(self) -> str:  # pragma: no cover
        proba = self.predict_proba_fn is not None
        return (
            f"LoadedModel(format={self.format!r}, "
            f"path={str(self.path)!r}, "
            f"has_proba={proba})"
        )


# -------------------------------------------------------------------
# Security warning
# -------------------------------------------------------------------

_PICKLE_WARNING = (
    "WARNING: Loading a pickle/joblib model can execute "
    "arbitrary code.  Only load models you trust."
)


def _emit_pickle_warning() -> None:
    """Print a security warning to stderr."""
    print(_PICKLE_WARNING, file=sys.stderr)
    logger.warning(_PICKLE_WARNING)


# -------------------------------------------------------------------
# Public API
# -------------------------------------------------------------------

def load_model(
    path: str | Path,
    fmt: str | None = None,
    output_type: str | None = None,
) -> LoadedModel:
    """Load a serialized model and return normalized callables.

    Auto-detects the format from the file extension when *fmt*
    is ``None``.  Supported formats:

    - **pickle** (``.pkl``, ``.pickle``): via ``joblib.load``
    - **joblib** (``.joblib``): via ``joblib.load``
    - **onnx** (``.onnx``): via ``onnxruntime``
    - **torch** (``.pt``, ``.pth``): via ``torch.load``
    - **keras** (``.h5``, ``.hdf5``, ``.keras``): via
      ``keras.models.load_model``

    Args:
        path: Path to the model file on disk.
        fmt: Explicit format override.  When ``None``,
            the format is inferred from the file extension.
        output_type: Hint for raw model output.  Set to
            ``"logits"`` to apply softmax wrapping on the
            ``predict_proba_fn`` output.

    Returns:
        A :class:`LoadedModel` with normalized predict
        callables.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If the format cannot be determined.
        ImportError: If the required library is not installed.
    """
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"Model file not found: {path}"
        )

    if fmt is None:
        fmt = _detect_format(path)

    loader = _FORMAT_LOADERS.get(fmt)
    if loader is None:
        supported = ", ".join(sorted(_FORMAT_LOADERS))
        raise ValueError(
            f"Unsupported model format {fmt!r}.  "
            f"Supported: {supported}"
        )

    logger.info("Loading model from %s (format=%s)", path, fmt)
    return loader(path, fmt, output_type)


# -------------------------------------------------------------------
# Format detection
# -------------------------------------------------------------------

def _detect_format(path: Path) -> str:
    """Infer model format from file extension."""
    suffix = path.suffix.lower()
    fmt = _EXT_FORMAT_MAP.get(suffix)
    if fmt is None:
        supported = ", ".join(sorted(_EXT_FORMAT_MAP))
        raise ValueError(
            f"Cannot determine model format from "
            f"extension {suffix!r}.  "
            f"Supported extensions: {supported}.  "
            f"Use the fmt parameter to specify explicitly."
        )
    return fmt


# -------------------------------------------------------------------
# Per-format loaders
# -------------------------------------------------------------------

def _load_pickle(
    path: Path,
    fmt: str,
    output_type: str | None,
) -> LoadedModel:
    """Load pickle or joblib model."""
    _emit_pickle_warning()

    try:
        import joblib
    except ImportError as exc:
        raise ImportError(
            "joblib is required to load pickle/joblib models.  "
            "Install it with: pip install joblib"
        ) from exc

    model = joblib.load(path)
    predict_fn = _extract_predict(model)
    proba_fn = _extract_predict_proba(model, output_type)
    return LoadedModel(predict_fn, proba_fn, model, fmt, path)


def _load_onnx(
    path: Path,
    fmt: str,
    output_type: str | None,
) -> LoadedModel:
    """Load ONNX model via onnxruntime."""
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise ImportError(
            "onnxruntime is required to load ONNX models.  "
            "Install it with: pip install onnxruntime"
        ) from exc

    session = ort.InferenceSession(str(path))
    input_name = session.get_inputs()[0].name
    output_names = [o.name for o in session.get_outputs()]

    def predict_fn(X: np.ndarray) -> np.ndarray:
        X_float = np.asarray(X, dtype=np.float32)
        results = session.run(
            output_names, {input_name: X_float}
        )
        return np.asarray(results[0])

    proba_fn: Callable[..., np.ndarray] | None = None
    if len(output_names) >= 2:
        def _proba(X: np.ndarray) -> np.ndarray:
            X_float = np.asarray(X, dtype=np.float32)
            results = session.run(
                output_names, {input_name: X_float}
            )
            raw = np.asarray(results[1])
            return _maybe_softmax(raw, output_type)

        proba_fn = _proba

    return LoadedModel(predict_fn, proba_fn, session, fmt, path)


def _load_torch(
    path: Path,
    fmt: str,
    output_type: str | None,
) -> LoadedModel:
    """Load PyTorch model."""
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "torch is required to load PyTorch models.  "
            "Install it with: pip install torch"
        ) from exc

    _emit_pickle_warning()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = torch.load(
            str(path),
            map_location="cpu",
            weights_only=False,
        )

    if hasattr(model, "eval"):
        model.eval()

    def predict_fn(X: np.ndarray) -> np.ndarray:
        X_tensor = torch.as_tensor(
            np.asarray(X, dtype=np.float32)
        )
        with torch.no_grad():
            output = model(X_tensor)
        raw = output.cpu().numpy()
        if raw.ndim == 2 and raw.shape[1] > 1:
            return np.argmax(raw, axis=1)
        return raw.ravel()

    def proba_fn(X: np.ndarray) -> np.ndarray:
        X_tensor = torch.as_tensor(
            np.asarray(X, dtype=np.float32)
        )
        with torch.no_grad():
            output = model(X_tensor)
        raw = output.cpu().numpy()
        return _maybe_softmax(raw, output_type)

    return LoadedModel(predict_fn, proba_fn, model, fmt, path)


def _load_keras(
    path: Path,
    fmt: str,
    output_type: str | None,
) -> LoadedModel:
    """Load Keras/TensorFlow model."""
    try:
        from tensorflow import keras  # type: ignore[import]
    except ImportError:
        try:
            import keras  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "tensorflow or keras is required to load "
                "Keras models.  Install with: "
                "pip install tensorflow"
            ) from exc

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = keras.models.load_model(str(path))

    def predict_fn(X: np.ndarray) -> np.ndarray:
        X_float = np.asarray(X, dtype=np.float32)
        raw = np.asarray(model.predict(X_float, verbose=0))
        if raw.ndim == 2 and raw.shape[1] > 1:
            return np.argmax(raw, axis=1)
        return raw.ravel()

    def proba_fn(X: np.ndarray) -> np.ndarray:
        X_float = np.asarray(X, dtype=np.float32)
        raw = np.asarray(model.predict(X_float, verbose=0))
        return _maybe_softmax(raw, output_type)

    return LoadedModel(predict_fn, proba_fn, model, fmt, path)


# -------------------------------------------------------------------
# Shared helpers
# -------------------------------------------------------------------

def _extract_predict(
    model: object,
) -> Callable[..., np.ndarray]:
    """Get a predict callable from a scikit-learn-style model."""
    if hasattr(model, "predict"):
        def _predict(X: np.ndarray) -> np.ndarray:
            return np.asarray(model.predict(X))  # type: ignore[union-attr]
        return _predict

    if callable(model):
        def _call(X: np.ndarray) -> np.ndarray:
            return np.asarray(model(X))  # type: ignore[operator]
        return _call

    raise TypeError(
        f"Model of type {type(model).__name__} has no "
        f"predict() method and is not callable.  "
        f"Cannot create a predict function."
    )


def _extract_predict_proba(
    model: object,
    output_type: str | None,
) -> Callable[..., np.ndarray] | None:
    """Get predict_proba callable if available."""
    if not hasattr(model, "predict_proba"):
        return None

    def _proba(X: np.ndarray) -> np.ndarray:
        raw = np.asarray(
            model.predict_proba(X)  # type: ignore[union-attr]
        )
        return _maybe_softmax(raw, output_type)

    return _proba


def _maybe_softmax(
    raw: np.ndarray,
    output_type: str | None,
) -> np.ndarray:
    """Apply softmax if output_type indicates logits."""
    if output_type != "logits":
        return raw
    if raw.ndim < 2:
        return raw
    # Numerically stable softmax
    shifted = raw - np.max(raw, axis=1, keepdims=True)
    exp_vals = np.exp(shifted)
    return exp_vals / np.sum(exp_vals, axis=1, keepdims=True)


# -------------------------------------------------------------------
# Loader registry
# -------------------------------------------------------------------

_FORMAT_LOADERS: dict[
    str,
    Callable[
        [Path, str, str | None],
        LoadedModel,
    ],
] = {
    "pickle": _load_pickle,
    "joblib": _load_pickle,
    "onnx": _load_onnx,
    "torch": _load_torch,
    "keras": _load_keras,
}
