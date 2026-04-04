"""Model-backed evaluation helpers for LLM domain assertions.

This module isolates heavyweight optional dependencies (sentence-transformers,
cross-encoder NLI models) from the pure-text helpers in ``_utils.py``.  Every
function lazy-loads its model on first call and caches via ``lru_cache`` so
subsequent invocations reuse the same instance.

No function in this module is imported at package load time — consumers
import directly when they need a specific backend, keeping the default
``import mltk`` path dependency-free.
"""

from __future__ import annotations

from functools import lru_cache

# Re-export from _utils — single source of truth (CR-1 fix).
from mltk.domains.llm._utils import _normalize as normalize_unicode  # noqa: F401

# ---------------------------------------------------------------------------
# Pinned model revisions (supply-chain defense — SEC-2)
# ---------------------------------------------------------------------------

_MODEL_REVISIONS: dict[str, str] = {
    "all-mpnet-base-v2": "e8c3b32edf5434bc",
    "all-MiniLM-L6-v2": "c22d4bce25e7e04e",
    "cross-encoder/nli-deberta-v3-base": "6c749ce3425cd33b",
}


# ---------------------------------------------------------------------------
# Embedding similarity
# ---------------------------------------------------------------------------

@lru_cache(maxsize=4)
def _load_sentence_model(model_name: str):  # noqa: ANN201
    """Load and cache a SentenceTransformer model.

    Uses pinned revisions for known models to defend against
    supply-chain attacks on HuggingFace Hub.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError(
            "sentence-transformers is required for "
            "embedding-based methods. "
            "Install with: pip install mltk[embedding]"
        ) from None
    revision = _MODEL_REVISIONS.get(model_name)
    kwargs: dict[str, str] = {}
    if revision:
        kwargs["revision"] = revision
    return SentenceTransformer(model_name, **kwargs)


def embedding_cosine_pairs(
    texts_a: list[str],
    texts_b: list[str],
    model_name: str = "all-mpnet-base-v2",
) -> list[float]:
    """Compute pairwise cosine similarity between two text lists.

    Args:
        texts_a: First list of texts.
        texts_b: Second list of texts (same length).
        model_name: Sentence-transformer model to use.

    Returns:
        List of cosine similarity scores (one per pair).
    """
    import numpy as np

    model = _load_sentence_model(model_name)
    emb_a = model.encode(texts_a, convert_to_numpy=True)
    emb_b = model.encode(texts_b, convert_to_numpy=True)

    scores: list[float] = []
    for a, b in zip(emb_a, emb_b, strict=True):
        norm_a = float(np.linalg.norm(a))
        norm_b = float(np.linalg.norm(b))
        if norm_a == 0 or norm_b == 0:
            scores.append(0.0)
        else:
            scores.append(float(np.dot(a, b) / (norm_a * norm_b)))
    return scores


def embedding_cosine_single(
    text_a: str,
    text_b: str,
    model_name: str = "all-mpnet-base-v2",
) -> float:
    """Cosine similarity between two individual texts."""
    return embedding_cosine_pairs([text_a], [text_b], model_name)[0]


# ---------------------------------------------------------------------------
# NLI (Natural Language Inference) — bidirectional entailment
# ---------------------------------------------------------------------------

@lru_cache(maxsize=4)
def _load_nli_model(model_name: str):  # noqa: ANN201
    """Load and cache a CrossEncoder NLI model.

    Uses pinned revisions for known models (SEC-2).
    """
    try:
        from sentence_transformers import CrossEncoder
    except ImportError:
        raise ImportError(
            "sentence-transformers is required for "
            "NLI-based methods. "
            "Install with: pip install mltk[nli]"
        ) from None
    revision = _MODEL_REVISIONS.get(model_name)
    kwargs = {}
    if revision:
        kwargs["model_kwargs"] = {"revision": revision}
    return CrossEncoder(model_name, **kwargs)


# Label order for cross-encoder/nli-deberta-v3-base:
# index 0 = contradiction, 1 = entailment, 2 = neutral
_NLI_LABELS = ["contradiction", "entailment", "neutral"]


def nli_entailment_score(
    premise: str,
    hypothesis: str,
    model_name: str = "cross-encoder/nli-deberta-v3-base",
) -> dict[str, float]:
    """Classify the NLI relationship between premise and hypothesis.

    Returns:
        Dict with keys ``contradiction``, ``entailment``, ``neutral``
        mapping to probability scores (sum to ~1.0), plus ``label``
        (the highest-scoring class).
    """
    import numpy as np

    model = _load_nli_model(model_name)
    logits = model.predict([(premise, hypothesis)])
    # logits shape: (1, 3)
    probs = _softmax(logits[0] if len(logits.shape) > 1 else logits)

    result = {label: float(probs[i]) for i, label in enumerate(_NLI_LABELS)}
    result["label"] = _NLI_LABELS[int(np.argmax(probs))]
    return result


def nli_bidirectional(
    text_a: str,
    text_b: str,
    model_name: str = "cross-encoder/nli-deberta-v3-base",
) -> dict[str, object]:
    """Check bidirectional entailment between two texts.

    Semantic equivalence = both directions are ``entailment``.

    Returns:
        Dict with ``forward`` (A->B scores), ``backward`` (B->A scores),
        ``equivalent`` (bool), ``contradiction`` (bool).
    """
    fwd = nli_entailment_score(text_a, text_b, model_name)
    bwd = nli_entailment_score(text_b, text_a, model_name)

    equivalent = fwd["label"] == "entailment" and bwd["label"] == "entailment"
    contradiction = fwd["label"] == "contradiction" or bwd["label"] == "contradiction"

    return {
        "forward": fwd,
        "backward": bwd,
        "equivalent": equivalent,
        "contradiction": contradiction,
    }


def _softmax(logits) -> list[float]:  # noqa: ANN001
    """Compute softmax over a 1-D array of logits."""
    import numpy as np
    exp = np.exp(logits - np.max(logits))
    return (exp / exp.sum()).tolist()
