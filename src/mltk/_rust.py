"""Bridge to optional Rust acceleration. Falls back to pure Python (scipy/numpy)."""

from __future__ import annotations

RUST_AVAILABLE = False

try:
    from mltk._mltk_rust import bertscore_precision_recall as _bertscore_pr_rust
    from mltk._mltk_rust import centroid_cosine_distance as _centroid_cosine_distance_rust
    from mltk._mltk_rust import chi_squared as _chi_squared_rust
    from mltk._mltk_rust import cosine_similarity as _cosine_similarity_rust
    from mltk._mltk_rust import js_divergence as _js_divergence_rust
    from mltk._mltk_rust import kl_divergence as _kl_divergence_rust
    from mltk._mltk_rust import ks_test as _ks_test_rust
    from mltk._mltk_rust import psi as _psi_rust
    from mltk._mltk_rust import scan_pii_rust as _scan_pii_rust
    from mltk._mltk_rust import wasserstein as _wasserstein_rust

    RUST_AVAILABLE = True
except ImportError:
    pass


def ks_test(reference: list[float], current: list[float]) -> tuple[float, float]:
    """Kolmogorov-Smirnov test. Returns (statistic, p_value).

    Args:
        reference: Reference distribution values.
        current: Current distribution values to compare.

    Returns:
        Tuple of (test_statistic, p_value). Small p_value indicates distributions differ.

    Example:
        >>> stat, p = ks_test([1.0, 2.0, 3.0], [1.1, 2.1, 3.1])
        >>> assert p > 0.05  # no significant difference
    """
    if RUST_AVAILABLE:
        return _ks_test_rust(reference, current)
    try:
        from scipy.stats import ks_2samp

        stat, p = ks_2samp(reference, current)
        return (float(stat), float(p))
    except ImportError as err:
        raise ImportError(
            "Either the Rust extension or scipy is required for KS test. "
            "Install with: pip install mltk[scipy]"
        ) from err


def psi(reference: list[float], current: list[float], bins: int = 10) -> float:
    """Population Stability Index. <0.1 stable, 0.1-0.2 moderate, >0.2 significant.

    Args:
        reference: Reference distribution values.
        current: Current distribution values to compare.
        bins: Number of histogram bins for discretization.

    Returns:
        PSI value as a float. <0.1 is stable, 0.1-0.2 moderate shift, >0.2 significant shift.

    Example:
        >>> score = psi([1.0, 2.0, 3.0, 4.0], [1.1, 2.1, 3.1, 4.1])
        >>> assert score < 0.1  # stable distribution
    """
    if RUST_AVAILABLE:
        return _psi_rust(reference, current, bins)
    import numpy as np

    ref = np.array(reference)
    cur = np.array(current)
    # Create equal-width bins spanning the full range of both distributions
    breakpoints = np.linspace(
        min(ref.min(), cur.min()),
        max(ref.max(), cur.max()),
        bins + 1,
    )
    # Compute proportion of values in each bin
    ref_pcts = np.histogram(ref, bins=breakpoints)[0] / len(ref)
    cur_pcts = np.histogram(cur, bins=breakpoints)[0] / len(cur)
    # Clip to avoid log(0) — small epsilon replaces zero-count bins
    ref_pcts = np.clip(ref_pcts, 1e-6, None)
    cur_pcts = np.clip(cur_pcts, 1e-6, None)
    # PSI formula: sum of (current - reference) * ln(current / reference) per bin
    return float(np.sum((cur_pcts - ref_pcts) * np.log(cur_pcts / ref_pcts)))


def kl_divergence(
    reference: list[float], current: list[float], bins: int = 10
) -> float:
    """Histogram-based KL divergence D_KL(P || Q).

    Args:
        reference: Reference distribution values (P).
        current: Current distribution values (Q).
        bins: Number of histogram bins for discretization.

    Returns:
        KL divergence value. 0 = identical distributions; higher = more divergence.

    Example:
        >>> kl = kl_divergence([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
        >>> assert kl < 1e-6
    """
    if RUST_AVAILABLE:
        return _kl_divergence_rust(reference, current, bins)
    import numpy as np

    ref = np.array(reference, dtype=float)
    cur = np.array(current, dtype=float)
    global_min = min(ref.min(), cur.min())
    global_max = max(ref.max(), cur.max())
    breakpoints = np.linspace(global_min, global_max, bins + 1)
    epsilon = 1e-6
    p = np.histogram(ref, bins=breakpoints)[0] / len(ref)
    q = np.histogram(cur, bins=breakpoints)[0] / len(cur)
    p = np.clip(p, epsilon, None)
    q = np.clip(q, epsilon, None)
    return float(np.sum(p * np.log(p / q)))


def chi_squared(
    observed: list[float], expected: list[float]
) -> tuple[float, float]:
    """Chi-squared goodness-of-fit test.

    Args:
        observed: Observed frequency counts.
        expected: Expected frequency counts.

    Returns:
        Tuple of (statistic, p_value). High statistic / low p_value means
        the observed distribution differs significantly from expected.

    Example:
        >>> stat, p = chi_squared([10, 20, 30], [10, 20, 30])
        >>> assert stat < 1e-10
    """
    if RUST_AVAILABLE:
        return _chi_squared_rust(observed, expected)
    try:
        from scipy.stats import chisquare

        stat, p = chisquare(f_obs=observed, f_exp=expected)
        return (float(stat), float(p))
    except ImportError:
        pass
    # Pure numpy fallback
    import numpy as np

    obs = np.array(observed, dtype=float)
    exp = np.array(expected, dtype=float)
    mask = exp > 1e-12
    stat = float(np.sum(((obs[mask] - exp[mask]) ** 2) / exp[mask]))
    # Very rough p-value via normal approximation (Wilson-Hilferty)
    df = len(observed) - 1
    if df <= 0:
        return (stat, 1.0)
    # Wilson-Hilferty: (chi2/df)^(1/3) ~ Normal(1 - 2/(9df), 2/(9df))
    h = 2.0 / (9.0 * df)
    z = ((stat / df) ** (1.0 / 3.0) - (1.0 - h)) / h**0.5
    # Approximate 1 - Phi(z)
    p_value = float(0.5 * (1.0 - float(np.sign(z)) * (1.0 - np.exp(-0.7 * abs(z)))))
    return (stat, max(0.0, min(1.0, p_value)))


def js_divergence(
    reference: list[float], current: list[float], bins: int = 10
) -> float:
    """Jensen-Shannon divergence, normalised to [0, 1].

    Args:
        reference: Reference distribution values.
        current: Current distribution values.
        bins: Number of histogram bins for discretization.

    Returns:
        JS divergence in [0, 1]. 0 = identical; 1 = completely disjoint.

    Example:
        >>> js = js_divergence([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
        >>> assert js < 1e-6
    """
    if RUST_AVAILABLE:
        return _js_divergence_rust(reference, current, bins)
    import numpy as np

    ref = np.array(reference, dtype=float)
    cur = np.array(current, dtype=float)
    global_min = min(ref.min(), cur.min())
    global_max = max(ref.max(), cur.max())
    breakpoints = np.linspace(global_min, global_max, bins + 1)
    epsilon = 1e-6
    p = np.clip(np.histogram(ref, bins=breakpoints)[0] / len(ref), epsilon, None)
    q = np.clip(np.histogram(cur, bins=breakpoints)[0] / len(cur), epsilon, None)
    m = (p + q) / 2.0
    kl_pm = float(np.sum(p * np.log(p / m)))
    kl_qm = float(np.sum(q * np.log(q / m)))
    js_nats = 0.5 * kl_pm + 0.5 * kl_qm
    return max(0.0, min(1.0, js_nats / float(np.log(2))))


def wasserstein(reference: list[float], current: list[float]) -> float:
    """Earth Mover's Distance (Wasserstein-1) between two 1-D distributions.

    Args:
        reference: Reference distribution values.
        current: Current distribution values.

    Returns:
        Wasserstein distance. 0 = identical distributions.

    Example:
        >>> d = wasserstein([0.0, 1.0, 2.0], [0.0, 1.0, 2.0])
        >>> assert d < 1e-10
    """
    if RUST_AVAILABLE:
        return _wasserstein_rust(reference, current)
    try:
        from scipy.stats import wasserstein_distance

        return float(wasserstein_distance(reference, current))
    except ImportError:
        pass
    # Pure numpy fallback: sorted CDF integral
    import numpy as np

    ref = np.sort(np.array(reference, dtype=float))
    cur = np.sort(np.array(current, dtype=float))
    all_vals = np.union1d(ref, cur)
    cdf_ref = np.searchsorted(ref, all_vals, side="right") / len(ref)
    cdf_cur = np.searchsorted(cur, all_vals, side="right") / len(cur)
    widths = np.diff(all_vals, prepend=all_vals[0])
    widths[0] = 0.0
    return float(np.sum(np.abs(cdf_ref - cdf_cur) * widths))


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors: dot(a, b) / (||a|| * ||b||).

    Args:
        a: First vector as a list of floats.
        b: Second vector as a list of floats.

    Returns:
        Cosine similarity in [-1, 1]. Returns 0.0 for zero-norm vectors.

    Example:
        >>> sim = cosine_similarity([1.0, 0.0], [0.0, 1.0])
        >>> assert abs(sim) < 1e-10  # orthogonal
    """
    if RUST_AVAILABLE:
        return _cosine_similarity_rust(a, b)
    import numpy as np

    va = np.array(a, dtype=np.float64)
    vb = np.array(b, dtype=np.float64)
    norm_a = np.linalg.norm(va)
    norm_b = np.linalg.norm(vb)
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    return float(np.dot(va, vb) / (norm_a * norm_b))


def centroid_cosine_distance(
    ref_embs: list[list[float]], cur_embs: list[list[float]]
) -> float:
    """Cosine distance between centroids of two embedding sets.

    Centroid = mean of all vectors. Distance = 1 - cosine_similarity(c_ref, c_cur).

    Args:
        ref_embs: Reference embeddings as list of float vectors.
        cur_embs: Current embeddings as list of float vectors.

    Returns:
        Cosine distance in [0, 2]. 0.0 = identical centroids.

    Example:
        >>> d = centroid_cosine_distance([[1.0, 0.0]], [[1.0, 0.0]])
        >>> assert d < 1e-10
    """
    if RUST_AVAILABLE:
        return _centroid_cosine_distance_rust(ref_embs, cur_embs)
    import numpy as np

    ref = np.array(ref_embs, dtype=np.float64)
    cur = np.array(cur_embs, dtype=np.float64)
    ref_centroid = ref.mean(axis=0)
    cur_centroid = cur.mean(axis=0)
    return float(1.0 - cosine_similarity(ref_centroid.tolist(), cur_centroid.tolist()))


def bertscore_precision_recall(
    ref_embs: list[list[float]], hyp_embs: list[list[float]]
) -> tuple[float, float, float]:
    """BERTScore precision, recall, and F1 via token-level cosine similarity.

    Precision: for each hypothesis token, max cosine similarity against all
               reference tokens → mean over hypothesis tokens.
    Recall:    for each reference token, max cosine similarity against all
               hypothesis tokens → mean over reference tokens.
    F1:        2 * P * R / (P + R), or 0.0 if P + R == 0.

    Args:
        ref_embs: Reference token embeddings, shape (N, D).
        hyp_embs: Hypothesis token embeddings, shape (M, D).

    Returns:
        Tuple of (precision, recall, f1).

    Example:
        >>> embs = [[1.0, 0.0], [0.0, 1.0]]
        >>> p, r, f1 = bertscore_precision_recall(embs, embs)
        >>> assert abs(f1 - 1.0) < 1e-10
    """
    if RUST_AVAILABLE:
        return _bertscore_pr_rust(ref_embs, hyp_embs)
    import numpy as np

    ref = np.array(ref_embs, dtype=np.float64)
    hyp = np.array(hyp_embs, dtype=np.float64)

    if ref.size == 0 or hyp.size == 0:
        return (0.0, 0.0, 0.0)

    # Precision: for each hyp token, max cosine sim against all ref tokens
    precision_scores = []
    for hyp_tok in hyp:
        sims = [cosine_similarity(hyp_tok.tolist(), ref_tok.tolist()) for ref_tok in ref]
        precision_scores.append(max(sims))
    precision = float(np.mean(precision_scores))

    # Recall: for each ref token, max cosine sim against all hyp tokens
    recall_scores = []
    for ref_tok in ref:
        sims = [cosine_similarity(ref_tok.tolist(), hyp_tok.tolist()) for hyp_tok in hyp]
        recall_scores.append(max(sims))
    recall = float(np.mean(recall_scores))

    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return (precision, recall, f1)


def scan_pii_fast(
    text: str, patterns: list[tuple[str, str]]
) -> list[tuple[str, int, int, str]]:
    """Scan text for PII using compiled regex patterns.

    Args:
        text: The text to scan.
        patterns: List of (pattern_name, regex_pattern) pairs.

    Returns:
        List of (pattern_name, start_pos, end_pos, matched_text) sorted by start_pos.

    Example:
        >>> hits = scan_pii_fast("Call 555-1234", [("phone", r"\\d{3}-\\d{4}")])
        >>> assert hits[0][0] == "phone"
    """
    if RUST_AVAILABLE:
        return _scan_pii_rust(text, patterns)
    # Pure Python re fallback
    import re

    results: list[tuple[str, int, int, str]] = []
    for name, pattern in patterns:
        compiled = re.compile(pattern)
        for mat in compiled.finditer(text):
            results.append((name, mat.start(), mat.end(), mat.group()))
    results.sort(key=lambda r: r[1])
    return results
