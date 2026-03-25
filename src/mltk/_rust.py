"""Bridge to optional Rust acceleration. Falls back to pure Python (scipy/numpy)."""

from __future__ import annotations

RUST_AVAILABLE = False

try:
    from mltk._mltk_rust import ks_test as _ks_test_rust
    from mltk._mltk_rust import psi as _psi_rust

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
