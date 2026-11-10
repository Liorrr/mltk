"""Dependency-free paired statistics for the approved parity specification.

See docs/api/parity.md for the public comparison and statistical contract. The
independent unit is an item; callers must reduce repeated trials and separate
strata before calling these functions. Invalid inputs raise TypeError/ValueError,
except TOST, whose contract reports invalid inputs as NOT_VERIFIED.

McNemar supports at most 1,000,000 total discordances. Interior Clopper-Pearson
intervals (0 < successes < n) support n <= 1,000,000. Larger beta-based binomial
calculations raise ValueError rather than return an unvalidated probability.
Clopper-Pearson endpoints use analytic formulas and allow larger n representable
as a finite float. These limits do not restrict Wilson, TOST, BH, or bootstrap.
"""

from __future__ import annotations

import math
import random
from statistics import NormalDist

# Independent SciPy grids at and below this limit cover central probabilities,
# significance boundaries and tails. Larger shapes lose lgamma-difference precision.
MAX_BETA_BINOMIAL_N = 1_000_000


def _supported_binomial_count(n: int) -> None:
    if n > MAX_BETA_BINOMIAL_N:
        raise ValueError(
            f"beta-based binomial count {n} exceeds supported limit {MAX_BETA_BINOMIAL_N}"
        )


def _integer(value: int, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer (not bool)")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _finite(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number (not bool)")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _probability(value: float, name: str, upper: float = 1.0) -> float:
    result = _finite(value, name)
    if not 0 < result < upper:
        raise ValueError(f"{name} must be strictly between 0 and {upper}")
    return result


def _values(values: list[float], name: str) -> list[float]:
    if not isinstance(values, list):
        raise TypeError(f"{name} must be a list")
    return [_finite(value, f"{name}[{i}]") for i, value in enumerate(values)]


def _counts(successes: int, n: int) -> None:
    _integer(successes, "successes")
    _integer(n, "n")
    if successes > n:
        raise ValueError("successes must not exceed n")


def wilson_interval(successes: int, n: int, confidence: float = 0.95) -> dict:
    """Two-sided Wilson score interval; an empty population has null bounds."""
    _counts(successes, n)
    confidence = _probability(confidence, "confidence")
    result = {"ci_low": None, "ci_high": None, "ci_method": "wilson", "n": n}
    if not n:
        return result
    # Use the lower tail so confidence near one never rounds the argument to 1.
    z = -NormalDist().inv_cdf((1 - confidence) / 2)
    p = successes / n
    z2_n = z * z / n
    denominator = 1 + z2_n
    center = (p + z2_n / 2) / denominator
    halfwidth = z * math.sqrt((p * (1 - p) + z2_n / 4) / n) / denominator
    result.update(ci_low=max(0.0, center - halfwidth), ci_high=min(1.0, center + halfwidth))
    # Preserve the exact boundary in spite of cancellation in center +/- halfwidth.
    if successes == 0:
        result["ci_low"] = 0.0
    if successes == n:
        result["ci_high"] = 1.0
    return result


def _beta_fraction(a: float, b: float, x: float) -> float:
    """Incomplete-beta continued fraction, evaluated with modified Lentz steps."""
    tiny = 1e-300

    def nonzero(value: float) -> float:
        return math.copysign(tiny, value) if abs(value) < tiny else value

    c = 1.0
    d = 1 / nonzero(1 - (a + b) * x / (a + 1))
    h = d
    for m in range(1, 10001):
        m2 = 2 * m
        aa = m * (b - m) * x / ((a + m2 - 1) * (a + m2))
        d = 1 / nonzero(1 + aa * d)
        c = nonzero(1 + aa / c)
        h *= d * c
        aa = -(a + m) * (a + b + m) * x / ((a + m2) * (a + m2 + 1))
        d = 1 / nonzero(1 + aa * d)
        c = nonzero(1 + aa / c)
        delta = d * c
        h *= delta
        if abs(delta - 1) <= 3e-14:
            return h
    raise ArithmeticError("incomplete beta did not converge")


def _regularized_beta(x: float, a: float, b: float) -> float:
    """Regularized incomplete beta for positive shapes, with tail symmetry."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    log_factor = (
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + a * math.log(x) + b * math.log1p(-x)
    )
    factor = math.exp(log_factor)
    if x < (a + 1) / (a + b + 2):
        value = factor * _beta_fraction(a, b, x) / a
    else:
        value = 1 - factor * _beta_fraction(b, a, 1 - x) / b
    return min(1.0, max(0.0, value))


def _student_t_sf(t: float, df: int) -> float:
    """Student-t survival probability, retaining precision in the small tail."""
    if math.isinf(t):
        return 0.0 if t > 0 else 1.0
    root_df = math.sqrt(df)
    if abs(t) < 1:
        ratio = (abs(t) / root_df) ** 2
        central = _regularized_beta(ratio / (1 + ratio), 0.5, df / 2)
        tail = (1 - central) / 2
    else:
        ratio = (root_df / abs(t)) ** 2
        if ratio < 1e-150:
            # I_x(a,b) ~ x**a/(a*B(a,b)); compute log(x) before squaring
            # underflows. The omitted relative term is O(x) in this regime.
            a = df / 2
            log_x = 2 * (math.log(root_df) - math.log(abs(t)))
            tail = math.exp(
                a * log_x - math.log(a) - math.log(2)
                + math.lgamma(a + 0.5) - math.lgamma(a) - math.lgamma(0.5)
            )
        else:
            tail = _regularized_beta(ratio / (1 + ratio), df / 2, 0.5) / 2
    return tail if t >= 0 else 1 - tail


def _student_t_cdf(t: float, df: int) -> float:
    """Student-t CDF via incomplete beta (df must be a positive integer)."""
    return _student_t_sf(-t, df)


def _student_t_isf(probability: float, df: int) -> float:
    """Positive inverse survival function for 0 < probability < 0.5."""
    low, high = 0.0, 1.0
    while _student_t_sf(high, df) > probability:
        high *= 2
        if not math.isfinite(high):
            raise ArithmeticError("Student-t quantile exceeds floating-point range")
    for _ in range(200):
        middle = low + (high - low) / 2
        if middle == low or middle == high:
            break
        if _student_t_sf(middle, df) > probability:
            low = middle
        else:
            high = middle
    return low + (high - low) / 2


def exact_mcnemar(b: int, c: int) -> float:
    """Exact two-sided binomial McNemar p-value for discordant item counts.

    Uses the binomial/incomplete-beta identity for large counts, avoiding huge
    powers or underflow from starting a binomial sum at its smallest term.
    Requires b+c <= MAX_BETA_BINOMIAL_N (1,000,000), including balanced counts;
    larger totals raise ValueError, with no normal-approximation fallback.
    """
    _integer(b, "b")
    _integer(c, "c")
    m = b + c
    _supported_binomial_count(m)
    k = min(b, c)
    if m == 0 or 2 * k >= m - 1:
        return 1.0
    if m <= 1024:
        term = total = 1
        for j in range(1, k + 1):
            term = term * (m - j + 1) // j
            total += term
        return min(1.0, (2 * total) / (1 << m))
    return min(1.0, 2 * _regularized_beta(0.5, m - k, k + 1))


def benjamini_hochberg(pvalues: list[float]) -> list[float]:
    """BH adjusted p-values in original order, for one caller-defined family."""
    values = _values(pvalues, "pvalues")
    if any(not 0 <= p <= 1 for p in values):
        raise ValueError("pvalues must be between 0 and 1 inclusive")
    n = len(values)
    order = sorted(range(n), key=values.__getitem__)
    adjusted = [0.0] * n
    running = 1.0
    for rank in range(n, 0, -1):
        index = order[rank - 1]
        running = min(running, values[index] * (n / rank))
        adjusted[index] = running
    return adjusted


def paired_tost(differences: list[float], margin: float, alpha: float = 0.05) -> dict:
    """Paired t TOST of mean candidate-minus-base differences in (-margin, margin).

    Alpha must be in (0, 0.5); the returned interval has confidence 1-2*alpha.
    PASS requires both one-sided p-values strictly below alpha. Empty/singleton
    data and invalid inputs are NOT_VERIFIED. For constant differences, the
    p-value is 0 strictly inside the bounds and 1 otherwise; reason identifies
    this deterministic degenerate case. No automatic margin is supplied.
    """
    result = {
        "status": "NOT_VERIFIED", "pvalue": None, "mean_difference": None,
        "n": len(differences) if isinstance(differences, list) else 0,
        "ci_low": None, "ci_high": None, "reason": "invalid_input",
    }
    try:
        values = _values(differences, "differences")
        margin = _finite(margin, "margin")
        if margin <= 0:
            raise ValueError("margin must be > 0")
        alpha = _probability(alpha, "alpha", 0.5)
    except (TypeError, ValueError):
        return result
    n = len(values)
    scale = max(map(abs, values), default=0.0) or 1.0
    scaled = [value / scale for value in values]
    mean = math.fsum(scaled) / n if n else None
    result["mean_difference"] = mean * scale if mean is not None else None
    if n < 2:
        result["reason"] = "insufficient_pairs"
        return result
    se = math.sqrt(math.fsum((value - mean) ** 2 for value in scaled) / (n - 1) / n)
    if se == 0:
        inside = abs(result["mean_difference"]) < margin
        result.update(
            status="PASS" if inside else "FAIL", pvalue=0.0 if inside else 1.0,
            ci_low=result["mean_difference"], ci_high=result["mean_difference"],
            reason=(
                "zero_variance_inside_margin" if inside else "zero_variance_at_or_outside_margin"
            ),
        )
        return result
    try:
        scaled_margin = margin / scale
        pvalue = max(
            _student_t_sf((mean + scaled_margin) / se, n - 1),
            _student_t_sf((scaled_margin - mean) / se, n - 1),
        )
        halfwidth = _student_t_isf(alpha, n - 1) * se
        low, high = (mean - halfwidth) * scale, (mean + halfwidth) * scale
        if not all(math.isfinite(value) for value in (pvalue, low, high)):
            raise ArithmeticError("TOST interval exceeds floating-point range")
    except (ArithmeticError, ValueError):
        result["reason"] = "numerical_error"
        return result
    passed = pvalue < alpha and low > -margin and high < margin
    result.update(
        status="PASS" if passed else "FAIL", pvalue=pvalue, ci_low=low, ci_high=high,
        reason="equivalent" if passed else "equivalence_not_established",
    )
    return result


def _percentile(sorted_values: list[float], probability: float) -> float:
    position = (len(sorted_values) - 1) * probability
    lower = int(position)
    fraction = position - lower
    upper = min(lower + 1, len(sorted_values) - 1)
    return sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction


def paired_bootstrap(
    differences: list[float], seed: int, resamples: int = 2000, confidence: float = 0.95,
) -> dict:
    """Seeded percentile CI for the paired mean, resampling n items per draw.

    Uses linear interpolation at (resamples-1)*q. Sorting the input makes
    reordering items deterministic; the local RNG never changes global state.
    Empty input returns null bounds/mean; one item yields a point interval.
    """
    values = sorted(_values(differences, "differences"))
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer (not bool)")
    _integer(resamples, "resamples", 1)
    confidence = _probability(confidence, "confidence")
    n = len(values)
    result = {
        "ci_low": None, "ci_high": None, "ci_method": "paired_bootstrap_percentile",
        "n": n, "mean_difference": None, "seed": seed,
        "resamples": resamples, "confidence": confidence,
    }
    if not n:
        return result
    scale = max(map(abs, values)) or 1.0
    scaled = [value / scale for value in values]
    rng = random.Random(seed)
    means = sorted(math.fsum(rng.choices(scaled, k=n)) / n for _ in range(resamples))
    tail = (1 - confidence) / 2
    result.update(
        mean_difference=math.fsum(scaled) / n * scale,
        ci_low=_percentile(means, tail) * scale,
        ci_high=_percentile(means, 1 - tail) * scale,
    )
    return result


def _beta_quantile(probability: float, a: int, b: int) -> float:
    low, high = 0.0, 1.0
    # A relative stopping condition also resolves small endpoint probabilities.
    for _ in range(1100):
        middle = low + (high - low) / 2
        if middle == low or middle == high:
            break
        if _regularized_beta(middle, a, b) < probability:
            low = middle
        else:
            high = middle
        if high - low <= 2e-14 * middle:
            break
    return low + (high - low) / 2


def clopper_pearson(successes: int, n: int, confidence: float = 0.95) -> dict:
    """Exact two-sided binomial CI, with analytic zero/all-success endpoints.

    Interior intervals require n <= MAX_BETA_BINOMIAL_N (1,000,000) or raise
    ValueError. Analytic endpoints accept larger n if convertible to a finite
    float. expm1 preserves the small zero-success upper bound without subtracting
    two nearly equal numbers; all-success lower bounds use exp directly.
    """
    _counts(successes, n)
    confidence = _probability(confidence, "confidence")
    result = {"ci_low": None, "ci_high": None, "ci_method": "clopper_pearson", "n": n}
    if not n:
        return result
    tail = (1 - confidence) / 2
    if successes in (0, n):
        log_endpoint = math.log(tail) / _finite(n, "n")
        result.update(
            ci_low=math.exp(log_endpoint) if successes else 0.0,
            ci_high=1.0 if successes else -math.expm1(log_endpoint),
        )
        return result
    _supported_binomial_count(n)
    result.update(
        ci_low=_beta_quantile(tail, successes, n - successes + 1),
        ci_high=1 - _beta_quantile(tail, n - successes, successes + 1),
    )
    return result
