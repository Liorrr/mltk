"""Synthetic statistical fixtures; SciPy is an optional independent reference."""

from __future__ import annotations

import json
import math
import random
import subprocess
import sys
from decimal import Decimal, localcontext
from pathlib import Path

import pytest

from mltk.core.statistics import (
    MAX_BETA_BINOMIAL_N,
    _regularized_beta,
    _student_t_cdf,
    _student_t_isf,
    _student_t_sf,
    benjamini_hochberg,
    clopper_pearson,
    exact_mcnemar,
    paired_bootstrap,
    paired_tost,
    wilson_interval,
)


def test_wilson_approved_acceptance_fixtures():
    # Independently tabulated Wilson score endpoints, including spec T4.
    assert wilson_interval(2, 20)["ci_high"] == pytest.approx(0.3010336452284873)
    interval = wilson_interval(0, 20)
    assert interval["ci_low"] == 0
    assert interval["ci_high"] == pytest.approx(0.1611251580528194)
    assert math.floor(interval["ci_high"] * 20) + 1 == 4
    assert wilson_interval(20, 20)["ci_high"] == 1
    assert wilson_interval(10, 20)["ci_low"] == pytest.approx(0.2992980081982123)


@pytest.mark.parametrize("function", [wilson_interval, clopper_pearson])
def test_rate_empty_and_confidence(function):
    empty = function(0, 0)
    assert empty["ci_low"] is empty["ci_high"] is None
    assert empty["n"] == 0
    assert set(empty) == {"ci_low", "ci_high", "ci_method", "n"}
    narrow, wide = function(3, 10, 0.9), function(3, 10, 0.99)
    assert wide["ci_low"] < narrow["ci_low"] < narrow["ci_high"] < wide["ci_high"]
    extreme = function(3, 10, math.nextafter(1.0, 0.0))
    assert 0 <= extreme["ci_low"] < extreme["ci_high"] <= 1


@pytest.mark.parametrize("function", [wilson_interval, clopper_pearson])
@pytest.mark.parametrize(
    ("successes", "n", "error"),
    [(True, 2, TypeError), (1, False, TypeError), (1.0, 2, TypeError),
     (1, 2.0, TypeError), (-1, 2, ValueError), (0, -1, ValueError),
     (3, 2, ValueError), (1, 0, ValueError), ("1", 2, TypeError)],
)
def test_rate_invalid_counts(function, successes, n, error):
    with pytest.raises(error):
        function(successes, n)


@pytest.mark.parametrize("confidence", [True, None, "0.95", 0, 1, -0.1, math.nan, math.inf])
@pytest.mark.parametrize("function", [wilson_interval, clopper_pearson])
def test_rate_invalid_confidence(function, confidence):
    with pytest.raises((TypeError, ValueError)):
        function(0, 0, confidence)


@pytest.mark.parametrize(
    ("b", "c", "expected"),
    [(0, 0, 1), (6, 0, 0.03125), (1, 0, 1), (2, 0, 0.5),
     (5, 1, 0.21875), (2000, 2000, 1), (2001, 2000, 1), (2000, 0, 0)],
)
def test_exact_mcnemar_fixtures(b, c, expected):
    assert exact_mcnemar(b, c) == expected
    assert exact_mcnemar(c, b) == expected


def test_mcnemar_against_exact_integer_sum():
    # This independent direct definition also exercises the large-count beta path.
    for b, c in [(5, 12), (500, 600), (5000, 5100)]:
        total = sum(math.comb(b + c, j) for j in range(min(b, c) + 1))
        expected = min(1, 2 * total / 2 ** (b + c))
        assert exact_mcnemar(b, c) == pytest.approx(expected, rel=1e-10)


@pytest.mark.parametrize("bad", [True, 1.0, None, "1", -1, math.inf])
def test_mcnemar_bad_counts(bad):
    with pytest.raises((TypeError, ValueError)):
        exact_mcnemar(bad, 1)
    with pytest.raises((TypeError, ValueError)):
        exact_mcnemar(1, bad)


def test_bh_order_ties_endpoints_and_no_mutation():
    values = [0.04, 0.001, 0.03, 0.01, 0.03]
    original = values[:]
    assert benjamini_hochberg(values) == pytest.approx([0.04, 0.005, 0.0375, 0.025, 0.0375])
    assert values == original
    assert benjamini_hochberg([]) == []
    assert benjamini_hochberg([0, 1, 1]) == [0, 1, 1]
    assert benjamini_hochberg([0.3]) == [0.3]
    sorted_adjusted = benjamini_hochberg(sorted(values))
    assert sorted_adjusted == sorted(sorted_adjusted)


@pytest.mark.parametrize("bad", [[True], [math.nan], [math.inf], [-0.1], [1.1], ["0.1"], (0.1,)])
def test_bh_bad_input(bad):
    with pytest.raises((TypeError, ValueError)):
        benjamini_hochberg(bad)


@pytest.mark.parametrize("t", [-100, -3, -1, -1e-8, 0, 1e-8, 1, 3, 100])
def test_student_t_analytic_cauchy_and_df_two(t):
    assert _student_t_cdf(t, 1) == pytest.approx(0.5 + math.atan(t) / math.pi, abs=2e-15)
    expected_df_two = 0.5 + t / (2 * math.sqrt(t * t + 2))
    assert _student_t_cdf(t, 2) == pytest.approx(expected_df_two, abs=2e-15)
    assert _student_t_cdf(t, 5) == pytest.approx(_student_t_sf(-t, 5), abs=1e-15)


@pytest.mark.parametrize("probability", [0.49, 0.25, 0.05, 0.001, 1e-10, 1e-100, 1e-200])
def test_student_t_inverse_analytic_cauchy(probability):
    expected = 1 / math.tan(math.pi * probability)
    assert _student_t_isf(probability, 1) == pytest.approx(expected, rel=1e-12)


def test_beta_analytic_fixtures():
    for x in [0.0001, 0.1, 0.5, 0.9, 0.9999]:
        assert _regularized_beta(x, 1, 1) == pytest.approx(x, abs=1e-14)
        assert _regularized_beta(x, 2, 1) == pytest.approx(x * x, abs=1e-14)
        assert _regularized_beta(x, 1, 2) == pytest.approx(2 * x - x * x, abs=1e-14)


def test_tost_analytic_two_item_fixture():
    result = paired_tost([-1, 1], 1)
    assert result["status"] == "FAIL"
    assert result["pvalue"] == pytest.approx(0.25, abs=1e-14)
    assert result["mean_difference"] == 0
    assert result["ci_low"] == pytest.approx(-6.313751514675041)
    assert result["ci_high"] == pytest.approx(6.313751514675041)
    passing = paired_tost([-1, 1], 10)
    assert passing["status"] == "PASS"
    assert passing["pvalue"] == pytest.approx(math.atan(0.1) / math.pi)
    assert set(result) == {
        "status", "pvalue", "mean_difference", "n", "ci_low", "ci_high", "reason",
    }


def test_tost_nonconstant_exact_boundary_fails():
    # Cauchy upper quartile is exactly 1; [-1, 1] has mean 0 and SE 1.
    result = paired_tost([-1, 1], margin=1, alpha=0.25)
    assert result["pvalue"] == pytest.approx(0.25, abs=1e-15)
    assert result["status"] == "FAIL"


def test_tost_analytic_three_item_fixture():
    result = paired_tost([-1, 0, 1], 1)
    assert result["pvalue"] == pytest.approx(0.5 - math.sqrt(3) / (2 * math.sqrt(5)))
    assert result["ci_high"] == pytest.approx(2.919985580353725 / math.sqrt(3))


@pytest.mark.parametrize("value", [-2, -1, -0.999, 0, 0.999, 1, 2])
def test_tost_zero_variance_strict_bounds(value):
    result = paired_tost([value] * 4, 1)
    inside = abs(value) < 1
    assert result["status"] == ("PASS" if inside else "FAIL")
    assert result["pvalue"] == (0.0 if inside else 1.0)
    assert result["ci_low"] == result["ci_high"] == value
    assert result["reason"].startswith("zero_variance_")


@pytest.mark.parametrize("differences", [[], [0.25]])
def test_tost_insufficient_pairs(differences):
    result = paired_tost(differences, 1)
    assert result["status"] == "NOT_VERIFIED"
    assert result["reason"] == "insufficient_pairs"
    assert result["n"] == len(differences)
    assert result["pvalue"] is result["ci_low"] is result["ci_high"] is None
    assert result["mean_difference"] == (0.25 if differences else None)


@pytest.mark.parametrize("margin", [None, True, "1", 0, -1, math.nan, math.inf, 10 ** 400])
def test_tost_invalid_margin(margin):
    result = paired_tost([0, 0], margin)
    assert result["status"] == "NOT_VERIFIED"
    assert result["reason"] == "invalid_input"
    assert result["pvalue"] is None


@pytest.mark.parametrize(
    "differences", [None, (0, 0), [True, 0], [math.nan, 0], [math.inf, 0], ["0", 0]],
)
def test_tost_invalid_differences(differences):
    assert paired_tost(differences, 1)["status"] == "NOT_VERIFIED"


@pytest.mark.parametrize("alpha", [True, "0.05", None, 0, 0.5, 1, math.nan, math.inf])
def test_tost_invalid_alpha(alpha):
    assert paired_tost([0, 0], 1, alpha)["status"] == "NOT_VERIFIED"


def test_tost_scale_invariance_and_finite_extremes():
    baseline = paired_tost([-0.1, 0.2, 0.1, -0.2], 1)
    for scale in (1e-250, 1e250):
        result = paired_tost([v * scale for v in [-0.1, 0.2, 0.1, -0.2]], scale)
        assert result["status"] == baseline["status"]
        assert result["pvalue"] == pytest.approx(baseline["pvalue"])
        assert result["ci_high"] / scale == pytest.approx(baseline["ci_high"])
    assert paired_tost([1e308, 1e308], 1.1e308)["status"] == "PASS"
    overflow = paired_tost([-1e308, 1e308], 1e308)
    assert overflow["status"] == "NOT_VERIFIED"
    assert overflow["reason"] == "numerical_error"
    json.dumps(overflow, allow_nan=False)


def test_bootstrap_independent_rng_percentiles_and_item_weighting():
    # Independently draw item indices and interpolate using ordinary arithmetic.
    values, seed, count = [0.0, 1.0, 3.0, 8.0], 712, 101
    rng = random.Random(seed)
    draws = sorted(sum(values[int(rng.random() * 4)] for _ in range(4)) / 4 for _ in range(count))
    result = paired_bootstrap(values, seed, count, 0.95)
    assert result["mean_difference"] == 3
    assert result["ci_low"] == pytest.approx((draws[2] + draws[3]) / 2)
    assert result["ci_high"] == pytest.approx((draws[97] + draws[98]) / 2)
    assert result["n"] == 4
    assert result["seed"] == seed
    assert result["resamples"] == count
    assert result["confidence"] == 0.95


def test_bootstrap_determinism_no_mutation_and_global_rng_isolation():
    values = [8, -3, 1, 2, 5]
    original = values[:]
    state = random.getstate()
    first = paired_bootstrap(values, 5)
    assert first == paired_bootstrap(values, 5)
    assert first == paired_bootstrap(list(reversed(values)), 5)
    assert random.getstate() == state
    assert values == original
    assert first != paired_bootstrap(values, 6)


def test_bootstrap_empty_singleton_constant_and_extreme():
    empty = paired_bootstrap([], -1)
    assert empty["n"] == 0
    assert empty["ci_low"] is empty["ci_high"] is empty["mean_difference"] is None
    for values in [[3], [3, 3, 3], [1e308, 1e308]]:
        result = paired_bootstrap(values, 4)
        assert result["ci_low"] == result["ci_high"] == result["mean_difference"] == values[0]
        json.dumps(result, allow_nan=False)
    assert paired_bootstrap([1, 2], 1, resamples=1)["ci_low"] is not None


@pytest.mark.parametrize("differences", [None, (1, 2), [True], [math.nan], [math.inf], ["1"]])
def test_bootstrap_invalid_differences(differences):
    with pytest.raises((TypeError, ValueError)):
        paired_bootstrap(differences, 1)


@pytest.mark.parametrize("kwargs", [
    {"seed": True}, {"seed": None}, {"seed": 1.0}, {"seed": "1"},
    {"resamples": True}, {"resamples": 0}, {"resamples": -1}, {"resamples": 1.5},
    {"confidence": True}, {"confidence": 0}, {"confidence": 1}, {"confidence": math.nan},
])
def test_bootstrap_invalid_options(kwargs):
    options = {"seed": 1, **kwargs}
    with pytest.raises((TypeError, ValueError)):
        paired_bootstrap([], **options)


def test_clopper_pearson_analytic_endpoints():
    # At x=0 and x=n the beta quantiles reduce to simple powers.
    assert clopper_pearson(0, 20)["ci_high"] == pytest.approx(1 - 0.025 ** (1 / 20))
    assert clopper_pearson(20, 20)["ci_low"] == pytest.approx(0.025 ** (1 / 20))
    assert clopper_pearson(0, 20)["ci_low"] == 0
    assert clopper_pearson(20, 20)["ci_high"] == 1


@pytest.mark.parametrize("n", [1, 20, 10**6, 10**16, 10**100, 10**308])
@pytest.mark.parametrize("confidence", [0.5, 0.95, math.nextafter(1.0, 0.0)])
def test_clopper_pearson_endpoints_against_high_precision_analytic(n, confidence):
    # Independent Decimal evaluation retains the complement even for n=10**308.
    with localcontext() as context:
        context.prec = 360
        tail = (Decimal(1) - Decimal.from_float(confidence)) / 2
        endpoint = (tail.ln() / Decimal(n)).exp()
        expected_low = float(endpoint)
        expected_high = float(Decimal(1) - endpoint)
    empty_successes = clopper_pearson(0, n, confidence)
    all_successes = clopper_pearson(n, n, confidence)
    assert empty_successes["ci_low"] == 0
    assert empty_successes["ci_high"] == pytest.approx(expected_high, rel=5e-15, abs=0)
    assert all_successes["ci_low"] == pytest.approx(expected_low, rel=5e-15, abs=0)
    assert all_successes["ci_high"] == 1


def test_clopper_pearson_endpoint_count_must_be_float_representable():
    for successes in (0, 10**400):
        with pytest.raises(ValueError, match="n must be finite"):
            clopper_pearson(successes, 10**400)


@pytest.mark.parametrize(
    ("b", "c"),
    [(500000, 500001), (500001, 500001), (0, 1000001),
     (49999990500000, 50000009500000)],
)
def test_mcnemar_rejects_unsupported_counts_including_significance_regression(b, c):
    # The last case formerly gave p=.04827699 versus the reference .05743313.
    with pytest.raises(ValueError, match="exceeds supported limit 1000000"):
        exact_mcnemar(b, c)


@pytest.mark.parametrize("n", [1000001, 10**16])
@pytest.mark.parametrize("position", ["first", "middle", "last"])
def test_clopper_pearson_rejects_unsupported_interior_counts(n, position):
    successes = {"first": 1, "middle": n // 2, "last": n - 1}[position]
    with pytest.raises(ValueError, match="exceeds supported limit 1000000"):
        clopper_pearson(successes, n)


def test_supported_binomial_boundary_fixed_independent_references():
    # Fixed SciPy reference fixtures also run when SciPy is unavailable.
    assert MAX_BETA_BINOMIAL_N == 1_000_000
    assert exact_mcnemar(500000, 500000) == 1
    assert exact_mcnemar(499000, 501000) == pytest.approx(0.0456082998653797, rel=5e-9)
    assert exact_mcnemar(499020, 500980) == pytest.approx(0.05011273253278041, rel=5e-9)
    assert exact_mcnemar(499019, 500981) < 0.05 < exact_mcnemar(499020, 500980)
    for successes, low, high in [
        (1, 2.5317807663794178e-08, 5.571630655166941e-06),
        (500000, 0.49901951919531184, 0.5009804808046882),
        (999999, 0.9999944283693448, 0.9999999746821924),
    ]:
        result = clopper_pearson(successes, 1000000)
        assert result["ci_low"] == pytest.approx(low, rel=2e-8, abs=1e-15)
        assert result["ci_high"] == pytest.approx(high, rel=2e-8, abs=1e-15)


@pytest.mark.parametrize("n", [1025, 10000, 100000, 999999, 1000000])
def test_supported_binomial_range_against_scipy(n):
    stats = pytest.importorskip("scipy.stats")
    counts = {0, 1, n // 2 - 1, n // 2}
    counts.update(max(0, n // 2 - round(z * math.sqrt(n) / 2))
                  for z in (0.1, 1, 1.95, 1.96, 1.97, 2, 3, 5, 8, 20, 30))
    for b in counts:
        expected = stats.binomtest(b, n, 0.5).pvalue
        assert exact_mcnemar(b, n - b) == pytest.approx(expected, rel=5e-9, abs=1e-300)
    for successes in (1, 2, n // 100, n // 2, n - 2, n - 1):
        for confidence in (0.5, 0.9, 0.95, 0.99, 0.999999, math.nextafter(1.0, 0.0)):
            tail = (1 - confidence) / 2
            expected_low = stats.beta.ppf(tail, successes, n - successes + 1)
            expected_high = stats.beta.isf(tail, successes + 1, n - successes)
            actual = clopper_pearson(successes, n, confidence)
            assert actual["ci_low"] == pytest.approx(expected_low, rel=0, abs=2e-12)
            assert actual["ci_high"] == pytest.approx(expected_high, rel=0, abs=2e-12)


@pytest.mark.parametrize("df", [1, 2, 3, 5, 30, 100, 1000, 1000000])
def test_student_t_against_scipy_small_n_and_tails(df):
    stats = pytest.importorskip("scipy.stats")
    for t in [-30, -10, -3, -1, -1e-8, 0, 1e-8, 0.1, 1, 3, 10, 30, 1e6]:
        assert _student_t_cdf(t, df) == pytest.approx(stats.t.cdf(t, df), rel=2e-8, abs=1e-300)
        assert _student_t_sf(t, df) == pytest.approx(stats.t.sf(t, df), rel=2e-8, abs=1e-300)
    for probability in [0.49, 0.05, 0.001, 1e-12]:
        assert _student_t_isf(probability, df) == pytest.approx(
            stats.t.isf(probability, df), rel=2e-8,
        )


@pytest.mark.parametrize("n", [2, 3, 5, 30, 100])
@pytest.mark.parametrize("alpha", [0.01, 0.05, 0.1])
def test_tost_against_independent_scipy_calculation(n, alpha):
    stats = pytest.importorskip("scipy.stats")
    values = [math.sin(i * 2.7) * 0.2 + 0.1 for i in range(n)]
    mean = sum(values) / n
    se = math.sqrt(sum((value - mean) ** 2 for value in values) / (n - 1) / n)
    for margin in [0.01, 0.1, 0.5, 2]:
        result = paired_tost(values, margin, alpha)
        expected = max(stats.t.sf((mean + margin) / se, n - 1),
                       stats.t.cdf((mean - margin) / se, n - 1))
        radius = stats.t.isf(alpha, n - 1) * se
        assert result["pvalue"] == pytest.approx(expected, rel=2e-10, abs=1e-15)
        assert result["ci_low"] == pytest.approx(mean - radius, rel=2e-10, abs=1e-13)
        assert result["ci_high"] == pytest.approx(mean + radius, rel=2e-10, abs=1e-13)
        assert result["status"] == ("PASS" if expected < alpha else "FAIL")


def test_binomial_intervals_mcnemar_and_bh_against_scipy():
    stats = pytest.importorskip("scipy.stats")
    for successes, n in [(0, 10), (1, 10), (5, 10), (10, 10), (432, 1000)]:
        for confidence in [0.9, 0.95, 0.99]:
            reference = stats.binomtest(successes, n)
            for function, method in [(wilson_interval, "wilson"), (clopper_pearson, "exact")]:
                expected = reference.proportion_ci(confidence, method)
                actual = function(successes, n, confidence)
                assert actual["ci_low"] == pytest.approx(expected.low, abs=1e-11)
                assert actual["ci_high"] == pytest.approx(expected.high, abs=1e-11)
    for b, c in [(0, 6), (9, 15), (500, 600), (499000, 501000), (3000, 4000)]:
        assert exact_mcnemar(b, c) == pytest.approx(stats.binomtest(b, b + c).pvalue, rel=2e-8)
    values = [0.3, 0.001, 0.1, 0.1, 0, 1]
    assert benjamini_hochberg(values) == pytest.approx(stats.false_discovery_control(values))


def test_module_runs_with_stdlib_only_and_no_network():
    # Load the file directly under -S: root mltk legitimately needs numpy/pandas.
    path = Path(__file__).resolve().parents[2] / "src" / "mltk" / "core" / "statistics.py"
    script = """
import importlib.util
import sys
def offline(event, args):
    if event.startswith('socket.'):
        raise AssertionError('network use is forbidden')
sys.addaudithook(offline)
spec = importlib.util.spec_from_file_location('standalone_statistics', sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert module.paired_tost([-1, 1], 10)['status'] == 'PASS'
assert module.exact_mcnemar(6, 0) == 0.03125
assert module.wilson_interval(0, 20)['ci_high'] < .162
assert module.clopper_pearson(0, 20)['ci_high'] < .169
assert module.benjamini_hochberg([0.01, 0.05]) == [0.02, 0.05]
assert module.paired_bootstrap([1, 1], 1)['ci_low'] == 1
assert not any(name in sys.modules for name in ('scipy', 'numpy', 'pandas'))
print('stdlib-only/offline OK')
"""
    result = subprocess.run([sys.executable, "-S", "-c", script, str(path)],
                            capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "stdlib-only/offline OK"
