"""Rust <-> Python numerical parity tests for every dual-engine function.

Each function in ``mltk._rust`` has a Rust fast path and a pure-Python
(numpy/scipy) fallback. These tests run identical inputs through BOTH
engines and assert the results agree. Divergence here means assertion
outcomes silently differ between machines with and without the compiled
extension — the worst kind of flakiness to debug.

Statistics are compared tightly (both engines compute the same exact
quantity). KS/chi-squared p-values are compared loosely: the engines use
different tail approximations (Stephens-corrected asymptotic Kolmogorov
vs scipy's exact/asymp switch), which is a documented, bounded
difference — not a bug.
"""

from __future__ import annotations

import numpy as np
import pytest

from mltk import _rust

pytestmark = pytest.mark.skipif(
    not _rust.RUST_AVAILABLE, reason="Rust extension not built"
)

_RNG = np.random.default_rng(42)
_NORMAL_300 = _RNG.normal(0.0, 1.0, 300).tolist()
_SHIFTED_250 = _RNG.normal(0.5, 1.2, 250).tolist()
_TINY_A = [1.0, 2.0, 3.0, 4.0, 5.0]
_TINY_B = [1.5, 2.5, 3.0, 4.5, 5.5]
_TIED_REF = [1.0] * 50 + [2.0] * 50
_TIED_CUR = [1.0] * 30 + [2.0] * 70

DIST_PAIRS = [
    pytest.param(_NORMAL_300, list(_NORMAL_300), id="identical"),
    pytest.param(_NORMAL_300, _SHIFTED_250, id="shifted"),
    pytest.param(_TINY_A, _TINY_B, id="tiny"),
    pytest.param(_TIED_REF, _TIED_CUR, id="heavy-ties"),
]


def _both(monkeypatch, fn, *args, **kwargs):
    """Return (rust_result, python_result) for the same call."""
    rust_result = fn(*args, **kwargs)
    monkeypatch.setattr(_rust, "RUST_AVAILABLE", False)
    python_result = fn(*args, **kwargs)
    monkeypatch.setattr(_rust, "RUST_AVAILABLE", True)
    return rust_result, python_result


# ===================================================================
# Distribution-distance functions
# ===================================================================


class TestKsTest:
    """ks_test parity: exact statistic, approximation-bounded p-value."""

    @pytest.mark.parametrize(("ref", "cur"), DIST_PAIRS)
    def test_statistic_parity(self, monkeypatch, ref, cur) -> None:
        (rust_stat, _), (py_stat, _) = _both(monkeypatch, _rust.ks_test, ref, cur)
        assert rust_stat == pytest.approx(py_stat, abs=1e-12)

    @pytest.mark.parametrize(("ref", "cur"), DIST_PAIRS)
    def test_p_value_parity(self, monkeypatch, ref, cur) -> None:
        (_, rust_p), (_, py_p) = _both(monkeypatch, _rust.ks_test, ref, cur)
        # Different tail approximations; bounded, not identical.
        assert rust_p == pytest.approx(py_p, abs=0.05)

    def test_identical_samples_statistic_is_zero(self) -> None:
        """Regression: the pre-fix Rust merge scored identical samples 1/n.

        Any value present in both samples made the two-pointer sweep
        measure the ECDF gap mid-jump, inflating D — false drift
        positives on integer or categorical-coded features.
        """
        stat, p = _rust.ks_test(_TINY_A, list(_TINY_A))
        assert stat == 0.0
        assert p == 1.0

    def test_tied_samples_match_scipy_exactly(self) -> None:
        """Regression: heavy cross-sample ties scored 0.7 pre-fix; true D=0.2."""
        stat, _ = _rust.ks_test(_TIED_REF, _TIED_CUR)
        assert stat == pytest.approx(0.2, abs=1e-12)


class TestHistogramDistances:
    """psi / kl_divergence / js_divergence parity (same binning scheme)."""

    @pytest.mark.parametrize(("ref", "cur"), DIST_PAIRS)
    def test_psi_parity(self, monkeypatch, ref, cur) -> None:
        rust_val, py_val = _both(monkeypatch, _rust.psi, ref, cur)
        assert rust_val == pytest.approx(py_val, rel=1e-9, abs=1e-12)

    @pytest.mark.parametrize(("ref", "cur"), DIST_PAIRS)
    def test_kl_parity(self, monkeypatch, ref, cur) -> None:
        rust_val, py_val = _both(monkeypatch, _rust.kl_divergence, ref, cur)
        assert rust_val == pytest.approx(py_val, rel=1e-9, abs=1e-12)

    @pytest.mark.parametrize(("ref", "cur"), DIST_PAIRS)
    def test_js_parity(self, monkeypatch, ref, cur) -> None:
        rust_val, py_val = _both(monkeypatch, _rust.js_divergence, ref, cur)
        assert rust_val == pytest.approx(py_val, rel=1e-9, abs=1e-12)

    @pytest.mark.parametrize("bins", [2, 5, 20])
    def test_psi_bins_parity(self, monkeypatch, bins) -> None:
        rust_val, py_val = _both(
            monkeypatch, _rust.psi, _NORMAL_300, _SHIFTED_250, bins
        )
        assert rust_val == pytest.approx(py_val, rel=1e-9, abs=1e-12)


class TestWasserstein:
    @pytest.mark.parametrize(("ref", "cur"), DIST_PAIRS)
    def test_parity(self, monkeypatch, ref, cur) -> None:
        rust_val, py_val = _both(monkeypatch, _rust.wasserstein, ref, cur)
        assert rust_val == pytest.approx(py_val, rel=1e-9, abs=1e-12)

    def test_unequal_lengths(self, monkeypatch) -> None:
        rust_val, py_val = _both(
            monkeypatch, _rust.wasserstein, [0.0, 1.0, 2.0], [0.5, 1.5]
        )
        assert rust_val == pytest.approx(py_val, rel=1e-9, abs=1e-12)


class TestWassersteinNumpyFallback:
    """The scipy-less numpy path must agree with the other two engines.

    ``_both`` cannot reach this path: with RUST_AVAILABLE off the bridge
    tries scipy first, and CI always has scipy — so the numpy integral
    was never exercised. It weighted each interval by the CDF gap at the
    RIGHT endpoint (where both CDFs already jumped) instead of the left,
    returning 0.0 for disjoint point masses — a silent false 'no drift'
    on installs with neither the extension nor scipy.
    """

    def _numpy_only(self, monkeypatch) -> None:
        import sys

        monkeypatch.setattr(_rust, "RUST_AVAILABLE", False)
        monkeypatch.setitem(sys.modules, "scipy.stats", None)

    @pytest.mark.parametrize(("ref", "cur"), DIST_PAIRS)
    def test_numpy_matches_rust(self, monkeypatch, ref, cur) -> None:
        rust_val = _rust.wasserstein(ref, cur)
        self._numpy_only(monkeypatch)
        numpy_val = _rust.wasserstein(ref, cur)
        assert numpy_val == pytest.approx(rust_val, rel=1e-9, abs=1e-12)

    def test_numpy_disjoint_point_masses(self, monkeypatch) -> None:
        """Regression: the pre-fix integral scored this 0.0; true W1 = 1."""
        self._numpy_only(monkeypatch)
        assert _rust.wasserstein([0.0], [1.0]) == pytest.approx(1.0)


class TestChiSquared:
    CASES = [
        pytest.param([10.0, 20.0, 30.0], [15.0, 20.0, 25.0], id="moderate"),
        pytest.param([10.0, 20.0, 30.0], [10.0, 20.0, 30.0], id="identical"),
        pytest.param(
            [100.0, 250.0, 300.0, 350.0], [200.0, 250.0, 300.0, 250.0], id="large"
        ),
    ]

    @pytest.mark.parametrize(("obs", "exp"), CASES)
    def test_statistic_parity(self, monkeypatch, obs, exp) -> None:
        (rust_stat, _), (py_stat, _) = _both(monkeypatch, _rust.chi_squared, obs, exp)
        assert rust_stat == pytest.approx(py_stat, rel=1e-9, abs=1e-12)

    @pytest.mark.parametrize(("obs", "exp"), CASES)
    def test_p_value_parity(self, monkeypatch, obs, exp) -> None:
        (_, rust_p), (_, py_p) = _both(monkeypatch, _rust.chi_squared, obs, exp)
        assert rust_p == pytest.approx(py_p, rel=1e-6, abs=1e-9)

    @pytest.mark.parametrize(
        ("obs", "exp"),
        [*CASES, pytest.param([50.0, 60.0, 40.0], [50.0, 50.0, 50.0], id="mid-p")],
    )
    def test_numpy_fallback_p_value(self, monkeypatch, obs, exp) -> None:
        """The scipy-less numpy path (Wilson-Hilferty + erfc) must track the
        exact p-value closely. ``_both`` can't reach it — with RUST_AVAILABLE
        off the bridge tries scipy first, and CI always has scipy. The
        pre-fix exp(-0.7|z|) normal-CDF approximation deviated by up to
        ~0.09 in p — enough to flip borderline assertions. Wilson-Hilferty
        itself is approximation-bounded, hence abs=5e-3 (not 1e-9).
        """
        import sys

        (_, rust_p) = _rust.chi_squared(obs, exp)
        monkeypatch.setattr(_rust, "RUST_AVAILABLE", False)
        monkeypatch.setitem(sys.modules, "scipy.stats", None)
        (_, numpy_p) = _rust.chi_squared(obs, exp)
        assert numpy_p == pytest.approx(rust_p, abs=5e-3)


# ===================================================================
# Vector functions
# ===================================================================


class TestCosineSimilarity:
    VEC_A = _RNG.normal(0.0, 1.0, 64).tolist()
    VEC_B = _RNG.normal(0.0, 1.0, 64).tolist()

    CASES = [
        pytest.param(VEC_A, VEC_B, id="random"),
        pytest.param(VEC_A, list(VEC_A), id="identical"),
        pytest.param([1.0, 0.0], [0.0, 1.0], id="orthogonal"),
        pytest.param([1.0, 1.0], [-1.0, -1.0], id="opposite"),
        pytest.param([0.0, 0.0], [1.0, 1.0], id="zero-norm"),
    ]

    @pytest.mark.parametrize(("a", "b"), CASES)
    def test_parity(self, monkeypatch, a, b) -> None:
        rust_val, py_val = _both(monkeypatch, _rust.cosine_similarity, a, b)
        assert rust_val == pytest.approx(py_val, abs=1e-12)


class TestCentroidCosineDistance:
    EMBS_REF = _RNG.normal(0.0, 1.0, (5, 16)).tolist()
    EMBS_CUR = _RNG.normal(0.3, 1.0, (7, 16)).tolist()

    def test_parity_random(self, monkeypatch) -> None:
        rust_val, py_val = _both(
            monkeypatch, _rust.centroid_cosine_distance, self.EMBS_REF, self.EMBS_CUR
        )
        assert rust_val == pytest.approx(py_val, abs=1e-12)

    def test_parity_identical(self, monkeypatch) -> None:
        rust_val, py_val = _both(
            monkeypatch,
            _rust.centroid_cosine_distance,
            self.EMBS_REF,
            [list(e) for e in self.EMBS_REF],
        )
        assert rust_val == pytest.approx(py_val, abs=1e-12)


class TestBertscore:
    EMBS_REF = _RNG.normal(0.0, 1.0, (5, 16)).tolist()
    EMBS_HYP = _RNG.normal(0.3, 1.0, (7, 16)).tolist()

    def test_parity_random(self, monkeypatch) -> None:
        rust_prf, py_prf = _both(
            monkeypatch, _rust.bertscore_precision_recall, self.EMBS_REF, self.EMBS_HYP
        )
        for rust_v, py_v in zip(rust_prf, py_prf, strict=True):
            assert rust_v == pytest.approx(py_v, abs=1e-12)

    def test_parity_identical(self, monkeypatch) -> None:
        rust_prf, py_prf = _both(
            monkeypatch,
            _rust.bertscore_precision_recall,
            self.EMBS_REF,
            [list(e) for e in self.EMBS_REF],
        )
        for rust_v, py_v in zip(rust_prf, py_prf, strict=True):
            assert rust_v == pytest.approx(py_v, abs=1e-12)


# ===================================================================
# PII scanning
# ===================================================================


class TestScanPiiFast:
    TEXT = (
        "Contact john@example.com or 555-123-4567, "
        "card 4111-1111-1111-1111, password: hunter2"
    )
    PATTERNS = [
        ("email", r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),
        ("phone", r"\b\d{3}[\s.\-]?\d{3}[\s.\-]?\d{4}\b"),
        ("password", r"(?i)(?:password|pwd|passwd|pass)\s*[:=]\s*\S+"),
    ]

    def test_hits_identical(self, monkeypatch) -> None:
        rust_hits, py_hits = _both(
            monkeypatch, _rust.scan_pii_fast, self.TEXT, self.PATTERNS
        )
        assert sorted(rust_hits) == sorted(py_hits)
        assert len(rust_hits) == 3

    def test_no_matches(self, monkeypatch) -> None:
        rust_hits, py_hits = _both(
            monkeypatch, _rust.scan_pii_fast, "nothing here", self.PATTERNS
        )
        assert rust_hits == py_hits == []


class TestKsNanPropagation:
    """NaN inputs must poison the result on both engines, matching scipy."""

    def test_nan_propagates_like_scipy(self, monkeypatch) -> None:
        """Regression: a single-sided NaN froze one merge pointer and
        returned a plausible-looking D over a prefix of the data.
        """
        ref = [1.0, float("nan"), 2.0]
        cur = [1.0, 2.0, 3.0]
        (rust_stat, rust_p), (py_stat, py_p) = _both(
            monkeypatch, _rust.ks_test, ref, cur
        )
        assert np.isnan(rust_stat)
        assert np.isnan(rust_p)
        assert np.isnan(py_stat)
        assert np.isnan(py_p)
