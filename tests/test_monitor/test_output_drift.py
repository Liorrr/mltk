"""Tests for assert_no_output_drift.

Verifies that behavioral drift in model output distributions is correctly
detected. Output drift is distinct from input feature drift: the model may
receive inputs that look statistically identical to training data but still
produce a shifted score distribution — due to weight updates, software
changes, or upstream preprocessing mutations.
"""

from __future__ import annotations

import numpy as np
import pytest

from mltk.core.assertion import MltkAssertionError
from mltk.monitor.drift_monitor import assert_no_output_drift


class TestAssertNoOutputDrift:
    """Tests for assert_no_output_drift."""

    def test_no_output_drift_identical(self) -> None:
        """SCENARIO: Reference and current outputs are drawn from the same distribution.
        WHY: Happy path — model deployed unchanged, both windows sample identically.
             KS test on identical distributions should return a high p-value.
        EXPECTED: pass=True, drift_detected == False.
        """
        rng = np.random.default_rng(10)
        scores = rng.random(1000).tolist()
        result = assert_no_output_drift(scores, scores)
        assert result.passed is True
        assert result.details["drift_detected"] is False

    def test_output_drift_detected(self) -> None:
        """SCENARIO: Current outputs are shifted by 0.5 relative to reference.
        WHY: A model that was retrained or fine-tuned may produce systematically
             higher or lower scores, which signals behavioral drift even when
             input data looks the same. A 0.5-unit shift on a [0,1] output is
             large enough that KS test should detect it reliably.
        EXPECTED: MltkAssertionError raised, drift_detected == True.
        """
        rng = np.random.default_rng(11)
        ref = rng.random(2000)
        # Shift current distribution by 0.5 (significant drift)
        cur = np.clip(rng.random(2000) + 0.5, 0, 1)

        with pytest.raises(MltkAssertionError) as exc:
            assert_no_output_drift(ref.tolist(), cur.tolist(), method="ks", threshold=0.05)
        result = exc.value.result
        assert result.details["drift_detected"] is True

    def test_output_drift_custom_threshold(self) -> None:
        """SCENARIO: Output distribution shifts slightly but a lenient threshold is used.
        WHY: Some applications tolerate small output shifts (e.g., re-calibration runs).
             A threshold of 1e-8 (near zero p-value tolerance) should pass for a
             tiny shift, while still detecting large shifts.
        EXPECTED: pass=True when threshold is set very leniently (threshold=1.0 accepts
                  any p-value since p <= 1.0 always, but we use p > threshold logic,
                  so we set threshold=0.0 to always pass).
        """
        rng = np.random.default_rng(12)
        ref = rng.random(500)
        # Slight shift — might or might not pass at default threshold=0.05
        cur = rng.random(500) + 0.05

        # threshold=0.0 means: pass if p_value > 0.0, which is always true
        result = assert_no_output_drift(
            ref.tolist(), cur.tolist(),
            method="ks",
            threshold=0.0,
        )
        assert result.passed is True

    def test_empty_outputs(self) -> None:
        """SCENARIO: Empty arrays passed (no production traffic in the current window).
        WHY: A newly deployed model may have zero predictions in its first monitoring
             window. The assertion must not crash — it should pass gracefully.
        EXPECTED: pass=True, drift_detected == False.
        """
        result = assert_no_output_drift([], [], method="ks", threshold=0.05)
        assert result.passed is True
        assert result.details["drift_detected"] is False

    def test_output_drift_psi_method(self) -> None:
        """SCENARIO: PSI method is used on heavily shifted outputs.
        WHY: Some teams prefer PSI over KS for its interpretability (PSI > 0.2
             is "significant shift" in financial ML). A PSI-based check should
             fire when distributions are very different.
        EXPECTED: MltkAssertionError raised when PSI exceeds threshold.
        """
        rng = np.random.default_rng(13)
        # Reference: uniform [0, 1]
        ref = rng.random(3000)
        # Current: concentrated near 0.9 (very different shape)
        cur = np.clip(rng.normal(0.9, 0.05, 3000), 0, 1)

        with pytest.raises(MltkAssertionError) as exc:
            assert_no_output_drift(ref.tolist(), cur.tolist(), method="psi", threshold=0.1)
        result = exc.value.result
        assert result.details["drift_detected"] is True
        assert result.details["method"] == "psi"

    def test_details_contain_statistic_and_method(self) -> None:
        """SCENARIO: Caller inspects TestResult details for audit logging.
        WHY: Downstream systems (dashboards, alerts) need the raw statistic and
             method name to display context without re-running the test.
        EXPECTED: result.details contains 'statistic', 'method', 'threshold'.
        """
        rng = np.random.default_rng(14)
        scores = rng.random(500).tolist()
        result = assert_no_output_drift(scores, scores, method="ks", threshold=0.05)
        assert "statistic" in result.details
        assert result.details["method"] == "ks"
        assert result.details["threshold"] == 0.05
