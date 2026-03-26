"""Tests for mltk.monitor -- production monitoring assertions.

Monitor tests validate assertions used for post-deployment model health:
- Degradation detection: catches gradual metric decline (concept drift,
  data quality decay) before it reaches crisis levels.
- SLA compliance: gates on latency and error rate thresholds defined in
  service-level agreements.

These are run on live metric streams, not just at training time.
"""

import pytest

from mltk.core.assertion import MltkAssertionError
from mltk.monitor import assert_no_degradation, assert_sla


class TestAssertNoDegradation:
    """Metric degradation detection tests.

    Validates that assert_no_degradation correctly identifies gradual
    metric decline using a sliding window comparison. This catches the
    "slow rot" failure mode where model quality degrades week over week.
    """

    def test_stable_metrics(self) -> None:
        """PASS: Metrics fluctuate normally within a stable range.

        WHY: Normal noise in production metrics (0.91-0.93 accuracy) should
        not trigger false alarms. The window-based comparison smooths out
        natural variance.
        Expected: result.passed is True.
        """
        history = [0.92, 0.91, 0.93, 0.92, 0.91, 0.92, 0.93, 0.91, 0.92, 0.93]
        result = assert_no_degradation(history, window=5, max_decline=0.05)
        assert result.passed is True

    def test_degradation_detected(self) -> None:
        """FAIL: Accuracy dropped from 0.95 to 0.65 over 10 data points.

        WHY: A steady decline from 95% to 65% accuracy means the model is
        losing predictive power -- likely due to concept drift or data quality
        degradation. This must trigger an alert before users notice.
        Expected: MltkAssertionError with "Degradation" in message.
        """
        history = [0.95, 0.94, 0.93, 0.92, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65]
        with pytest.raises(MltkAssertionError) as exc:
            assert_no_degradation(history, window=5, max_decline=0.05)
        assert "Degradation" in str(exc.value)

    def test_insufficient_history(self) -> None:
        """PASS: Too few data points for meaningful comparison.

        WHY: A newly deployed model has only 2 metric snapshots. The window
        of 5 cannot be filled, so comparison is meaningless. Should pass
        (informational) rather than block or error.
        Expected: result.passed is True.
        """
        result = assert_no_degradation([0.9, 0.9], window=5)
        assert result.passed is True


class TestAssertSLA:
    """SLA compliance tests.

    Validates that assert_sla correctly gates on latency and error rate
    thresholds defined in service-level agreements. SLA breaches have
    contractual and financial consequences.
    """

    def test_sla_compliant(self) -> None:
        """PASS: Latency and error rate both within SLA bounds.

        WHY: P99 latency at 100ms vs 500ms limit, error rate at 0.5% vs
        1% limit. Both comfortably within SLA. This is the healthy
        production state.
        Expected: result.passed is True.
        """
        result = assert_sla(
            latency_p99=100.0, error_rate=0.005,
            thresholds={"latency_p99_ms": 500.0, "error_rate": 0.01}
        )
        assert result.passed is True

    def test_latency_breach(self) -> None:
        """FAIL: P99 latency at 600ms exceeds 500ms SLA.

        WHY: Latency breaches cause user-facing degradation, timeout cascades,
        and may trigger contractual penalties. This must fire an alert so
        the on-call team can investigate (model too large, infra issue, etc.).
        Expected: MltkAssertionError with "SLA breach" in message.
        """
        with pytest.raises(MltkAssertionError) as exc:
            assert_sla(
                latency_p99=600.0,
                thresholds={"latency_p99_ms": 500.0}
            )
        assert "SLA breach" in str(exc.value)

    def test_error_rate_breach(self) -> None:
        """FAIL: Error rate at 5% exceeds 1% SLA.

        WHY: A 5x error rate spike means the model is crashing or returning
        garbage for 1 in 20 requests. Even if latency is fine, this
        violates the reliability SLA.
        Expected: MltkAssertionError raised.
        """
        with pytest.raises(MltkAssertionError):
            assert_sla(
                error_rate=0.05,
                thresholds={"error_rate": 0.01}
            )

    def test_sla_no_metrics_provided(self) -> None:
        """EDGE: assert_sla called with neither latency_p99 nor error_rate.

        WHY: Neither metric is measured so there are no violations — the result
             should pass (vacuously compliant) rather than crash.
        Expected: result.passed is True, zero violations.
        """
        result = assert_sla()
        assert result.passed is True
        assert result.details["violations"] == []

    def test_no_degradation_exact_window_boundary(self) -> None:
        """EDGE: History length == window — earlier slice is history[:1].

        WHY: When len(arr) == window, earlier = arr[:1] (single element).
             Tests the boundary branch where arr[:-window] would be empty.
             Stable values should still pass.
        """
        history = [0.90, 0.90, 0.90, 0.90, 0.90]
        result = assert_no_degradation(history, window=5, max_decline=0.05)
        assert result.passed is True
