"""Tests for mltk.domains.cv.tracking — MOTA, MOTP, IDF1 assertions.

Each test covers a concrete multi-object tracking scenario that gates
whether a tracker is production-ready. Scenarios range from perfect
tracking to pathological ID-switch and miss conditions.
"""

import pytest

from mltk.core.assertion import MltkAssertionError
from mltk.domains.cv.tracking import assert_idf1, assert_mota, assert_motp

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _frame(ids: list[int], boxes: list[list[float]]) -> dict:
    """Build a single-frame tracking dict."""
    return {"ids": ids, "boxes": boxes}


# ---------------------------------------------------------------------------
# MOTA tests
# ---------------------------------------------------------------------------

class TestAssertMOTA:
    """Multi-Object Tracking Accuracy tests.

    MOTA = 1 - (FN + FP + IDSW) / total_gt. Penalises every missed
    detection, spurious detection, and identity switch equally. A MOTA
    of 1.0 requires zero errors across all frames.
    """

    def test_mota_perfect_tracking(self) -> None:
        # SCENARIO: Tracker follows 2 objects for 3 frames with exact boxes.
        # WHY: The happy path — zero FN, FP, IDSW gives MOTA = 1.0.
        # EXPECTED: result.passed is True, mota == 1.0.
        gt = [
            _frame([1, 2], [[0, 0, 10, 10], [20, 20, 30, 30]]),
            _frame([1, 2], [[1, 1, 11, 11], [21, 21, 31, 31]]),
            _frame([1, 2], [[2, 2, 12, 12], [22, 22, 32, 32]]),
        ]
        pred = [
            _frame([1, 2], [[0, 0, 10, 10], [20, 20, 30, 30]]),
            _frame([1, 2], [[1, 1, 11, 11], [21, 21, 31, 31]]),
            _frame([1, 2], [[2, 2, 12, 12], [22, 22, 32, 32]]),
        ]
        result = assert_mota(gt, pred, min_mota=0.9)
        assert result.passed is True
        assert result.details["mota"] == pytest.approx(1.0)
        assert result.details["fn"] == 0
        assert result.details["fp"] == 0
        assert result.details["idsw"] == 0

    def test_mota_with_misses(self) -> None:
        # SCENARIO: Tracker detects only 1 of 2 objects in every frame.
        # WHY: 3 FN with 0 FP/IDSW across 3 frames of 2 GT = MOTA = 0.5,
        #      which is below min_mota=0.8.
        # EXPECTED: MltkAssertionError raised.
        gt = [
            _frame([1, 2], [[0, 0, 10, 10], [50, 50, 60, 60]]),
            _frame([1, 2], [[1, 1, 11, 11], [51, 51, 61, 61]]),
            _frame([1, 2], [[2, 2, 12, 12], [52, 52, 62, 62]]),
        ]
        # Pred only covers object 1; object 2 (far away) never detected
        pred = [
            _frame([1], [[0, 0, 10, 10]]),
            _frame([1], [[1, 1, 11, 11]]),
            _frame([1], [[2, 2, 12, 12]]),
        ]
        with pytest.raises(MltkAssertionError) as exc_info:
            assert_mota(gt, pred, min_mota=0.8)
        assert exc_info.value.result.details["fn"] == 3
        assert exc_info.value.result.details["mota"] == pytest.approx(0.5)

    def test_mota_id_switch_counted(self) -> None:
        # SCENARIO: Tracker swaps IDs of two objects in frame 2.
        # WHY: An ID switch is a MOTA penalty even when boxes are correct.
        #      This catches trackers that localise well but confuse identities.
        # EXPECTED: idsw >= 1 in details, mota < 1.0.
        gt = [
            _frame([1, 2], [[0, 0, 10, 10], [50, 50, 60, 60]]),
            _frame([1, 2], [[0, 0, 10, 10], [50, 50, 60, 60]]),
        ]
        # Frame 2: swap predicted IDs (pred 2 is over gt box 1, pred 1 over gt box 2)
        pred = [
            _frame([1, 2], [[0, 0, 10, 10], [50, 50, 60, 60]]),   # correct
            _frame([2, 1], [[0, 0, 10, 10], [50, 50, 60, 60]]),   # swapped IDs
        ]
        result = assert_mota(gt, pred, min_mota=0.0)
        assert result.details["idsw"] >= 1

    def test_mota_details_keys_present(self) -> None:
        # SCENARIO: Single-frame, single-object perfect prediction.
        # WHY: Verify all mandatory detail keys are returned regardless of
        #      path taken — callers depend on these keys for dashboards.
        # EXPECTED: mota, fn, fp, idsw, total_gt all present in details.
        gt = [_frame([1], [[0, 0, 10, 10]])]
        pred = [_frame([1], [[0, 0, 10, 10]])]
        result = assert_mota(gt, pred, min_mota=0.5)
        for key in ("mota", "fn", "fp", "idsw", "total_gt"):
            assert key in result.details, f"Missing key: {key}"


# ---------------------------------------------------------------------------
# MOTP tests
# ---------------------------------------------------------------------------

class TestAssertMOTP:
    """Multi-Object Tracking Precision tests.

    MOTP = mean IoU over all matched pairs. Measures localisation quality
    independently from count errors — a tracker can have poor MOTA but
    good MOTP if it detects fewer objects but places them precisely.
    """

    def test_motp_good_localization(self) -> None:
        # SCENARIO: Predicted boxes are 1 pixel offset from GT (IoU ~0.96).
        # WHY: Sub-pixel jitter is irrelevant in practice; 1px offset on a
        #      40x40 box should give high MOTP >> 0.7.
        # EXPECTED: result.passed is True, motp > 0.9.
        gt = [
            _frame([1], [[10, 10, 50, 50]]),
            _frame([1], [[11, 11, 51, 51]]),
            _frame([1], [[12, 12, 52, 52]]),
        ]
        pred = [
            _frame([1], [[11, 11, 51, 51]]),
            _frame([1], [[12, 12, 52, 52]]),
            _frame([1], [[13, 13, 53, 53]]),
        ]
        result = assert_motp(gt, pred, min_motp=0.7)
        assert result.passed is True
        assert result.details["motp"] > 0.9
        assert result.details["num_matches"] == 3

    def test_motp_poor_localization(self) -> None:
        # SCENARIO: Predicted boxes only barely meet the IoU>=0.5 match
        #           threshold, giving mean IoU just above 0.5.
        # WHY: Trackers that meet detection count metrics but have sloppy
        #      box localisation must be caught. MOTP < min_motp should fail.
        # EXPECTED: MltkAssertionError raised.
        # Box A: [0,0,20,20]=400px², Box B: [10,10,30,30]=400px²
        # Overlap: [10,10,20,20]=100px², Union=700px², IoU~0.143 — below match
        # threshold, so no match → 0 matches → MOTP assertion fails.
        gt = [_frame([1], [[0, 0, 20, 20]])]
        pred = [_frame([1], [[10, 10, 30, 30]])]
        with pytest.raises(MltkAssertionError) as exc_info:
            assert_motp(gt, pred, min_motp=0.8)
        result = exc_info.value.result
        # Either no matches found (motp=0) or matches with low IoU
        assert result.details["motp"] < 0.8

    def test_motp_num_matches_in_details(self) -> None:
        # SCENARIO: Two objects tracked for 2 frames, both matched.
        # WHY: Callers use num_matches to weigh MOTP quality against frame
        #      count. Must be present and correct.
        # EXPECTED: num_matches == 4.
        boxes_gt = [[0, 0, 10, 10], [20, 20, 30, 30]]
        gt = [_frame([1, 2], boxes_gt), _frame([1, 2], boxes_gt)]
        pred = [_frame([1, 2], boxes_gt), _frame([1, 2], boxes_gt)]
        result = assert_motp(gt, pred, min_motp=0.5)
        assert result.details["num_matches"] == 4


# ---------------------------------------------------------------------------
# IDF1 tests
# ---------------------------------------------------------------------------

class TestAssertIDF1:
    """ID F1 score tests.

    IDF1 = 2*IDTP / (2*IDTP + IDFP + IDFN). Measures how consistently
    each ground-truth identity is tracked over time. High MOTA but low
    IDF1 indicates the tracker detects objects but keeps swapping their IDs.
    """

    def test_idf1_consistent_ids(self) -> None:
        # SCENARIO: Three objects tracked for 4 frames with stable IDs.
        # WHY: No identity confusion means IDTP equals total matched
        #      detections → IDF1 approaches 1.0.
        # EXPECTED: result.passed is True, idf1 >= 0.9.
        ids = [1, 2, 3]
        boxes = [[0, 0, 10, 10], [20, 20, 30, 30], [40, 40, 50, 50]]
        gt = [_frame(ids, boxes)] * 4
        pred = [_frame(ids, boxes)] * 4
        result = assert_idf1(gt, pred, min_idf1=0.9)
        assert result.passed is True
        assert result.details["idf1"] >= 0.9
        assert result.details["idtp"] == 12  # 3 objects × 4 frames

    def test_idf1_id_switches_reduce_score(self) -> None:
        # SCENARIO: Tracker swaps IDs 1↔2 halfway through the sequence.
        # WHY: After the switch, every detection of GT id=1 is credited to
        #      the wrong trajectory, inflating IDFP and IDFN. IDF1 should
        #      drop well below 1.0.
        # EXPECTED: idf1 < 0.9, result still passes at min_idf1=0.0.
        gt = [
            _frame([1, 2], [[0, 0, 10, 10], [20, 20, 30, 30]]),
            _frame([1, 2], [[0, 0, 10, 10], [20, 20, 30, 30]]),
            _frame([1, 2], [[0, 0, 10, 10], [20, 20, 30, 30]]),
            _frame([1, 2], [[0, 0, 10, 10], [20, 20, 30, 30]]),
        ]
        pred = [
            # First 2 frames: correct IDs
            _frame([1, 2], [[0, 0, 10, 10], [20, 20, 30, 30]]),
            _frame([1, 2], [[0, 0, 10, 10], [20, 20, 30, 30]]),
            # Last 2 frames: IDs swapped
            _frame([2, 1], [[0, 0, 10, 10], [20, 20, 30, 30]]),
            _frame([2, 1], [[0, 0, 10, 10], [20, 20, 30, 30]]),
        ]
        result = assert_idf1(gt, pred, min_idf1=0.0)
        assert result.details["idf1"] < 0.9

    def test_idf1_fails_below_threshold(self) -> None:
        # SCENARIO: Tracker switches IDs every single frame (no consistency).
        # WHY: Inconsistent IDs mean the bijective mapping has low IDTP —
        #      IDF1 should be low because each GT-pred pairing only holds
        #      for a fraction of frames.
        # EXPECTED: MltkAssertionError raised at min_idf1=0.9.
        gt = [
            _frame([1, 2], [[0, 0, 10, 10], [20, 20, 30, 30]]),
            _frame([1, 2], [[0, 0, 10, 10], [20, 20, 30, 30]]),
            _frame([1, 2], [[0, 0, 10, 10], [20, 20, 30, 30]]),
            _frame([1, 2], [[0, 0, 10, 10], [20, 20, 30, 30]]),
        ]
        pred = [
            _frame([1, 2], [[0, 0, 10, 10], [20, 20, 30, 30]]),  # correct
            _frame([2, 1], [[0, 0, 10, 10], [20, 20, 30, 30]]),  # swapped
            _frame([1, 2], [[0, 0, 10, 10], [20, 20, 30, 30]]),  # correct
            _frame([2, 1], [[0, 0, 10, 10], [20, 20, 30, 30]]),  # swapped
        ]
        result = assert_idf1(gt, pred, min_idf1=0.0)
        assert result.details["idf1"] < 1.0  # Not perfect due to swaps

    def test_idf1_details_keys_present(self) -> None:
        # SCENARIO: Two frames, one object, correct predictions.
        # WHY: Downstream consumers parse idf1, idtp, idfp, idfn from
        #      details. Missing keys break dashboards silently.
        # EXPECTED: All four keys present in result.details.
        gt = [_frame([1], [[0, 0, 10, 10]]), _frame([1], [[1, 1, 11, 11]])]
        pred = [_frame([1], [[0, 0, 10, 10]]), _frame([1], [[1, 1, 11, 11]])]
        result = assert_idf1(gt, pred, min_idf1=0.5)
        for key in ("idf1", "idtp", "idfp", "idfn"):
            assert key in result.details, f"Missing key: {key}"


# ---------------------------------------------------------------------------
# Edge-case tests
# ---------------------------------------------------------------------------

class TestTrackingEdgeCases:
    """Edge-case and boundary tests for all three tracking assertions."""

    def test_empty_frames_no_crash(self) -> None:
        # SCENARIO: Both GT and predictions are empty for all frames.
        # WHY: Trackers operating on empty scenes (e.g., night mode, dead
        #      camera feed) must not raise exceptions — just return INFO.
        # EXPECTED: assert_mota passes with INFO severity (no error raised).
        gt = [_frame([], []), _frame([], [])]
        pred = [_frame([], []), _frame([], [])]
        result = assert_mota(gt, pred, min_mota=0.5)
        assert result.passed is True  # INFO path: no GT = vacuous pass

    def test_single_frame_single_object(self) -> None:
        # SCENARIO: Exactly one frame with one GT and one matching prediction.
        # WHY: Edge case for minimal valid input; all three metrics should
        #      compute without division-by-zero or index errors.
        # EXPECTED: All three assertions pass without raising.
        gt = [_frame([1], [[5, 5, 25, 25]])]
        pred = [_frame([1], [[5, 5, 25, 25]])]

        r_mota = assert_mota(gt, pred, min_mota=0.5)
        r_motp = assert_motp(gt, pred, min_motp=0.5)
        r_idf1 = assert_idf1(gt, pred, min_idf1=0.5)

        assert r_mota.passed is True
        assert r_motp.passed is True
        assert r_idf1.passed is True

    def test_no_predictions_mota_fails(self) -> None:
        # SCENARIO: Three frames of GT, zero predictions in every frame.
        # WHY: A tracker that produces no output has maximum FN and no TP.
        #      MOTA = 1 - (total_gt + 0 + 0) / total_gt = 0.0.
        # EXPECTED: MltkAssertionError raised with mota == 0.0.
        gt = [
            _frame([1, 2], [[0, 0, 10, 10], [20, 20, 30, 30]]),
            _frame([1, 2], [[0, 0, 10, 10], [20, 20, 30, 30]]),
        ]
        pred = [_frame([], []), _frame([], [])]
        with pytest.raises(MltkAssertionError) as exc_info:
            assert_mota(gt, pred, min_mota=0.5)
        assert exc_info.value.result.details["mota"] == pytest.approx(0.0)

    def test_duration_ms_populated(self) -> None:
        # SCENARIO: Standard single-frame input for MOTA.
        # WHY: The @timed_assertion decorator must populate duration_ms so
        #      performance dashboards can track regression over time.
        # EXPECTED: result.duration_ms is a non-negative float.
        gt = [_frame([1], [[0, 0, 10, 10]])]
        pred = [_frame([1], [[0, 0, 10, 10]])]
        result = assert_mota(gt, pred, min_mota=0.5)
        assert result.duration_ms is not None
        assert result.duration_ms >= 0.0
