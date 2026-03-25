"""Multi-object tracking testing — MOTA, MOTP, IDF1 for tracker evaluation.

CLEAR-MOT standard: MOTA and MOTP jointly characterise tracker accuracy.
IDF1 measures identity preservation across the full sequence.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from mltk.core.assertion import assert_true, timed_assertion
from mltk.core.result import Severity, TestResult
from mltk.domains.cv.detection import compute_iou

# IoU threshold for a detection to count as a true match
_MATCH_IOU = 0.5
_EMPTY_BOXES = np.empty((0, 4), dtype=np.float64)


def _parse_frame(frame: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Extract ids and boxes arrays from a per-frame dict."""
    ids = np.asarray(frame.get("ids", []), dtype=int)
    raw = frame.get("boxes", _EMPTY_BOXES)
    boxes = np.asarray(raw, dtype=np.float64).reshape(-1, 4)
    return ids, boxes


def _match_frame(
    gt_boxes: np.ndarray,
    pred_boxes: np.ndarray,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Greedily match predictions to GT boxes by IoU >= 0.5.

    Args:
        gt_boxes: (N, 4) ground-truth boxes for one frame.
        pred_boxes: (M, 4) predicted boxes for one frame.

    Returns:
        Tuple of (matched_pairs, unmatched_gt_indices, unmatched_pred_indices).
        matched_pairs is a list of (gt_idx, pred_idx) tuples.
    """
    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        return [], list(range(len(gt_boxes))), list(range(len(pred_boxes)))

    iou_mat = compute_iou(gt_boxes, pred_boxes)  # (N, M)

    matched: list[tuple[int, int]] = []
    used_gt: set[int] = set()
    used_pred: set[int] = set()

    # Greedy: sort candidate pairs by IoU descending, take best-first
    n_gt, n_pred = iou_mat.shape
    candidates = [
        (iou_mat[i, j], i, j)
        for i in range(n_gt)
        for j in range(n_pred)
        if iou_mat[i, j] >= _MATCH_IOU
    ]
    candidates.sort(key=lambda x: x[0], reverse=True)

    for _, gi, pi in candidates:
        if gi not in used_gt and pi not in used_pred:
            matched.append((gi, pi))
            used_gt.add(gi)
            used_pred.add(pi)

    unmatched_gt = [i for i in range(n_gt) if i not in used_gt]
    unmatched_pred = [j for j in range(n_pred) if j not in used_pred]
    return matched, unmatched_gt, unmatched_pred


@timed_assertion
def assert_mota(
    gt_tracks: list[dict[str, Any]],
    pred_tracks: list[dict[str, Any]],
    min_mota: float = 0.5,
) -> TestResult:
    """Assert Multi-Object Tracking Accuracy (MOTA) meets threshold.

    MOTA = 1 - (FN + FP + IDSW) / total_gt.
    Counts false negatives, false positives, and identity switches over
    all frames. MOTA can be negative when errors exceed GT count.

    Args:
        gt_tracks: Per-frame ground-truth list. Each element is a dict with
            ``ids`` (array of int, length N) and ``boxes`` (array Nx4,
            [x1, y1, x2, y2]) for that frame.
        pred_tracks: Per-frame prediction list. Same format as gt_tracks.
        min_mota: Minimum required MOTA score (default 0.5).

    Returns:
        TestResult with mota, fn, fp, idsw, total_gt in details.

    Example:
        >>> gt = [{"ids": [1], "boxes": [[0, 0, 10, 10]]}]
        >>> pred = [{"ids": [1], "boxes": [[0, 0, 10, 10]]}]
        >>> assert_mota(gt, pred, min_mota=0.5)
    """
    total_gt = 0
    fn = 0
    fp = 0
    idsw = 0

    # id_map tracks the last known pred-id assigned to each gt-id
    id_map: dict[int, int] = {}

    for frame_gt, frame_pred in zip(gt_tracks, pred_tracks, strict=False):
        gt_ids, gt_boxes = _parse_frame(frame_gt)
        pred_ids, pred_boxes = _parse_frame(frame_pred)

        total_gt += len(gt_ids)

        matched, unmatched_gt, unmatched_pred = _match_frame(gt_boxes, pred_boxes)

        fn += len(unmatched_gt)
        fp += len(unmatched_pred)

        # Count ID switches: matched gt had a different pred-id previously
        for gi, pi in matched:
            gt_id = int(gt_ids[gi]) if gi < len(gt_ids) else -1
            pred_id = int(pred_ids[pi]) if pi < len(pred_ids) else -1
            prev_pred_id = id_map.get(gt_id)
            if prev_pred_id is not None and prev_pred_id != pred_id:
                idsw += 1
            id_map[gt_id] = pred_id

    if total_gt == 0:
        return assert_true(
            True,
            name="cv.mota",
            message="No ground-truth objects to evaluate",
            severity=Severity.INFO,
            mota=1.0,
            fn=0,
            fp=fp,
            idsw=0,
            total_gt=0,
        )

    mota = 1.0 - (fn + fp + idsw) / total_gt
    passed = mota >= min_mota
    message = (
        f"MOTA: {mota:.4f} >= {min_mota}"
        if passed
        else f"MOTA: {mota:.4f} < {min_mota} (FN={fn}, FP={fp}, IDSW={idsw})"
    )

    return assert_true(
        passed,
        name="cv.mota",
        message=message,
        severity=Severity.CRITICAL,
        mota=round(mota, 6),
        fn=fn,
        fp=fp,
        idsw=idsw,
        total_gt=total_gt,
    )


@timed_assertion
def assert_motp(
    gt_tracks: list[dict[str, Any]],
    pred_tracks: list[dict[str, Any]],
    min_motp: float = 0.5,
) -> TestResult:
    """Assert Multi-Object Tracking Precision (MOTP) meets threshold.

    MOTP = mean IoU of all correctly matched detection pairs across every
    frame. A value near 1.0 indicates tight box localisation; near 0.5
    means boxes barely overlap the GT.

    Args:
        gt_tracks: Per-frame ground-truth list. Each element is a dict with
            ``ids`` (array of int) and ``boxes`` (array Nx4).
        pred_tracks: Per-frame prediction list. Same format as gt_tracks.
        min_motp: Minimum required MOTP score (default 0.5).

    Returns:
        TestResult with motp and num_matches in details.

    Example:
        >>> gt = [{"ids": [1], "boxes": [[0, 0, 10, 10]]}]
        >>> pred = [{"ids": [1], "boxes": [[0, 0, 10, 10]]}]
        >>> assert_motp(gt, pred, min_motp=0.5)
    """
    iou_sum = 0.0
    num_matches = 0

    for frame_gt, frame_pred in zip(gt_tracks, pred_tracks, strict=False):
        _, gt_boxes = _parse_frame(frame_gt)
        _, pred_boxes = _parse_frame(frame_pred)

        if len(gt_boxes) == 0 or len(pred_boxes) == 0:
            continue

        iou_mat = compute_iou(gt_boxes, pred_boxes)
        matched, _, _ = _match_frame(gt_boxes, pred_boxes)

        for gi, pi in matched:
            iou_sum += float(iou_mat[gi, pi])
            num_matches += 1

    if num_matches == 0:
        return assert_true(
            False,
            name="cv.motp",
            message="No matched pairs found across all frames",
            severity=Severity.CRITICAL,
            motp=0.0,
            num_matches=0,
        )

    motp = iou_sum / num_matches
    passed = motp >= min_motp
    message = (
        f"MOTP: {motp:.4f} >= {min_motp} ({num_matches} matched pairs)"
        if passed
        else f"MOTP: {motp:.4f} < {min_motp} ({num_matches} matched pairs)"
    )

    return assert_true(
        passed,
        name="cv.motp",
        message=message,
        severity=Severity.CRITICAL,
        motp=round(motp, 6),
        num_matches=num_matches,
    )


@timed_assertion
def assert_idf1(
    gt_tracks: list[dict[str, Any]],
    pred_tracks: list[dict[str, Any]],
    min_idf1: float = 0.5,
) -> TestResult:
    """Assert ID F1 score (IDF1) meets threshold.

    IDF1 = 2 * IDTP / (2 * IDTP + IDFP + IDFN), where:
    - IDTP: detections where the predicted ID matches the most-associated GT ID
    - IDFP: predicted detections that are false or have the wrong ID
    - IDFN: GT detections that were missed or assigned a wrong ID

    The most-associated GT-to-pred mapping is determined by majority vote
    across all matched frames (each GT id is paired with the pred id it
    co-occurs with most often).

    Args:
        gt_tracks: Per-frame ground-truth list. Each element is a dict with
            ``ids`` (array of int) and ``boxes`` (array Nx4).
        pred_tracks: Per-frame prediction list. Same format as gt_tracks.
        min_idf1: Minimum required IDF1 score (default 0.5).

    Returns:
        TestResult with idf1, idtp, idfp, idfn in details.

    Example:
        >>> gt = [{"ids": [1, 2], "boxes": [[0,0,10,10],[20,20,30,30]]}]
        >>> pred = [{"ids": [1, 2], "boxes": [[0,0,10,10],[20,20,30,30]]}]
        >>> assert_idf1(gt, pred, min_idf1=0.5)
    """
    # co_occurrence[gt_id][pred_id] = number of matched frames
    from collections import defaultdict

    co_occur: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    total_gt_dets: dict[int, int] = defaultdict(int)
    total_pred_dets: dict[int, int] = defaultdict(int)

    for frame_gt, frame_pred in zip(gt_tracks, pred_tracks, strict=False):
        gt_ids, gt_boxes = _parse_frame(frame_gt)
        pred_ids, pred_boxes = _parse_frame(frame_pred)

        for gid in gt_ids:
            total_gt_dets[int(gid)] += 1
        for pid in pred_ids:
            total_pred_dets[int(pid)] += 1

        matched, _, _ = _match_frame(gt_boxes, pred_boxes)
        for gi, pi in matched:
            if gi < len(gt_ids) and pi < len(pred_ids):
                co_occur[int(gt_ids[gi])][int(pred_ids[pi])] += 1

    if not total_gt_dets and not total_pred_dets:
        return assert_true(
            True,
            name="cv.idf1",
            message="No objects in any frame",
            severity=Severity.INFO,
            idf1=1.0,
            idtp=0,
            idfp=0,
            idfn=0,
        )

    # Build best bijective mapping: each GT id → its most-co-occurring pred id
    # (Each pred id can be assigned to at most one GT id — greedy by count)
    mapping: dict[int, int] = {}
    used_pred_ids: set[int] = set()

    # Sort GT ids by their best co-occurrence count descending for greedy assignment
    sorted_gt = sorted(
        co_occur.keys(),
        key=lambda g: max(co_occur[g].values()) if co_occur[g] else 0,
        reverse=True,
    )
    for gt_id in sorted_gt:
        best_pred = max(
            (pid for pid in co_occur[gt_id] if pid not in used_pred_ids),
            key=lambda pid: co_occur[gt_id][pid],
            default=None,
        )
        if best_pred is not None:
            mapping[gt_id] = best_pred
            used_pred_ids.add(best_pred)

    # Compute IDTP, IDFP, IDFN
    idtp = 0
    for gt_id, pred_id in mapping.items():
        idtp += co_occur[gt_id][pred_id]

    total_gt_count = sum(total_gt_dets.values())
    total_pred_count = sum(total_pred_dets.values())

    idfn = total_gt_count - idtp
    idfp = total_pred_count - idtp

    denom = 2 * idtp + idfp + idfn
    idf1 = (2 * idtp / denom) if denom > 0 else 0.0

    passed = idf1 >= min_idf1
    message = (
        f"IDF1: {idf1:.4f} >= {min_idf1} (IDTP={idtp}, IDFP={idfp}, IDFN={idfn})"
        if passed
        else f"IDF1: {idf1:.4f} < {min_idf1} (IDTP={idtp}, IDFP={idfp}, IDFN={idfn})"
    )

    return assert_true(
        passed,
        name="cv.idf1",
        message=message,
        severity=Severity.CRITICAL,
        idf1=round(idf1, 6),
        idtp=idtp,
        idfp=idfp,
        idfn=idfn,
    )
