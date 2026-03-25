"""Computer vision testing — IoU, mAP, frame accuracy, temporal consistency, top-K, face."""

from mltk.domains.cv.classification import assert_topk_accuracy
from mltk.domains.cv.detection import assert_iou, assert_map, compute_iou
from mltk.domains.cv.face import assert_face_far
from mltk.domains.cv.tracking import assert_idf1, assert_mota, assert_motp
from mltk.domains.cv.video import assert_frame_accuracy, assert_temporal_consistency

__all__ = [
    "compute_iou",
    "assert_iou",
    "assert_map",
    "assert_frame_accuracy",
    "assert_temporal_consistency",
    "assert_topk_accuracy",
    "assert_face_far",
    "assert_mota",
    "assert_motp",
    "assert_idf1",
]
