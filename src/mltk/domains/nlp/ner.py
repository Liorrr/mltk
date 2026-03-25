"""Named Entity Recognition testing -- entity-level F1 scoring."""

from __future__ import annotations

from mltk.core.assertion import assert_true, timed_assertion
from mltk.core.result import Severity, TestResult


@timed_assertion
def assert_ner_f1(
    y_true_entities: list[list[tuple[str, int, int]]],
    y_pred_entities: list[list[tuple[str, int, int]]],
    min_f1: float = 0.8,
) -> TestResult:
    """Assert entity-level F1 score meets threshold.

    Args:
        y_true_entities: Ground truth entities per document.
            Each entity is (label, start, end).
        y_pred_entities: Predicted entities per document.
        min_f1: Minimum required F1 score.

    Returns:
        TestResult with precision, recall, F1.

    Example:
        >>> true = [[("PER", 0, 5)]]
        >>> pred = [[("PER", 0, 5), ("ORG", 10, 15)]]
        >>> assert_ner_f1(true, pred, min_f1=0.5)
    """
    tp = 0
    fp = 0
    fn = 0

    for true_ents, pred_ents in zip(y_true_entities, y_pred_entities, strict=False):
        true_set = set(true_ents)
        pred_set = set(pred_ents)
        tp += len(true_set & pred_set)
        fp += len(pred_set - true_set)
        fn += len(true_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    passed = f1 >= min_f1
    message = (
        f"NER F1: {f1:.4f} >= {min_f1} (P={precision:.4f}, R={recall:.4f})"
        if passed
        else f"NER F1: {f1:.4f} < {min_f1} (P={precision:.4f}, R={recall:.4f})"
    )

    return assert_true(
        passed, name="nlp.ner_f1", message=message,
        severity=Severity.CRITICAL,
        f1=f1, precision=precision, recall=recall,
        tp=tp, fp=fp, fn=fn, min_f1=min_f1,
    )
