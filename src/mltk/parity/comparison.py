"""Pair classification with explicit precedence and source evidence flags."""

from __future__ import annotations

from mltk.parity.models import value_key

CHANGED_CLASSES = frozenset({"CHANGED_NO_ORACLE", "NEG_FLIP", "POS_FLIP", "CHANGED_SAME_VERDICT"})
UNRESOLVED_CLASSES = frozenset({"EXCLUDED", "INDETERMINATE"})


def classify_pair(
    base: dict,
    candidate: dict,
    *,
    base_source: str | None = None,
    candidate_source: str | None = None,
) -> dict:
    """Classify reduced states; caller supplies stable item/layer/stratum IDs."""
    source_unknown = base_source is None or candidate_source is None
    source_changed = not source_unknown and base_source != candidate_source
    a, b = base["status"], candidate["status"]
    reasons = []
    if a == "NOT_SCORABLE":
        reasons.append("BASE_NOT_SCORABLE")
    if b == "NOT_SCORABLE":
        reasons.append("CANDIDATE_NOT_SCORABLE")
    if reasons:
        classification, group = "EXCLUDED", "Unconcluded"
    elif a == b == "FAILED":
        classification, group = "FAILED_BOTH", "Failure"
    elif a == "FAILED":
        classification, group = "FAILED_BASE", "Failure"
    elif b == "FAILED":
        classification, group = "FAILED_CAND", "Failure"
    elif "INDETERMINATE" in (a, b):
        classification, group = "INDETERMINATE", "Unconcluded"
    elif value_key(base["value"]) == value_key(candidate["value"]):
        classification, group = "UNCHANGED", "Unchanged"
    elif source_changed:
        classification, group = "CHANGED_SOURCE", "Excluded"
    elif base["oracle"] is None or candidate["oracle"] is None:
        classification, group = "CHANGED_NO_ORACLE", "Changed"
    elif base["oracle"] is True and candidate["oracle"] is False:
        classification, group = "NEG_FLIP", "Changed"
    elif base["oracle"] is False and candidate["oracle"] is True:
        classification, group = "POS_FLIP", "Changed"
    else:
        classification, group = "CHANGED_SAME_VERDICT", "Changed"
    if classification != "UNCHANGED" and not reasons:
        reasons.append(classification)
    if source_changed:
        reasons.append("SOURCE_CHANGED")
    return {
        "classification": classification,
        "group": group,
        "reason": reasons[0] if reasons else None,
        "reasons": reasons,
        "from_failure": classification == "FAILED_CAND" and base["oracle"] is True,
        "source_unknown": source_unknown,
        "source_changed": source_changed,
    }
