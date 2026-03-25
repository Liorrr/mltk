"""Ticket deduplication — prevent spam when tests fail repeatedly.

Uses content hashing to detect duplicate failures and cooldown periods
to avoid creating tickets for the same issue within a time window.
"""

from __future__ import annotations

import hashlib
import time
from typing import Any


class TicketDecisionEngine:
    """Decide whether to create, update, or skip a ticket.

    Rules:
    1. Severity threshold — only create for CRITICAL/HIGH
    2. Content hash dedup — same test + assertion = same ticket
    3. Cooldown period — don't recreate within N seconds

    Args:
        severity_threshold: Minimum severity to create ticket ("CRITICAL" or "HIGH").
        cooldown_seconds: Minimum seconds between tickets for same failure.

    Example:
        >>> engine = TicketDecisionEngine(severity_threshold="HIGH", cooldown_seconds=3600)
        >>> engine.should_create({"test_name": "test_drift", "severity": "CRITICAL"})
        True
    """

    def __init__(
        self,
        severity_threshold: str = "HIGH",
        cooldown_seconds: int = 21600,  # 6 hours
    ) -> None:
        self.severity_threshold = severity_threshold
        self.cooldown_seconds = cooldown_seconds
        self._recent_hashes: dict[str, float] = {}
        self._severity_rank = {"CRITICAL": 3, "HIGH": 2, "MEDIUM": 1, "LOW": 0}

    def _hash_failure(self, failure: dict[str, Any]) -> str:
        """Generate content hash for dedup."""
        parts = [
            failure.get("test_name", ""),
            failure.get("assertion_type", ""),
            failure.get("metric_name", ""),
        ]
        key = "#".join(parts)
        return hashlib.sha256(key.encode()).hexdigest()[:12]

    def should_create(self, failure: dict[str, Any]) -> bool:
        """Decide whether to create a new ticket for this failure.

        Args:
            failure: Dict with test_name, severity, assertion_type, etc.

        Returns:
            True if a new ticket should be created.

        Example:
            >>> engine.should_create({"test_name": "test_x", "severity": "LOW"})
            False  # below severity threshold
        """
        # Rule 1: Severity threshold
        severity = failure.get("severity", "LOW")
        threshold_rank = self._severity_rank.get(self.severity_threshold, 2)
        failure_rank = self._severity_rank.get(severity, 0)
        if failure_rank < threshold_rank:
            return False

        # Rule 2: Content hash dedup + cooldown
        content_hash = self._hash_failure(failure)
        now = time.time()

        if content_hash in self._recent_hashes:
            last_created = self._recent_hashes[content_hash]
            if now - last_created < self.cooldown_seconds:
                return False  # Too recent, skip

        # Record this creation
        self._recent_hashes[content_hash] = now
        return True

    def get_hash(self, failure: dict[str, Any]) -> str:
        """Get content hash for a failure (for labeling tickets).

        Args:
            failure: Dict with test failure details.

        Returns:
            12-character hex hash string.
        """
        return self._hash_failure(failure)
