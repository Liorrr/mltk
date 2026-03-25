"""Chat engine — rule-based Q&A about mltk test results."""

from __future__ import annotations

import json
import re
from pathlib import Path

# ---------------------------------------------------------------------------
# Known test categories used for recommendations
# ---------------------------------------------------------------------------

_ALL_CATEGORIES: list[tuple[str, str, str]] = [
    # (prefix, label, suggestion)
    ("data.schema",    "schema",        "Add schema testing with assert_schema()"),
    ("data.drift",     "drift",         "Add drift testing with assert_no_drift()"),
    ("data.pii",       "PII",           "Add PII scanning with assert_no_pii()"),
    ("data.labels",    "labels",        "Add label testing with assert_label_distribution()"),
    ("data.freshness", "freshness",     "Add freshness testing with assert_freshness()"),
    ("model.metric",   "metrics",       "Add metric testing with assert_metric_above()"),
    ("model.bias",     "bias/fairness", "Add bias testing with assert_no_bias()"),
    ("model.slice",    "slicing",       "Add slicing tests with assert_slice_metric()"),
    ("model.calib",    "calibration",   "Add calibration tests with assert_calibration()"),
    ("model.adv",      "adversarial",   "Add adversarial tests with assert_robust()"),
    ("inference.latency",    "latency",    "Add latency tests with assert_latency()"),
    ("inference.throughput", "throughput", "Add throughput tests with assert_throughput()"),
    ("inference.contract",   "contract",   "Add contract tests with assert_inference_contract()"),
    ("pipeline",       "pipeline",      "Add pipeline tests with assert_pipeline_reproducible()"),
    ("monitor",        "monitoring",    "Add monitoring tests with assert_no_degradation()"),
]

# Minimum categories expected in a well-tested ML system
_RECOMMENDED_PREFIXES = [
    "data.schema",
    "data.drift",
    "model.metric",
    "model.bias",
    "inference.latency",
    "monitor",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_HELP_TEXT = """\
Available commands:
  what failed?      — list all failed tests
  why did X fail?   — show failure details for test X (use test name or keyword)
  summary           — overall pass/fail count and score
  recommend         — suggest missing test categories
  score             — ML Test Score breakdown by category
  drift             — drift-related test results
  bias / fairness   — bias and fairness test results
  slow / slowest    — top 5 slowest tests by duration
  help              — show this message"""


def _pct(value: float) -> str:
    return f"{value:.1f}%"


# ---------------------------------------------------------------------------
# ChatEngine
# ---------------------------------------------------------------------------

class ChatEngine:
    """Analyze test results and answer questions.

    Not an LLM — uses pattern matching + structured analysis.
    """

    def __init__(self, results_path: str | Path | None = None) -> None:
        """Load test results from JSON file."""
        self.results: list[dict] = []
        if results_path:
            self._load(results_path)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load(self, path: str | Path) -> None:
        """Load results from JSON.

        Accepts two formats:
        - A JSON array of result objects (bare list).
        - A JSON object with a ``"results"`` key (produced by --mltk-export-json).
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Results file not found: {path}")
        raw = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(raw, list):
            self.results = raw
        elif isinstance(raw, dict):
            self.results = raw.get("results", [])
        else:
            self.results = []

    # ------------------------------------------------------------------
    # Public query API
    # ------------------------------------------------------------------

    def ask(self, question: str) -> str:
        """Answer a question about the test results.

        Supported questions (pattern matched):

        - "what failed?" / "failures" / "show failures" → list failed tests
        - "why did X fail?" / "why X" → show failure details for test X
        - "summary" / "status" / "overview" → pass/fail/score summary
        - "what should I test?" / "recommend" → suggest missing test types
        - "score" / "ml test score" → ML Test Score breakdown
        - "drift" / "data drift" → drift-related test results
        - "bias" / "fairness" → bias-related test results
        - "slow" / "slowest" / "performance" → slowest tests by duration
        - "help" → list available commands
        """
        q = question.strip().lower()

        # help
        if q in ("help", "?", "commands", "what can you do"):
            return _HELP_TEXT

        # summary / status / overview
        if any(kw in q for kw in ("summary", "status", "overview", "stats")):
            return self._format_summary()

        # score
        if re.search(r"\bscore\b", q):
            return self._format_score()

        # what failed / failures / show failures
        if re.search(r"(what\s+failed|failures|show\s+fail|list\s+fail)", q):
            return self._format_failures()

        # recommend / what should I test / missing
        if re.search(r"(recommend|what should|missing|should i test)", q):
            return self._format_recommendations()

        # drift
        if re.search(r"\bdrift\b", q):
            return self._format_category("drift", ["data.drift", "drift"])

        # bias / fairness
        if re.search(r"\b(bias|fairness|fair)\b", q):
            return self._format_category("bias/fairness", ["model.bias", "bias", "fairness"])

        # slow / slowest / performance
        if re.search(r"\b(slow|slowest|performance|duration)\b", q):
            return self._format_slowest()

        # why did X fail / why X
        why_match = re.search(
            r"why\s+(?:did\s+)?(?:\"?([^\"?]+?)\"?\s+)?(?:fail|failed)?",
            q,
        )
        if why_match:
            target = (why_match.group(1) or "").strip()
            return self._format_why(target)

        # fallback
        return (
            "I didn't understand that question.\n"
            "Type 'help' to see available commands."
        )

    # ------------------------------------------------------------------
    # Structured data accessors
    # ------------------------------------------------------------------

    def get_failures(self) -> list[dict]:
        """Return all failed test results."""
        return [r for r in self.results if not r.get("passed", True)]

    def get_summary(self) -> dict:
        """Return summary stats: total, passed, failed, score, duration."""
        total = len(self.results)
        passed = sum(1 for r in self.results if r.get("passed", True))
        failed = total - passed
        score = (passed / total * 100) if total else 0.0
        duration = sum(r.get("duration_ms", 0.0) for r in self.results)
        return {
            "total": total,
            "passed": passed,
            "failed": failed,
            "score": score,
            "duration_ms": duration,
        }

    def get_recommendations(self) -> list[str]:
        """Suggest tests that should be added based on what's missing.

        Checks which test categories are present (by name prefix) and
        returns a suggestion string for each missing recommended category.
        """
        present = {r.get("name", "") for r in self.results}

        def _has_prefix(prefix: str) -> bool:
            return any(name.startswith(prefix) for name in present)

        missing: list[str] = []
        for prefix, _label, suggestion in _ALL_CATEGORIES:
            if prefix in _RECOMMENDED_PREFIXES and not _has_prefix(prefix):
                missing.append(suggestion)

        return missing

    def get_slowest(self, n: int = 5) -> list[dict]:
        """Return N slowest tests by duration."""
        sorted_results = sorted(
            self.results,
            key=lambda r: r.get("duration_ms", 0.0),
            reverse=True,
        )
        return sorted_results[:n]

    # ------------------------------------------------------------------
    # Internal formatters
    # ------------------------------------------------------------------

    def _format_summary(self) -> str:
        if not self.results:
            return "No results loaded. Run pytest --mltk-export-json first."
        s = self.get_summary()
        status = "PASS" if s["failed"] == 0 else "FAIL"
        lines = [
            f"Status: {status}",
            f"  Total:    {s['total']}",
            f"  Passed:   {s['passed']}",
            f"  Failed:   {s['failed']}",
            f"  Score:    {_pct(s['score'])}",
            f"  Duration: {s['duration_ms']:.1f} ms",
        ]
        return "\n".join(lines)

    def _format_score(self) -> str:
        if not self.results:
            return "No results loaded. Run pytest --mltk-export-json first."

        # Group by top-level category prefix
        categories: dict[str, list[dict]] = {}
        for r in self.results:
            name = r.get("name", "")
            prefix = name.split(".")[0] if "." in name else "other"
            categories.setdefault(prefix, []).append(r)

        lines = ["ML Test Score by category:"]
        for cat, results in sorted(categories.items()):
            total = len(results)
            passed = sum(1 for r in results if r.get("passed", True))
            pct = passed / total * 100
            lines.append(f"  {cat:<20} {passed}/{total} ({_pct(pct)})")

        overall = self.get_summary()
        lines.append("")
        lines.append(f"Overall: {_pct(overall['score'])} ({overall['passed']}/{overall['total']})")
        return "\n".join(lines)

    def _format_failures(self) -> str:
        failures = self.get_failures()
        if not self.results:
            return "No results loaded. Run pytest --mltk-export-json first."
        if not failures:
            return "All tests passed! No failures."

        lines = [f"{len(failures)} test(s) failed:"]
        for r in failures:
            name = r.get("name", "unknown")
            message = r.get("message", "")
            lines.append(f"  - {name}: {message}")
        return "\n".join(lines)

    def _format_why(self, target: str) -> str:
        if not self.results:
            return "No results loaded. Run pytest --mltk-export-json first."

        if not target:
            return "Specify a test name. Example: why did model.bias fail?"

        # Find the best match among failures (then all results)
        target_lower = target.lower()
        candidates = [r for r in self.results if target_lower in r.get("name", "").lower()]
        if not candidates:
            candidates = [
                r for r in self.results
                if target_lower in r.get("message", "").lower()
            ]
        if not candidates:
            return (
                f"No test matching '{target}' found.\n"
                f"Type 'what failed?' to see failed tests."
            )

        result = candidates[0]
        name = result.get("name", "unknown")
        passed = result.get("passed", True)
        message = result.get("message", "")
        details = result.get("details", {})
        duration_ms = result.get("duration_ms", 0.0)
        status = "PASS" if passed else "FAIL"

        lines = [
            f"{name} — {status}",
            f"  {message}",
        ]
        if details:
            for k, v in details.items():
                lines.append(f"  {k}: {v}")
        lines.append(f"  duration: {duration_ms:.2f} ms")
        return "\n".join(lines)

    def _format_category(self, label: str, prefixes: list[str]) -> str:
        if not self.results:
            return "No results loaded. Run pytest --mltk-export-json first."

        matched = [
            r for r in self.results
            if any(
                r.get("name", "").lower().startswith(p)
                or p in r.get("name", "").lower()
                for p in prefixes
            )
        ]

        if not matched:
            return f"No {label} tests found in results."

        total = len(matched)
        passed = sum(1 for r in matched if r.get("passed", True))
        lines = [f"{label} tests: {passed}/{total} passed"]
        for r in matched:
            name = r.get("name", "unknown")
            status = "PASS" if r.get("passed", True) else "FAIL"
            message = r.get("message", "")
            lines.append(f"  [{status}] {name}: {message}")
        return "\n".join(lines)

    def _format_slowest(self) -> str:
        if not self.results:
            return "No results loaded. Run pytest --mltk-export-json first."

        slowest = self.get_slowest(5)
        if not slowest:
            return "No results to show."

        lines = ["Slowest 5 tests:"]
        for i, r in enumerate(slowest, 1):
            name = r.get("name", "unknown")
            duration_ms = r.get("duration_ms", 0.0)
            lines.append(f"  {i}. {name}: {duration_ms:.2f} ms")
        return "\n".join(lines)

    def _format_recommendations(self) -> str:
        recs = self.get_recommendations()
        if not recs:
            if not self.results:
                return "No results loaded — load results first to get recommendations."
            return "Good coverage! No missing test categories detected."

        lines = ["Missing test categories:"]
        for rec in recs:
            lines.append(f"  - {rec}")
        return "\n".join(lines)
