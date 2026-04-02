"""mltk security-scan CLI command -- red team catalog scanning.

Uses the canonical attack catalog from
``mltk.domains.llm.red_team.catalog`` instead of maintaining
a separate hardcoded payload list. Grading is delegated to
``mltk.domains.llm.red_team._grading._check_compromised``
which provides category-aware vulnerability detection.
Mutations are handled by
``mltk.domains.llm.red_team.mutations.mutate_payloads``
which applies 8 encoding techniques (Base64, ROT13,
leetspeak, homoglyphs, zero-width, mixed-case, markdown
injection, HTML entities).
"""

from __future__ import annotations

import importlib
import json
from collections import defaultdict
from collections.abc import Callable
from typing import Any

from mltk.domains.llm.red_team._grading import (
    _check_compromised,
)
from mltk.domains.llm.red_team.catalog import (
    ATTACK_CATALOG,
    AttackCategory,
    AttackPayload,
)
from mltk.domains.llm.red_team.mutations import (
    mutate_payloads,
)

# -------------------------------------------------------------------
# Category CLI mapping
# -------------------------------------------------------------------

# Maps lowercase CLI strings (used with ``--category``) to
# ``AttackCategory`` enum members. Users pass snake_case
# names on the command line; this dict resolves them.
_CLI_TO_CATEGORY: dict[str, AttackCategory] = {
    member.value: member for member in AttackCategory
}

# Maps ``AttackCategory`` enum members to human-readable
# display names used in table output and JSON reports.
_CATEGORY_DISPLAY_NAMES: dict[AttackCategory, str] = {
    AttackCategory.PROMPT_INJECTION: "Prompt Injection",
    AttackCategory.JAILBREAK: "Jailbreak",
    AttackCategory.DATA_EXTRACTION: "Data Extraction",
    AttackCategory.HARMFUL_CONTENT: "Harmful Content",
    AttackCategory.EXCESSIVE_AGENCY: "Excessive Agency",
    AttackCategory.SYSTEM_PROMPT_THEFT: (
        "System Prompt Theft"
    ),
    AttackCategory.ENCODING_BYPASS: "Encoding Bypass",
}


def _display_name(category: AttackCategory) -> str:
    """Return human-readable display name for a category."""
    return _CATEGORY_DISPLAY_NAMES.get(
        category, category.value,
    )


def _flatten_catalog(
    categories: list[AttackCategory] | None = None,
) -> list[AttackPayload]:
    """Flatten ``ATTACK_CATALOG`` into a single payload list.

    When *categories* is ``None``, all categories are
    included. Otherwise only payloads whose category is
    in the given list are returned.

    Args:
        categories: Optional filter -- only include these
            categories.

    Returns:
        Flat list of ``AttackPayload`` objects.
    """
    payloads: list[AttackPayload] = []
    for cat, cat_payloads in ATTACK_CATALOG.items():
        if categories is not None and cat not in categories:
            continue
        payloads.extend(cat_payloads)
    return payloads


# -------------------------------------------------------------------
# Backward-compatible aliases
# -------------------------------------------------------------------

# Legacy flat list of dicts (``{"category": ..., "payload": ...}``)
# generated from the canonical catalog. Provided so that existing
# tests and downstream code that imported ``_RED_TEAM_CATALOG`` or
# ``_generate_mutations`` from this module continue to work.

_RED_TEAM_CATALOG: list[dict[str, str]] = [
    {
        "category": _display_name(p.category),
        "payload": p.payload_text,
    }
    for p in _flatten_catalog()
]


def _generate_mutations(
    payloads: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Generate encoding mutations (legacy dict format).

    Produces two mutations per payload (leetspeak and
    reversed), preserving the original behavior and
    return type. The CLI path uses ``mutate_payloads``
    from the canonical mutations module (8 techniques);
    this wrapper exists solely for backward compatibility
    with code that used the old ``_generate_mutations``
    API.

    New code should use ``mutate_payloads`` directly.
    """
    mutated: list[dict[str, str]] = []
    for entry in payloads:
        text = entry["payload"]
        cat = entry["category"] + " (mutation)"
        # leetspeak
        leet = text.translate(
            str.maketrans(
                "aeiostAEIOST", "431057431057",
            )
        )
        mutated.append(
            {"category": cat, "payload": leet}
        )
        # reversed
        mutated.append(
            {"category": cat, "payload": text[::-1]}
        )
    return mutated


# -------------------------------------------------------------------
# Dynamic import helper
# -------------------------------------------------------------------

def _import_model_fn(path: str) -> Callable[..., Any]:
    """Import ``'module.path:function_name'``.

    Raises ``ValueError`` if the format is wrong and
    propagates ``ImportError`` / ``AttributeError`` if
    the module or function cannot be found.
    """
    module_path, _, fn_name = path.rpartition(":")
    if not module_path or not fn_name:
        raise ValueError(
            f"Expected 'module:function' format, got '{path}'. "
            f"Example: 'myapp.llm:chat_fn'"
        )
    module = importlib.import_module(module_path)
    fn = getattr(module, fn_name)
    if not callable(fn):
        raise ValueError(
            f"'{fn_name}' in '{module_path}' is not callable. "
            f"It must be a function that accepts a string prompt "
            f"and returns a string response."
        )
    return fn


# -------------------------------------------------------------------
# Scanning engine
# -------------------------------------------------------------------

def _coerce_catalog(
    catalog: list[AttackPayload] | list[dict[str, str]],
) -> list[AttackPayload]:
    """Ensure *catalog* is a list of ``AttackPayload``.

    Accepts both the canonical ``AttackPayload`` format and
    the legacy ``dict[str, str]`` format (with ``"category"``
    and ``"payload"`` keys). Legacy dicts are converted by
    looking up the display name in
    ``_CATEGORY_DISPLAY_NAMES``; unrecognized categories
    default to ``PROMPT_INJECTION``.

    This enables backward compatibility with test fixtures
    and downstream code that still passes dict-based catalogs.
    """
    if not catalog:
        return []
    first = catalog[0]
    if isinstance(first, AttackPayload):
        return catalog  # type: ignore[return-value]

    # Legacy dict format -- convert.
    name_to_cat: dict[str, AttackCategory] = {
        v.lower(): k
        for k, v in _CATEGORY_DISPLAY_NAMES.items()
    }
    result: list[AttackPayload] = []
    for entry in catalog:
        raw_cat = entry["category"]  # type: ignore[index]
        cat_enum = name_to_cat.get(
            raw_cat.lower(),
            AttackCategory.PROMPT_INJECTION,
        )
        result.append(
            AttackPayload(
                category=cat_enum,
                payload_text=entry["payload"],  # type: ignore[index]
                description="legacy payload",
                owasp_id="LLM01",
            )
        )
    return result


def _run_scan(
    model_fn: Callable[..., Any],
    catalog: list[AttackPayload] | list[dict[str, str]],
    verbose: bool = False,
) -> dict[str, Any]:
    """Run payloads against *model_fn* and return results.

    Accepts both ``AttackPayload`` objects and legacy dicts
    (``{"category": ..., "payload": ...}``). Legacy dicts
    are transparently converted via ``_coerce_catalog``.

    Each payload is sent to *model_fn* and the response is
    graded using ``_check_compromised`` from the canonical
    grading module. This provides category-aware detection
    (e.g. PROMPT_INJECTION looks for compliance markers,
    JAILBREAK checks for engagement without refusal, etc.)
    instead of a single flat regex list.

    Returns a dict keyed by human-readable category name
    with ``tested``, ``resisted``, and ``details`` per
    category.
    """
    typed_catalog = _coerce_catalog(catalog)

    categories: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "tested": 0,
            "resisted": 0,
            "details": [],
        }
    )

    for entry in typed_catalog:
        cat_name = _display_name(entry.category)
        payload_text = entry.payload_text
        categories[cat_name]["tested"] += 1

        try:
            response = str(model_fn(payload_text))
        except Exception:
            # Model error counts as resisted (not
            # exploitable via this payload).
            categories[cat_name]["resisted"] += 1
            if verbose:
                categories[cat_name]["details"].append({
                    "payload": payload_text[:60],
                    "result": "ERROR",
                    "resisted": True,
                })
            continue

        vulnerable = _check_compromised(
            response, entry.category,
        )

        if not vulnerable:
            categories[cat_name]["resisted"] += 1

        if verbose:
            categories[cat_name]["details"].append({
                "payload": payload_text[:60],
                "result": (
                    "VULNERABLE" if vulnerable
                    else "RESISTED"
                ),
                "resisted": not vulnerable,
            })

    return dict(categories)


# -------------------------------------------------------------------
# Output formatters
# -------------------------------------------------------------------

def _format_table(
    target: str,
    categories: dict[str, dict[str, Any]],
    threshold: float,
    mutations_on: bool,
    verbose: bool,
) -> str:
    """Format scan results as a human-readable table."""
    total_tested = sum(
        c["tested"] for c in categories.values()
    )
    total_resisted = sum(
        c["resisted"] for c in categories.values()
    )
    overall_rate = (
        total_resisted / total_tested
        if total_tested > 0 else 1.0
    )
    passed = overall_rate >= threshold

    lines: list[str] = []
    lines.append("mltk Security Scan Report")
    lines.append("=" * 26)
    mutations_label = "ON" if mutations_on else "OFF"
    lines.append(f"Target: {target}")
    lines.append(
        f"Categories: {len(categories)} | "
        f"Payloads: {total_tested} | "
        f"Mutations: {mutations_label}"
    )
    lines.append("")

    header = (
        f"{'Category':<26}"
        f"{'Tested':>7}"
        f"{'Resisted':>10}"
        f"{'Rate':>8}"
    )
    lines.append(header)
    sep = "\u2500" * 53
    lines.append(sep)

    for cat_name, data in sorted(categories.items()):
        tested = data["tested"]
        resisted = data["resisted"]
        rate = (
            resisted / tested if tested > 0 else 1.0
        )
        rate_str = f"{rate * 100:.1f}%"
        flag = (
            "  \u26a0" if rate < threshold else ""
        )
        lines.append(
            f"{cat_name:<26}"
            f"{tested:>7}"
            f"{resisted:>10}"
            f"{rate_str:>8}"
            f"{flag}"
        )
        if verbose and data.get("details"):
            for d in data["details"]:
                status = d["result"]
                snip = d["payload"]
                lines.append(f"    {status}: {snip}")

    lines.append(sep)
    total_rate_str = f"{overall_rate * 100:.1f}%"
    result_flag = (
        "  \u2713" if passed else "  \u26a0"
    )
    lines.append(
        f"{'TOTAL':<26}"
        f"{total_tested:>7}"
        f"{total_resisted:>10}"
        f"{total_rate_str:>8}"
        f"{result_flag}"
    )
    lines.append("")
    result_word = "PASS" if passed else "FAIL"
    lines.append(
        f"Threshold: {threshold * 100:.1f}% | "
        f"Result: {result_word}"
    )
    return "\n".join(lines)


def _format_json(
    target: str,
    categories: dict[str, dict[str, Any]],
    threshold: float,
) -> str:
    """Format scan results as a JSON string."""
    total_tested = sum(
        c["tested"] for c in categories.values()
    )
    total_resisted = sum(
        c["resisted"] for c in categories.values()
    )
    overall_rate = (
        total_resisted / total_tested
        if total_tested > 0 else 1.0
    )
    passed = overall_rate >= threshold

    cat_out: dict[str, Any] = {}
    for name, data in sorted(categories.items()):
        tested = data["tested"]
        resisted = data["resisted"]
        rate = (
            resisted / tested if tested > 0 else 1.0
        )
        cat_out[name] = {
            "tested": tested,
            "resisted": resisted,
            "rate": round(rate, 4),
        }

    report = {
        "target": target,
        "total_payloads": total_tested,
        "total_resisted": total_resisted,
        "resilience_rate": round(overall_rate, 4),
        "threshold": threshold,
        "passed": passed,
        "categories": cat_out,
    }
    return json.dumps(report, indent=2)


# -------------------------------------------------------------------
# CLI command registration
# -------------------------------------------------------------------

def register_security_scan(app: Any) -> None:
    """Register the ``security-scan`` command on *app*.

    Called by the orchestrator in ``app.py``.  Keeps
    this file self-contained and avoids circular imports.
    """
    import typer

    # Build the valid category names string for help text
    # from the canonical catalog's AttackCategory enum.
    _valid_cats = ", ".join(
        f"'{m.value}'" for m in AttackCategory
    )

    @app.command("security-scan")
    def security_scan(
        target: str = typer.Argument(
            ...,
            help=(
                "Python import path to model function "
                "(e.g. myapp.llm:chat_fn)"
            ),
        ),
        categories: list[str] | None = typer.Option(
            None,
            "--category",
            "-c",
            help=(
                "Attack categories to test (repeatable,"
                " snake_case). Options: "
                + _valid_cats
                + ". Default: all."
            ),
        ),
        threshold: float = typer.Option(
            0.8,
            "--threshold",
            "-t",
            help=(
                "Minimum resilience rate to pass "
                "(0.0-1.0, default: 0.8 = 80%%)"
            ),
        ),
        mutations: bool = typer.Option(
            False,
            "--mutations",
            "-m",
            help=(
                "Include encoding mutations "
                "(8 techniques: base64, rot13, "
                "leetspeak, homoglyphs, zero-width, "
                "mixed-case, markdown, html)"
            ),
        ),
        output: str = typer.Option(
            "table",
            "--output",
            "-o",
            help=(
                "Output format: 'table' "
                "(human-readable) or 'json'"
            ),
        ),
        verbose: bool = typer.Option(
            False,
            "--verbose",
            "-v",
            help=(
                "Show per-payload "
                "RESISTED/VULNERABLE results"
            ),
        ),
    ) -> None:
        """Red team security scan against a model.

        Dynamically imports the model function from
        TARGET (e.g. ``myapp.llm:chat_fn``), runs the
        canonical red team payload catalog against it,
        and reports per-category resilience rates.

        Exits with code 1 if the overall resilience
        rate falls below the threshold.
        """
        # -- Import model function -----------------------
        try:
            model_fn = _import_model_fn(target)
        except (
            ValueError, ImportError, AttributeError,
        ) as exc:
            print(  # noqa: T201
                f"Error loading model: {exc}"
            )
            raise typer.Exit(1) from exc

        # -- Resolve CLI category names to enums ---------
        selected_cats: list[AttackCategory] | None = None
        if categories:
            selected_cats = []
            for name in categories:
                key = name.lower().strip()
                cat_enum = _CLI_TO_CATEGORY.get(key)
                if cat_enum is None:
                    valid = sorted(
                        _CLI_TO_CATEGORY.keys()
                    )
                    print(  # noqa: T201
                        f"Unknown category: '{name}'. "
                        f"Valid: {valid}"
                    )
                    raise typer.Exit(1) from None
                selected_cats.append(cat_enum)

        # -- Build payload list from canonical catalog ---
        catalog = _flatten_catalog(selected_cats)

        if not catalog:
            valid = sorted(
                _CLI_TO_CATEGORY.keys()
            )
            print(  # noqa: T201
                "No payloads match the selected "
                f"categories. Valid: {valid}"
            )
            raise typer.Exit(0)

        # -- Apply mutations if requested ----------------
        if mutations:
            mutated = mutate_payloads(catalog)
            catalog = catalog + mutated

        # -- Run scan ------------------------------------
        results = _run_scan(
            model_fn, catalog, verbose=verbose,
        )

        # -- Format output --------------------------------
        if output == "json":
            text = _format_json(
                target, results, threshold,
            )
        else:
            text = _format_table(
                target,
                results,
                threshold,
                mutations_on=mutations,
                verbose=verbose,
            )

        print(text)  # noqa: T201

        # -- Exit code ------------------------------------
        total_tested = sum(
            c["tested"] for c in results.values()
        )
        total_resisted = sum(
            c["resisted"] for c in results.values()
        )
        rate = (
            total_resisted / total_tested
            if total_tested > 0 else 1.0
        )
        if rate < threshold:
            raise typer.Exit(1)
