"""Tests for mltk.domains.llm.mcp — MCP evaluation assertions."""

from __future__ import annotations

import sys
import types
from unittest.mock import patch

import pytest

from mltk.core.assertion import MltkAssertionError
from mltk.domains.llm.mcp import (
    McpResourceAccess,
    McpToolCall,
    McpTrace,
    _parse_server_tool,
    assert_mcp_context_window,
    assert_mcp_error_recovery,
    assert_mcp_resource_access,
    assert_mcp_tool_schema_conformance,
    assert_mcp_tool_selection,
)
from mltk.domains.llm.trace import AgentTrace, ToolCall

# ---------------------------------------------------------------
# jsonschema mock infrastructure
# ---------------------------------------------------------------

def _make_jsonschema_mock():
    """Build a mock jsonschema module with realistic errors."""
    mod = types.ModuleType("jsonschema")

    class _ValidationError(Exception):
        def __init__(self, message):
            self.message = message
            super().__init__(message)

    class _SchemaError(Exception):
        def __init__(self, message):
            self.message = message
            super().__init__(message)

    mod.ValidationError = _ValidationError
    mod.SchemaError = _SchemaError

    def _validate(instance, schema):
        """Minimal schema validator for tests."""
        s_type = schema.get("type")
        # Malformed schema type check
        valid_types = {
            "object", "array", "string",
            "integer", "number", "boolean", "null",
        }
        if s_type and s_type not in valid_types:
            raise _SchemaError(
                f"Unknown type: {s_type!r}"
            )
        if s_type == "object" and isinstance(instance, dict):
            props = schema.get("properties", {})
            required = schema.get("required", [])
            additional = schema.get(
                "additionalProperties", True,
            )
            for r in required:
                if r not in instance:
                    raise _ValidationError(
                        f"{r!r} is a required property"
                    )
            for key, val in instance.items():
                if key in props:
                    p_schema = props[key]
                    _validate(val, p_schema)
                elif additional is False:
                    raise _ValidationError(
                        f"Additional properties are "
                        f"not allowed ('{key}')"
                    )
        elif s_type == "string":
            if not isinstance(instance, str):
                raise _ValidationError(
                    f"{instance!r} is not of type 'string'"
                )
            min_len = schema.get("minLength")
            if min_len and len(instance) < min_len:
                raise _ValidationError(
                    f"{instance!r} is too short"
                )
        elif s_type == "integer":
            if not isinstance(instance, int):
                raise _ValidationError(
                    f"{instance!r} is not of type 'integer'"
                )
            mn = schema.get("minimum")
            mx = schema.get("maximum")
            if mn is not None and instance < mn:
                raise _ValidationError(
                    f"{instance} is less than {mn}"
                )
            if mx is not None and instance > mx:
                raise _ValidationError(
                    f"{instance} is greater than {mx}"
                )
        elif s_type == "array":
            if not isinstance(instance, list):
                raise _ValidationError(
                    f"{instance!r} is not of type 'array'"
                )
            items_schema = schema.get("items")
            if items_schema:
                for item in instance:
                    _validate(item, items_schema)
        elif s_type == "boolean":
            if not isinstance(instance, bool):
                raise _ValidationError(
                    f"{instance!r} is not of type 'boolean'"
                )

    mod.validate = _validate
    return mod


@pytest.fixture(autouse=True)
def _mock_jsonschema():
    """Inject mock jsonschema for all tests."""
    mock_mod = _make_jsonschema_mock()
    with patch.dict(sys.modules, {"jsonschema": mock_mod}):
        yield mock_mod


# ---------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------


@pytest.fixture
def sample_schema():
    """JSON Schema for a search tool."""
    return {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "minLength": 1,
            },
            "limit": {
                "type": "integer",
                "minimum": 1,
                "maximum": 100,
            },
        },
        "required": ["query"],
    }


@pytest.fixture
def strict_schema():
    """Schema that forbids extra properties."""
    return {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
        },
        "required": ["path"],
        "additionalProperties": False,
    }


@pytest.fixture
def nested_schema():
    """Schema with nested object."""
    return {
        "type": "object",
        "properties": {
            "filters": {
                "type": "object",
                "properties": {
                    "date": {"type": "string"},
                    "count": {"type": "integer"},
                },
                "required": ["date"],
            },
        },
        "required": ["filters"],
    }


@pytest.fixture
def array_schema():
    """Schema with array parameter."""
    return {
        "type": "object",
        "properties": {
            "tags": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "required": ["tags"],
    }


@pytest.fixture
def sample_mcp_trace():
    """Trace with two tool calls and one resource."""
    return McpTrace(
        tool_calls=[
            McpToolCall(
                name="search",
                server="docs",
                arguments={"query": "test"},
                schema={},
            ),
            McpToolCall(
                name="read_file",
                server="fs",
                arguments={"path": "/readme.md"},
            ),
        ],
        resource_accesses=[
            McpResourceAccess(
                uri="file:///readme.md",
                server="fs",
                content_tokens=500,
            ),
        ],
        total_tokens=5000,
        model_context_limit=100000,
    )


# ===============================================================
# Schema Conformance (12 tests)
# ===============================================================


class TestSchemaConformance:
    """JSON Schema conformance for MCP tool args."""

    def test_schema_valid_args(
        self, sample_schema
    ) -> None:
        # SCENARIO: Args match the schema exactly.
        # WHY: Correct args must pass validation.
        # EXPECTED: passes.
        result = assert_mcp_tool_schema_conformance(
            sample_schema,
            {"query": "hello", "limit": 10},
            tool_name="search",
        )
        assert result.passed is True
        assert result.details["errors"] == []

    def test_schema_wrong_type(
        self, sample_schema
    ) -> None:
        # SCENARIO: limit is string instead of int.
        # WHY: Type mismatch is a schema violation.
        # EXPECTED: raises MltkAssertionError.
        with pytest.raises(MltkAssertionError) as exc:
            assert_mcp_tool_schema_conformance(
                sample_schema,
                {"query": "hello", "limit": "ten"},
            )
        r = exc.value.result
        assert len(r.details["errors"]) > 0

    def test_schema_missing_required(
        self, sample_schema
    ) -> None:
        # SCENARIO: Required field "query" is omitted.
        # WHY: Missing required field is a violation.
        # EXPECTED: raises MltkAssertionError.
        with pytest.raises(MltkAssertionError) as exc:
            assert_mcp_tool_schema_conformance(
                sample_schema,
                {"limit": 5},
            )
        r = exc.value.result
        assert len(r.details["errors"]) > 0

    def test_schema_extra_field_forbidden(
        self, strict_schema
    ) -> None:
        # SCENARIO: Extra field with
        # additionalProperties=false.
        # WHY: Strict schemas reject unknown keys.
        # EXPECTED: raises MltkAssertionError.
        with pytest.raises(MltkAssertionError):
            assert_mcp_tool_schema_conformance(
                strict_schema,
                {"path": "/a.txt", "verbose": True},
            )

    def test_schema_extra_field_allowed(
        self, sample_schema
    ) -> None:
        # SCENARIO: Extra field when
        # additionalProperties not set.
        # WHY: Permissive schemas accept extra keys.
        # EXPECTED: passes.
        result = assert_mcp_tool_schema_conformance(
            sample_schema,
            {"query": "hi", "extra_key": 42},
        )
        assert result.passed is True

    def test_schema_value_out_of_range(
        self, sample_schema
    ) -> None:
        # SCENARIO: limit=999 exceeds maximum=100.
        # WHY: Range constraints must be enforced.
        # EXPECTED: raises MltkAssertionError.
        with pytest.raises(MltkAssertionError):
            assert_mcp_tool_schema_conformance(
                sample_schema,
                {"query": "hi", "limit": 999},
            )

    def test_schema_nested_object(
        self, nested_schema
    ) -> None:
        # SCENARIO: Nested object with valid fields.
        # WHY: Deep validation traverses objects.
        # EXPECTED: passes.
        result = assert_mcp_tool_schema_conformance(
            nested_schema,
            {
                "filters": {
                    "date": "2025-01-01",
                    "count": 5,
                },
            },
        )
        assert result.passed is True

    def test_schema_empty_args_with_required(
        self, sample_schema
    ) -> None:
        # SCENARIO: Empty dict when "query" required.
        # WHY: Empty args violate required constraint.
        # EXPECTED: raises MltkAssertionError.
        with pytest.raises(MltkAssertionError):
            assert_mcp_tool_schema_conformance(
                sample_schema, {},
            )

    def test_schema_empty_args_no_required(
        self,
    ) -> None:
        # SCENARIO: Empty dict with no required fields.
        # WHY: No constraints means empty is valid.
        # EXPECTED: passes.
        schema = {
            "type": "object",
            "properties": {
                "verbose": {"type": "boolean"},
            },
        }
        result = assert_mcp_tool_schema_conformance(
            schema, {},
        )
        assert result.passed is True

    def test_schema_import_error(self) -> None:
        # SCENARIO: jsonschema is not installed.
        # WHY: Graceful error with clear message.
        # EXPECTED: raises MltkAssertionError with
        # import_error detail.
        with patch.dict(
            sys.modules, {"jsonschema": None}
        ):
            with pytest.raises(
                MltkAssertionError
            ) as exc:
                assert_mcp_tool_schema_conformance(
                    {"type": "object"}, {},
                )
            r = exc.value.result
            assert r.details["import_error"] is True
            assert "jsonschema" in r.message

    def test_schema_malformed_schema(self) -> None:
        # SCENARIO: Schema type is invalid.
        # WHY: Must not crash; report graceful error.
        # EXPECTED: raises MltkAssertionError with
        # errors mentioning malformed schema.
        bad_schema = {"type": "not_a_real_type"}
        with pytest.raises(MltkAssertionError) as exc:
            assert_mcp_tool_schema_conformance(
                bad_schema, {"x": 1},
            )
        r = exc.value.result
        assert len(r.details["errors"]) > 0
        assert "malformed" in r.details["errors"][0].lower()

    def test_schema_array_type(
        self, array_schema
    ) -> None:
        # SCENARIO: Array parameter with valid items.
        # WHY: Array types must validate item schemas.
        # EXPECTED: passes.
        result = assert_mcp_tool_schema_conformance(
            array_schema,
            {"tags": ["a", "b", "c"]},
        )
        assert result.passed is True

    def test_schema_array_wrong_item_type(
        self, array_schema
    ) -> None:
        # SCENARIO: Array items are ints, not strings.
        # WHY: Item-level type validation must catch
        # wrong element types within arrays.
        # EXPECTED: raises MltkAssertionError.
        with pytest.raises(MltkAssertionError) as exc:
            assert_mcp_tool_schema_conformance(
                array_schema,
                {"tags": [1, 2, 3]},
            )
        r = exc.value.result
        assert len(r.details["errors"]) > 0


# ===============================================================
# Tool Selection (8 tests)
# ===============================================================


class TestToolSelection:
    """MCP tool selection with server namespace."""

    def test_selection_exact_match(
        self, sample_mcp_trace
    ) -> None:
        # SCENARIO: All expected tools used, no extras.
        # WHY: Perfect selection must pass.
        # EXPECTED: precision=recall=1.0.
        result = assert_mcp_tool_selection(
            sample_mcp_trace,
            ["docs::search", "fs::read_file"],
        )
        assert result.passed is True
        assert result.details["precision"] == 1.0
        assert result.details["recall"] == 1.0

    def test_selection_missing_tool(
        self, sample_mcp_trace
    ) -> None:
        # SCENARIO: Expected a tool not called.
        # WHY: Missing tools = incomplete work.
        # EXPECTED: raises MltkAssertionError.
        with pytest.raises(MltkAssertionError) as exc:
            assert_mcp_tool_selection(
                sample_mcp_trace,
                [
                    "docs::search",
                    "fs::read_file",
                    "db::query",
                ],
            )
        r = exc.value.result
        missing = r.details["missing_tools"]
        assert "db::query" in missing

    def test_selection_extra_tool(
        self, sample_mcp_trace
    ) -> None:
        # SCENARIO: Trace has more tools than expected.
        # WHY: Extra tools reduce precision.
        # EXPECTED: raises MltkAssertionError.
        with pytest.raises(MltkAssertionError) as exc:
            assert_mcp_tool_selection(
                sample_mcp_trace,
                ["docs::search"],
            )
        r = exc.value.result
        assert r.details["precision"] < 1.0

    def test_selection_server_namespace(self) -> None:
        # SCENARIO: "read" on server "fs" matches
        # expected "fs::read".
        # WHY: Namespace routing must be correct.
        # EXPECTED: passes.
        trace = McpTrace(
            tool_calls=[
                McpToolCall(
                    name="read",
                    server="fs",
                    arguments={},
                ),
            ],
        )
        result = assert_mcp_tool_selection(
            trace, ["fs::read"],
        )
        assert result.passed is True

    def test_selection_wrong_server(self) -> None:
        # SCENARIO: "read" on server "db" != "fs::read".
        # WHY: Cross-server confusion is a routing bug.
        # EXPECTED: raises MltkAssertionError.
        trace = McpTrace(
            tool_calls=[
                McpToolCall(
                    name="read",
                    server="db",
                    arguments={},
                ),
            ],
        )
        with pytest.raises(MltkAssertionError) as exc:
            assert_mcp_tool_selection(
                trace, ["fs::read"],
            )
        r = exc.value.result
        assert "fs::read" in r.details["missing_tools"]

    def test_selection_no_namespace(self) -> None:
        # SCENARIO: Plain tool names without prefix.
        # WHY: Tools without namespace use bare names.
        # EXPECTED: passes.
        trace = McpTrace(
            tool_calls=[
                McpToolCall(
                    name="search",
                    arguments={"q": "test"},
                ),
            ],
        )
        result = assert_mcp_tool_selection(
            trace, ["search"],
        )
        assert result.passed is True

    def test_selection_empty_trace(self) -> None:
        # SCENARIO: No tool calls in trace.
        # WHY: Empty trace with expected = failure.
        # EXPECTED: raises MltkAssertionError.
        trace = McpTrace(tool_calls=[])
        with pytest.raises(MltkAssertionError):
            assert_mcp_tool_selection(
                trace, ["search"],
            )

    def test_selection_server_filter(self) -> None:
        # SCENARIO: Filter by server="docs" only.
        # WHY: server param narrows scope.
        # EXPECTED: passes matching only docs tools.
        trace = McpTrace(
            tool_calls=[
                McpToolCall(
                    name="search",
                    server="docs",
                    arguments={},
                ),
                McpToolCall(
                    name="read",
                    server="fs",
                    arguments={},
                ),
            ],
        )
        result = assert_mcp_tool_selection(
            trace,
            ["docs::search"],
            server="docs",
        )
        assert result.passed is True


# ===============================================================
# Resource Access (8 tests)
# ===============================================================


class TestResourceAccess:
    """MCP resource URI access patterns."""

    def test_resource_all_expected_accessed(
        self, sample_mcp_trace
    ) -> None:
        # SCENARIO: All required URIs were accessed.
        # WHY: Complete coverage of required resources.
        # EXPECTED: passes.
        result = assert_mcp_resource_access(
            sample_mcp_trace,
            expected_uris=["file:///readme.md"],
        )
        assert result.passed is True

    def test_resource_missing_expected(
        self, sample_mcp_trace
    ) -> None:
        # SCENARIO: Required URI was not accessed.
        # WHY: Missing resources = incomplete work.
        # EXPECTED: raises MltkAssertionError.
        with pytest.raises(MltkAssertionError) as exc:
            assert_mcp_resource_access(
                sample_mcp_trace,
                expected_uris=[
                    "file:///readme.md",
                    "file:///config.json",
                ],
            )
        r = exc.value.result
        assert len(r.details["errors"]) > 0

    def test_resource_forbidden_accessed(self) -> None:
        # SCENARIO: A forbidden URI was accessed.
        # WHY: Forbidden access = security violation.
        # EXPECTED: raises MltkAssertionError.
        trace = McpTrace(
            resource_accesses=[
                McpResourceAccess(
                    uri="file:///etc/passwd",
                ),
            ],
        )
        with pytest.raises(MltkAssertionError) as exc:
            assert_mcp_resource_access(
                trace,
                forbidden_uris=[
                    "file:///etc/passwd",
                ],
            )
        r = exc.value.result
        assert "forbidden" in r.details["errors"][0]

    def test_resource_no_forbidden(
        self, sample_mcp_trace
    ) -> None:
        # SCENARIO: No forbidden URIs accessed, all
        # expected present.
        # WHY: Clean access pattern should pass.
        # EXPECTED: passes.
        result = assert_mcp_resource_access(
            sample_mcp_trace,
            expected_uris=["file:///readme.md"],
            forbidden_uris=["file:///etc/shadow"],
        )
        assert result.passed is True

    def test_resource_max_reads_ok(self) -> None:
        # SCENARIO: Under the max reads limit.
        # WHY: Reasonable fetch count should pass.
        # EXPECTED: passes.
        trace = McpTrace(
            resource_accesses=[
                McpResourceAccess(
                    uri="file:///a.txt",
                ),
                McpResourceAccess(
                    uri="file:///b.txt",
                ),
            ],
        )
        result = assert_mcp_resource_access(
            trace, max_reads=5,
        )
        assert result.passed is True

    def test_resource_max_reads_exceeded(self) -> None:
        # SCENARIO: More reads than max_reads allows.
        # WHY: Over-fetching wastes context budget.
        # EXPECTED: raises MltkAssertionError.
        accesses = [
            McpResourceAccess(
                uri=f"file:///{i}.txt",
            )
            for i in range(10)
        ]
        trace = McpTrace(resource_accesses=accesses)
        with pytest.raises(MltkAssertionError):
            assert_mcp_resource_access(
                trace, max_reads=3,
            )

    def test_resource_empty_trace(self) -> None:
        # SCENARIO: No resource accesses, no constraints.
        # WHY: At least one constraint required.
        # EXPECTED: raises ValueError.
        trace = McpTrace(resource_accesses=[])
        with pytest.raises(
            ValueError, match="At least one constraint"
        ):
            assert_mcp_resource_access(trace)

    def test_resource_uri_patterns(self) -> None:
        # SCENARIO: Various URI formats.
        # WHY: URIs compared as strings; format agnostic.
        # EXPECTED: passes when all expected present.
        trace = McpTrace(
            resource_accesses=[
                McpResourceAccess(
                    uri="file:///workspace/data.csv",
                ),
                McpResourceAccess(
                    uri="https://api.example.com/v1",
                ),
                McpResourceAccess(
                    uri="db://main/users",
                ),
            ],
        )
        result = assert_mcp_resource_access(
            trace,
            expected_uris=[
                "file:///workspace/data.csv",
                "db://main/users",
            ],
        )
        assert result.passed is True


# ===============================================================
# Context Window (6 tests)
# ===============================================================


class TestContextWindow:
    """MCP context window utilization checks."""

    def test_context_under_limit(self) -> None:
        # SCENARIO: 50% utilization (5K / 10K).
        # WHY: Well under limit should pass easily.
        # EXPECTED: passes.
        trace = McpTrace(
            total_tokens=5000,
            model_context_limit=10000,
        )
        result = assert_mcp_context_window(
            trace, max_utilization=0.9,
        )
        assert result.passed is True
        assert result.details["utilization"] == 0.5

    def test_context_over_limit(self) -> None:
        # SCENARIO: 95% utilization exceeds max 90%.
        # WHY: Near-full context risks truncation.
        # EXPECTED: raises MltkAssertionError.
        trace = McpTrace(
            total_tokens=9500,
            model_context_limit=10000,
        )
        with pytest.raises(MltkAssertionError) as exc:
            assert_mcp_context_window(
                trace, max_utilization=0.9,
            )
        r = exc.value.result
        assert r.details["utilization"] == 0.95

    def test_context_at_boundary(self) -> None:
        # SCENARIO: Exactly 90% with max=0.9.
        # WHY: Boundary should pass (<=).
        # EXPECTED: passes.
        trace = McpTrace(
            total_tokens=9000,
            model_context_limit=10000,
        )
        result = assert_mcp_context_window(
            trace, max_utilization=0.9,
        )
        assert result.passed is True

    def test_context_zero_tokens(self) -> None:
        # SCENARIO: Zero tokens used.
        # WHY: Empty trace is valid, 0% utilization.
        # EXPECTED: passes.
        trace = McpTrace(
            total_tokens=0,
            model_context_limit=100000,
        )
        result = assert_mcp_context_window(trace)
        assert result.passed is True
        u = result.details["utilization"]
        assert u == 0.0

    def test_context_custom_limit(self) -> None:
        # SCENARIO: Override limit via param.
        # WHY: Explicit param takes precedence.
        # EXPECTED: passes with explicit limit.
        trace = McpTrace(
            total_tokens=50000,
            model_context_limit=200000,
        )
        result = assert_mcp_context_window(
            trace,
            model_context_limit=100000,
            max_utilization=0.9,
        )
        assert result.passed is True
        assert result.details["utilization"] == 0.5

    def test_context_from_trace(
        self, sample_mcp_trace
    ) -> None:
        # SCENARIO: Use trace.model_context_limit (100K).
        # WHY: Trace limit is the default source.
        # EXPECTED: passes (5K / 100K = 5%).
        result = assert_mcp_context_window(
            sample_mcp_trace,
        )
        assert result.passed is True
        u = result.details["utilization"]
        assert abs(u - 0.05) < 1e-9

    def test_context_zero_limit_raises(self) -> None:
        # SCENARIO: model_context_limit=0 is invalid.
        # WHY: Zero limit causes division by zero;
        # must raise ValueError with clear message.
        # EXPECTED: raises ValueError.
        trace = McpTrace(
            total_tokens=100,
            model_context_limit=0,
        )
        with pytest.raises(
            ValueError, match="must be > 0"
        ):
            assert_mcp_context_window(trace)

    def test_context_no_limit_anywhere_raises(
        self,
    ) -> None:
        # SCENARIO: No limit in param or trace.
        # WHY: Cannot compute utilization without a
        # denominator.
        # EXPECTED: raises ValueError.
        trace = McpTrace(total_tokens=5000)
        with pytest.raises(
            ValueError, match="must be > 0"
        ):
            assert_mcp_context_window(trace)


# ===============================================================
# Error Recovery (6 tests)
# ===============================================================


class TestErrorRecovery:
    """MCP error recovery — same-tool retry detection."""

    def test_recovery_no_errors(self) -> None:
        # SCENARIO: All tool calls succeed.
        # WHY: Clean trace needs no recovery.
        # EXPECTED: passes.
        trace = McpTrace(
            tool_calls=[
                McpToolCall(
                    name="search",
                    arguments={"q": "a"},
                    result="ok",
                ),
                McpToolCall(
                    name="read",
                    arguments={"p": "/x"},
                    result="ok",
                ),
            ],
        )
        result = assert_mcp_error_recovery(trace)
        assert result.passed is True

    def test_recovery_single_error(self) -> None:
        # SCENARIO: One error then success on same tool.
        # WHY: Single retry is normal behavior.
        # EXPECTED: passes.
        trace = McpTrace(
            tool_calls=[
                McpToolCall(
                    name="search",
                    arguments={"q": "a"},
                    error="timeout",
                ),
                McpToolCall(
                    name="search",
                    arguments={"q": "a"},
                    result="ok",
                ),
            ],
        )
        result = assert_mcp_error_recovery(trace)
        assert result.passed is True

    def test_recovery_retry_loop(self) -> None:
        # SCENARIO: 4x same tool+args error (max=3).
        # WHY: Stuck retry loop wastes resources.
        # EXPECTED: raises MltkAssertionError.
        calls = [
            McpToolCall(
                name="fetch",
                arguments={"url": "http://x"},
                error="500 error",
            )
            for _ in range(4)
        ]
        trace = McpTrace(tool_calls=calls)
        with pytest.raises(MltkAssertionError) as exc:
            assert_mcp_error_recovery(trace)
        r = exc.value.result
        assert r.details["violation_count"] > 0

    def test_recovery_different_tools(self) -> None:
        # SCENARIO: Errors on different tools.
        # WHY: Diverse errors = agent is adapting.
        # EXPECTED: passes.
        trace = McpTrace(
            tool_calls=[
                McpToolCall(
                    name="search",
                    arguments={"q": "a"},
                    error="timeout",
                ),
                McpToolCall(
                    name="fetch",
                    arguments={"url": "/b"},
                    error="404",
                ),
                McpToolCall(
                    name="read",
                    arguments={"p": "/c"},
                    error="perm denied",
                ),
            ],
        )
        result = assert_mcp_error_recovery(trace)
        assert result.passed is True

    def test_recovery_same_tool_different_args(
        self,
    ) -> None:
        # SCENARIO: Same tool with different args.
        # WHY: Changing args = agent is adapting.
        # EXPECTED: passes (not a blind retry loop).
        trace = McpTrace(
            tool_calls=[
                McpToolCall(
                    name="search",
                    arguments={"q": "alpha"},
                    error="no results",
                ),
                McpToolCall(
                    name="search",
                    arguments={"q": "beta"},
                    error="no results",
                ),
                McpToolCall(
                    name="search",
                    arguments={"q": "gamma"},
                    error="no results",
                ),
                McpToolCall(
                    name="search",
                    arguments={"q": "delta"},
                    error="no results",
                ),
            ],
        )
        result = assert_mcp_error_recovery(trace)
        assert result.passed is True

    def test_recovery_max_retries_custom(self) -> None:
        # SCENARIO: Custom max_same_tool_retries=1.
        # WHY: Strict limit catches 2 identical retries.
        # EXPECTED: raises MltkAssertionError.
        trace = McpTrace(
            tool_calls=[
                McpToolCall(
                    name="api_call",
                    arguments={"id": 1},
                    error="rate limit",
                ),
                McpToolCall(
                    name="api_call",
                    arguments={"id": 1},
                    error="rate limit",
                ),
            ],
        )
        with pytest.raises(MltkAssertionError):
            assert_mcp_error_recovery(
                trace, max_same_tool_retries=1,
            )

    def test_recovery_empty_trace(self) -> None:
        # SCENARIO: No tool calls at all.
        # WHY: Empty trace has no errors to detect.
        # EXPECTED: passes with max_retries_seen=0.
        trace = McpTrace(tool_calls=[])
        result = assert_mcp_error_recovery(trace)
        assert result.passed is True
        assert result.details["max_retries_seen"] == 0
        assert result.details["retry_loops"] == []

    def test_recovery_error_then_success_different_tool(
        self,
    ) -> None:
        # SCENARIO: Error on tool A, then success on
        # tool B (different tool).
        # WHY: Switching tools after error = good
        # recovery strategy.
        # EXPECTED: passes.
        trace = McpTrace(
            tool_calls=[
                McpToolCall(
                    name="read_file",
                    arguments={"path": "/missing"},
                    error="File not found",
                ),
                McpToolCall(
                    name="search",
                    arguments={"q": "alternative"},
                    result="found it",
                ),
            ],
        )
        result = assert_mcp_error_recovery(trace)
        assert result.passed is True


# ===============================================================
# Helper / Dataclass (9 tests)
# ===============================================================


class TestHelpers:
    """Helper functions and dataclass properties."""

    def test_parse_server_tool_with_namespace(
        self,
    ) -> None:
        # SCENARIO: "fs::read" -> ("fs", "read").
        # WHY: Namespace parsing is foundational.
        # EXPECTED: correct tuple.
        assert _parse_server_tool("fs::read") == (
            "fs", "read",
        )

    def test_parse_server_tool_no_namespace(
        self,
    ) -> None:
        # SCENARIO: "search" has no namespace.
        # WHY: Bare names return empty server.
        # EXPECTED: ("", "search").
        assert _parse_server_tool("search") == (
            "", "search",
        )

    def test_parse_server_tool_double_colon(
        self,
    ) -> None:
        # SCENARIO: "a::b::c" splits on first "::".
        # WHY: Tool names might contain "::" in the
        # name portion (e.g., nested namespaces).
        # EXPECTED: ("a", "b::c") -- split("::", 1).
        assert _parse_server_tool("a::b::c") == (
            "a", "b::c",
        )

    def test_mcp_trace_extends_agent_trace(
        self, sample_mcp_trace
    ) -> None:
        # SCENARIO: McpTrace is AgentTrace subclass.
        # WHY: Backward compatibility requirement.
        # EXPECTED: isinstance check passes.
        assert isinstance(
            sample_mcp_trace, AgentTrace,
        )

    def test_mcp_tool_call_extends_tool_call(
        self,
    ) -> None:
        # SCENARIO: McpToolCall is ToolCall subclass.
        # WHY: Must work with existing assertions.
        # EXPECTED: isinstance check passes.
        tc = McpToolCall(
            name="read",
            server="fs",
            arguments={},
        )
        assert isinstance(tc, ToolCall)

    def test_mcp_trace_backward_compat(
        self, sample_mcp_trace
    ) -> None:
        # SCENARIO: AgentTrace properties work.
        # WHY: step_count and failed_calls must work.
        # EXPECTED: step_count=2, no failed calls.
        assert sample_mcp_trace.step_count == 2
        assert sample_mcp_trace.failed_calls == []

    def test_mcp_trace_tool_names_with_server(
        self, sample_mcp_trace
    ) -> None:
        # SCENARIO: tool_names includes server prefix.
        # WHY: MCP tools are namespaced by server.
        # EXPECTED: ["docs::search", "fs::read_file"].
        names = sample_mcp_trace.tool_names
        assert names == [
            "docs::search", "fs::read_file",
        ]

    def test_mcp_trace_tool_names_mixed(self) -> None:
        # SCENARIO: Mix of McpToolCall and ToolCall.
        # WHY: Trace can contain both types.
        # EXPECTED: server prefix only when present.
        trace = McpTrace(
            tool_calls=[
                McpToolCall(
                    name="search",
                    server="docs",
                    arguments={},
                ),
                ToolCall(
                    name="calculate",
                    arguments={},
                ),
            ],
        )
        names = trace.tool_names
        assert names == [
            "docs::search", "calculate",
        ]

    def test_mcp_resource_access_defaults(
        self,
    ) -> None:
        # SCENARIO: McpResourceAccess with defaults.
        # WHY: Dataclass defaults must be sensible.
        # EXPECTED: All optional fields have defaults.
        ra = McpResourceAccess(uri="file:///x.txt")
        assert ra.uri == "file:///x.txt"
        assert ra.server == ""
        assert ra.content_tokens == 0
        assert ra.result is None
        assert ra.error is None
        assert ra.duration_ms == 0.0

    def test_mcp_tool_call_server_default(
        self,
    ) -> None:
        # SCENARIO: McpToolCall defaults.
        # WHY: Server default is empty string.
        # EXPECTED: server="" and schema={}.
        tc = McpToolCall(
            name="search", arguments={},
        )
        assert tc.server == ""
        assert tc.schema == {}
