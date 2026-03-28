"""Tests for mltk.domains.codegen -- code generation evaluation assertions.

LLM-generated code looks plausible but frequently crashes, fails tests,
contains security vulnerabilities, or is unnecessarily complex.  These
tests validate four assertions that catch those failures:

1. Execution: does generated code actually run?
2. Test passing: does generated code satisfy a test suite?
3. Vulnerability scanning: does generated code contain dangerous patterns?
4. Complexity: is generated code maintainable?

Subprocess calls are mocked to avoid executing arbitrary code in CI and
to keep tests fast and deterministic.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from mltk.core.assertion import MltkAssertionError
from mltk.domains.codegen import (
    assert_code_complexity,
    assert_code_executes,
    assert_code_passes_tests,
    assert_no_code_vulnerabilities,
)

# ------------------------------------------------------------------
# assert_code_executes
# ------------------------------------------------------------------


class TestCodeExecutes:
    """Execution tests -- does generated code run without errors?

    LLMs produce code with undefined variables, missing imports, and
    type mismatches.  These tests verify that the assertion correctly
    distinguishes runnable code from broken code.
    """

    @patch("mltk.domains.codegen._run_code_in_subprocess")
    def test_valid_python_passes(self, mock_run: MagicMock) -> None:
        """PASS: Simple valid Python code executes successfully.

        WHY: The most basic check -- ``x = 1 + 2; print(x)`` must
        produce returncode=0.  If this fails, the LLM output is
        fundamentally broken.
        Expected: passed=True, returncode=0.
        """
        mock_run.return_value = (0, "3\n", "")
        code = "x = 1 + 2\nprint(x)"
        result = assert_code_executes(code)
        assert result.passed is True
        assert result.details["returncode"] == 0
        assert result.details["language"] == "python"
        assert result.name == "codegen.executes"

    @patch("mltk.domains.codegen._run_code_in_subprocess")
    def test_syntax_error_fails(self, mock_run: MagicMock) -> None:
        """FAIL: Code with a syntax error crashes on execution.

        WHY: ``def foo(`` is incomplete syntax.  Python exits with
        returncode=1 and a SyntaxError traceback.  The assertion
        must catch this and report the error.
        Expected: MltkAssertionError raised.
        """
        mock_run.return_value = (
            1,
            "",
            "  File \"tmp.py\", line 1\n"
            "    def foo(\n"
            "SyntaxError: unexpected EOF",
        )
        with pytest.raises(MltkAssertionError) as exc:
            assert_code_executes("def foo(")
        result = exc.value.result
        assert result.details["returncode"] == 1
        assert "SyntaxError" in result.details["stderr"]

    @patch("mltk.domains.codegen._run_code_in_subprocess")
    def test_timeout_fails(self, mock_run: MagicMock) -> None:
        """FAIL: Infinite loop code times out.

        WHY: LLMs sometimes generate infinite loops (``while True:
        pass``).  The subprocess must be killed after the timeout,
        and the assertion must report the timeout clearly.
        Expected: MltkAssertionError raised, returncode=-1.
        """
        mock_run.return_value = (
            -1, "", "Timed out after 5.0s"
        )
        with pytest.raises(MltkAssertionError) as exc:
            assert_code_executes(
                "while True: pass", timeout_seconds=5.0
            )
        result = exc.value.result
        assert result.details["returncode"] == -1
        assert "timed out" in result.message.lower()

    def test_empty_code_passes(self) -> None:
        """PASS: Empty code executes trivially.

        WHY: An empty string is valid Python (does nothing).
        The assertion should not crash on empty input.
        Expected: passed=True without invoking subprocess.
        """
        result = assert_code_executes("")
        assert result.passed is True
        assert result.details["returncode"] == 0

    def test_whitespace_only_passes(self) -> None:
        """PASS: Whitespace-only code executes trivially.

        WHY: Code that is only spaces/newlines should be treated
        the same as empty code.
        Expected: passed=True.
        """
        result = assert_code_executes("   \n\n  ")
        assert result.passed is True

    def test_unsupported_language_fails(self) -> None:
        """FAIL: Non-Python language is not supported.

        WHY: The module currently only supports Python execution.
        Attempting to run JavaScript should fail with a clear
        message, not silently do something wrong.
        Expected: MltkAssertionError raised.
        """
        with pytest.raises(MltkAssertionError) as exc:
            assert_code_executes(
                "console.log('hi')", language="javascript"
            )
        result = exc.value.result
        assert "Unsupported language" in result.message
        assert result.details["language"] == "javascript"

    @patch("mltk.domains.codegen._run_code_in_subprocess")
    def test_details_include_stdout(
        self, mock_run: MagicMock
    ) -> None:
        """PASS: stdout is captured in details.

        WHY: When debugging why generated code behaves
        unexpectedly, engineers need to see what it printed.
        Expected: stdout in details matches subprocess output.
        """
        mock_run.return_value = (0, "hello world\n", "")
        result = assert_code_executes("print('hello world')")
        assert result.passed is True
        assert "hello world" in result.details["stdout"]


# ------------------------------------------------------------------
# assert_code_passes_tests
# ------------------------------------------------------------------


class TestCodePassesTests:
    """Test passing -- does generated code satisfy test cases?

    Code that runs is not necessarily correct.  These tests verify
    that the assertion correctly detects when generated code passes
    or fails its test suite.
    """

    @patch("mltk.domains.codegen._run_code_in_subprocess")
    def test_correct_code_passes(
        self, mock_run: MagicMock
    ) -> None:
        """PASS: Code + passing tests produce returncode=0.

        WHY: ``def add(a, b): return a + b`` with
        ``assert add(2, 3) == 5`` is correct.  The assertion
        must recognize this as a pass.
        Expected: passed=True.
        """
        mock_run.return_value = (0, "", "")
        code = "def add(a, b): return a + b"
        tests = "assert add(2, 3) == 5\nassert add(0, 0) == 0"
        result = assert_code_passes_tests(code, tests)
        assert result.passed is True
        assert result.name == "codegen.passes_tests"

    @patch("mltk.domains.codegen._run_code_in_subprocess")
    def test_wrong_code_fails(
        self, mock_run: MagicMock
    ) -> None:
        """FAIL: Code that returns wrong results fails tests.

        WHY: ``def add(a, b): return a - b`` is a common LLM
        mistake (wrong operator).  The tests catch this.
        Expected: MltkAssertionError raised.
        """
        mock_run.return_value = (
            1,
            "",
            "Traceback (most recent call last):\n"
            "  File \"tmp.py\", line 3\n"
            "AssertionError",
        )
        code = "def add(a, b): return a - b"
        tests = "assert add(2, 3) == 5"
        with pytest.raises(MltkAssertionError) as exc:
            assert_code_passes_tests(code, tests)
        result = exc.value.result
        assert result.details["returncode"] == 1

    @patch("mltk.domains.codegen._run_code_in_subprocess")
    def test_timeout_fails(
        self, mock_run: MagicMock
    ) -> None:
        """FAIL: Tests that hang are killed by timeout.

        WHY: Generated code with infinite recursion or loops
        causes test execution to hang.  The timeout prevents
        CI from blocking forever.
        Expected: MltkAssertionError raised.
        """
        mock_run.return_value = (
            -1, "", "Timed out after 30.0s"
        )
        with pytest.raises(MltkAssertionError):
            assert_code_passes_tests(
                "def f(): f()",
                "f()",
                timeout_seconds=30.0,
            )

    def test_empty_code_and_tests_passes(self) -> None:
        """PASS: Empty code and empty tests -- trivially passing.

        WHY: Edge case -- nothing to test means nothing to fail.
        Expected: passed=True.
        """
        result = assert_code_passes_tests("", "")
        assert result.passed is True


# ------------------------------------------------------------------
# assert_no_code_vulnerabilities
# ------------------------------------------------------------------


class TestNoCodeVulnerabilities:
    """Vulnerability scanning -- does generated code contain
    dangerous patterns?

    LLMs freely generate eval(), exec(), shell=True, and
    hardcoded passwords.  These tests verify that the scanner
    detects each pattern correctly.
    """

    def test_clean_code_passes(self) -> None:
        """PASS: Code with no dangerous patterns.

        WHY: ``def add(a, b): return a + b`` is safe.  The
        scanner should find zero vulnerabilities.
        Expected: passed=True, n_findings=0.
        """
        code = "def add(a, b):\n    return a + b\n"
        result = assert_no_code_vulnerabilities(code)
        assert result.passed is True
        assert result.details["n_findings"] == 0
        assert result.name == "codegen.no_vulnerabilities"

    def test_eval_detected(self) -> None:
        """FAIL: eval() call detected.

        WHY: ``eval(user_input)`` is the most dangerous pattern --
        arbitrary code execution.  LLMs generate it casually for
        "flexible" input parsing.
        Expected: MltkAssertionError, finding with rule="eval(".
        """
        code = "result = eval(user_input)\n"
        with pytest.raises(MltkAssertionError) as exc:
            assert_no_code_vulnerabilities(code)
        result = exc.value.result
        assert result.details["n_findings"] >= 1
        rules_found = {
            f["rule"] for f in result.details["vulnerabilities"]
        }
        assert "eval(" in rules_found

    def test_exec_detected(self) -> None:
        """FAIL: exec() call detected.

        WHY: ``exec(code_string)`` is arbitrary code execution.
        LLMs use it for "dynamic code generation" which is a
        security disaster in production.
        Expected: MltkAssertionError, finding with rule="exec(".
        """
        code = "exec(code_string)\n"
        with pytest.raises(MltkAssertionError) as exc:
            assert_no_code_vulnerabilities(code)
        result = exc.value.result
        rules_found = {
            f["rule"] for f in result.details["vulnerabilities"]
        }
        assert "exec(" in rules_found

    def test_shell_true_detected(self) -> None:
        """FAIL: subprocess with shell=True detected.

        WHY: ``subprocess.run(cmd, shell=True)`` is command
        injection.  If ``cmd`` contains user input, an attacker
        can run arbitrary shell commands.
        Expected: MltkAssertionError, finding with rule="shell=True".
        """
        code = (
            "import subprocess\n"
            "subprocess.run(cmd, shell=True)\n"
        )
        with pytest.raises(MltkAssertionError) as exc:
            assert_no_code_vulnerabilities(code)
        result = exc.value.result
        rules_found = {
            f["rule"] for f in result.details["vulnerabilities"]
        }
        assert "shell=True" in rules_found

    def test_hardcoded_password_detected(self) -> None:
        """FAIL: Hardcoded password detected.

        WHY: ``password = "s3cr3t"`` in source code is a
        credential leak.  LLMs generate placeholder passwords
        that end up in production.
        Expected: MltkAssertionError, finding with
        rule="hardcoded_password".
        """
        code = 'password = "s3cr3t_value"\n'
        with pytest.raises(MltkAssertionError) as exc:
            assert_no_code_vulnerabilities(code)
        result = exc.value.result
        rules_found = {
            f["rule"] for f in result.details["vulnerabilities"]
        }
        assert "hardcoded_password" in rules_found

    def test_multiple_vulnerabilities(self) -> None:
        """FAIL: Multiple different vulnerabilities in one file.

        WHY: LLM-generated code often has multiple issues.  The
        scanner must find ALL of them, not just the first.
        Expected: n_findings >= 3.
        """
        code = (
            "result = eval(user_input)\n"
            "exec(dynamic_code)\n"
            'password = "hunter2"\n'
        )
        with pytest.raises(MltkAssertionError) as exc:
            assert_no_code_vulnerabilities(code)
        result = exc.value.result
        assert result.details["n_findings"] >= 3

    def test_custom_rules(self) -> None:
        """PASS: Custom rules -- only check what is specified.

        WHY: Teams may want to check only specific patterns.
        Passing ``rules=["eval("]`` should ignore exec(), passwords,
        etc.
        Expected: passed=True when code has exec() but rules
        only check for eval().
        """
        code = "exec(code_string)\n"
        result = assert_no_code_vulnerabilities(
            code, rules=["eval("]
        )
        assert result.passed is True
        assert result.details["n_findings"] == 0

    def test_syntax_error_is_finding(self) -> None:
        """FAIL: Unparseable code is reported as a finding.

        WHY: Code with syntax errors cannot be statically analyzed.
        The scanner reports the syntax error as a vulnerability
        since the code may hide dangerous patterns behind the
        parse failure.
        Expected: MltkAssertionError, finding with
        rule="syntax_error".
        """
        code = "def foo(\n"
        with pytest.raises(MltkAssertionError) as exc:
            assert_no_code_vulnerabilities(code)
        result = exc.value.result
        rules_found = {
            f["rule"] for f in result.details["vulnerabilities"]
        }
        assert "syntax_error" in rules_found

    def test_os_system_detected(self) -> None:
        """FAIL: os.system() call detected.

        WHY: ``os.system(cmd)`` is command injection -- the shell
        interprets the entire string, including any injected
        commands.
        Expected: MltkAssertionError, finding with
        rule="os.system(".
        """
        code = "import os\nos.system('rm -rf /')\n"
        with pytest.raises(MltkAssertionError) as exc:
            assert_no_code_vulnerabilities(code)
        result = exc.value.result
        rules_found = {
            f["rule"] for f in result.details["vulnerabilities"]
        }
        assert "os.system(" in rules_found


# ------------------------------------------------------------------
# assert_code_complexity
# ------------------------------------------------------------------


class TestCodeComplexity:
    """Complexity tests -- is generated code maintainable?

    LLMs produce verbose, deeply nested code.  These tests verify
    that the complexity assertion correctly measures cyclomatic
    complexity and line count.
    """

    def test_simple_function_passes(self) -> None:
        """PASS: A simple function has low complexity.

        WHY: ``def add(a, b): return a + b`` has cyclomatic
        complexity of 1 (just the function entry, no branches).
        Expected: passed=True, max_function_complexity=1.
        """
        code = "def add(a, b):\n    return a + b\n"
        result = assert_code_complexity(
            code, max_cyclomatic=10, max_lines=200
        )
        assert result.passed is True
        assert result.details["max_function_complexity"] == 1
        assert result.details["per_function"]["add"] == 1
        assert result.name == "codegen.complexity"

    def test_deeply_nested_fails(self) -> None:
        """FAIL: A function with many branches exceeds threshold.

        WHY: A function with 12 if/elif branches has cyclomatic
        complexity of 13.  With max_cyclomatic=10, this should
        fail, signaling that the LLM-generated code needs
        refactoring.
        Expected: MltkAssertionError raised.
        """
        # Build a function with many branches
        lines = ["def complex_fn(x):"]
        for i in range(12):
            indent = "    "
            if i == 0:
                lines.append(f"{indent}if x == {i}:")
            else:
                lines.append(f"{indent}elif x == {i}:")
            lines.append(f"{indent}    return {i}")
        lines.append("    return -1")
        code = "\n".join(lines) + "\n"

        with pytest.raises(MltkAssertionError) as exc:
            assert_code_complexity(
                code, max_cyclomatic=10, max_lines=200
            )
        result = exc.value.result
        assert result.details["max_function_complexity"] > 10

    def test_too_many_lines_fails(self) -> None:
        """FAIL: Code exceeds line count threshold.

        WHY: 250 lines of generated code when the limit is 200
        means the LLM was overly verbose.  This signals that the
        prompt or model needs adjustment.
        Expected: MltkAssertionError raised.
        """
        lines = ["x = 1"] * 250
        code = "\n".join(lines) + "\n"
        with pytest.raises(MltkAssertionError) as exc:
            assert_code_complexity(
                code, max_cyclomatic=10, max_lines=200
            )
        result = exc.value.result
        assert result.details["total_lines"] > 200
        assert "lines" in result.message.lower()

    def test_no_functions_passes(self) -> None:
        """PASS: Code with no functions has zero complexity.

        WHY: Module-level code (just assignments, prints) has no
        function-level complexity to measure.  Zero <= any threshold.
        Expected: passed=True, max_function_complexity=0.
        """
        code = "x = 1\ny = 2\nprint(x + y)\n"
        result = assert_code_complexity(
            code, max_cyclomatic=10, max_lines=200
        )
        assert result.passed is True
        assert result.details["max_function_complexity"] == 0
        assert result.details["per_function"] == {}

    def test_multiple_functions(self) -> None:
        """PASS: Multiple functions -- worst complexity is reported.

        WHY: If one function has complexity 2 and another has
        complexity 5, the max is 5.  The per_function dict
        should contain both.
        Expected: max_function_complexity = max of all functions.
        """
        code = (
            "def simple(x):\n"
            "    return x + 1\n"
            "\n"
            "def branchy(x):\n"
            "    if x > 0:\n"
            "        if x > 10:\n"
            "            return 'big'\n"
            "        return 'small'\n"
            "    return 'negative'\n"
        )
        result = assert_code_complexity(
            code, max_cyclomatic=10, max_lines=200
        )
        assert result.passed is True
        pf = result.details["per_function"]
        assert "simple" in pf
        assert "branchy" in pf
        assert pf["simple"] == 1
        assert pf["branchy"] == 3  # 1 base + 2 if-nodes
        assert result.details["max_function_complexity"] == 3

    def test_syntax_error_still_checks_lines(self) -> None:
        """PASS: Code with syntax errors still has lines counted.

        WHY: If the code cannot be parsed, function complexity is
        zero (no functions detected).  But line count is still
        checked.  Short unparseable code should pass the line
        check.
        Expected: passed=True (complexity=0, lines within limit).
        """
        code = "def foo(\n"
        result = assert_code_complexity(
            code, max_cyclomatic=10, max_lines=200
        )
        assert result.passed is True
        assert result.details["max_function_complexity"] == 0
        assert result.details["total_lines"] == 1

    def test_for_and_while_add_complexity(self) -> None:
        """PASS: Loops contribute to cyclomatic complexity.

        WHY: ``for`` and ``while`` are decision points.  A
        function with one for-loop and one while-loop has
        complexity 3 (1 base + 1 for + 1 while).
        Expected: per_function["loopy"] == 3.
        """
        code = (
            "def loopy(items):\n"
            "    for item in items:\n"
            "        pass\n"
            "    while True:\n"
            "        break\n"
        )
        result = assert_code_complexity(
            code, max_cyclomatic=10, max_lines=200
        )
        assert result.passed is True
        assert result.details["per_function"]["loopy"] == 3

    def test_empty_code_passes(self) -> None:
        """PASS: Empty code has zero complexity and zero lines.

        WHY: Edge case -- empty string should not crash the
        assertion.
        Expected: passed=True.
        """
        result = assert_code_complexity(
            "", max_cyclomatic=10, max_lines=200
        )
        assert result.passed is True
        assert result.details["total_lines"] == 0
        assert result.details["max_function_complexity"] == 0
