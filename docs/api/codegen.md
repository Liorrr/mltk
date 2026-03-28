# Code Generation Evaluation

mltk provides **four assertions** for testing LLM-generated code: execution, test passing, vulnerability scanning, and complexity analysis.  These go beyond token-level metrics (BLEU, CodeBLEU) to evaluate whether generated code actually *works*, *is correct*, *is safe*, and *is maintainable*.

## Why Test Generated Code?

Every major AI coding tool (Copilot, Cursor, Claude, ChatGPT) generates code that *looks* correct but contains hidden defects:

| Failure Mode | What Happens | Which Assertion Catches It |
|---|---|---|
| **Runtime crash** | Undefined variables, import errors, type mismatches | `assert_code_executes` |
| **Wrong behavior** | Code runs but returns incorrect results | `assert_code_passes_tests` |
| **Security holes** | eval(), exec(), shell=True, hardcoded passwords | `assert_no_code_vulnerabilities` |
| **Unmaintainable** | Deeply nested spaghetti, 500-line functions | `assert_code_complexity` |

A generated function can have 95% CodeBLEU similarity to the reference while producing completely wrong output.  These assertions catch what similarity metrics cannot.

## Security Model

All code execution uses **subprocess isolation**:

- Code is written to a temporary file and executed as a **separate process**
- `subprocess.run()` with `shell=False` -- no shell injection possible
- Configurable **timeout** kills runaway processes (infinite loops, recursion)
- **No `eval()` or `exec()`** is ever used on the generated code
- stdout and stderr are captured and truncated to prevent memory issues

This means generated code cannot access the parent process memory, modify test state, or escape the execution sandbox.

---

## assert_code_executes

The most basic check -- does the generated code run without errors?

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `code` | `str` | required | Source code to execute |
| `timeout_seconds` | `float` | `10.0` | Max seconds before killing the process |
| `language` | `str` | `"python"` | Programming language (currently Python only) |

**Returns:** TestResult with `returncode`, `stdout`, `stderr`, `timeout_seconds`, `language`

**Example:**

```python
from mltk.domains.codegen import assert_code_executes

# Valid code -- passes
code = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

print(fibonacci(10))
"""
result = assert_code_executes(code, timeout_seconds=5.0)
assert result.passed
assert result.details["returncode"] == 0

# Broken code -- fails with MltkAssertionError
bad_code = "print(undefined_variable)"
# This raises MltkAssertionError because returncode != 0
```

**When to use:** As the first gate in any code generation pipeline.  If the code does not run, nothing else matters.

---

## assert_code_passes_tests

Code that runs is not necessarily correct.  This assertion combines generated code with test code and verifies all tests pass.

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `code` | `str` | required | Generated source code (functions, classes) |
| `test_code` | `str` | required | Test code with assert statements |
| `timeout_seconds` | `float` | `30.0` | Max seconds before killing the process |

**Returns:** TestResult with `returncode`, `stdout`, `stderr`, `timeout_seconds`

**Example:**

```python
from mltk.domains.codegen import assert_code_passes_tests

# LLM-generated code
code = """
def merge_sorted(a, b):
    result = []
    i = j = 0
    while i < len(a) and j < len(b):
        if a[i] <= b[j]:
            result.append(a[i])
            i += 1
        else:
            result.append(b[j])
            j += 1
    result.extend(a[i:])
    result.extend(b[j:])
    return result
"""

# Test suite that defines the contract
test_code = """
assert merge_sorted([], []) == []
assert merge_sorted([1, 3, 5], [2, 4, 6]) == [1, 2, 3, 4, 5, 6]
assert merge_sorted([1], []) == [1]
assert merge_sorted([], [1]) == [1]
assert merge_sorted([1, 1], [1, 1]) == [1, 1, 1, 1]
"""

result = assert_code_passes_tests(code, test_code)
assert result.passed
```

**When to use:** After `assert_code_executes` passes.  Execution without correctness is meaningless -- this is where you verify the LLM actually solved the problem.

---

## assert_no_code_vulnerabilities

Scans generated code for security anti-patterns without executing it.

**Default rules:**

| Rule | Pattern | Why It Is Dangerous |
|---|---|---|
| `eval(` | `eval(user_input)` | Arbitrary code execution -- attacker controls what runs |
| `exec(` | `exec(code_string)` | Arbitrary code execution via dynamic code |
| `__import__(` | `__import__('os')` | Bypasses import controls, loads arbitrary modules |
| `shell=True` | `subprocess.run(cmd, shell=True)` | Shell injection -- attacker injects commands via input |
| `os.system(` | `os.system('rm -rf /')` | Command injection via shell interpretation |
| `hardcoded_password` | `password = "s3cr3t"` | Credentials leaked in source code |

The scanner also reports **syntax errors** as findings, since unparseable code cannot be statically analyzed and may hide vulnerabilities.

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `code` | `str` | required | Source code to scan |
| `rules` | `list[str] \| None` | `None` (all defaults) | Custom list of patterns to check |

**Returns:** TestResult with `vulnerabilities` (list of `{rule, line, snippet}`), `n_findings`, `rules_checked`

**Example:**

```python
from mltk.domains.codegen import assert_no_code_vulnerabilities

# Safe code -- passes
safe_code = """
import json

def parse_config(path):
    with open(path) as f:
        return json.load(f)
"""
result = assert_no_code_vulnerabilities(safe_code)
assert result.passed
assert result.details["n_findings"] == 0

# Dangerous code -- fails
dangerous_code = """
import subprocess

def run_command(user_input):
    result = eval(user_input)
    subprocess.run(user_input, shell=True)
    return result
"""
# Raises MltkAssertionError -- eval() and shell=True detected

# Custom rules -- only check for specific patterns
result = assert_no_code_vulnerabilities(
    "exec(code)", rules=["eval("]
)
# Passes -- exec() is not in the custom rule list
```

**When to use:** Before deploying any LLM-generated code.  Security vulnerabilities in generated code are the highest-risk failure mode because the code *works* -- it just opens attack vectors.

---

## assert_code_complexity

Measures cyclomatic complexity per function and total line count.

**Cyclomatic complexity** counts decision points in a function: `if`, `elif`, `for`, `while`, `except`, `with`, `assert`, and boolean operators (`and`/`or`).  Each decision point adds a new execution path through the function.

| Complexity | Interpretation | Action |
|---|---|---|
| 1-5 | Simple, easy to test | No action needed |
| 6-10 | Moderate, manageable | Consider simplifying |
| 11-20 | Complex, hard to test | Refactor into smaller functions |
| 21+ | Untestable, error-prone | Must refactor before deployment |

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `code` | `str` | required | Source code to analyze |
| `max_cyclomatic` | `int` | `10` | Max complexity for any single function |
| `max_lines` | `int` | `200` | Max total lines of code |

**Returns:** TestResult with `max_function_complexity`, `max_cyclomatic`, `total_lines`, `max_lines`, `per_function` (dict of function name to complexity)

**Example:**

```python
from mltk.domains.codegen import assert_code_complexity

# Simple code -- passes
code = """
def add(a, b):
    return a + b

def is_positive(x):
    if x > 0:
        return True
    return False
"""
result = assert_code_complexity(code, max_cyclomatic=10, max_lines=200)
assert result.passed
assert result.details["per_function"]["add"] == 1
assert result.details["per_function"]["is_positive"] == 2

# LLM-generated spaghetti -- fails
spaghetti = """
def process(data):
    if data is None:
        return None
    elif isinstance(data, str):
        if len(data) > 100:
            if data.startswith("http"):
                # ... 15 more nested branches ...
                pass
    # ... continues for 300 lines
"""
# Raises MltkAssertionError -- complexity > 10 or lines > 200
```

**When to use:** As a quality gate for generated code.  High complexity means more bugs, harder testing, and painful maintenance.  If the LLM generates a 300-line function with complexity 25, the prompt needs refinement.

---

## Integration: CI/CD Pipeline for LLM-Generated Code

Use all four assertions together as a quality gate:

```python
from mltk.domains.codegen import (
    assert_code_executes,
    assert_code_passes_tests,
    assert_no_code_vulnerabilities,
    assert_code_complexity,
)


def evaluate_generated_code(code: str, test_code: str):
    """Full evaluation pipeline for LLM-generated code."""

    # Gate 1: Does it run?
    assert_code_executes(code, timeout_seconds=10.0)

    # Gate 2: Is it correct?
    assert_code_passes_tests(code, test_code, timeout_seconds=30.0)

    # Gate 3: Is it safe?
    assert_no_code_vulnerabilities(code)

    # Gate 4: Is it maintainable?
    assert_code_complexity(code, max_cyclomatic=10, max_lines=200)
```

In pytest:

```python
import pytest
from mltk.domains.codegen import (
    assert_code_executes,
    assert_code_passes_tests,
    assert_no_code_vulnerabilities,
    assert_code_complexity,
)


class TestLLMOutput:
    """Test suite for validating LLM-generated code quality."""

    def test_fibonacci_executes(self):
        code = generate_code("Write a fibonacci function")
        assert_code_executes(code)

    def test_fibonacci_correct(self):
        code = generate_code("Write a fibonacci function")
        tests = "assert fibonacci(10) == 55\nassert fibonacci(0) == 0"
        assert_code_passes_tests(code, tests)

    def test_fibonacci_secure(self):
        code = generate_code("Write a fibonacci function")
        assert_no_code_vulnerabilities(code)

    def test_fibonacci_maintainable(self):
        code = generate_code("Write a fibonacci function")
        assert_code_complexity(code, max_cyclomatic=5, max_lines=20)
```
