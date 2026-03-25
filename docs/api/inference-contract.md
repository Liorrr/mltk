# API Contract Testing

Contract testing validates that ML inference functions accept expected inputs and produce expected outputs. Catches training-serving skew where preprocessing differs between training and serving code.

**Module:** `mltk.inference.contract`

---

## assert_api_contract

Assert inference function input/output matches schema.

```python
from mltk.inference import assert_api_contract

output_schema = {"type": "object", "properties": {"prediction": {"type": "number"}}}
assert_api_contract(model.predict, input_data, output_schema=output_schema)
```

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `func` | `Callable` | *(required)* | Inference function to test |
| `input_data` | `Any` | *(required)* | Sample input to pass to func |
| `input_schema` | `dict \| None` | `None` | JSON Schema for input validation |
| `output_schema` | `dict \| None` | `None` | JSON Schema for output validation |

### Returns

`TestResult` with details:
- `errors` -- list of validation error strings
- `has_input_schema` -- whether input schema was provided
- `has_output_schema` -- whether output schema was provided

### How it works

1. If `input_schema` is provided, validates `input_data` against it
2. Calls `func(input_data)` -- if the function raises, the assertion fails immediately
3. If `output_schema` is provided, validates the return value against it

Uses `jsonschema.Draft7Validator` when `jsonschema` is installed. Falls back to basic type checking (type, required fields) when it is not.

### Example

```python
import pytest
from mltk.inference import assert_api_contract

@pytest.mark.ml_inference
def test_prediction_contract(model):
    """Verify model accepts and returns expected formats."""
    input_data = {"features": [1.0, 2.0, 3.0]}
    input_schema = {
        "type": "object",
        "properties": {"features": {"type": "array"}},
        "required": ["features"],
    }
    output_schema = {
        "type": "object",
        "properties": {"prediction": {"type": "number"}},
        "required": ["prediction"],
    }
    assert_api_contract(
        model.predict,
        input_data,
        input_schema=input_schema,
        output_schema=output_schema,
    )
```

### Edge Cases

- **Function raises**: If `func(input_data)` raises any exception, the assertion fails with `CRITICAL` severity and the exception type and message in the result.
- **No schemas**: If both `input_schema` and `output_schema` are `None`, the assertion only checks that `func(input_data)` does not raise.
- **jsonschema not installed**: Falls back to basic type checking (validates `type` and `required` fields only).

---
