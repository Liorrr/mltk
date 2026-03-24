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
| `func` | `Callable` | *(required)* | Inference function |
| `input_data` | `Any` | *(required)* | Sample input to pass to func |
| `input_schema` | `dict \| None` | `None` | JSON Schema for input validation |
| `output_schema` | `dict \| None` | `None` | JSON Schema for output validation |

---
