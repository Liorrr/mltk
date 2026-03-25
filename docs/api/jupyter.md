# Jupyter Notebook Integration

Rich HTML display for mltk test results in Jupyter notebooks. No extra install needed.

**Module:** `mltk.core.result` (built-in `_repr_html_` methods)

---

## Quick Start

```python
# In a Jupyter notebook cell:
from mltk.data import assert_schema, assert_no_nulls
import pandas as pd

df = pd.read_csv("data.csv")
result = assert_schema(df, {"id": "int64", "value": "float64"})
result  # Rich HTML display with pass/fail badge
```

## TestResult Display

Each `TestResult` renders as a colored card:
- Green badge for passed, red for failed
- Test name, message, duration
- Expandable details dict

## TestSuite Display

A `TestSuite` renders as a summary dashboard:
- Pass/fail counts with score percentage
- Per-test breakdown table
- Color-coded rows

## display_report

```python
from mltk.jupyter import display_report

suite = TestSuite()
# ... add results ...
display_report(suite)  # Renders full inline report
```

---
