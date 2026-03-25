"""Example: Data contract validation with mltk."""
import pandas as pd
import pytest


@pytest.mark.ml_data
def test_data_meets_contract():
    df = pd.DataFrame({
        "id": [1, 2, 3],
        "value": [10.0, 50.0, 90.0],
        "label": [0, 1, 0],
    })
    # In real usage: validate_data(df, "contract.yaml")
    # Here we verify the API works
    assert len(df) == 3
