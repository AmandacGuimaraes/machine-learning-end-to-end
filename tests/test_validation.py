import pandas as pd
import pytest
from src.data.validation import validate_dataset

def test_valid_dataset():
    df = pd.DataFrame({"feature": range(20), "target": [0, 1] * 10})
    assert validate_dataset(df, "target")["rows"] == 20

def test_missing_target():
    with pytest.raises(ValueError):
        validate_dataset(pd.DataFrame({"x": range(20)}), "target")
